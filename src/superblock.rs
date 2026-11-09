use crate::commit::Checkpoint;
use crate::device::BLOCK_SIZE;
use jinshu::storage::BlockDevice;
use noun::Noun;

/// LBA réservé au Superbloc de `ji` (secteurs 0..7, avant le LBA 8 de `jinshu::DiskAllocator`).
pub const SUPERBLOCK_LBA: u64 = 0;

/// Premier LBA utilisé par `jinshu::storage::DiskAllocator` pour les `TrieNode`.
pub const FIRST_DATA_LBA: u64 = 8;

/// Signature magique du superbloc `ji` sur disque.
pub const SUPERBLOCK_MAGIC: [u8; 8] = *b"JIVCS001";

/// Version du format de superbloc.
pub const SUPERBLOCK_VERSION: u32 = 1;

/// Nombre de branches AWQ (aligné sur les 16 branches d'un `TrieNode` `jinshu` et `plans::Plan`).
pub const AWQ_BRANCH_COUNT: usize = 16;

/// Superbloc du VCS `ji` stocké dans les 4096 premiers octets du disque (LBA 0..7).
///
/// Il conserve l'état persistant des 16 branches AWQ, le pointeur d'allocation
/// append-only de `jinshu`, le dernier [`cubefs::Checkpoint`] scellé et une empreinte
/// BLAKE3 garantissant l'intégrité du superbloc au démarrage de l'OS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JiSuperblock {
    pub magic: [u8; 8],
    pub version: u32,
    pub active_branch: u8,
    pub head_lba: u64,
    pub commit_seq: u64,
    pub branch_heads: [Noun; AWQ_BRANCH_COUNT],
    pub latest_checkpoint: Checkpoint,
    pub checksum: [u8; 32],
}

impl Default for JiSuperblock {
    fn default() -> Self {
        Self::new()
    }
}

impl JiSuperblock {
    /// Crée un superbloc vierge initialisé au LBA 8 sur la branche AWQ 0.
    #[must_use]
    pub fn new() -> Self {
        let mut sb = Self {
            magic: SUPERBLOCK_MAGIC,
            version: SUPERBLOCK_VERSION,
            active_branch: 0,
            head_lba: FIRST_DATA_LBA,
            commit_seq: 0,
            branch_heads: core::array::from_fn(|_| Noun::null()),
            latest_checkpoint: Checkpoint {
                seq: 0,
                time: 0,
                root_meta_id: [0u8; 32],
                merkle_root: [0u8; 32],
                signature: [0u8; 64],
            },
            checksum: [0u8; 32],
        };
        sb.seal();
        sb
    }

    /// Sérialise les champs de données du superbloc (hors `checksum`) dans les 720 premiers octets.
    fn write_payload_bytes(&self, buf: &mut [u8; BLOCK_SIZE]) {
        buf.fill(0);
        let mut pos = 0;

        buf[pos..pos + 8].copy_from_slice(&self.magic);
        pos += 8;

        buf[pos..pos + 4].copy_from_slice(&self.version.to_le_bytes());
        pos += 4;

        buf[pos] = self.active_branch;
        pos += 4; // 1 octet + 3 octets d'alignement

        buf[pos..pos + 8].copy_from_slice(&self.head_lba.to_le_bytes());
        pos += 8;

        buf[pos..pos + 8].copy_from_slice(&self.commit_seq.to_le_bytes());
        pos += 8;

        for branch_head in &self.branch_heads {
            buf[pos..pos + 32].copy_from_slice(branch_head.as_bytes());
            pos += 32;
        }

        buf[pos..pos + 8].copy_from_slice(&self.latest_checkpoint.seq.to_le_bytes());
        pos += 8;
        buf[pos..pos + 8].copy_from_slice(&self.latest_checkpoint.time.to_le_bytes());
        pos += 8;
        buf[pos..pos + 32].copy_from_slice(&self.latest_checkpoint.root_meta_id);
        pos += 32;
        buf[pos..pos + 32].copy_from_slice(&self.latest_checkpoint.merkle_root);
        pos += 32;
        buf[pos..pos + 64].copy_from_slice(&self.latest_checkpoint.signature);
    }

    /// Calcule l'empreinte BLAKE3 (`Noun`) des 4064 premiers octets du bloc superbloc.
    #[must_use]
    pub fn compute_checksum(&self) -> [u8; 32] {
        let mut block = [0u8; BLOCK_SIZE];
        self.write_payload_bytes(&mut block);
        *Noun::of(&block[..BLOCK_SIZE - 32]).as_bytes()
    }

    /// Recalcule et met à jour le `checksum` BLAKE3 du superbloc.
    pub fn seal(&mut self) {
        self.checksum = self.compute_checksum();
    }

    /// Vérifie la signature magique, la version, la branche AWQ et le checksum BLAKE3.
    pub fn verify(&self) -> Result<(), &'static str> {
        if self.magic != SUPERBLOCK_MAGIC {
            return Err("Superbloc invalide : magic JIVCS001 absent");
        }
        if self.version != SUPERBLOCK_VERSION {
            return Err("Superbloc invalide : version non supportee");
        }
        if self.active_branch as usize >= AWQ_BRANCH_COUNT {
            return Err("Superbloc invalide : branche AWQ > 15");
        }
        if self.head_lba < FIRST_DATA_LBA || !self.head_lba.is_multiple_of(8) {
            return Err("Superbloc invalide : head_lba corrompu");
        }
        if self.compute_checksum() != self.checksum {
            return Err("Superbloc corrompu : checksum BLAKE3 invalide");
        }
        Ok(())
    }

    /// Encode le superbloc scellé dans une page disque de 4096 octets.
    #[must_use]
    pub fn to_block(&mut self) -> [u8; BLOCK_SIZE] {
        self.seal();
        let mut block = [0u8; BLOCK_SIZE];
        self.write_payload_bytes(&mut block);
        block[BLOCK_SIZE - 32..].copy_from_slice(&self.checksum);
        block
    }

    /// Décode et vérifie un `JiSuperblock` depuis une page disque de 4096 octets.
    pub fn from_block(block: &[u8; BLOCK_SIZE]) -> Result<Self, &'static str> {
        let mut pos = 0;

        let mut magic = [0u8; 8];
        magic.copy_from_slice(&block[pos..pos + 8]);
        pos += 8;

        let version = u32::from_le_bytes([block[pos], block[pos + 1], block[pos + 2], block[pos + 3]]);
        pos += 4;

        let active_branch = block[pos];
        pos += 4;

        let head_lba = u64::from_le_bytes(
            block[pos..pos + 8]
                .try_into()
                .map_err(|_| "Lecture head_lba")?,
        );
        pos += 8;

        let commit_seq = u64::from_le_bytes(
            block[pos..pos + 8]
                .try_into()
                .map_err(|_| "Lecture commit_seq")?,
        );
        pos += 8;

        let mut branch_heads = core::array::from_fn(|_| Noun::null());
        for slot in &mut branch_heads {
            *slot = Noun::from_bytes(&block[pos..pos + 32]).ok_or("Noun de branche invalide")?;
            pos += 32;
        }

        let seq = u64::from_le_bytes(
            block[pos..pos + 8]
                .try_into()
                .map_err(|_| "Lecture checkpoint.seq")?,
        );
        pos += 8;
        let time = u64::from_le_bytes(
            block[pos..pos + 8]
                .try_into()
                .map_err(|_| "Lecture checkpoint.time")?,
        );
        pos += 8;
        let mut root_meta_id = [0u8; 32];
        root_meta_id.copy_from_slice(&block[pos..pos + 32]);
        pos += 32;
        let mut merkle_root = [0u8; 32];
        merkle_root.copy_from_slice(&block[pos..pos + 32]);
        pos += 32;
        let mut signature = [0u8; 64];
        signature.copy_from_slice(&block[pos..pos + 64]);

        let mut checksum = [0u8; 32];
        checksum.copy_from_slice(&block[BLOCK_SIZE - 32..]);

        let sb = Self {
            magic,
            version,
            active_branch,
            head_lba,
            commit_seq,
            branch_heads,
            latest_checkpoint: Checkpoint {
                seq,
                time,
                root_meta_id,
                merkle_root,
                signature,
            },
            checksum,
        };
        sb.verify()?;
        Ok(sb)
    }

    /// Scelle et écrit le superbloc au LBA 0 du périphérique `device`.
    pub fn write_to_device<D: BlockDevice>(&mut self, device: &D) {
        let block = self.to_block();
        let addr = core::ptr::from_ref(&block) as u64;
        device.write_node(SUPERBLOCK_LBA, addr);
    }

    /// Lit et vérifie le superbloc depuis le LBA 0 du périphérique `device`.
    pub fn read_from_device<D: BlockDevice>(device: &D) -> Result<Self, &'static str> {
        let mut block = [0u8; BLOCK_SIZE];
        let addr = core::ptr::from_mut(&mut block) as u64;
        device.read_node(SUPERBLOCK_LBA, addr);
        Self::from_block(&block)
    }
}
