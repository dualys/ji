use alloc::vec;
use alloc::vec::Vec;
use core::cell::{Cell, RefCell};
use jinshu::storage::BlockDevice;

/// Taille d'un secteur NVMe standard en octets.
pub const SECTOR_SIZE: usize = 512;

/// Nombre de secteurs NVMe de 512 octets composant un bloc `TrieNode` de 4096 octets.
pub const SECTORS_PER_BLOCK: u64 = 8;

/// Taille d'un bloc logique (`TrieNode` / `BTreeNode` / `JiSuperblock`) en octets.
pub const BLOCK_SIZE: usize = 4096;

/// Périphérique de blocs en mémoire implémentant [`jinshu::storage::BlockDevice`].
///
/// Permet d'exécuter et de tester `ji` en environnement `#![no_std]` (noyau Amentys
/// sur RAM disk ou tests sous Linux) avec la même sémantique LBA que le driver NVMe.
pub struct MemBlockDevice {
    blocks: RefCell<Vec<[u8; BLOCK_SIZE]>>,
    read_ops: Cell<u64>,
    write_ops: Cell<u64>,
}

impl MemBlockDevice {
    /// Crée un disque virtuel contenant `num_blocks` blocs de 4096 octets
    /// (soit `num_blocks * 8` secteurs LBA de 512 octets).
    #[must_use]
    pub fn new(num_blocks: usize) -> Self {
        Self {
            blocks: RefCell::new(vec![[0u8; BLOCK_SIZE]; num_blocks]),
            read_ops: Cell::new(0),
            write_ops: Cell::new(0),
        }
    }

    /// Retourne la capacité totale du disque en nombre de secteurs LBA (512 octets).
    #[must_use]
    pub fn total_lbas(&self) -> u64 {
        (self.blocks.borrow().len() as u64) * SECTORS_PER_BLOCK
    }

    /// Retourne le nombre d'opérations de lecture effectuées sur le périphérique.
    #[must_use]
    pub fn read_count(&self) -> u64 {
        self.read_ops.get()
    }

    /// Retourne le nombre d'opérations d'écriture effectuées sur le périphérique.
    #[must_use]
    pub fn write_count(&self) -> u64 {
        self.write_ops.get()
    }

    /// Lit directement un bloc de 4096 octets à l'adresse `lba` (multiple de 8).
    pub fn read_block(&self, lba: u64) -> Result<[u8; BLOCK_SIZE], &'static str> {
        if !lba.is_multiple_of(SECTORS_PER_BLOCK) {
            return Err("LBA non aligne sur 4096 octets (8 secteurs)");
        }
        let idx = usize::try_from(lba / SECTORS_PER_BLOCK).map_err(|_| "Index LBA invalide")?;
        let blocks = self.blocks.borrow();
        let block = blocks.get(idx).ok_or("LBA hors limites du disque")?;
        self.read_ops.set(self.read_ops.get() + 1);
        Ok(*block)
    }

    /// Écrit directement un bloc de 4096 octets à l'adresse `lba` (multiple de 8).
    pub fn write_block(&self, lba: u64, data: &[u8; BLOCK_SIZE]) -> Result<(), &'static str> {
        if !lba.is_multiple_of(SECTORS_PER_BLOCK) {
            return Err("LBA non aligne sur 4096 octets (8 secteurs)");
        }
        let idx = usize::try_from(lba / SECTORS_PER_BLOCK).map_err(|_| "Index LBA invalide")?;
        let mut blocks = self.blocks.borrow_mut();
        let slot = blocks.get_mut(idx).ok_or("LBA hors limites du disque")?;
        slot.copy_from_slice(data);
        self.write_ops.set(self.write_ops.get() + 1);
        Ok(())
    }
}

impl BlockDevice for MemBlockDevice {
    fn write_node(&self, lba: u64, data_phys_addr: u64) {
        let idx = (lba / SECTORS_PER_BLOCK) as usize;
        let mut blocks = self.blocks.borrow_mut();
        if let Some(slot) = blocks.get_mut(idx) {
            // SAFETY: `data_phys_addr` pointe vers un bloc valide de 4096 octets
            // (`TrieNode` ou `JiSuperblock`) fourni par `StorageEngine` ou `JiSuperblock`.
            let src =
                unsafe { core::slice::from_raw_parts(data_phys_addr as *const u8, BLOCK_SIZE) };
            slot.copy_from_slice(src);
            self.write_ops.set(self.write_ops.get() + 1);
        }
    }

    fn read_node(&self, lba: u64, dest_phys_addr: u64) {
        let idx = (lba / SECTORS_PER_BLOCK) as usize;
        let blocks = self.blocks.borrow();
        if let Some(slot) = blocks.get(idx) {
            // SAFETY: `dest_phys_addr` pointe vers un buffer de 4096 octets alloué par l'appelant.
            let dst =
                unsafe { core::slice::from_raw_parts_mut(dest_phys_addr as *mut u8, BLOCK_SIZE) };
            dst.copy_from_slice(slot);
            self.read_ops.set(self.read_ops.get() + 1);
        }
    }
}
