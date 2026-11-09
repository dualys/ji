use crate::commit::{Checkpoint, JiCommit};
use crate::superblock::{AWQ_BRANCH_COUNT, FIRST_DATA_LBA, JiSuperblock};
use crate::tree::{
    DirRecord, FLAG_BLOB, FLAG_DIR, FLAG_INODE, MAX_INODE_CHUNKS, MAX_MULTI_CHUNK_FILE_SIZE,
    TRIE_PAYLOAD_CAPACITY, VAL_TYPE_DIRECTORY, VAL_TYPE_INODE, blob_logical_len, compute_node_noun,
    lookup_dir_entry, make_blob_node, make_delta_blob_node, make_dict_blob_node,
    make_empty_dir_node, make_multi_chunk_inode_node, make_rs_blob_node, make_seed_blob_node,
    make_xor_key_blob_node, make_xor_ref_blob_node, nucleate_page_byte, read_dir_entries,
    read_inode_extents, split_path, upsert_dir_entry, write_dir_entries,
};
use alloc::collections::BTreeMap;
use alloc::vec;
use heapless::{String, Vec};
use jinshu::node::TrieNode;
use jinshu::ocean::CoreOceanBuilder;
use jinshu::router::{NounIndex, SemanticRouter};
use jinshu::storage::{BlockDevice, StorageEngine};
use kpack::Opcode;
use noun::Noun;
use plans::Plan;
use plans::layer::{Capabilities, Layer, Layers};
use plans::sceau::{Sceau, Verdict};
use wuu::StackString;

/// État d'un OS monté via `ji mount <commit_noun>`.
///
/// Contient le [`JiCommit`] restauré, le [`Plan`] d'exécution reconstruit avec ses [`Layer`]s,
/// le [`Sceau`] cryptographique et le [`Verdict`] de pesée (`Verdict::Accept`).
#[derive(Debug, Clone)]
pub struct MountedOs {
    pub commit_noun: Noun,
    pub commit: JiCommit,
    pub plan: Plan,
    pub sceau: Sceau,
    pub verdict: Verdict,
    pub branch: u8,
}

/// Moteur principal du VCS `ji` orchestrant `noun`, `kpack`, `cubefs`, `jinshu` et `plans`.
pub struct JiEngine<'a, D: BlockDevice> {
    pub superblock: JiSuperblock,
    pub storage: StorageEngine<'a, D>,
    pub disk_index: NounIndex,
    pub core_ocean: BTreeMap<Noun, TrieNode>,
    pub virtual_ocean: BTreeMap<Noun, TrieNode>,
    pub working_root: Noun,
    pub phoenix: Layers,
    pub active_plan: Plan,
    branch_roots: [Noun; AWQ_BRANCH_COUNT],
}

impl<'a, D: BlockDevice> JiEngine<'a, D> {
    /// Initialise un nouveau dépôt/disque VCS `ji` (`ji init`) sur le périphérique `device`.
    ///
    /// 1. Crée un `TrieNode` répertoire racine (`FLAG_DIR`) vide et le persiste au LBA 8.
    /// 2. Initialise la couche fondamentale `phoenix` et le [`Plan`] sur la branche AWQ `0`.
    /// 3. Crée le premier commit d'initialisation (`seq = 1`) scellé par un [`Sceau`] et un [`Checkpoint`].
    /// 4. Écrit le [`JiSuperblock`] au LBA `0`.
    pub fn init(
        device: &'a D,
        total_lbas: u64,
        phoenix_layer: Layer,
    ) -> Result<Self, &'static str> {
        let mut storage = StorageEngine::new(total_lbas, device);
        let mut disk_index = NounIndex::new();
        let mut core_ocean = BTreeMap::new();

        // 1. Création et persistance du répertoire racine vide au LBA 8
        let root_dir = make_empty_dir_node();
        let root_noun = compute_node_noun(&root_dir);
        let root_addr = core::ptr::from_ref(&root_dir) as u64;
        let lba = storage.persist_node(root_addr)?;
        disk_index.insert(root_noun.clone(), lba)?;
        core_ocean.insert(root_noun.clone(), root_dir);

        // 2. Initialisation de Phoenix et du Plan sur la branche 0
        let mut phoenix = Layers::new(root_noun.clone());
        phoenix.add_layer(phoenix_layer.clone())?;
        let mut active_plan = Plan::new(root_noun.clone(), &mut phoenix, 0)?;
        active_plan.add_layer(phoenix_layer)?;

        let superblock = JiSuperblock::new();
        let branch_roots = core::array::from_fn(|_| root_noun.clone());

        let mut engine = Self {
            superblock,
            storage,
            disk_index,
            core_ocean,
            virtual_ocean: BTreeMap::new(),
            working_root: root_noun,
            phoenix,
            active_plan,
            branch_roots,
        };

        // 3. Premier commit fondateur (Genesis) + persistance du Superbloc au LBA 0
        engine.commit("genesis: initial OS state", 1, 1000)?;
        Ok(engine)
    }

    /// Rouvre à froid un disque VCS `ji` déjà initialisé (au redémarrage de l'OS) :
    ///
    /// 1. Relit et vérifie le [`JiSuperblock`] au LBA `0` (magic, version, bornes et checksum BLAKE3).
    /// 2. Scanne les blocs de `FIRST_DATA_LBA` (`8`) jusqu'à `superblock.head_lba` pour reconstruire
    ///    automatiquement le [`NounIndex`] (`disk_index`) et l'état du `DiskAllocator` `jinshu`.
    /// 3. Monte automatiquement le commit `branch_heads[active_branch]` (préchargement Merkle dans
    ///    `core_ocean`, désérialisation des [`Layer`]s depuis le disque et vérification [`Sceau::weigh`]).
    pub fn open(device: &'a D, total_lbas: u64) -> Result<Self, &'static str> {
        let superblock = JiSuperblock::read_from_device(device)?;
        if superblock.head_lba > total_lbas {
            return Err("Superbloc invalide : head_lba depasse la capacite du disque");
        }

        let active_branch = superblock.active_branch;
        let active_commit_noun = superblock.branch_heads[active_branch as usize].clone();
        if active_commit_noun.is_null() {
            return Err("Superbloc invalide : aucun commit sur la branche AWQ active");
        }

        // 1. Reconstruction du DiskAllocator et du NounIndex par scan séquentiel des LBAs alloués
        let mut storage = StorageEngine::new(total_lbas, device);
        let mut disk_index = NounIndex::new();
        let mut scratch_node = TrieNode::new();
        let scratch_addr = core::ptr::from_mut(&mut scratch_node) as u64;

        let mut lba = FIRST_DATA_LBA;
        while lba < superblock.head_lba {
            let allocated_lba = storage
                .allocator
                .allocate_node()
                .ok_or("Capacite disque insuffisante lors de la reconstruction du DiskAllocator")?;
            if allocated_lba != lba {
                return Err("Desynchronisation LBA lors de la reconstruction du DiskAllocator");
            }
            storage.fetch_node(lba, scratch_addr)?;
            let noun = compute_node_noun(&scratch_node);
            disk_index.insert(noun, lba)?;
            lba += 8;
        }

        // 2. Lecture et validation du commit actif (Checkpoint + Sceau + Layers désérialisées)
        let active_commit_lba = disk_index
            .get_lba(&active_commit_noun)
            .ok_or("Commit actif introuvable dans le NounIndex reconstruit")?;
        storage.fetch_node(active_commit_lba, scratch_addr)?;
        let active_commit = JiCommit::from_trie_node(&scratch_node)?;

        let mut core_ocean = BTreeMap::new();
        core_ocean.insert(active_commit_noun.clone(), scratch_node.clone());

        let base_layer = active_commit
            .layers
            .first()
            .cloned()
            .unwrap_or_else(Self::default_kernel_layer);
        let mut phoenix = Layers::new(active_commit.root_tree.clone());
        phoenix.add_layer(base_layer)?;

        let active_plan = Plan::new(
            active_commit.root_tree.clone(),
            &mut phoenix,
            active_branch,
        )?;

        let mut branch_roots = core::array::from_fn(|_| active_commit.root_tree.clone());
        for (b, head_noun) in superblock.branch_heads.iter().enumerate() {
            if !head_noun.is_null()
                && let Some(b_lba) = disk_index.get_lba(head_noun)
            {
                storage.fetch_node(b_lba, scratch_addr)?;
                if let Ok(b_commit) = JiCommit::from_trie_node(&scratch_node) {
                    branch_roots[b] = b_commit.root_tree;
                }
            }
        }

        let mut engine = Self {
            superblock,
            storage,
            disk_index,
            core_ocean,
            virtual_ocean: BTreeMap::new(),
            working_root: active_commit.root_tree,
            phoenix,
            active_plan,
            branch_roots,
        };

        // 3. Montage complet du commit actif (CoreOceanBuilder::deep_load + pesée Sceau::weigh)
        engine.mount(&active_commit_noun, Some(active_branch))?;
        Ok(engine)
    }

    /// Exécute une opération sur un [`SemanticRouter`] `jinshu` connecté aux océans du moteur.
    fn with_router<R, F>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut SemanticRouter<'_, D>) -> R,
    {
        let device = self.storage.device;
        let allocator = core::mem::replace(
            &mut self.storage.allocator,
            jinshu::storage::DiskAllocator::new(0),
        );
        let storage = StorageEngine { allocator, device };
        let disk_index = core::mem::take(&mut self.disk_index);
        let virtual_ocean = core::mem::take(&mut self.virtual_ocean);

        let mut router = SemanticRouter {
            disk_storage: storage,
            disk_index,
            core_ocean: &self.core_ocean,
            virtual_ocean,
        };

        let result = f(&mut router);

        self.storage.allocator = router.disk_storage.allocator;
        self.disk_index = router.disk_index;
        self.virtual_ocean = router.virtual_ocean;
        result
    }

    /// Récupère un `TrieNode` en cascade (`virtual_ocean` -> `core_ocean` -> `disk_storage`)
    /// via [`SemanticRouter::fetch_node_cascade`].
    pub fn fetch_node(&mut self, noun: &Noun) -> Result<TrieNode, &'static str> {
        let mut dest = TrieNode::new();
        let addr = core::ptr::from_mut(&mut dest) as u64;
        self.with_router(|router| router.fetch_node_cascade(noun, addr, addr))?;
        Ok(dest)
    }

    /// Lecture sans mutation via `virtual_ocean`, `core_ocean` ou `disk_storage`.
    pub fn fetch_node_ref(&self, noun: &Noun) -> Result<TrieNode, &'static str> {
        if let Some(node) = self.virtual_ocean.get(noun) {
            return Ok(node.clone());
        }
        if let Some(node) = self.core_ocean.get(noun) {
            return Ok(node.clone());
        }
        let lba = self
            .disk_index
            .get_lba(noun)
            .ok_or("Erreur critique : Noun introuvable dans tous les oceans")?;
        let mut dest = TrieNode::new();
        let addr = core::ptr::from_mut(&mut dest) as u64;
        self.storage.fetch_node(lba, addr)?;
        Ok(dest)
    }

    /// Résout un chemin (plat ou hiérarchique avec `/`) depuis une racine d'arbre `root_tree`
    /// et retourne `(val_type, target_noun)`.
    pub fn resolve_entry_in_tree(
        &self,
        root_tree: &Noun,
        path: &str,
    ) -> Result<(u8, Noun), &'static str> {
        let parts = split_path(path)?;
        let mut current_dir_noun = root_tree.clone();

        for (i, &segment) in parts.iter().enumerate() {
            let dir_node = self.fetch_node_ref(&current_dir_noun)?;
            if dir_node.flags != FLAG_DIR {
                return Err("Le noeud intermediaire n'est pas un repertoire CubeFS (FLAG_DIR)");
            }
            let entry =
                lookup_dir_entry(&dir_node, segment).ok_or("Fichier introuvable dans l'arbre")?;
            let target_noun =
                Noun::from_bytes(&entry.val_data).ok_or("Noun d'entree CubeFS invalide")?;

            if i + 1 == parts.len() {
                return Ok((entry.val_type, target_noun));
            }
            if entry.val_type != VAL_TYPE_DIRECTORY {
                return Err("Un segment parent du chemin n'est pas un repertoire");
            }
            current_dir_noun = target_noun;
        }

        Err("Chemin introuvable")
    }

    /// Insère ou met à jour récursivement une entrée à l'adresse `segments` dans le sous-arbre `dir_noun`,
    /// en créant automatiquement les sous-répertoires intermédiaires (`FLAG_DIR`) en Copy-on-Write.
    fn upsert_path_segments(
        &mut self,
        dir_noun: &Noun,
        segments: &[&str],
        val_type: u8,
        target: Noun,
    ) -> Result<Noun, &'static str> {
        let current_dir = self.fetch_node(dir_noun)?;
        if current_dir.flags != FLAG_DIR {
            return Err("Noeud repertoire invalide (FLAG_DIR attendu)");
        }

        if segments.len() == 1 {
            let updated_dir = upsert_dir_entry(&current_dir, segments[0], val_type, target)?;
            return Ok(self.with_router(|router| router.write_virtual_node(updated_dir)));
        }

        let head = segments[0];
        let child_dir_noun = if let Some(entry) = lookup_dir_entry(&current_dir, head) {
            if entry.val_type != VAL_TYPE_DIRECTORY {
                return Err("Conflit de chemin : une entree fichier porte deja le nom du dossier");
            }
            Noun::from_bytes(&entry.val_data).ok_or("Noun de sous-repertoire invalide")?
        } else {
            let empty_sub = make_empty_dir_node();
            self.with_router(|router| router.write_virtual_node(empty_sub))
        };

        let new_child_noun =
            self.upsert_path_segments(&child_dir_noun, &segments[1..], val_type, target)?;
        let updated_dir =
            upsert_dir_entry(&current_dir, head, VAL_TYPE_DIRECTORY, new_child_noun)?;
        Ok(self.with_router(|router| router.write_virtual_node(updated_dir)))
    }

    /// Met à jour un chemin `path` dans `working_root` et synchronise `active_plan.directory`.
    fn set_working_path_entry(
        &mut self,
        path: &str,
        val_type: u8,
        target: Noun,
    ) -> Result<Noun, &'static str> {
        let parts = split_path(path)?;
        let working_root = self.working_root.clone();
        let new_root = self.upsert_path_segments(&working_root, &parts, val_type, target.clone())?;
        self.working_root = new_root;
        self.active_plan.directory = self.working_root.clone();
        Ok(target)
    }

    /// Crée un sous-répertoire CubeFS (`ji mkdir <path>`) et tous ses répertoires parents en Copy-on-Write.
    pub fn mkdir(&mut self, path: &str) -> Result<Noun, &'static str> {
        let empty_dir = make_empty_dir_node();
        let dir_noun = self.with_router(|router| router.write_virtual_node(empty_dir));
        self.set_working_path_entry(path, VAL_TYPE_DIRECTORY, dir_noun)
    }

    /// Écrit un fichier dans le `virtual_ocean` (Copy-on-Write en RAM) :
    /// - Si `data.len() <= 3576` (`TRIE_PAYLOAD_CAPACITY`) : stocké dans un seul `TrieNode` `FLAG_BLOB` (`Lit` ou `Rle`).
    /// - Si `data.len() > 3576` (jusqu'à `57 216` octets) : découpé automatiquement en chunks `FLAG_BLOB` et
    ///   indexé par un `TrieNode` `FLAG_INODE` contenant un [`cubefs::Inode`] et ses [`cubefs::Extent`]s.
    /// - Supporte les chemins hiérarchiques avec sous-répertoires (ex. `"/boot/kernel.sys"`).
    pub fn stage_file(&mut self, path: &str, data: &[u8]) -> Result<Noun, &'static str> {
        if data.len() > MAX_MULTI_CHUNK_FILE_SIZE {
            return Err("Fichier depasse la capacite maximale d'un Inode multi-chunks (57216 octets)");
        }

        let target_noun = if data.len() <= TRIE_PAYLOAD_CAPACITY {
            let blob_node = make_blob_node(data)?;
            self.with_router(|router| router.write_virtual_node(blob_node))
        } else {
            let mut chunk_nouns: Vec<Noun, MAX_INODE_CHUNKS> = Vec::new();
            let mut chunk_lengths: Vec<u32, MAX_INODE_CHUNKS> = Vec::new();

            for slice in data.chunks(TRIE_PAYLOAD_CAPACITY) {
                let chunk_node = make_blob_node(slice)?;
                let chunk_noun = self.with_router(|router| router.write_virtual_node(chunk_node));
                chunk_nouns
                    .push(chunk_noun)
                    .map_err(|_| "Trop de chunks pour un Inode")?;
                chunk_lengths
                    .push(u32::try_from(slice.len()).unwrap_or(0))
                    .map_err(|_| "Trop de chunks pour un Inode")?;
            }

            let inode_node = make_multi_chunk_inode_node(
                chunk_nouns.as_slice(),
                chunk_lengths.as_slice(),
                data.len() as u64,
            )?;
            self.with_router(|router| router.write_virtual_node(inode_node))
        };

        self.set_working_path_entry(path, VAL_TYPE_INODE, target_noun)
    }

    /// Écrit une nouvelle version d'un fichier dans le `virtual_ocean` en l'encodant sous forme de
    /// recette delta [`kpack::Opcode::Delta`] par rapport à sa version précédente dans l'arbre courant.
    pub fn stage_file_delta(&mut self, path: &str, new_data: &[u8]) -> Result<Noun, &'static str> {
        let (val_type, base_noun) = self
            .resolve_entry_in_tree(&self.working_root, path)
            .map_err(|_| "Fichier parent introuvable pour l'encodage Delta")?;
        if val_type != VAL_TYPE_INODE {
            return Err("La cible pour l'encodage Delta n'est pas un fichier");
        }

        let mut base_buf = [0u8; TRIE_PAYLOAD_CAPACITY];
        let base_len = self.read_blob(&base_noun, &mut base_buf)?;

        let delta_node = make_delta_blob_node(base_noun, &base_buf[..base_len], new_data)?;
        let delta_noun = self.with_router(|router| router.write_virtual_node(delta_node));

        self.set_working_path_entry(path, VAL_TYPE_INODE, delta_noun)
    }

    /// Écrit un fichier dans le `virtual_ocean` en le compressant par dictionnaire [`kpack::Opcode::Dict`]
    /// vis-à-vis d'un fichier dictionnaire `dict_path` déjà présent dans l'arbre de travail.
    pub fn stage_file_dict(
        &mut self,
        path: &str,
        dict_path: &str,
        word_size: usize,
        data: &[u8],
    ) -> Result<Noun, &'static str> {
        let (_, dict_noun) = self
            .resolve_entry_in_tree(&self.working_root, dict_path)
            .map_err(|_| "Fichier dictionnaire de reference introuvable dans l'arbre")?;
        let mut dict_buf = [0u8; TRIE_PAYLOAD_CAPACITY];
        let dict_len = self.read_blob(&dict_noun, &mut dict_buf)?;

        let dict_node = make_dict_blob_node(dict_noun, &dict_buf[..dict_len], word_size, data)?;
        let node_noun = self.with_router(|router| router.write_virtual_node(dict_node));
        self.set_working_path_entry(path, VAL_TYPE_INODE, node_noun)
    }

    /// Écrit un fichier encodé par [`kpack::Opcode::Xor`] avec une clé constante `xor_key`.
    pub fn stage_file_xor_key(
        &mut self,
        path: &str,
        xor_key: u8,
        data: &[u8],
    ) -> Result<Noun, &'static str> {
        let xor_node = make_xor_key_blob_node(xor_key, data)?;
        let xor_noun = self.with_router(|router| router.write_virtual_node(xor_node));
        self.set_working_path_entry(path, VAL_TYPE_INODE, xor_noun)
    }

    /// Écrit une nouvelle version d'un fichier encodée par [`kpack::Opcode::Xor`] différentiel
    /// vis-à-vis de la version précédente du fichier (référencée en branche `0`).
    pub fn stage_file_xor_delta(
        &mut self,
        path: &str,
        new_data: &[u8],
    ) -> Result<Noun, &'static str> {
        let (_, base_noun) = self
            .resolve_entry_in_tree(&self.working_root, path)
            .map_err(|_| "Fichier parent introuvable pour l'encodage Xor differentiel")?;
        let mut base_buf = [0u8; TRIE_PAYLOAD_CAPACITY];
        let base_len = self.read_blob(&base_noun, &mut base_buf)?;

        let xor_node = make_xor_ref_blob_node(base_noun, &base_buf[..base_len], new_data)?;
        let xor_noun = self.with_router(|router| router.write_virtual_node(xor_node));
        self.set_working_path_entry(path, VAL_TYPE_INODE, xor_noun)
    }

    /// Écrit un fichier encodé par formule générative [`kpack::Opcode::Seed`] (`xorshift32` + patchs épars).
    pub fn stage_file_seed(
        &mut self,
        path: &str,
        seed: u32,
        data: &[u8],
    ) -> Result<Noun, &'static str> {
        let seed_node = make_seed_blob_node(seed, data)?;
        let seed_noun = self.with_router(|router| router.write_virtual_node(seed_node));
        self.set_working_path_entry(path, VAL_TYPE_INODE, seed_noun)
    }

    /// Écrit un fichier encodé par décalage bit-à-bit [`kpack::Opcode::Rs`].
    pub fn stage_file_rs(
        &mut self,
        path: &str,
        shift: u8,
        data: &[u8],
    ) -> Result<Noun, &'static str> {
        let rs_node = make_rs_blob_node(shift, data)?;
        let rs_noun = self.with_router(|router| router.write_virtual_node(rs_node));
        self.set_working_path_entry(path, VAL_TYPE_INODE, rs_noun)
    }

    /// Applique une mutation atomique Copy-on-Write d'un octet via [`cubefs::Nucleator::insert_byte`]
    /// sur le fichier `path` et retourne `(nouveau_noun, octet_ejecte)`.
    pub fn nucleate_working_file_byte(
        &mut self,
        path: &str,
        offset: usize,
        new_byte: u8,
    ) -> Result<(Noun, u8), &'static str> {
        let (_, current_noun) = self.resolve_entry_in_tree(&self.working_root, path)?;
        let mut page = [0u8; 4096];
        let current_len = self.read_blob(&current_noun, &mut page)?;
        if offset > current_len || offset >= 4096 {
            return Err("Offset hors bornes pour Nucleator::insert_byte");
        }

        let (mutated_page, ejected_byte) = nucleate_page_byte(&page, offset, new_byte)
            .ok_or("Echec de Nucleator::insert_byte")?;
        let new_len = (current_len + 1).min(4096);
        let new_noun = self.stage_file(path, &mutated_page[..new_len])?;
        Ok((new_noun, ejected_byte))
    }

    /// Reconstruit le contenu d'un nœud de fichier (`FLAG_BLOB` ou `FLAG_INODE` multi-chunks)
    /// en exécutant ses recettes `kpack` (`Lit`, `Rle`, `Delta`, `Dict`, `Xor`, `Seed`, `Rs`).
    pub fn read_blob(&self, blob_noun: &Noun, output: &mut [u8]) -> Result<usize, &'static str> {
        let node = self.fetch_node_ref(blob_noun)?;

        if node.flags == FLAG_INODE {
            let (_inode, extents) = read_inode_extents(&node)?;
            let mut cursor = 0usize;
            for ext in &extents {
                if cursor >= output.len() {
                    break;
                }
                let chunk_noun =
                    Noun::from_bytes(&ext.first_chunk_id).ok_or("Noun de chunk Extent invalide")?;
                let remaining = output.len() - cursor;
                let max_chunk = (ext.last_len as usize).min(remaining);
                let read_n =
                    self.read_blob(&chunk_noun, &mut output[cursor..cursor + max_chunk])?;
                cursor += read_n;
            }
            return Ok(cursor);
        }

        if node.flags != FLAG_BLOB {
            return Err("Le Noun demande n'est ni un FLAG_BLOB ni un FLAG_INODE");
        }

        let target_len = blob_logical_len(&node).min(output.len());
        output[..target_len].fill(0);

        let needs_ref = matches!(
            Opcode::from_u8(node.opcode),
            Some(Opcode::Delta | Opcode::Dict)
        ) || (node.opcode == Opcode::Xor as u8 && node.has_branch(0));

        if needs_ref {
            let base_noun = node
                .get_branch(0)
                .ok_or("Blob kpack corrompu : branche 0 (reference parente) absente")?;
            let mut ref_buf = [0u8; TRIE_PAYLOAD_CAPACITY];
            let ref_len = self.read_blob(base_noun, &mut ref_buf)?;
            node.execute_recipe(&mut output[..target_len], Some(&ref_buf[..ref_len]));
        } else {
            node.execute_recipe(&mut output[..target_len], None);
        }

        Ok(target_len)
    }

    /// Lit un fichier (chemin plat ou avec sous-répertoires `/`) depuis n'importe quelle racine d'arbre `root_tree`.
    pub fn read_file(
        &self,
        root_tree: &Noun,
        path: &str,
        output: &mut [u8],
    ) -> Result<usize, &'static str> {
        let (val_type, blob_noun) = self.resolve_entry_in_tree(root_tree, path)?;
        if val_type != VAL_TYPE_INODE {
            return Err("Le chemin demande designe un repertoire et non un fichier");
        }
        self.read_blob(&blob_noun, output)
    }

    /// Lit un fichier dans l'espace de travail courant (`working_root`).
    pub fn read_working_file(&self, path: &str, output: &mut [u8]) -> Result<usize, &'static str> {
        self.read_file(&self.working_root, path, output)
    }

    /// Ajoute une couche système [`Layer`] (`plans`) au [`Plan`] actif avant le prochain commit.
    pub fn add_os_layer(&mut self, layer: Layer) -> Result<(), &'static str> {
        self.active_plan.add_layer(layer)
    }

    /// Retourne le nombre de nœuds modifiés en attente dans le `virtual_ocean` (staging area).
    #[must_use]
    pub fn staged_node_count(&self) -> usize {
        self.virtual_ocean.len()
    }

    /// Persiste un `TrieNode` sur le disque NVMe s'il n'y est pas déjà indexé, et l'ajoute à `core_ocean`.
    fn persist_node_if_needed(&mut self, noun: &Noun, node: &TrieNode) -> Result<u64, &'static str> {
        if let Some(existing_lba) = self.disk_index.get_lba(noun) {
            self.core_ocean
                .entry(noun.clone())
                .or_insert_with(|| node.clone());
            return Ok(existing_lba);
        }

        let addr = core::ptr::from_ref(node) as u64;
        let lba = self.storage.persist_node(addr)?;
        self.disk_index.insert(noun.clone(), lba)?;
        self.core_ocean.insert(noun.clone(), node.clone());
        Ok(lba)
    }

    /// Valide l'état courant (`virtual_ocean` + `active_plan`) dans un nouveau [`JiCommit`] immuable :
    ///
    /// 1. Persiste tous les `TrieNode` de `virtual_ocean` vers `StorageEngine` + `NounIndex`.
    /// 2. Scelle le [`Plan`] actif avec [`Sceau::birth`].
    /// 3. Génère et signe le [`Checkpoint`] `cubefs`.
    /// 4. Persiste le `TrieNode` du commit et met à jour le [`JiSuperblock`] au LBA `0`.
    pub fn commit(
        &mut self,
        message: &str,
        timestamp: u64,
        budget: u32,
    ) -> Result<Noun, &'static str> {
        self.commit_internal(message, timestamp, budget, Noun::null())
    }

    fn commit_internal(
        &mut self,
        message: &str,
        timestamp: u64,
        budget: u32,
        second_parent: Noun,
    ) -> Result<Noun, &'static str> {
        // 1. Flush de tous les nœuds du VirtualOcean vers le disque NVMe et CoreOcean
        let staged: alloc::vec::Vec<(Noun, TrieNode)> =
             core::mem::take(&mut self.virtual_ocean).into_iter().collect();
        for (noun, node) in &staged {
            self.persist_node_if_needed(noun, node)?;
        }

        // 2. Alignement du Plan sur la racine d'arbre courante et la branche AWQ active
        let branch = self.superblock.active_branch;
        self.active_plan.branch_index = branch;
        self.active_plan.directory = self.working_root.clone();

        // 3. Création du Sceau cryptographique (`plans::sceau::Sceau`)
        let sceau = Sceau::birth(&self.active_plan, budget);
        if sceau.weigh(&self.active_plan) != Verdict::Accept {
            return Err("Echec de verification du Sceau lors de la creation du commit");
        }

        // 4. Création du Checkpoint signé (`cubefs::Checkpoint`)
        let seq = self.superblock.commit_seq + 1;
        let parent = self.superblock.branch_heads[branch as usize].clone();
        let root_meta_id = *self.working_root.as_bytes();
        let merkle_root = *sceau.root.as_bytes();
        let signature = JiCommit::compute_checkpoint_signature(
            seq,
            timestamp,
            &root_meta_id,
            &merkle_root,
            &parent,
            branch,
        );

        let checkpoint = Checkpoint {
            seq,
            time: timestamp,
            root_meta_id,
            merkle_root,
            signature,
        };

        let mut layers_snapshot: Vec<Layer, 16> = Vec::new();
        for layer in self.active_plan.get_layers() {
            layers_snapshot
                .push(layer.clone())
                .map_err(|_| "Trop de layers pour un commit (max 16)")?;
        }

        let mut msg_stack = String::<128>::new();
        let _ = core::fmt::Write::write_str(&mut msg_stack, message);

        let ji_commit = JiCommit {
            seq,
            timestamp,
            branch,
            parent,
            second_parent,
            root_tree: self.working_root.clone(),
            checkpoint,
            sceau,
            layers: layers_snapshot,
            message: msg_stack,
        };

        // 5. Persistance du TrieNode de commit sur le disque
        let commit_node = ji_commit.to_trie_node();
        let commit_noun = compute_node_noun(&commit_node);
        let commit_lba = self.persist_node_if_needed(&commit_noun, &commit_node)?;

        // 6. Mise à jour et persistance du Superbloc au LBA 0
        self.superblock.commit_seq = seq;
        self.superblock.head_lba = commit_lba + 8;
        self.superblock.branch_heads[branch as usize] = commit_noun.clone();
        self.superblock.latest_checkpoint = checkpoint;
        self.superblock.write_to_device(self.storage.device);

        self.branch_roots[branch as usize] = self.working_root.clone();
        Ok(commit_noun)
    }

    /// Charge et vérifie un [`JiCommit`] (avec ses [`Layer`]s désérialisées) depuis le store par son [`Noun`].
    pub fn load_commit(&self, commit_noun: &Noun) -> Result<JiCommit, &'static str> {
        let node = self.fetch_node_ref(commit_noun)?;
        JiCommit::from_trie_node(&node)
    }

    /// Monte un commit (`ji mount <commit_noun> [target_branch]`) et restaure l'état exact de l'OS :
    ///
    /// 1. Charge et vérifie le [`JiCommit`] et la signature de son [`Checkpoint`].
    /// 2. Précharge récursivement tout l'arbre Merkle du commit dans `core_ocean` via [`CoreOceanBuilder::deep_load`].
    /// 3. Reconstruit le [`Plan`] d'origine du commit et le pèse avec [`Sceau::weigh`] (`Verdict::Accept` exigé).
    /// 4. Bascule l'espace de travail (`working_root`, `virtual_ocean`, `active_plan`) sur l'état exact du commit
    ///    sur la branche `target_branch` (ou la branche du commit par défaut).
    pub fn mount(
        &mut self,
        commit_noun: &Noun,
        target_branch: Option<u8>,
    ) -> Result<MountedOs, &'static str> {
        let commit = self.load_commit(commit_noun)?;

        // 1. Préchargement profond de l'arbre Merkle du commit dans CoreOcean via `jinshu::ocean::CoreOceanBuilder`
        let mut dma_buf = TrieNode::new();
        let dma_addr = core::ptr::from_mut(&mut dma_buf) as u64;
        CoreOceanBuilder::deep_load(
            &commit.root_tree,
            &mut self.storage,
            &self.disk_index,
            &mut self.core_ocean,
            dma_addr,
            dma_addr,
        )?;

        // 2. Reconstruction du Plan exact du commit et vérification cryptographique du Sceau
        let mut commit_plan = Plan::new(commit.root_tree.clone(), &mut self.phoenix, commit.branch)?;
        for layer in &commit.layers {
            commit_plan.add_layer(layer.clone())?;
        }

        let verdict = commit.sceau.weigh(&commit_plan);
        if let Verdict::Refuse(reason) = verdict {
            return Err(reason);
        }

        // 3. Installation du Plan sur la branche cible (sandbox AWQ ou branche d'origine)
        let mount_branch = target_branch.unwrap_or(commit.branch);
        if mount_branch as usize >= AWQ_BRANCH_COUNT {
            return Err("Branche AWQ invalide (doit etre dans 0..=15)");
        }

        let mut mounted_plan =
            Plan::new(commit.root_tree.clone(), &mut self.phoenix, mount_branch)?;
        for layer in &commit.layers {
            mounted_plan.add_layer(layer.clone())?;
        }
        let mounted_sceau = if mount_branch == commit.branch {
            commit.sceau.clone()
        } else {
            Sceau::birth(&mounted_plan, commit.sceau.budget)
        };

        // 4. Réinitialisation de l'espace de travail Copy-on-Write sur le commit monté
        self.virtual_ocean.clear();
        self.working_root = commit.root_tree.clone();
        self.active_plan = mounted_plan.clone();
        self.superblock.active_branch = mount_branch;
        self.branch_roots[mount_branch as usize] = commit.root_tree.clone();
        if self.superblock.branch_heads[mount_branch as usize].is_null()
            || target_branch.is_none()
        {
            self.superblock.branch_heads[mount_branch as usize] = commit_noun.clone();
        }
        self.superblock.write_to_device(self.storage.device);

        Ok(MountedOs {
            commit_noun: commit_noun.clone(),
            commit,
            plan: mounted_plan,
            sceau: mounted_sceau,
            verdict,
            branch: mount_branch,
        })
    }

    /// Change de branche AWQ (`0..=15`). Si la branche cible ne possède pas encore de commit,
    /// elle hérite de la tête et de l'arbre courant (fork instantané en $O(1)$).
    pub fn switch_branch(&mut self, branch: u8) -> Result<Noun, &'static str> {
        if branch as usize >= AWQ_BRANCH_COUNT {
            return Err("Branche AWQ invalide (doit etre dans 0..=15)");
        }

        let current_branch = self.superblock.active_branch;
        let current_head = self.superblock.branch_heads[current_branch as usize].clone();

        if self.superblock.branch_heads[branch as usize].is_null() && !current_head.is_null() {
            self.superblock.branch_heads[branch as usize] = current_head;
            self.branch_roots[branch as usize] = self.working_root.clone();
        }

        let target_head = self.superblock.branch_heads[branch as usize].clone();
        if !target_head.is_null() {
            let mounted = self.mount(&target_head, Some(branch))?;
            return Ok(mounted.commit_noun);
        }

        self.superblock.active_branch = branch;
        self.active_plan.branch_index = branch;
        self.superblock.write_to_device(self.storage.device);
        Ok(Noun::null())
    }

    /// Fusionne `source_branch` dans `target_branch` en utilisant :
    /// - [`SemanticRouter::merge_trees`] (`jinshu`) pour la fusion géométrique des `TrieNode`,
    /// - [`Plan::merge_with`] (`plans`) pour la fusion commutative et déterministe des `Layer`s,
    ///
    /// puis crée un commit de fusion scellé sur `target_branch`.
    pub fn merge_branches(
        &mut self,
        source_branch: u8,
        target_branch: u8,
        message: &str,
        timestamp: u64,
        budget: u32,
    ) -> Result<Noun, &'static str> {
        if source_branch as usize >= AWQ_BRANCH_COUNT || target_branch as usize >= AWQ_BRANCH_COUNT
        {
            return Err("Branche AWQ invalide (0..=15)");
        }

        let source_commit_noun = self.superblock.branch_heads[source_branch as usize].clone();
        let target_commit_noun = self.superblock.branch_heads[target_branch as usize].clone();
        if source_commit_noun.is_null() || target_commit_noun.is_null() {
            return Err("Impossible de fusionner une branche sans commit");
        }

        let source_commit = self.load_commit(&source_commit_noun)?;
        let target_commit = self.load_commit(&target_commit_noun)?;

        // 1. Fusion géométrique des arbres via `jinshu::router::SemanticRouter::merge_trees`
        let mut scratch_nodes = vec![TrieNode::new(); 8];
        let scratch_addr = scratch_nodes.as_mut_ptr() as u64;
        let _merged_raw_noun = self.with_router(|router| {
            router.merge_trees(
                &target_commit.root_tree,
                &source_commit.root_tree,
                scratch_addr,
                scratch_addr,
            )
        })?;

        // 2. Synchronisation récursive du catalogue `cubefs` des deux arbres (sous-répertoires compris)
        let merged_tree_noun =
            self.merge_dir_nodes_recursive(&target_commit.root_tree, &source_commit.root_tree)?;

        // 3. Fusion commutative des Plans (`plans::Plan::merge_with`)
        let mut plan_target = Plan::new(
            target_commit.root_tree.clone(),
            &mut self.phoenix,
            target_branch,
        )?;
        for l in &target_commit.layers {
            plan_target.add_layer(l.clone())?;
        }

        let mut plan_source = Plan::new(
            source_commit.root_tree.clone(),
            &mut self.phoenix,
            source_branch,
        )?;
        for l in &source_commit.layers {
            plan_source.add_layer(l.clone())?;
        }

        let mut merged_plan = plan_target.merge_with(&plan_source)?;
        merged_plan.branch_index = target_branch;
        merged_plan.directory = merged_tree_noun.clone();

        // 4. Création du commit de merge sur `target_branch`
        self.superblock.active_branch = target_branch;
        self.working_root = merged_tree_noun;
        self.active_plan = merged_plan;

        self.commit_internal(message, timestamp, budget, source_commit_noun)
    }

    /// Fusionne récursivement deux répertoires `FLAG_DIR` (`target_dir_noun` et `source_dir_noun`),
    /// en fusionnant en profondeur les sous-répertoires (`VAL_TYPE_DIRECTORY`) portant le même nom.
    fn merge_dir_nodes_recursive(
        &mut self,
        target_dir_noun: &Noun,
        source_dir_noun: &Noun,
    ) -> Result<Noun, &'static str> {
        let target_dir = self.fetch_node(target_dir_noun)?;
        let source_dir = self.fetch_node(source_dir_noun)?;
        let mut merged_records = read_dir_entries(&target_dir)?;
        let source_records = read_dir_entries(&source_dir)?;

        for srec in source_records {
            if let Some(pos) = merged_records
                .iter()
                .position(|r| r.name.as_str() == srec.name.as_str())
            {
                if merged_records[pos].val_type == VAL_TYPE_DIRECTORY
                    && srec.val_type == VAL_TYPE_DIRECTORY
                {
                    let existing_target = merged_records[pos].target.clone();
                    let sub_merged =
                        self.merge_dir_nodes_recursive(&existing_target, &srec.target)?;
                    merged_records[pos].target = sub_merged;
                } else {
                    merged_records[pos].target = srec.target;
                    merged_records[pos].val_type = srec.val_type;
                }
            } else {
                let mut used_mask: u16 = 0;
                for r in &merged_records {
                    used_mask |= 1 << r.slot;
                }
                let slot = if (used_mask & (1 << srec.slot)) == 0 {
                    srec.slot
                } else {
                    (0..16u8)
                        .find(|&s| (used_mask & (1 << s)) == 0)
                        .ok_or("Repertoire sature lors du merge")?
                };
                merged_records
                    .push(DirRecord { slot, ..srec })
                    .map_err(|_| "Repertoire sature lors du merge")?;
            }
        }

        let synced_dir = write_dir_entries(&merged_records)?;
        Ok(self.with_router(|router| router.write_virtual_node(synced_dir)))
    }

    /// Parcourt la chaîne des parents depuis la tête de la branche `branch` et retourne l'historique des commits.
    pub fn history(&self, branch: u8) -> Result<Vec<(Noun, JiCommit), 16>, &'static str> {
        if branch as usize >= AWQ_BRANCH_COUNT {
            return Err("Branche AWQ invalide");
        }
        let mut out: Vec<(Noun, JiCommit), 16> = Vec::new();
        let mut cursor = self.superblock.branch_heads[branch as usize].clone();

        while !cursor.is_null() && out.len() < 16 {
            let commit = self.load_commit(&cursor)?;
            let next = commit.parent.clone();
            let _ = out.push((cursor, commit));
            cursor = next;
        }
        Ok(out)
    }

    /// Liste les entrées (`DirRecord`) d'un répertoire donné (`""` ou `"/"` pour la racine, ou ex. `"/boot/drivers"`).
    pub fn list_dir_at(
        &self,
        root_tree: &Noun,
        path: &str,
    ) -> Result<Vec<DirRecord, 16>, &'static str> {
        let trimmed = path.trim_matches('/');
        let dir_noun = if trimmed.is_empty() {
            root_tree.clone()
        } else {
            let (val_type, target) = self.resolve_entry_in_tree(root_tree, trimmed)?;
            if val_type != VAL_TYPE_DIRECTORY {
                return Err("Le chemin specifie n'est pas un repertoire");
            }
            target
        };
        let dir_node = self.fetch_node_ref(&dir_noun)?;
        if dir_node.flags != FLAG_DIR {
            return Err("Noeud repertoire invalide (FLAG_DIR attendu)");
        }
        read_dir_entries(&dir_node)
    }

    /// Liste les entrées d'un répertoire dans l'arbre de travail courant (`working_root`).
    pub fn list_working_dir(&self, path: &str) -> Result<Vec<DirRecord, 16>, &'static str> {
        self.list_dir_at(&self.working_root, path)
    }

    /// Liste les entrées fichiers (`DirRecord`) de la racine de l'arbre de travail courant (`working_root`).
    pub fn list_working_files(&self) -> Result<Vec<DirRecord, 16>, &'static str> {
        self.list_dir_at(&self.working_root, "/")
    }

    /// Recherche un commit dans l'historique de toutes les branches à partir d'un préfixe hexadécimal (ex. `"a1b2c3d4"`).
    pub fn resolve_commit_prefix(&self, hex_prefix: &str) -> Result<Noun, &'static str> {
        for b in 0..AWQ_BRANCH_COUNT as u8 {
            let hist = self.history(b)?;
            for (noun, _) in hist {
                let formatted = format_noun_short(&noun);
                if formatted.as_str().starts_with(hex_prefix) {
                    return Ok(noun);
                }
            }
        }
        Err("Aucun commit ne correspond a ce prefixe Noun")
    }

    /// Crée une couche système par défaut pour initialiser le moteur rapidement.
    #[must_use]
    pub fn default_kernel_layer() -> Layer {
        Layer::new(
            "amentys-core",
            1,
            "Amentys OS Kernel & VFS Base Layer",
            Noun::of(b"amentys-core-v1"),
            Capabilities::ReadWriteExecute,
        )
    }
}

/// Formate les 4 premiers octets (8 caractères hexadécimaux) d'un [`Noun`] dans une [`StackString`] sans allocation.
#[must_use]
pub fn format_noun_short(noun: &Noun) -> StackString<256> {
    let bytes = noun.as_bytes();
    wuu::format_stack!(
        "{:02x}{:02x}{:02x}{:02x}",
        bytes[0],
        bytes[1],
        bytes[2],
        bytes[3]
    )
}
