use crate::commit::{Checkpoint, JiCommit};
use crate::superblock::{AWQ_BRANCH_COUNT, FIRST_DATA_LBA, JiSuperblock};
use crate::tree::{
    DirRecord, FLAG_BLOB, FLAG_DIR, TRIE_PAYLOAD_CAPACITY, VAL_TYPE_INODE, compute_node_noun,
    lookup_dir_entry, make_blob_node, make_delta_blob_node, make_empty_dir_node, read_dir_entries,
    upsert_dir_entry, write_dir_entries,
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

    /// Écrit un fichier dans le `virtual_ocean` (Copy-on-Write en RAM) avec compression `kpack` (`Lit` ou `Rle`).
    pub fn stage_file(&mut self, name: &str, data: &[u8]) -> Result<Noun, &'static str> {
        let blob_node = make_blob_node(data)?;
        let blob_noun = self.with_router(|router| router.write_virtual_node(blob_node));

        let working_root = self.working_root.clone();
        let current_dir = self.fetch_node(&working_root)?;
        let updated_dir = upsert_dir_entry(&current_dir, name, VAL_TYPE_INODE, blob_noun.clone())?;
        let new_root = self.with_router(|router| router.write_virtual_node(updated_dir));

        self.working_root = new_root;
        self.active_plan.directory = self.working_root.clone();
        Ok(blob_noun)
    }

    /// Écrit une nouvelle version d'un fichier dans le `virtual_ocean` en l'encodant sous forme de
    /// recette delta [`kpack::Opcode::Delta`] par rapport à sa version précédente dans l'arbre courant.
    pub fn stage_file_delta(&mut self, name: &str, new_data: &[u8]) -> Result<Noun, &'static str> {
        let working_root = self.working_root.clone();
        let current_dir = self.fetch_node(&working_root)?;
        let entry = lookup_dir_entry(&current_dir, name)
            .ok_or("Fichier parent introuvable pour l'encodage Delta")?;
        let base_noun =
            Noun::from_bytes(&entry.val_data).ok_or("Noun de base invalide dans le repertoire")?;

        let mut base_buf = [0u8; TRIE_PAYLOAD_CAPACITY];
        let base_len = self.read_blob(&base_noun, &mut base_buf)?;

        let delta_node = make_delta_blob_node(base_noun, &base_buf[..base_len], new_data)?;
        let delta_noun = self.with_router(|router| router.write_virtual_node(delta_node));

        let updated_dir =
            upsert_dir_entry(&current_dir, name, VAL_TYPE_INODE, delta_noun.clone())?;
        let new_root = self.with_router(|router| router.write_virtual_node(updated_dir));

        self.working_root = new_root;
        self.active_plan.directory = self.working_root.clone();
        Ok(delta_noun)
    }

    /// Reconstruit le contenu d'un blob `TrieNode` en exécutant sa recette `kpack`
    /// (avec résolution récursive de la branche `0` si `opcode == Opcode::Delta`).
    pub fn read_blob(&self, blob_noun: &Noun, output: &mut [u8]) -> Result<usize, &'static str> {
        let node = self.fetch_node_ref(blob_noun)?;
        if node.flags != FLAG_BLOB {
            return Err("Le Noun demande n'est pas un blob de donnees (FLAG_BLOB)");
        }

        let target_len = (node.param as usize).min(output.len());
        output[..target_len].fill(0);

        if node.opcode == Opcode::Delta as u8 {
            let base_noun = node
                .get_branch(0)
                .ok_or("Blob Delta corrompu : branche 0 (reference parente) absente")?;
            let mut ref_buf = [0u8; TRIE_PAYLOAD_CAPACITY];
            let ref_len = self.read_blob(base_noun, &mut ref_buf)?;
            node.execute_recipe(&mut output[..target_len], Some(&ref_buf[..ref_len]));
        } else {
            node.execute_recipe(&mut output[..target_len], None);
        }

        Ok(target_len)
    }

    /// Lit un fichier nommé depuis n'importe quelle racine d'arbre `root_tree` (commit passé ou `working_root`).
    pub fn read_file(
        &self,
        root_tree: &Noun,
        name: &str,
        output: &mut [u8],
    ) -> Result<usize, &'static str> {
        let dir_node = self.fetch_node_ref(root_tree)?;
        let entry = lookup_dir_entry(&dir_node, name).ok_or("Fichier introuvable dans l'arbre")?;
        let blob_noun = Noun::from_bytes(&entry.val_data).ok_or("Noun de fichier invalide")?;
        self.read_blob(&blob_noun, output)
    }

    /// Lit un fichier dans l'espace de travail courant (`working_root`).
    pub fn read_working_file(&self, name: &str, output: &mut [u8]) -> Result<usize, &'static str> {
        self.read_file(&self.working_root, name, output)
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

        // 2. Synchronisation du catalogue `cubefs` des deux arbres (priorité à `source_branch` en cas de collision)
        let target_dir = self.fetch_node(&target_commit.root_tree)?;
        let source_dir = self.fetch_node(&source_commit.root_tree)?;
        let mut merged_records = read_dir_entries(&target_dir)?;
        let source_records = read_dir_entries(&source_dir)?;

        for srec in source_records {
            if let Some(existing) = merged_records
                .iter_mut()
                .find(|r| r.name.as_str() == srec.name.as_str())
            {
                existing.target = srec.target;
                existing.val_type = srec.val_type;
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
        let merged_tree_noun = self.with_router(|router| router.write_virtual_node(synced_dir));

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

    /// Liste les entrées fichiers (`DirRecord`) de l'arbre de travail courant (`working_root`).
    pub fn list_working_files(&self) -> Result<Vec<DirRecord, 16>, &'static str> {
        let dir_node = self.fetch_node_ref(&self.working_root)?;
        if dir_node.flags != FLAG_DIR {
            return Err("Racine courante invalide (FLAG_DIR attendu)");
        }
        read_dir_entries(&dir_node)
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
