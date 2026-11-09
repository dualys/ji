use crate::tree::{FLAG_COMMIT, TRIE_PAYLOAD_CAPACITY};
use alloc::boxed::Box;
use alloc::string::String as HeapString;
use heapless::{String, Vec};
use jinshu::node::TrieNode;
use kpack::Opcode;
use noun::Noun;
use plans::Plan;
use plans::layer::{Capabilities, Layer, Layers};
use plans::sceau::{Sceau, Verdict};

const COMMIT_MAGIC: [u8; 4] = *b"JICM";

/// Point de sauvegarde racine signé (layout identique à `cubefs::Checkpoint`,
/// redéfini ici car `cubefs 0.1.1` possède un attribut `#[cfg(test)]` devant `Checkpoint`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Checkpoint {
    pub seq: u64,
    pub time: u64,
    pub root_meta_id: [u8; 32],
    pub merkle_root: [u8; 32],
    pub signature: [u8; 64],
}

/// Représente un Commit immuable dans `ji`.
///
/// Un `JiCommit` relie :
/// - L'arbre Merkle du système de fichiers (`root_tree: Noun` dans `jinshu` / `cubefs`),
/// - Le [`Checkpoint`] signé,
/// - Le [`plans::sceau::Sceau`] cryptographique scellant les couches système (`Layer`) et leurs `Capabilities`,
/// - La filiation VCS (`parent` et `second_parent` pour les fusions de branches AWQ).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JiCommit {
    pub seq: u64,
    pub timestamp: u64,
    pub branch: u8,
    pub parent: Noun,
    pub second_parent: Noun,
    pub root_tree: Noun,
    pub checkpoint: Checkpoint,
    pub sceau: Sceau,
    pub layers: Vec<Layer, 16>,
    pub message: String<128>,
}

impl JiCommit {
    /// Calcule la signature BLAKE3 sur 64 octets d'un [`Checkpoint`] et de sa filiation via [`Noun::keyed`].
    #[must_use]
    pub fn compute_checkpoint_signature(
        seq: u64,
        time: u64,
        root_meta_id: &[u8; 32],
        merkle_root: &[u8; 32],
        parent: &Noun,
        branch: u8,
    ) -> [u8; 64] {
        let mut msg = [0u8; 113];
        msg[0..8].copy_from_slice(&seq.to_le_bytes());
        msg[8..16].copy_from_slice(&time.to_le_bytes());
        msg[16] = branch;
        msg[17..49].copy_from_slice(root_meta_id);
        msg[49..81].copy_from_slice(merkle_root);
        msg[81..113].copy_from_slice(parent.as_bytes());

        let key_lo = b"JI-CHECKPOINT-SIG-V1-LO-AMENTYS!";
        let key_hi = b"JI-CHECKPOINT-SIG-V1-HI-AMENTYS!";
        let part_lo = Noun::keyed(key_lo, &msg);
        let part_hi = Noun::keyed(key_hi, &msg);

        let mut sig = [0u8; 64];
        sig[0..32].copy_from_slice(part_lo.as_bytes());
        sig[32..64].copy_from_slice(part_hi.as_bytes());
        sig
    }

    /// Vérifie que la signature 64 octets du `Checkpoint` correspond exactement aux métadonnées du commit.
    #[must_use]
    pub fn verify_checkpoint(&self) -> bool {
        if self.checkpoint.root_meta_id != *self.root_tree.as_bytes() {
            return false;
        }
        if self.checkpoint.merkle_root != *self.sceau.root.as_bytes() {
            return false;
        }
        let expected = Self::compute_checkpoint_signature(
            self.seq,
            self.timestamp,
            &self.checkpoint.root_meta_id,
            &self.checkpoint.merkle_root,
            &self.parent,
            self.branch,
        );
        self.checkpoint.signature == expected
    }

    /// Encapsule ce `JiCommit` dans un [`TrieNode`] de 4096 octets pour persistance dans `jinshu`.
    ///
    /// Les pointeurs Merkle sont placés dans `branches` :
    /// - `branch[0]` : `root_tree` (racine des fichiers de l'OS)
    /// - `branch[1]` : `parent` (si non nul)
    /// - `branch[2]` : `second_parent` (si commit de merge)
    #[must_use]
    pub fn to_trie_node(&self) -> TrieNode {
        let mut node = TrieNode::new();
        node.opcode = Opcode::Lit as u8;
        node.flags = FLAG_COMMIT;
        node.set_branch(0, self.root_tree.clone());
        if !self.parent.is_null() {
            node.set_branch(1, self.parent.clone());
        }
        if !self.second_parent.is_null() {
            node.set_branch(2, self.second_parent.clone());
        }

        let mut pos = 0;
        node.payload[pos..pos + 4].copy_from_slice(&COMMIT_MAGIC);
        pos += 4;

        node.payload[pos..pos + 8].copy_from_slice(&self.seq.to_le_bytes());
        pos += 8;

        node.payload[pos..pos + 8].copy_from_slice(&self.timestamp.to_le_bytes());
        pos += 8;

        node.payload[pos] = self.branch;
        pos += 1;

        node.payload[pos..pos + 32].copy_from_slice(self.parent.as_bytes());
        pos += 32;

        node.payload[pos..pos + 32].copy_from_slice(self.second_parent.as_bytes());
        pos += 32;

        node.payload[pos..pos + 32].copy_from_slice(self.root_tree.as_bytes());
        pos += 32;

        // Checkpoint
        node.payload[pos..pos + 8].copy_from_slice(&self.checkpoint.seq.to_le_bytes());
        pos += 8;
        node.payload[pos..pos + 8].copy_from_slice(&self.checkpoint.time.to_le_bytes());
        pos += 8;
        node.payload[pos..pos + 32].copy_from_slice(&self.checkpoint.root_meta_id);
        pos += 32;
        node.payload[pos..pos + 32].copy_from_slice(&self.checkpoint.merkle_root);
        pos += 32;
        node.payload[pos..pos + 64].copy_from_slice(&self.checkpoint.signature);
        pos += 64;

        // Sceau (root + caps + budget + layers)
        node.payload[pos..pos + 32].copy_from_slice(self.sceau.root.as_bytes());
        pos += 32;
        node.payload[pos] = encode_capabilities(&self.sceau.caps);
        pos += 1;
        node.payload[pos..pos + 4].copy_from_slice(&self.sceau.budget.to_le_bytes());
        pos += 4;
        node.payload[pos] = self.sceau.layers;
        pos += 1;

        // Message de commit
        let msg_bytes = self.message.as_str().as_bytes();
        let msg_len = msg_bytes.len().min(128);
        node.payload[pos..pos + 2]
            .copy_from_slice(&u16::try_from(msg_len).unwrap_or(0).to_le_bytes());
        pos += 2;
        node.payload[pos..pos + msg_len].copy_from_slice(&msg_bytes[..msg_len]);
        pos += msg_len;

        // Sérialisation directe des Layers (`plans::layer::Layer`) dans le TrieNode du commit
        let layer_count = u8::try_from(self.layers.len().min(16)).unwrap_or(0);
        node.payload[pos] = layer_count;
        pos += 1;

        for layer in self.layers.iter().take(layer_count as usize) {
            let name_bytes = layer.name.as_bytes();
            let name_len = name_bytes.len().min(255);
            let desc_bytes = layer.description.as_bytes();
            let desc_len = desc_bytes.len().min(1024);
            let required = 4 + 1 + 32 + 1 + name_len + 2 + desc_len;
            if pos + required > TRIE_PAYLOAD_CAPACITY {
                break;
            }

            node.payload[pos..pos + 4].copy_from_slice(&layer.version.to_le_bytes());
            pos += 4;

            node.payload[pos] = encode_capabilities(&layer.capabilities);
            pos += 1;

            node.payload[pos..pos + 32].copy_from_slice(layer.root.as_bytes());
            pos += 32;

            node.payload[pos] = u8::try_from(name_len).unwrap_or(0);
            pos += 1;
            node.payload[pos..pos + name_len].copy_from_slice(&name_bytes[..name_len]);
            pos += name_len;

            node.payload[pos..pos + 2]
                .copy_from_slice(&u16::try_from(desc_len).unwrap_or(0).to_le_bytes());
            pos += 2;
            node.payload[pos..pos + desc_len].copy_from_slice(&desc_bytes[..desc_len]);
            pos += desc_len;
        }

        node.param = u32::try_from(pos).unwrap_or(0);
        node
    }

    /// Reconstruit le [`Plan`] du commit à partir de ses [`Layer`]s et vérifie son [`Sceau`] via [`Sceau::weigh`].
    pub fn verify_sceau(&self) -> Result<Verdict, &'static str> {
        let mut phoenix = Layers::new(self.root_tree.clone());
        if let Some(first) = self.layers.first() {
            phoenix.add_layer(first.clone())?;
        } else {
            phoenix.add_layer(Layer::new(
                "amentys-core",
                1,
                "Amentys OS Kernel & VFS Base Layer",
                self.root_tree.clone(),
                Capabilities::None,
            ))?;
        }

        let mut plan = Plan::new(self.root_tree.clone(), &mut phoenix, self.branch)?;
        for layer in &self.layers {
            plan.add_layer(layer.clone())?;
        }
        Ok(self.sceau.weigh(&plan))
    }

    /// Reconstruit un `JiCommit` (incluant ses [`Layer`]s sérialisées) depuis son [`TrieNode`] stocké dans `jinshu`.
    pub fn from_trie_node(node: &TrieNode) -> Result<Self, &'static str> {
        if node.flags != FLAG_COMMIT {
            return Err("TrieNode invalide : ce n'est pas un JiCommit (FLAG_COMMIT)");
        }
        if node.payload.len() < TRIE_PAYLOAD_CAPACITY {
            return Err("Payload de commit tronque");
        }

        let mut pos = 0;
        if node.payload[pos..pos + 4] != COMMIT_MAGIC {
            return Err("Magic JICM invalide dans le TrieNode de commit");
        }
        pos += 4;

        let seq = u64::from_le_bytes(
            node.payload[pos..pos + 8]
                .try_into()
                .map_err(|_| "Lecture seq")?,
        );
        pos += 8;

        let timestamp = u64::from_le_bytes(
            node.payload[pos..pos + 8]
                .try_into()
                .map_err(|_| "Lecture timestamp")?,
        );
        pos += 8;

        let branch = node.payload[pos];
        pos += 1;

        let parent =
            Noun::from_bytes(&node.payload[pos..pos + 32]).ok_or("Parent Noun invalide")?;
        pos += 32;

        let second_parent =
            Noun::from_bytes(&node.payload[pos..pos + 32]).ok_or("Second parent Noun invalide")?;
        pos += 32;

        let root_tree =
            Noun::from_bytes(&node.payload[pos..pos + 32]).ok_or("Root tree Noun invalide")?;
        pos += 32;

        let cp_seq = u64::from_le_bytes(
            node.payload[pos..pos + 8]
                .try_into()
                .map_err(|_| "Lecture cp_seq")?,
        );
        pos += 8;
        let cp_time = u64::from_le_bytes(
            node.payload[pos..pos + 8]
                .try_into()
                .map_err(|_| "Lecture cp_time")?,
        );
        pos += 8;
        let mut root_meta_id = [0u8; 32];
        root_meta_id.copy_from_slice(&node.payload[pos..pos + 32]);
        pos += 32;
        let mut merkle_root = [0u8; 32];
        merkle_root.copy_from_slice(&node.payload[pos..pos + 32]);
        pos += 32;
        let mut signature = [0u8; 64];
        signature.copy_from_slice(&node.payload[pos..pos + 64]);
        pos += 64;

        let sceau_root =
            Noun::from_bytes(&node.payload[pos..pos + 32]).ok_or("Sceau root Noun invalide")?;
        pos += 32;
        let sceau_caps = decode_capabilities(node.payload[pos])?;
        pos += 1;
        let sceau_budget = u32::from_le_bytes(
            node.payload[pos..pos + 4]
                .try_into()
                .map_err(|_| "Lecture sceau_budget")?,
        );
        pos += 4;
        let sceau_layers = node.payload[pos];
        pos += 1;

        let msg_len =
            u16::from_le_bytes([node.payload[pos], node.payload[pos + 1]]) as usize;
        pos += 2;
        if msg_len > 128 || pos + msg_len > node.payload.len() {
            return Err("Longueur de message de commit invalide");
        }
        let msg_str = core::str::from_utf8(&node.payload[pos..pos + msg_len])
            .map_err(|_| "Message de commit non UTF-8")?;
        let mut message = String::<128>::new();
        let _ = core::fmt::Write::write_str(&mut message, msg_str);
        pos += msg_len;

        if pos >= node.payload.len() {
            return Err("Payload de commit tronque avant layer_count");
        }
        let layer_count = node.payload[pos];
        pos += 1;
        if layer_count > 16 {
            return Err("Nombre de Layers invalide dans le TrieNode de commit (max 16)");
        }
        if layer_count != sceau_layers {
            return Err("Incoherence entre layer_count du commit et Sceau.layers");
        }

        let mut layers: Vec<Layer, 16> = Vec::new();
        for _ in 0..layer_count {
            if pos + 4 + 1 + 32 + 1 > node.payload.len() {
                return Err("Layer tronquee dans le TrieNode de commit");
            }
            let version = u32::from_le_bytes(
                node.payload[pos..pos + 4]
                    .try_into()
                    .map_err(|_| "Lecture layer.version")?,
            );
            pos += 4;

            let capabilities = decode_capabilities(node.payload[pos])?;
            pos += 1;

            let root =
                Noun::from_bytes(&node.payload[pos..pos + 32]).ok_or("Layer root Noun invalide")?;
            pos += 32;

            let name_len = node.payload[pos] as usize;
            pos += 1;
            if pos + name_len + 2 > node.payload.len() {
                return Err("Nom de Layer tronque dans le TrieNode de commit");
            }
            let name_str = core::str::from_utf8(&node.payload[pos..pos + name_len])
                .map_err(|_| "Nom de Layer non UTF-8")?;
            pos += name_len;

            let desc_len =
                u16::from_le_bytes([node.payload[pos], node.payload[pos + 1]]) as usize;
            pos += 2;
            if pos + desc_len > node.payload.len() {
                return Err("Description de Layer tronquee dans le TrieNode de commit");
            }
            let desc_str = core::str::from_utf8(&node.payload[pos..pos + desc_len])
                .map_err(|_| "Description de Layer non UTF-8")?;
            pos += desc_len;

            let layer = Layer::new(
                intern_layer_str(name_str),
                version,
                intern_layer_str(desc_str),
                root,
                capabilities,
            );
            layers
                .push(layer)
                .map_err(|_| "Trop de Layers dans le TrieNode de commit")?;
        }

        let commit = Self {
            seq,
            timestamp,
            branch,
            parent,
            second_parent,
            root_tree,
            checkpoint: Checkpoint {
                seq: cp_seq,
                time: cp_time,
                root_meta_id,
                merkle_root,
                signature,
            },
            sceau: Sceau {
                root: sceau_root,
                caps: sceau_caps,
                budget: sceau_budget,
                layers: sceau_layers,
            },
            layers,
            message,
        };

        if !commit.verify_checkpoint() {
            return Err("Signature cryptographique du Checkpoint invalide");
        }
        if let Verdict::Refuse(reason) = commit.verify_sceau()? {
            return Err(reason);
        }

        Ok(commit)
    }
}

fn intern_layer_str(s: &str) -> &'static str {
    match s {
        "" => "",
        "amentys-core" => "amentys-core",
        "Amentys OS Kernel & VFS Base Layer" => "Amentys OS Kernel & VFS Base Layer",
        other => Box::leak(HeapString::from(other).into_boxed_str()),
    }
}

const fn encode_capabilities(c: &Capabilities) -> u8 {
    match c {
        Capabilities::None => 0,
        Capabilities::Read => 1,
        Capabilities::Write => 2,
        Capabilities::Execute => 3,
        Capabilities::ReadWrite => 4,
        Capabilities::ReadExecute => 5,
        Capabilities::WriteExecute => 6,
        Capabilities::ReadWriteExecute => 7,
        Capabilities::All => 8,
    }
}

const fn decode_capabilities(tag: u8) -> Result<Capabilities, &'static str> {
    match tag {
        0 => Ok(Capabilities::None),
        1 => Ok(Capabilities::Read),
        2 => Ok(Capabilities::Write),
        3 => Ok(Capabilities::Execute),
        4 => Ok(Capabilities::ReadWrite),
        5 => Ok(Capabilities::ReadExecute),
        6 => Ok(Capabilities::WriteExecute),
        7 => Ok(Capabilities::ReadWriteExecute),
        8 => Ok(Capabilities::All),
        _ => Err("Capabilities tag inconnu dans le Sceau"),
    }
}
