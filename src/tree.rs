use cubefs::{BTreeEntry, BTreeIterator, BTreeNode, Extent, Inode};
use heapless::{String, Vec};
use jinshu::node::TrieNode;
use kpack::Opcode;
use noun::Noun;

/// Taille maximale du payload d'un `TrieNode` (`jinshu::node::TrieNode`).
pub const TRIE_PAYLOAD_CAPACITY: usize = 3576;

/// Indicateur `TrieNode.flags` pour un nœud de données contenant une recette `kpack`.
pub const FLAG_BLOB: u8 = 0x01;

/// Indicateur `TrieNode.flags` pour un nœud répertoire contenant un catalogue `cubefs`.
pub const FLAG_DIR: u8 = 0x02;

/// Indicateur `TrieNode.flags` pour un nœud de commit `JiCommit`.
pub const FLAG_COMMIT: u8 = 0x03;

/// Type d'entrée `cubefs` : Inode / Fichier recette `kpack`.
pub const VAL_TYPE_INODE: u8 = 0;

/// Type d'entrée `cubefs` : Sous-répertoire.
pub const VAL_TYPE_DIRECTORY: u8 = 1;

/// Type d'entrée `cubefs` : Donnée inline.
pub const VAL_TYPE_INLINE: u8 = 2;

/// Entrée nommée d'un répertoire `cubefs` rattachée à l'une des 16 branches d'un `TrieNode`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirRecord {
    pub slot: u8,
    pub hash: u64,
    pub name: String<64>,
    pub val_type: u8,
    pub target: Noun,
}

/// Calcule le [`Noun`] exact d'un [`TrieNode`] selon la convention de `jinshu::SemanticRouter::write_virtual_node`.
#[must_use]
pub fn compute_node_noun(node: &TrieNode) -> Noun {
    let raw_hash = node.calculate_noun();
    Noun::of(&raw_hash)
}

/// Calcule un hash 64 bits rapide (`cubefs::BTreeKeyHeader.hash`) et le slot nibble `0..15` associé à un nom.
#[must_use]
pub fn name_hash_and_slot(name: &str) -> (u64, u8) {
    let digest = Noun::of(name.as_bytes());
    let bytes = digest.as_bytes();
    let hash = u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]);
    let slot = bytes[0] & 0x0F;
    (hash, slot)
}

/// Crée un `TrieNode` de données exécutant une recette `kpack` (`Opcode::Lit` ou `Opcode::Rle` automatique).
pub fn make_blob_node(data: &[u8]) -> Result<TrieNode, &'static str> {
    if data.len() > TRIE_PAYLOAD_CAPACITY {
        return Err("Blob trop grand pour le payload d'un seul TrieNode (max 3576 octets)");
    }

    let mut node = TrieNode::new();
    node.flags = FLAG_BLOB;
    node.param = u32::try_from(data.len()).map_err(|_| "Taille de blob invalide")?;

    // Compression automatique RLE si tous les octets sont identiques et len >= 4
    if data.len() >= 4 && data.iter().all(|&b| b == data[0]) {
        node.opcode = Opcode::Rle as u8;
        node.payload[0] = data[0];
    } else {
        node.opcode = Opcode::Lit as u8;
        node.payload[..data.len()].copy_from_slice(data);
    }

    Ok(node)
}

/// Génère un programme d'instructions `kpack::Opcode::Delta` permettant de reconstruire `target` à partir de `base`.
///
/// Format `kpack::execute_delta` :
/// - `0x01` + `offset: u16 (LE)` + `size: u16 (LE)` : copie depuis `base`
/// - `0x02` + `size: u16 (LE)` + `bytes` : insère des octets littéraux
/// - `0x00` : fin du programme delta
pub fn build_delta_program(
    base: &[u8],
    target: &[u8],
) -> Result<Vec<u8, TRIE_PAYLOAD_CAPACITY>, &'static str> {
    if target.len() > TRIE_PAYLOAD_CAPACITY {
        return Err("Cible Delta depasse la capacite d'un TrieNode");
    }

    // 1. Longueur du préfixe commun avec `base`
    let max_prefix = base.len().min(target.len());
    let mut prefix_len = 0;
    while prefix_len < max_prefix && base[prefix_len] == target[prefix_len] {
        prefix_len += 1;
    }

    // 2. Longueur du suffixe commun avec `base` (sans chevaucher le préfixe)
    let mut suffix_len = 0;
    while suffix_len < (base.len() - prefix_len)
        && suffix_len < (target.len() - prefix_len)
        && base[base.len() - 1 - suffix_len] == target[target.len() - 1 - suffix_len]
    {
        suffix_len += 1;
    }

    let mid_target = &target[prefix_len..target.len() - suffix_len];
    let mut prog: Vec<u8, TRIE_PAYLOAD_CAPACITY> = Vec::new();

    // Instruction de copie du préfixe
    if prefix_len > 0 {
        let len_u16 = u16::try_from(prefix_len).map_err(|_| "Prefixe delta trop long")?;
        prog.push(0x01).map_err(|_| "Overflow programme delta")?;
        prog.extend_from_slice(&0u16.to_le_bytes())
            .map_err(|_| "Overflow programme delta")?;
        prog.extend_from_slice(&len_u16.to_le_bytes())
            .map_err(|_| "Overflow programme delta")?;
    }

    // Instruction d'ajout de la tranche centrale modifiée
    if !mid_target.is_empty() {
        let add_u16 = u16::try_from(mid_target.len()).map_err(|_| "Segment delta trop long")?;
        prog.push(0x02).map_err(|_| "Overflow programme delta")?;
        prog.extend_from_slice(&add_u16.to_le_bytes())
            .map_err(|_| "Overflow programme delta")?;
        prog.extend_from_slice(mid_target)
            .map_err(|_| "Overflow programme delta")?;
    }

    // Instruction de copie du suffixe depuis `base`
    if suffix_len > 0 {
        let ref_offset =
            u16::try_from(base.len() - suffix_len).map_err(|_| "Offset suffixe delta invalide")?;
        let len_u16 = u16::try_from(suffix_len).map_err(|_| "Suffixe delta trop long")?;
        prog.push(0x01).map_err(|_| "Overflow programme delta")?;
        prog.extend_from_slice(&ref_offset.to_le_bytes())
            .map_err(|_| "Overflow programme delta")?;
        prog.extend_from_slice(&len_u16.to_le_bytes())
            .map_err(|_| "Overflow programme delta")?;
    }

    // Fin du programme delta
    prog.push(0x00).map_err(|_| "Overflow programme delta")?;
    Ok(prog)
}

/// Crée un `TrieNode` de données codé en delta (`kpack::Opcode::Delta`) par rapport à un `base_noun` parent.
///
/// Le `base_noun` de référence est attaché à la branche `0` du `TrieNode` afin que le graphe Merkle
/// de `jinshu` (`CoreOceanBuilder::deep_load`) précharge automatiquement la chaîne de deltas.
pub fn make_delta_blob_node(
    base_noun: Noun,
    base_data: &[u8],
    new_data: &[u8],
) -> Result<TrieNode, &'static str> {
    let prog = build_delta_program(base_data, new_data)?;
    let mut node = TrieNode::new();
    node.opcode = Opcode::Delta as u8;
    node.flags = FLAG_BLOB;
    node.param = u32::try_from(new_data.len()).map_err(|_| "Taille de blob delta invalide")?;
    node.set_branch(0, base_noun);
    node.payload[..prog.len()].copy_from_slice(&prog);
    Ok(node)
}

/// Crée un `TrieNode` répertoire (`FLAG_DIR`) vide.
#[must_use]
pub fn make_empty_dir_node() -> TrieNode {
    let mut node = TrieNode::new();
    node.opcode = Opcode::Lit as u8;
    node.flags = FLAG_DIR;
    node.param = 0; // `param` stocke `entry_count` pour `cubefs::BTreeIterator`
    node
}

/// Extrait toutes les entrées d'un `TrieNode` répertoire en utilisant [`cubefs::BTreeIterator`].
pub fn read_dir_entries(dir_node: &TrieNode) -> Result<Vec<DirRecord, 16>, &'static str> {
    if dir_node.flags != FLAG_DIR {
        return Err("Le TrieNode n'est pas un repertoire CubeFS (FLAG_DIR)");
    }

    let entry_count = u16::try_from(dir_node.param).unwrap_or(0);
    let iter = BTreeIterator::new(&dir_node.payload, entry_count);
    let mut records: Vec<DirRecord, 16> = Vec::new();

    for entry in iter {
        let name_str =
            core::str::from_utf8(entry.name).map_err(|_| "Nom d'entree CubeFS non UTF-8")?;
        let mut stack_name = String::<64>::new();
        let _ = core::fmt::Write::write_str(&mut stack_name, name_str);

        let target = Noun::from_bytes(&entry.val_data).ok_or("Noun d'entree CubeFS invalide")?;
        let slot = find_branch_for_noun(dir_node, &target)
            .unwrap_or_else(|| name_hash_and_slot(name_str).1);

        records
            .push(DirRecord {
                slot,
                hash: entry.hash,
                name: stack_name,
                val_type: entry.val_type,
                target,
            })
            .map_err(|_| "Trop d'entrees dans le repertoire (max 16 branches par TrieNode)")?;
    }

    Ok(records)
}

fn find_branch_for_noun(dir_node: &TrieNode, target: &Noun) -> Option<u8> {
    (0..16u8).find(|&i| dir_node.has_branch(i) && dir_node.get_branch(i) == Some(target))
}

/// Encode une liste d'entrées [`DirRecord`] dans un `TrieNode` répertoire (`FLAG_DIR`)
/// au format binaire exact de [`cubefs::BTreeIterator`].
pub fn write_dir_entries(records: &[DirRecord]) -> Result<TrieNode, &'static str> {
    if records.len() > 16 {
        return Err("Un TrieNode repertoire ne peut contenir plus de 16 branches");
    }

    let mut node = make_empty_dir_node();
    let mut offset = 0usize;

    for rec in records {
        let name_bytes = rec.name.as_str().as_bytes();
        let name_len = u16::try_from(name_bytes.len()).map_err(|_| "Nom de fichier trop long")?;
        let entry_size = 8 + 2 + name_bytes.len() + 1 + 32;
        if offset + entry_size > TRIE_PAYLOAD_CAPACITY {
            return Err("Payload du repertoire CubeFS plein");
        }

        node.payload[offset..offset + 8].copy_from_slice(&rec.hash.to_le_bytes());
        offset += 8;

        node.payload[offset..offset + 2].copy_from_slice(&name_len.to_le_bytes());
        offset += 2;

        node.payload[offset..offset + name_bytes.len()].copy_from_slice(name_bytes);
        offset += name_bytes.len();

        node.payload[offset] = rec.val_type;
        offset += 1;

        node.payload[offset..offset + 32].copy_from_slice(rec.target.as_bytes());
        offset += 32;

        node.set_branch(rec.slot, rec.target.clone());
    }

    node.param = u32::try_from(records.len()).unwrap_or(0);
    Ok(node)
}

/// Insère ou met à jour une entrée nommée dans un `TrieNode` répertoire (Copy-on-Write).
pub fn upsert_dir_entry(
    dir_node: &TrieNode,
    name: &str,
    val_type: u8,
    target: Noun,
) -> Result<TrieNode, &'static str> {
    if name.is_empty() || name.len() > 64 {
        return Err("Nom de fichier invalide (1..=64 octets)");
    }

    let mut records = read_dir_entries(dir_node)?;
    let (hash, preferred_slot) = name_hash_and_slot(name);

    // Si le nom existe déjà, on met à jour sa cible sur le même slot de branche
    if let Some(existing) = records.iter_mut().find(|r| r.name.as_str() == name) {
        existing.target = target;
        existing.val_type = val_type;
        return write_dir_entries(&records);
    }

    // Sinon on choisit `preferred_slot` s'il est libre, ou le premier slot libre `0..16`
    let mut used_mask: u16 = 0;
    for rec in &records {
        used_mask |= 1 << rec.slot;
    }

    let chosen_slot = if (used_mask & (1 << preferred_slot)) == 0 {
        preferred_slot
    } else {
        (0..16u8)
            .find(|&s| (used_mask & (1 << s)) == 0)
            .ok_or("Les 16 branches du repertoire racine sont occupees")?
    };

    let mut stack_name = String::<64>::new();
    let _ = core::fmt::Write::write_str(&mut stack_name, name);

    records
        .push(DirRecord {
            slot: chosen_slot,
            hash,
            name: stack_name,
            val_type,
            target,
        })
        .map_err(|_| "Repertoire plein")?;

    write_dir_entries(&records)
}

/// Recherche une entrée par son nom dans un `TrieNode` répertoire via [`cubefs::BTreeIterator`].
#[must_use]
pub fn lookup_dir_entry<'a>(dir_node: &'a TrieNode, name: &str) -> Option<BTreeEntry<'a>> {
    if dir_node.flags != FLAG_DIR {
        return None;
    }
    let count = u16::try_from(dir_node.param).unwrap_or(0);
    let mut iter = BTreeIterator::new(&dir_node.payload, count);
    iter.find(|entry| entry.name == name.as_bytes())
}

/// Projette un `TrieNode` répertoire `jinshu` vers un [`cubefs::BTreeNode`] de 4096 octets
/// et construit l'[`Inode`] + [`Extent`] d'un blob pour [`cubefs::KpackReader`].
#[must_use]
pub fn to_cubefs_btree_node(dir_node: &TrieNode) -> BTreeNode {
    let mut payload = [0u8; 4056];
    payload[..TRIE_PAYLOAD_CAPACITY].copy_from_slice(&dir_node.payload);
    let entry_count = u16::try_from(dir_node.param).unwrap_or(0);

    let mut bnode = BTreeNode {
        magic: *b"BNOD",
        version: 1,
        level: 0,
        entry_count,
        payload,
        blake3_hash: [0u8; 32],
    };
    bnode.blake3_hash = dir_node.calculate_noun();
    bnode
}

/// Construit le couple `(`[`Inode`]`, `[`Extent`]`)` de `cubefs` pour un blob `TrieNode`.
#[must_use]
pub fn blob_inode_and_extent(blob_noun: &Noun, blob_node: &TrieNode) -> (Inode, Extent) {
    let len = u64::from(blob_node.param);
    let inode = Inode {
        id: *blob_noun.as_bytes(),
        size: len,
    };
    let extent = Extent {
        file_off: 0,
        len: len.max(1),
        first_chunk_id: *blob_noun.as_bytes(),
        count: 1,
        last_len: blob_node.param,
    };
    (inode, extent)
}
