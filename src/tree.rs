use cubefs::{BTreeEntry, BTreeIterator, BTreeNode, Extent, Inode, Nucleator};
use heapless::{String, Vec};
use jinshu::node::TrieNode;
use kpack::Opcode;
use noun::Noun;

/// Taille maximale du payload d'un `TrieNode` (`jinshu::node::TrieNode`).
pub const TRIE_PAYLOAD_CAPACITY: usize = 3576;

/// Capacité utile d'un `TrieNode` lorsque les 2 derniers octets stockent la taille logique (`Dict`, `Xor`, `Seed`, `Rs`).
pub const TRIE_RECIPE_DATA_CAPACITY: usize = TRIE_PAYLOAD_CAPACITY - 2;

/// Nombre maximal de chunks directs par nœud `FLAG_INODE` (16 branches AWQ).
pub const MAX_INODE_CHUNKS: usize = 16;

/// Taille maximale d'un fichier multi-chunks sur un seul niveau d'`Inode` (`16 * 3576 = 57 216` octets).
pub const MAX_MULTI_CHUNK_FILE_SIZE: usize = MAX_INODE_CHUNKS * TRIE_PAYLOAD_CAPACITY;

/// Indicateur `TrieNode.flags` pour un nœud de données contenant une recette `kpack`.
pub const FLAG_BLOB: u8 = 0x01;

/// Indicateur `TrieNode.flags` pour un nœud répertoire contenant un catalogue `cubefs`.
pub const FLAG_DIR: u8 = 0x02;

/// Indicateur `TrieNode.flags` pour un nœud de commit `JiCommit`.
pub const FLAG_COMMIT: u8 = 0x03;

/// Indicateur `TrieNode.flags` pour un nœud `cubefs::Inode` multi-chunks contenant des `cubefs::Extent`s.
pub const FLAG_INODE: u8 = 0x04;

/// Type d'entrée `cubefs` : Inode / Fichier recette `kpack`.
pub const VAL_TYPE_INODE: u8 = 0;

/// Type d'entrée `cubefs` : Sous-répertoire.
pub const VAL_TYPE_DIRECTORY: u8 = 1;

/// Type d'entrée `cubefs` : Donnée inline.
pub const VAL_TYPE_INLINE: u8 = 2;

const INODE_MAGIC: [u8; 4] = *b"JINO";

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

/// Découpe un chemin hiérarchique (ex. `"/boot/drivers/nvme.sys"`) en segments non vides (max 8 niveaux).
pub fn split_path(path: &str) -> Result<Vec<&str, 8>, &'static str> {
    let trimmed = path.trim_matches('/');
    if trimmed.is_empty() {
        return Err("Chemin de fichier vide");
    }

    let mut parts: Vec<&str, 8> = Vec::new();
    for segment in trimmed.split('/') {
        if segment.is_empty() {
            continue;
        }
        if segment.len() > 64 {
            return Err("Segment de chemin trop long (max 64 octets)");
        }
        parts
            .push(segment)
            .map_err(|_| "Profondeur de chemin maximale depassee (max 8 niveaux)")?;
    }

    if parts.is_empty() {
        return Err("Chemin de fichier invalide");
    }
    Ok(parts)
}

/// Encode la taille logique dans les 2 derniers octets du payload d'un `TrieNode` (`3574..3576`).
fn set_trailer_logical_len(node: &mut TrieNode, len: usize) -> Result<(), &'static str> {
    let len_u16 = u16::try_from(len).map_err(|_| "Taille logique de recette invalide")?;
    let bytes = len_u16.to_le_bytes();
    node.payload[TRIE_RECIPE_DATA_CAPACITY] = bytes[0];
    node.payload[TRIE_RECIPE_DATA_CAPACITY + 1] = bytes[1];
    Ok(())
}

/// Retourne la taille logique d'un `TrieNode` blob quel que soit son [`Opcode`] `kpack`.
#[must_use]
pub fn blob_logical_len(node: &TrieNode) -> usize {
    match Opcode::from_u8(node.opcode) {
        Some(Opcode::Lit | Opcode::Rle | Opcode::Delta) => node.param as usize,
        Some(Opcode::Dict | Opcode::Xor | Opcode::Seed | Opcode::Rs) => {
            let trailer = u16::from_le_bytes([
                node.payload[TRIE_RECIPE_DATA_CAPACITY],
                node.payload[TRIE_RECIPE_DATA_CAPACITY + 1],
            ]) as usize;
            if trailer > 0 {
                trailer
            } else {
                node.param as usize
            }
        }
        None => 0,
    }
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

/// Crée un `TrieNode` codé par dictionnaire (`kpack::Opcode::Dict`) à partir d'un dictionnaire `dict_noun`
/// découpé en mots de `word_size` octets.
///
/// Le `dict_noun` de référence est attaché à la branche `0` du `TrieNode`.
pub fn make_dict_blob_node(
    dict_noun: Noun,
    dict_data: &[u8],
    word_size: usize,
    target_data: &[u8],
) -> Result<TrieNode, &'static str> {
    if word_size == 0 || dict_data.len() < word_size {
        return Err("Dictionnaire kpack invalide ou word_size nul");
    }
    if !target_data.len().is_multiple_of(word_size) {
        return Err("La taille des donnees doit etre un multiple de word_size pour Opcode::Dict");
    }
    let word_count = target_data.len() / word_size;
    if word_count > TRIE_RECIPE_DATA_CAPACITY || target_data.len() > TRIE_PAYLOAD_CAPACITY {
        return Err("Donnees trop volumineuses pour un seul TrieNode Dict");
    }

    let max_dict_words = (dict_data.len() / word_size).min(256);
    let mut node = TrieNode::new();
    node.opcode = Opcode::Dict as u8;
    node.flags = FLAG_BLOB;
    node.param = u32::try_from(word_size).map_err(|_| "word_size invalide")?;
    node.set_branch(0, dict_noun);

    for (idx, chunk) in target_data.chunks_exact(word_size).enumerate() {
        let found = (0..max_dict_words)
            .find(|&w| &dict_data[w * word_size..(w + 1) * word_size] == chunk)
            .ok_or("Mot introuvable dans le dictionnaire de reference kpack")?;
        node.payload[idx] = u8::try_from(found).map_err(|_| "Index de mot Dict > 255")?;
    }

    set_trailer_logical_len(&mut node, target_data.len())?;
    Ok(node)
}

/// Crée un `TrieNode` codé par `kpack::Opcode::Xor` avec une clé constante `xor_key` (sans référence).
pub fn make_xor_key_blob_node(xor_key: u8, data: &[u8]) -> Result<TrieNode, &'static str> {
    if data.len() > TRIE_RECIPE_DATA_CAPACITY {
        return Err("Blob trop grand pour Opcode::Xor (max 3574 octets)");
    }

    let mut node = TrieNode::new();
    node.opcode = Opcode::Xor as u8;
    node.flags = FLAG_BLOB;
    node.param = u32::from(xor_key);
    for (i, &b) in data.iter().enumerate() {
        node.payload[i] = b ^ xor_key;
    }
    set_trailer_logical_len(&mut node, data.len())?;
    Ok(node)
}

/// Crée un `TrieNode` codé par `kpack::Opcode::Xor` différentiel vis-à-vis d'un `base_noun` parent (attaché en branche `0`).
pub fn make_xor_ref_blob_node(
    base_noun: Noun,
    base_data: &[u8],
    new_data: &[u8],
) -> Result<TrieNode, &'static str> {
    if new_data.len() > TRIE_RECIPE_DATA_CAPACITY {
        return Err("Blob trop grand pour Opcode::Xor (max 3574 octets)");
    }
    if base_data.len() < new_data.len() {
        return Err("Le buffer de reference Xor doit etre au moins aussi grand que la cible");
    }

    let mut node = TrieNode::new();
    node.opcode = Opcode::Xor as u8;
    node.flags = FLAG_BLOB;
    node.param = 0;
    node.set_branch(0, base_noun);
    for i in 0..new_data.len() {
        node.payload[i] = new_data[i] ^ base_data[i];
    }
    set_trailer_logical_len(&mut node, new_data.len())?;
    Ok(node)
}

/// Crée un `TrieNode` codé par formule générative `kpack::Opcode::Seed` (`xorshift32` + patchs épars).
///
/// Génère le flux `xorshift32(seed)` et n'enregistre dans le payload que les positions `(pos: u16, val: u8)`
/// qui diffèrent du flux synthétique.
pub fn make_seed_blob_node(seed: u32, target_data: &[u8]) -> Result<TrieNode, &'static str> {
    if target_data.len() > TRIE_RECIPE_DATA_CAPACITY {
        return Err("Blob trop grand pour Opcode::Seed (max 3574 octets)");
    }

    let mut state = if seed == 0 { 1 } else { seed };
    let mut node = TrieNode::new();
    node.opcode = Opcode::Seed as u8;
    node.flags = FLAG_BLOB;
    node.param = seed;

    let mut patch_count: u16 = 0;
    let mut offset = 2usize;

    for (pos, &target_byte) in target_data.iter().enumerate() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let synth_byte = (state & 0xFF) as u8;

        if target_byte != synth_byte {
            if offset + 3 > TRIE_RECIPE_DATA_CAPACITY {
                return Err("Trop de patchs epars pour le payload Opcode::Seed");
            }
            let pos_le = u16::try_from(pos)
                .map_err(|_| "Offset de patch Seed invalide")?
                .to_le_bytes();
            node.payload[offset] = pos_le[0];
            node.payload[offset + 1] = pos_le[1];
            node.payload[offset + 2] = target_byte;
            offset += 3;
            patch_count += 1;
        }
    }

    let count_le = patch_count.to_le_bytes();
    node.payload[0] = count_le[0];
    node.payload[1] = count_le[1];
    set_trailer_logical_len(&mut node, target_data.len())?;
    Ok(node)
}

/// Crée un `TrieNode` codé par décalage bit-à-bit `kpack::Opcode::Rs` (`>> shift`).
///
/// Encode chaque octet `b` sous la forme `b << shift` afin que l'exécution de `Opcode::Rs` restitue `b`.
pub fn make_rs_blob_node(shift: u8, target_data: &[u8]) -> Result<TrieNode, &'static str> {
    let shift = shift & 0x07;
    if target_data.len() > TRIE_RECIPE_DATA_CAPACITY {
        return Err("Blob trop grand pour Opcode::Rs (max 3574 octets)");
    }

    let mut node = TrieNode::new();
    node.opcode = Opcode::Rs as u8;
    node.flags = FLAG_BLOB;
    node.param = u32::from(shift);

    for (i, &b) in target_data.iter().enumerate() {
        if (b >> (8 - shift.max(1))) != 0 && shift > 0 {
            return Err("Octet non representable sans perte par decalage Opcode::Rs");
        }
        node.payload[i] = b << shift;
    }

    set_trailer_logical_len(&mut node, target_data.len())?;
    Ok(node)
}

/// Applique une mutation atomique Copy-on-Write sur une page de 4096 octets via [`cubefs::Nucleator::insert_byte`].
#[must_use]
pub fn nucleate_page_byte(
    original: &[u8; 4096],
    offset: usize,
    new_byte: u8,
) -> Option<([u8; 4096], u8)> {
    Nucleator::insert_byte(original, offset, new_byte)
}

/// Crée un `TrieNode` d'`Inode` multi-chunks (`FLAG_INODE`) référençant jusqu'à 16 chunks `FLAG_BLOB`
/// et sérialisant dans son payload la structure [`cubefs::Inode`] suivie des [`cubefs::Extent`]s.
pub fn make_multi_chunk_inode_node(
    chunk_nouns: &[Noun],
    chunk_lengths: &[u32],
    total_size: u64,
) -> Result<TrieNode, &'static str> {
    if chunk_nouns.is_empty() || chunk_nouns.len() > MAX_INODE_CHUNKS {
        return Err("Nombre de chunks invalide pour un Inode (1..=16)");
    }
    if chunk_nouns.len() != chunk_lengths.len() {
        return Err("Incoherence entre chunk_nouns et chunk_lengths");
    }

    let mut node = TrieNode::new();
    node.opcode = Opcode::Lit as u8;
    node.flags = FLAG_INODE;
    node.param = u32::try_from(total_size).map_err(|_| "Taille totale d'Inode trop grande")?;

    let mut extents: Vec<Extent, MAX_INODE_CHUNKS> = Vec::new();
    let mut file_off = 0u64;

    for (i, (chunk_noun, &chunk_len)) in chunk_nouns.iter().zip(chunk_lengths.iter()).enumerate() {
        let slot = u8::try_from(i).unwrap_or(0);
        node.set_branch(slot, chunk_noun.clone());

        let ext = Extent {
            file_off,
            len: u64::from(chunk_len),
            first_chunk_id: *chunk_noun.as_bytes(),
            count: 1,
            last_len: chunk_len,
        };
        extents.push(ext).map_err(|_| "Trop d'Extents")?;
        file_off += u64::from(chunk_len);
    }

    let inode_id = *Noun::of(chunk_nouns[0].as_bytes()).as_bytes();
    let inode = Inode {
        id: inode_id,
        size: total_size,
    };

    // Sérialisation dans `node.payload` :
    // [0..4]   : INODE_MAGIC ("JINO")
    // [4..36]  : inode.id ([u8; 32])
    // [36..44] : inode.size (u64 LE)
    // [44..46] : extent_count (u16 LE)
    // Pour chaque Extent (56 octets) :
    //   file_off (8) + len (8) + first_chunk_id (32) + count (4) + last_len (4)
    let mut pos = 0usize;
    node.payload[pos..pos + 4].copy_from_slice(&INODE_MAGIC);
    pos += 4;
    node.payload[pos..pos + 32].copy_from_slice(&inode.id);
    pos += 32;
    node.payload[pos..pos + 8].copy_from_slice(&inode.size.to_le_bytes());
    pos += 8;
    let ext_count = u16::try_from(extents.len()).unwrap_or(0);
    node.payload[pos..pos + 2].copy_from_slice(&ext_count.to_le_bytes());
    pos += 2;

    for ext in &extents {
        node.payload[pos..pos + 8].copy_from_slice(&ext.file_off.to_le_bytes());
        pos += 8;
        node.payload[pos..pos + 8].copy_from_slice(&ext.len.to_le_bytes());
        pos += 8;
        node.payload[pos..pos + 32].copy_from_slice(&ext.first_chunk_id);
        pos += 32;
        node.payload[pos..pos + 4].copy_from_slice(&ext.count.to_le_bytes());
        pos += 4;
        node.payload[pos..pos + 4].copy_from_slice(&ext.last_len.to_le_bytes());
        pos += 4;
    }

    Ok(node)
}

/// Désérialise l'[`Inode`] et la table des [`Extent`]s `cubefs` depuis un `TrieNode` `FLAG_INODE`.
pub fn read_inode_extents(
    node: &TrieNode,
) -> Result<(Inode, Vec<Extent, MAX_INODE_CHUNKS>), &'static str> {
    if node.flags != FLAG_INODE {
        return Err("Le TrieNode n'est pas un Inode multi-chunks (FLAG_INODE)");
    }
    if node.payload[0..4] != INODE_MAGIC {
        return Err("Magic JINO invalide dans le TrieNode Inode");
    }

    let mut id = [0u8; 32];
    id.copy_from_slice(&node.payload[4..36]);
    let size = u64::from_le_bytes(
        node.payload[36..44]
            .try_into()
            .map_err(|_| "Lecture inode.size")?,
    );
    let ext_count = u16::from_le_bytes([node.payload[44], node.payload[45]]) as usize;
    if ext_count == 0 || ext_count > MAX_INODE_CHUNKS {
        return Err("Nombre d'Extents invalide dans le TrieNode Inode");
    }

    let mut pos = 46usize;
    let mut extents: Vec<Extent, MAX_INODE_CHUNKS> = Vec::new();
    for _ in 0..ext_count {
        let file_off = u64::from_le_bytes(
            node.payload[pos..pos + 8]
                .try_into()
                .map_err(|_| "Lecture extent.file_off")?,
        );
        pos += 8;
        let len = u64::from_le_bytes(
            node.payload[pos..pos + 8]
                .try_into()
                .map_err(|_| "Lecture extent.len")?,
        );
        pos += 8;
        let mut first_chunk_id = [0u8; 32];
        first_chunk_id.copy_from_slice(&node.payload[pos..pos + 32]);
        pos += 32;
        let count = u32::from_le_bytes(
            node.payload[pos..pos + 4]
                .try_into()
                .map_err(|_| "Lecture extent.count")?,
        );
        pos += 4;
        let last_len = u32::from_le_bytes(
            node.payload[pos..pos + 4]
                .try_into()
                .map_err(|_| "Lecture extent.last_len")?,
        );
        pos += 4;

        extents
            .push(Extent {
                file_off,
                len,
                first_chunk_id,
                count,
                last_len,
            })
            .map_err(|_| "Trop d'Extents")?;
    }

    Ok((Inode { id, size }, extents))
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
    let mut claimed_slots: u16 = 0;

    for entry in iter {
        let name_str =
            core::str::from_utf8(entry.name).map_err(|_| "Nom d'entree CubeFS non UTF-8")?;
        let mut stack_name = String::<64>::new();
        let _ = core::fmt::Write::write_str(&mut stack_name, name_str);

        let target = Noun::from_bytes(&entry.val_data).ok_or("Noun d'entree CubeFS invalide")?;
        let slot = find_unclaimed_branch_for_noun(dir_node, &target, claimed_slots)
            .unwrap_or_else(|| name_hash_and_slot(name_str).1);
        claimed_slots |= 1 << slot;

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

fn find_unclaimed_branch_for_noun(
    dir_node: &TrieNode,
    target: &Noun,
    claimed_slots: u16,
) -> Option<u8> {
    (0..16u8).find(|&i| {
        (claimed_slots & (1 << i)) == 0
            && dir_node.has_branch(i)
            && dir_node.get_branch(i) == Some(target)
    })
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
            .ok_or("Les 16 branches du repertoire sont occupees")?
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
    let len = u64::try_from(blob_logical_len(blob_node)).unwrap_or(0);
    let inode = Inode {
        id: *blob_noun.as_bytes(),
        size: len,
    };
    let extent = Extent {
        file_off: 0,
        len: len.max(1),
        first_chunk_id: *blob_noun.as_bytes(),
        count: 1,
        last_len: u32::try_from(len).unwrap_or(0),
    };
    (inode, extent)
}
