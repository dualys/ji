# `ji` — Système de Contrôle de Version `#![no_std]` & Moteur Time-Travel OS

**`ji`** est le système de contrôle de version (VCS) immuable, adressé par le contenu et **`#![no_std]`** conçu pour l'OS **Amentys**. Contrairement à un VCS classique posé par-dessus un système de fichiers mutable, `ji` unifie dans un même graphe Merkle de pages de **4096 octets** :
- le stockage bloc LBA (`jinshu`),
- le système de fichiers par recettes Copy-on-Write (`cubefs` + `kpack`),
- et l'isolation d'exécution scellée cryptographiquement (`plans` + `Sceau`).

Monter un commit (`ji mount <noun>`) permet ainsi de **restaurer instantanément en $O(1)$ l'état exact de l'OS** (arborescence, binaires, couches `Layer` et capacités `Capabilities`) tel qu'il était au moment du commit, sur l'une des **16 branches AWQ** (`awq/0` .. `awq/15`).

---

## 1. Écosystème des Crates

| Crate | Version | Rôle dans `ji` |
| :--- | :---: | :--- |
| **`noun`** | `0.1.0` | **Identité cryptographique (BLAKE3 256 bits)** : chaque bloc de 4 Ko, fichier, répertoire, couche `Layer` ou commit est adressé par son `Noun` de 32 octets. |
| **`kpack`** | `0.1.0` | **Compression & Patching par formules (`no_std`)** : exécute les 7 opcodes (`Lit`, `Delta`, `Rle`, `Seed`, `Dict`, `Xor`, `Rs`) pour reconstruire les fichiers sans allocation dynamique. |
| **`cubefs`** | `0.1.1` | **Système de fichiers CoW** : catalogue de répertoires (`BTreeNode`, `BTreeIterator`, `BTreeEntry`), fichiers multi-chunks (`Inode`, `Extent`, `KpackReader`), `Checkpoint` signé et mutation atomique par page (`Nucleator`). |
| **`jinshu`** | `0.1.0` | **Moteur Merkle & Routage 3 niveaux** : nœuds Radix Trie 16-aires (`TrieNode` de 4096 octets), routage en cascade `SemanticRouter` (`virtual_ocean` $\rightarrow$ `core_ocean` $\rightarrow$ `disk_storage`), préchargement `CoreOceanBuilder::deep_load` et fusion géométrique `merge_trees`. |
| **`plans`** | `0.1.0` | **Isolation d'exécution & Scellement** : lie un arbre de fichiers (`directory: Noun`) et une pile de couches système (`Layer`, `Layers`, `LayerFace::Recto/Verso`, `Capabilities`) à une branche AWQ (`0..=15`), scellés et vérifiés par `Sceau::weigh` (`Verdict::Accept`). |
| **`wuu`** | `0.2.0` | **Interface CLI & TUI Zero-Heap** : parseur d'arguments `Command`/`Arg`, tables (`Table`), diffs colorisés (`Diff`), barres de progression (`ProgressBar`), prompts interactifs (`Prompt::noun_select`, `Prompt::text`) et dashboard multi-panneaux (`Wuu::render_dashboard`). |
| **`prisme`** | `0.1.0` | Point d'ancrage pour l'exécution isolée des plans montés. |

---

## 2. Architecture du Projet (`src/`)

```text
src/
├── lib.rs         # Point d'entrée de la bibliothèque (#![no_std]) et suite de tests d'intégration
├── main.rs        # Exécutable CLI/TUI (compatible bare-metal #![no_std] et hôte Linux avec --disk)
├── device.rs      # Implémentations de jinshu::storage::BlockDevice (MemBlockDevice & FileBlockDevice)
├── superblock.rs  # Superbloc JiSuperblock persisté au LBA 0 (magic "JIVCS001", 16 têtes AWQ, checksum BLAKE3)
├── tree.rs        # Arbres CubeFS, sous-dossiers, Inodes multi-chunks, les 7 opcodes Kpack et Nucleator
├── commit.rs      # Structure JiCommit (parent, racine, Checkpoint signé, Sceau et sérialisation des Layers)
├── engine.rs      # Moteur JiEngine (init, open à froid, stage, mkdir, commit, mount Time-Travel, merge)
└── cli.rs         # Commandes CLI, prompts clavier interactifs et Dashboard TUI via wuu
```

### Layout Disque (Pages de 4096 octets = 8 secteurs LBA de 512 octets)
- **LBA `0..7` (Bloc 0)** : `JiSuperblock` contenant le magic `JIVCS001`, la tête d'allocation `head_lba`, la branche active `active_branch`, les 16 têtes de branches `branch_heads: [Noun; 16]`, le dernier `cubefs::Checkpoint` et le checksum d'intégrité BLAKE3.
- **LBA `8..N` (Blocs 1+)** : Stockage append-only des `TrieNode` de 4096 octets (`FLAG_BLOB`, `FLAG_INODE`, `FLAG_DIR`, `FLAG_COMMIT`) géré par `jinshu::storage::StorageEngine`.

---

## 3. Fonctionnalités Clés

1. **Démarrage à froid (`JiEngine::open`)** :
   - Relit et vérifie le `JiSuperblock` au LBA `0`.
   - Reconstruit automatiquement le `DiskAllocator` et le `NounIndex` en scannant les blocs de `LBA 8` à `head_lba`.
   - Désérialise les `Layer`s directement depuis le `TrieNode` du commit actif, vérifie le `Sceau` et remonte le `Plan` de l'OS.
2. **Arborescence hiérarchique & Fichiers Multi-Chunks (`cubefs`)** :
   - Support des chemins multi-niveaux (ex. `/boot/kernel.sys`, `/sys/drivers/nvme/init.bin`) via des nœuds `FLAG_DIR` encodant des entrées `cubefs::BTreeEntry`.
   - Découpage automatique des fichiers volumineux (`> 3576` octets, jusqu'à `57 216` octets par `Inode` direct) en 16 chunks `FLAG_BLOB` fédérés par un nœud `FLAG_INODE` sérialisant `cubefs::Inode` et `cubefs::Extent`s.
3. **Les 7 Opcodes `kpack` & Copy-on-Write `Nucleator`** :
   - `Opcode::Lit` (littéral) et `Opcode::Rle` (compression automatique des répétitions d'octets).
   - `Opcode::Delta` (programme de patch binaire `0x01 COPY` / `0x02 INSERT` calculé automatiquement depuis la version parente).
   - `Opcode::Dict` (compression par dictionnaire de mots de taille fixe référencé par son `Noun`).
   - `Opcode::Xor` (masquage par clé ou delta XOR vis-à-vis d'un blob parent).
   - `Opcode::Seed` (génération synthétique `xorshift32` + liste de patchs épars `(pos: u16, val: u8)`).
   - `Opcode::Rs` (décalage bit-à-bit `>> shift`).
   - `cubefs::Nucleator::insert_byte` pour muter un octet précis en Copy-on-Write sans réécrire les autres chunks d'un fichier.
4. **Time-Travel OS (`ji mount`) & Fusion (`ji merge`)** :
   - `engine.mount(&commit_noun, Some(branch))` précharge l'arbre Merkle du commit dans `core_ocean` (`CoreOceanBuilder::deep_load`), reconstruit le `Plan`, pèse le `Sceau` (`Sceau::weigh` $\rightarrow$ `Verdict::Accept`) et ouvre un `virtual_ocean` isolé prêt à lire n'importe quel fichier du commit en $O(1)$.
   - `engine.merge_branches(src, dst, time, msg)` combine la fusion géométrique des `TrieNode` (`SemanticRouter::merge_trees`), la fusion récursive des sous-répertoires `cubefs` et la fusion commutative des couches d'OS (`Plan::merge_with`).

---

## 4. Commandes CLI & TUI (`wuu`)

Sous Linux, `ji` peut travailler soit en mémoire, soit directement sur une image disque persistante via `--disk <fichier.img>` :

```bash
# Initialiser ou inspecter une image disque d'OS
cargo run -- --disk amentys.img init
cargo run -- --disk amentys.img status

# Créer des sous-dossiers et stager des fichiers (Lit/Rle, Delta, Xor, Seed)
cargo run -- --disk amentys.img mkdir /boot/drivers
cargo run -- --disk amentys.img stage /boot/kernel.sys "AMENTYS_KERNEL_V1"
cargo run -- --disk amentys.img stage -d /boot/kernel.sys "AMENTYS_KERNEL_V2"
cargo run -- --disk amentys.img stage -x 42 /etc/secret.cfg "TOKEN_42"
cargo run -- --disk amentys.img stage -s 305419896 /boot/synth.bin "ABCD"

# Lister un répertoire et afficher le contenu d'un fichier
cargo run -- --disk amentys.img ls /boot
cargo run -- --disk amentys.img cat /boot/kernel.sys

# Valider un commit scellé par Sceau et consulter l'historique
cargo run -- --disk amentys.img commit -m "Systeme de base v1"
cargo run -- --disk amentys.img log

# Comparer deux fichiers/commits et monter un commit sur une branche AWQ (Time-Travel)
cargo run -- --disk amentys.img diff <noun_a> <noun_b>
cargo run -- --disk amentys.img mount <commit_noun_prefix> -b 2

# Fusionner deux branches AWQ
cargo run -- --disk amentys.img merge 2 0

# Afficher le tableau de bord TUI multi-panneaux (wuu Dashboard)
cargo run -- --disk amentys.img dashboard
```

---

## 5. Exemple d'utilisation en Rust (`#![no_std]`)

```rust
use ji::device::MemBlockDevice;
use ji::engine::JiEngine;
use noun::Noun;
use plans::layer::{Capabilities, Layer};

// 1. Initialisation du disque LBA et de la couche système initiale
let device = MemBlockDevice::new(256);
let base_layer = Layer::new("amentys-core", 1, "Noyau de base", Noun::of(b"core-v1"), Capabilities::ROOT);
let mut engine = JiEngine::init(&device, &[base_layer], 1_700_000_000).unwrap();

// 2. Staging de fichiers dans des sous-dossiers et premier commit
engine.stage_file("/boot/kernel.sys", b"KERNEL_V1").unwrap();
let commit_v1 = engine.commit(1_700_000_010, "Release v1").unwrap();

// 3. Mise à jour par patch Delta kpack et second commit
engine.stage_file_delta("/boot/kernel.sys", b"KERNEL_V2_PATCHED").unwrap();
let _commit_v2 = engine.commit(1_700_000_020, "Release v2").unwrap();

// 4. Time-Travel : remonter le commit v1 sur la branche AWQ 1
let mounted_v1 = engine.mount(&commit_v1, Some(1)).unwrap();
assert_eq!(
    engine.read_mounted_file(&mounted_v1, "/boot/kernel.sys").unwrap().as_slice(),
    b"KERNEL_V1"
);
```

---

## 6. Compilation & Tests

```bash
# Exécuter l'ensemble des suites de tests unitaires et d'intégration
cargo test

# Vérifier l'absence d'avertissements Clippy
cargo clippy --all-targets -- -D warnings
```
