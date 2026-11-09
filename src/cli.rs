use crate::engine::{JiEngine, MountedOs, format_noun_short};
use crate::superblock::FIRST_DATA_LBA;
use crate::tree::{TRIE_PAYLOAD_CAPACITY, VAL_TYPE_DIRECTORY};
use core::fmt::Write;
use heapless::Vec;
use jinshu::storage::BlockDevice;
use noun::Noun;
use wuu::{
    Arg, ColorMode, Command, Context, Diff, DropDownTerm, MessageLevel, Panel, ProgressBar, Prompt,
    PromptKey, PromptResult, PromptState, StackString, Status, Table, TableStyle, Wuu,
    write_ko_custom, write_ok_custom, write_styled_message,
};

const STATUS_ARGS: [Arg<'static>; 1] = [Arg::new("short")
    .short('s')
    .long("short")
    .help("Affichage compact")];

const STAGE_ARGS: [Arg<'static>; 7] = [
    Arg::new("delta")
        .short('d')
        .long("delta")
        .help("Encode par recette Delta kpack vis-a-vis de la version parente"),
    Arg::new("xor")
        .short('x')
        .long("xor")
        .help("Encode par recette Xor differentielle vis-a-vis de la version parente"),
    Arg::new("dict")
        .long("dict")
        .takes_value("DICT_PATH")
        .help("Encode par recette Dict kpack en referencant un fichier dictionnaire"),
    Arg::new("word_size")
        .short('w')
        .long("word-size")
        .takes_value("BYTES")
        .help("Taille des mots en octets pour --dict (defaut: 4)"),
    Arg::new("seed")
        .long("seed")
        .takes_value("SEED")
        .help("Encode par formule generative Seed kpack (xorshift32 + patchs epars)"),
    Arg::new("name")
        .takes_value("PATH")
        .required()
        .help("Chemin du fichier ou composant OS (ex. /boot/kernel.sys)"),
    Arg::new("content")
        .takes_value("DATA")
        .required()
        .help("Contenu du fichier"),
];

const MKDIR_ARGS: [Arg<'static>; 1] = [Arg::new("path")
    .takes_value("DIR_PATH")
    .required()
    .help("Chemin du sous-repertoire CubeFS a creer")];

const LS_ARGS: [Arg<'static>; 1] = [Arg::new("path")
    .takes_value("DIR_PATH")
    .help("Chemin du repertoire CubeFS a lister (defaut: /)")];

const CAT_ARGS: [Arg<'static>; 1] = [Arg::new("path")
    .takes_value("FILE_PATH")
    .required()
    .help("Chemin du fichier a afficher")];

const COMMIT_ARGS: [Arg<'static>; 2] = [
    Arg::new("message")
        .short('m')
        .long("message")
        .takes_value("MSG")
        .required()
        .help("Message decrivant le commit OS"),
    Arg::new("budget")
        .short('b')
        .long("budget")
        .takes_value("BUDGET")
        .help("Budget de capacites pour le Sceau"),
];

const LOG_ARGS: [Arg<'static>; 1] = [Arg::new("branch")
    .short('b')
    .long("branch")
    .takes_value("BRANCH")
    .help("Branche AWQ (0..15) a inspecter")];

const MOUNT_ARGS: [Arg<'static>; 2] = [
    Arg::new("branch")
        .short('b')
        .long("branch")
        .takes_value("BRANCH")
        .help("Branche AWQ cible (0..15) ou monter le commit"),
    Arg::new("commit")
        .takes_value("NOUN_PREFIX")
        .required()
        .help("Prefixe hex du Noun du commit a monter"),
];

const MERGE_ARGS: [Arg<'static>; 2] = [
    Arg::new("message")
        .short('m')
        .long("message")
        .takes_value("MSG")
        .help("Message du commit de fusion"),
    Arg::new("source")
        .takes_value("BRANCH")
        .required()
        .help("Branche AWQ source (0..15) a fusionner dans la branche active"),
];

const DIFF_ARGS: [Arg<'static>; 3] = [
    Arg::new("commit_a")
        .takes_value("NOUN_A")
        .required()
        .help("Premier commit"),
    Arg::new("commit_b")
        .takes_value("NOUN_B")
        .required()
        .help("Second commit"),
    Arg::new("file")
        .takes_value("FILE")
        .required()
        .help("Chemin du fichier a comparer"),
];

const DASHBOARD_ARGS: [Arg<'static>; 2] = [
    Arg::new("cols")
        .long("cols")
        .takes_value("COLS")
        .help("Largeur du dashboard en colonnes (defaut: 80)"),
    Arg::new("rows")
        .long("rows")
        .takes_value("ROWS")
        .help("Hauteur du dashboard en lignes (defaut: 24)"),
];

const SUBCOMMANDS: [Command<'static>; 11] = [
    Command::new("status")
        .about("Affiche l'etat de la branche AWQ, du Plan et du VirtualOcean")
        .args(&STATUS_ARGS),
    Command::new("stage")
        .about("Ecrit ou patche (Lit/Rle/Delta/Dict/Xor/Seed) un fichier dans VirtualOcean")
        .args(&STAGE_ARGS),
    Command::new("mkdir")
        .about("Cree un sous-repertoire CubeFS dans VirtualOcean")
        .args(&MKDIR_ARGS),
    Command::new("ls")
        .about("Liste les entrees d'un repertoire ou sous-repertoire CubeFS")
        .args(&LS_ARGS),
    Command::new("cat")
        .about("Reconstruit et affiche le contenu d'un fichier par sa recette kpack")
        .args(&CAT_ARGS),
    Command::new("commit")
        .about("Scelle le Plan (Sceau), signe le Checkpoint CubeFS et persiste sur NVMe")
        .args(&COMMIT_ARGS),
    Command::new("log")
        .about("Affiche l'historique des commits et Sceaux de la branche AWQ")
        .args(&LOG_ARGS),
    Command::new("mount")
        .about("Monte un commit (Time-Travel OS) et verifie son Sceau cryptographique")
        .args(&MOUNT_ARGS),
    Command::new("merge")
        .about("Fusionne une branche AWQ via SemanticRouter::merge_trees et Plan::merge_with")
        .args(&MERGE_ARGS),
    Command::new("diff")
        .about("Affiche le diff unifie d'un fichier entre deux commits")
        .args(&DIFF_ARGS),
    Command::new("dashboard")
        .about("Affiche le tableau de bord TUI Wuu (Branches AWQ, Oceans, Sceau, CubeFS)")
        .args(&DASHBOARD_ARGS),
];

/// Retourne la définition CLI principale `ji` construite avec [`wuu::Command`].
#[must_use]
pub const fn ji_cli() -> Command<'static> {
    Command::new("ji")
        .about("VCS immuable par contenu (Noun / Kpack / CubeFS / Jinshu / Plans)")
        .subcommands(&SUBCOMMANDS)
}

/// Exécute une commande CLI `ji` sur `engine` et écrit le rendu formaté via `wuu` dans `output`.
pub fn execute_cli<D: BlockDevice, W: Write>(
    engine: &mut JiEngine<'_, D>,
    argv: &[&str],
    color_mode: ColorMode,
    output: &mut W,
) -> Result<(), &'static str> {
    if argv.is_empty() || argv[0] == "help" || argv[0] == "--help" || argv[0] == "-h" {
        let cmd = ji_cli();
        cmd.write_help(output).map_err(|_| "Erreur d'ecriture aide")?;
        return Ok(());
    }

    let root_cmd = ji_cli();
    let parsed = root_cmd
        .parse(argv)
        .map_err(|_| "Commande ou option CLI invalide")?;

    let Some(sub_name) = parsed.subcommand_name() else {
        root_cmd
            .write_help(output)
            .map_err(|_| "Erreur d'ecriture aide")?;
        return Ok(());
    };
    let sub = parsed
        .subcommand_matches()
        .map_err(|_| "Arguments de sous-commande invalides")?
        .ok_or("Sous-commande absente")?;

    match sub_name {
        "status" => render_status(engine, color_mode, output),
        "stage" => {
            let name = sub.value("name").ok_or("Chemin de fichier requis")?;
            let content = sub.value("content").ok_or("Contenu requis")?;
            let bytes = content.as_bytes();

            let blob_noun = if sub.flag("delta") {
                engine.stage_file_delta(name, bytes)?
            } else if sub.flag("xor") {
                engine.stage_file_xor_delta(name, bytes)?
            } else if let Some(dict_path) = sub.value("dict") {
                let word_size = sub
                    .value("word_size")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(4);
                engine.stage_file_dict(name, dict_path, word_size, bytes)?
            } else if let Some(seed_str) = sub.value("seed") {
                let seed = seed_str
                    .parse::<u32>()
                    .map_err(|_| "Valeur --seed invalide")?;
                engine.stage_file_seed(name, seed, bytes)?
            } else {
                engine.stage_file(name, bytes)?
            };

            let short = format_noun_short(&blob_noun);
            let desc = wuu::format_stack!("staged {} ({})", name, short.as_str());
            write_ok_custom(output, desc.as_str(), "virtual", 60, color_mode)
                .map_err(|_| "Erreur d'affichage")?;
            Ok(())
        }
        "mkdir" => {
            let path = sub.value("path").ok_or("Chemin de repertoire requis")?;
            let dir_noun = engine.mkdir(path)?;
            let short = format_noun_short(&dir_noun);
            let desc = wuu::format_stack!("mkdir {} ({})", path, short.as_str());
            write_ok_custom(output, desc.as_str(), "dir", 60, color_mode)
                .map_err(|_| "Erreur d'affichage")?;
            Ok(())
        }
        "ls" => {
            let path = sub.value("path").unwrap_or("/");
            render_ls(engine, path, output)
        }
        "cat" => {
            let path = sub.value("path").ok_or("Chemin de fichier requis")?;
            let mut buf = [0u8; TRIE_PAYLOAD_CAPACITY];
            let n = engine.read_working_file(path, &mut buf)?;
            let text = core::str::from_utf8(&buf[..n]).unwrap_or("<binary>");
            writeln!(output, "{text}").map_err(|_| "Erreur d'ecriture cat")?;
            Ok(())
        }
        "commit" => {
            let msg = sub.value("message").ok_or("Message -m requis")?;
            let budget = sub
                .value("budget")
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(1000);
            let ts = (engine.superblock.commit_seq + 1) * 10;
            let commit_noun = engine.commit(msg, ts, budget)?;
            let short = format_noun_short(&commit_noun);
            let desc = wuu::format_stack!("commit {} ({})", short.as_str(), msg);
            write_ok_custom(output, desc.as_str(), "sealed", 60, color_mode)
                .map_err(|_| "Erreur d'affichage")?;
            Ok(())
        }
        "log" => {
            let branch = sub
                .value("branch")
                .and_then(|s| s.parse::<u8>().ok())
                .unwrap_or(engine.superblock.active_branch);
            render_log(engine, branch, output)
        }
        "mount" => {
            let prefix = sub.value("commit").ok_or("Prefixe de commit requis")?;
            let target_branch = sub.value("branch").and_then(|s| s.parse::<u8>().ok());
            let commit_noun = engine.resolve_commit_prefix(prefix)?;
            match engine.mount(&commit_noun, target_branch) {
                Ok(mounted) => {
                    let short = format_noun_short(&mounted.commit_noun);
                    let desc = wuu::format_stack!(
                        "mount {} on branch awq/{}",
                        short.as_str(),
                        mounted.branch
                    );
                    write_ok_custom(output, desc.as_str(), "mounted", 60, color_mode)
                        .map_err(|_| "Erreur d'affichage")?;
                    Ok(())
                }
                Err(reason) => {
                    let _ =
                        write_ko_custom(output, "mount refused by Sceau", reason, 60, color_mode);
                    Err(reason)
                }
            }
        }
        "merge" => {
            let src_str = sub.value("source").ok_or("Branche source requise")?;
            let src_branch = src_str
                .parse::<u8>()
                .map_err(|_| "Numero de branche AWQ invalide")?;
            let dst_branch = engine.superblock.active_branch;
            let msg = sub.value("message").unwrap_or("merge: AWQ branch fusion");
            let ts = (engine.superblock.commit_seq + 1) * 10;
            let merge_noun = engine.merge_branches(src_branch, dst_branch, msg, ts, 1000)?;
            let short = format_noun_short(&merge_noun);
            let desc = wuu::format_stack!(
                "merged awq/{} -> awq/{} ({})",
                src_branch,
                dst_branch,
                short.as_str()
            );
            write_ok_custom(output, desc.as_str(), "merged", 60, color_mode)
                .map_err(|_| "Erreur d'affichage")?;
            Ok(())
        }
        "diff" => {
            let prefix_a = sub.value("commit_a").ok_or("commit_a manquant")?;
            let prefix_b = sub.value("commit_b").ok_or("commit_b manquant")?;
            let file = sub.value("file").ok_or("file manquant")?;
            render_diff(engine, prefix_a, prefix_b, file, color_mode, output)
        }
        "dashboard" => {
            let cols = sub
                .value("cols")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(80);
            let rows = sub
                .value("rows")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(24);
            render_dashboard(engine, cols, rows, color_mode, output)
        }
        _ => Err("Sous-commande inconnue"),
    }
}

/// Affiche le contenu d'un répertoire ou sous-répertoire CubeFS sous forme de [`wuu::Table`].
pub fn render_ls<D: BlockDevice, W: Write>(
    engine: &JiEngine<'_, D>,
    path: &str,
    output: &mut W,
) -> Result<(), &'static str> {
    let entries = engine.list_working_dir(path)?;
    let mut slot_strs: Vec<StackString<256>, 16> = Vec::new();
    let mut type_strs: Vec<&'static str, 16> = Vec::new();
    let mut noun_strs: Vec<StackString<256>, 16> = Vec::new();

    for e in &entries {
        let _ = slot_strs.push(wuu::format_stack!("awq/{}", e.slot));
        let kind = if e.val_type == VAL_TYPE_DIRECTORY {
            "DIR"
        } else {
            "FILE"
        };
        let _ = type_strs.push(kind);
        let _ = noun_strs.push(format_noun_short(&e.target));
    }

    let mut cell_rows: Vec<[&str; 4], 16> = Vec::new();
    for i in 0..entries.len() {
        let _ = cell_rows.push([
            slot_strs[i].as_str(),
            type_strs[i],
            entries[i].name.as_str(),
            noun_strs[i].as_str(),
        ]);
    }

    let mut row_refs: Vec<&[&str], 16> = Vec::new();
    for r in &cell_rows {
        let _ = row_refs.push(r.as_slice());
    }

    let headers = ["SLOT", "TYPE", "NAME", "NOUN"];
    let table = Table::new(&headers, row_refs.as_slice()).with_style(TableStyle::Ascii);
    table
        .write_to(output)
        .map_err(|_| "Erreur d'ecriture table ls")?;
    Ok(())
}

/// Affiche l'état courant du VCS `ji` avec [`wuu::Table`] et [`wuu::write_styled_message`].
pub fn render_status<D: BlockDevice, W: Write>(
    engine: &JiEngine<'_, D>,
    color_mode: ColorMode,
    output: &mut W,
) -> Result<(), &'static str> {
    let branch = engine.superblock.active_branch;
    let head = &engine.superblock.branch_heads[branch as usize];
    let head_short = format_noun_short(head);
    let root_short = format_noun_short(&engine.working_root);
    let staged = engine.staged_node_count();

    let header_msg = wuu::format_stack!(
        "branch=awq/{} head={} root={} virtual_ocean={} layers={}",
        branch,
        head_short.as_str(),
        root_short.as_str(),
        staged,
        engine.active_plan.get_layers().len()
    );
    write_styled_message(output, MessageLevel::Info, header_msg.as_str(), color_mode)
        .map_err(|_| "Erreur d'ecriture status")?;

    let files = engine.list_working_files()?;
    let mut slot_strs: Vec<StackString<256>, 16> = Vec::new();
    let mut noun_strs: Vec<StackString<256>, 16> = Vec::new();

    for f in &files {
        let _ = slot_strs.push(wuu::format_stack!("awq/{}", f.slot));
        let _ = noun_strs.push(format_noun_short(&f.target));
    }

    let mut cell_rows: Vec<[&str; 3], 16> = Vec::new();
    for i in 0..files.len() {
        let _ = cell_rows.push([
            slot_strs[i].as_str(),
            files[i].name.as_str(),
            noun_strs[i].as_str(),
        ]);
    }

    let mut row_refs: Vec<&[&str], 16> = Vec::new();
    for r in &cell_rows {
        let _ = row_refs.push(r.as_slice());
    }

    let headers = ["SLOT", "ENTRY", "NOUN"];
    let table = Table::new(&headers, row_refs.as_slice()).with_style(TableStyle::Ascii);
    table
        .write_to(output)
        .map_err(|_| "Erreur d'ecriture table status")?;
    Ok(())
}

/// Affiche l'historique des commits d'une branche AWQ sous forme de [`wuu::Table`].
pub fn render_log<D: BlockDevice, W: Write>(
    engine: &JiEngine<'_, D>,
    branch: u8,
    output: &mut W,
) -> Result<(), &'static str> {
    let history = engine.history(branch)?;

    let mut seq_strs: Vec<StackString<256>, 16> = Vec::new();
    let mut noun_strs: Vec<StackString<256>, 16> = Vec::new();
    let mut sceau_strs: Vec<StackString<256>, 16> = Vec::new();

    for (noun, commit) in &history {
        let _ = seq_strs.push(wuu::format_stack!("#{}", commit.seq));
        let _ = noun_strs.push(format_noun_short(noun));
        let _ = sceau_strs.push(format_noun_short(&commit.sceau.root));
    }

    let mut cell_rows: Vec<[&str; 4], 16> = Vec::new();
    for i in 0..history.len() {
        let _ = cell_rows.push([
            seq_strs[i].as_str(),
            noun_strs[i].as_str(),
            sceau_strs[i].as_str(),
            history[i].1.message.as_str(),
        ]);
    }

    let mut row_refs: Vec<&[&str], 16> = Vec::new();
    for r in &cell_rows {
        let _ = row_refs.push(r.as_slice());
    }

    let headers = ["SEQ", "COMMIT", "SCEAU", "MESSAGE"];
    let table = Table::new(&headers, row_refs.as_slice()).with_style(TableStyle::Ascii);
    table
        .write_to(output)
        .map_err(|_| "Erreur d'ecriture table log")?;
    Ok(())
}

/// Génère et affiche un diff unifié entre deux commits pour `file` via [`wuu::Diff`].
pub fn render_diff<D: BlockDevice, W: Write>(
    engine: &JiEngine<'_, D>,
    prefix_a: &str,
    prefix_b: &str,
    file: &str,
    color_mode: ColorMode,
    output: &mut W,
) -> Result<(), &'static str> {
    let noun_a = engine.resolve_commit_prefix(prefix_a)?;
    let noun_b = engine.resolve_commit_prefix(prefix_b)?;
    let commit_a = engine.load_commit(&noun_a)?;
    let commit_b = engine.load_commit(&noun_b)?;

    let mut buf_a = [0u8; TRIE_PAYLOAD_CAPACITY];
    let mut buf_b = [0u8; TRIE_PAYLOAD_CAPACITY];
    let len_a = engine
        .read_file(&commit_a.root_tree, file, &mut buf_a)
        .unwrap_or(0);
    let len_b = engine
        .read_file(&commit_b.root_tree, file, &mut buf_b)
        .unwrap_or(0);

    let str_a = core::str::from_utf8(&buf_a[..len_a]).unwrap_or("<binary>");
    let str_b = core::str::from_utf8(&buf_b[..len_b]).unwrap_or("<binary>");

    let mut diff_buf = StackString::<1024>::new();
    let _ = writeln!(&mut diff_buf, "--- a/{file}");
    let _ = writeln!(&mut diff_buf, "+++ b/{file}");
    let _ = writeln!(&mut diff_buf, "@@ -1 +1 @@");
    if str_a != str_b {
        if !str_a.is_empty() {
            let _ = writeln!(&mut diff_buf, "-{str_a}");
        }
        if !str_b.is_empty() {
            let _ = writeln!(&mut diff_buf, "+{str_b}");
        }
    }

    Diff::new(diff_buf.as_str())
        .write_to(output, color_mode)
        .map_err(|_| "Erreur d'ecriture diff")?;
    Ok(())
}

/// Affiche le Dashboard TUI `wuu` (`Wuu::render_dashboard` + `Panel` + `DropDownTerm` + `ProgressBar`)
/// à la fois sur le framebuffer `os_terminal` (s'il est attaché) et dans `output`.
pub fn render_dashboard<D: BlockDevice, W: Write>(
    engine: &JiEngine<'_, D>,
    cols: usize,
    rows: usize,
    color_mode: ColorMode,
    output: &mut W,
) -> Result<(), &'static str> {
    let branch = engine.superblock.active_branch;
    let head = &engine.superblock.branch_heads[branch as usize];
    let head_short = format_noun_short(head);
    let root_short = format_noun_short(&engine.working_root);
    let used_nodes = ((engine.superblock.head_lba.saturating_sub(FIRST_DATA_LBA)) / 8) as usize;
    let core_count = engine.core_ocean.len();
    let virt_count = engine.staged_node_count();
    let layer_count = engine.active_plan.get_layers().len();

    let active_branches = engine
        .superblock
        .branch_heads
        .iter()
        .filter(|n| !n.is_null())
        .count();

    let branch_panel_text = wuu::format_stack!(
        "Active: awq/{}\nHead: {}\nCommit seq: #{}\nInitialized AWQ: {}/16",
        branch,
        head_short.as_str(),
        engine.superblock.commit_seq,
        active_branches
    );

    let ocean_panel_text = wuu::format_stack!(
        "VirtualOcean: {} nodes\nCoreOcean: {} nodes\nDisk LBA head: {}\nPersisted nodes: {}",
        virt_count,
        core_count,
        engine.superblock.head_lba,
        used_nodes
    );

    let plan_panel_text = wuu::format_stack!(
        "Directory: {}\nLayers: {}\nCapabilities: {:?}",
        root_short.as_str(),
        layer_count,
        engine.active_plan.effective_capabilities()
    );

    let files = engine.list_working_files()?;
    let mut tree_panel_text = StackString::<256>::new();
    let _ = writeln!(&mut tree_panel_text, "Entries: {}", files.len());
    for f in files.iter().take(4) {
        let kind = if f.val_type == VAL_TYPE_DIRECTORY {
            "D"
        } else {
            "F"
        };
        let _ = writeln!(&mut tree_panel_text, "[{}] {}", kind, f.name.as_str());
    }

    let ocean_status = if virt_count > 0 {
        Status::Alert
    } else {
        Status::Normal
    };

    let panels = [
        Panel::new(
            "AWQ Branches",
            branch_panel_text.as_str(),
            Status::Focused,
            DropDownTerm::new("awq"),
        ),
        Panel::new(
            "Jinshu Oceans",
            ocean_panel_text.as_str(),
            ocean_status,
            DropDownTerm::new("ocean"),
        ),
        Panel::new(
            "Plans & Sceau",
            plan_panel_text.as_str(),
            Status::Normal,
            DropDownTerm::new("plans"),
        ),
        Panel::new(
            "CubeFS Tree",
            tree_panel_text.as_str(),
            Status::Normal,
            DropDownTerm::open_with("ji", head_short.as_str(), "status"),
        ),
    ];

    // 1. Rendu sur le framebuffer/terminal global `wuu::TERMINAL` si configuré (sans effacer l'écran ANSI si absent)
    if wuu::TERMINAL.lock().is_some() {
        let ui = Wuu::new(Context::Server { cols, rows });
        ui.render_dashboard("JI VCS - AMENTYS OS", &panels);
    }

    // 2. Rendu structuré dans `output` avec `wuu::ProgressBar` et récapitulatif des panneaux
    write_styled_message(
        output,
        MessageLevel::Info,
        "=== JI VCS DASHBOARD (AMENTYS OS) ===",
        color_mode,
    )
    .map_err(|_| "Erreur d'ecriture titre dashboard")?;

    let mut progress = ProgressBar::with_message(64, "NVMe LBA Store");
    progress.set_width(16);
    progress.set_position(used_nodes.min(64));
    progress
        .write_finish(output)
        .map_err(|_| "Erreur d'ecriture ProgressBar")?;

    for p in &panels {
        writeln!(output, "[{}] {}", p.title, p.content.replace('\n', " | "))
            .map_err(|_| "Erreur d'ecriture panel dashboard")?;
    }
    Ok(())
}

/// Sélectionne interactivement un commit [`Noun`] dans l'historique de la branche `branch`
/// via [`wuu::Prompt::noun_select`] et le monte avec [`JiEngine::mount`].
pub fn interactive_mount_prompt<D: BlockDevice, W: Write>(
    engine: &mut JiEngine<'_, D>,
    branch: u8,
    target_branch: Option<u8>,
    keys: &[PromptKey],
    output: &mut W,
) -> Result<Option<MountedOs>, &'static str> {
    let history = engine.history(branch)?;
    let mut nouns: Vec<Noun, 16> = Vec::new();
    for (n, _) in &history {
        let _ = nouns.push(n.clone());
    }

    let mut prompt = Prompt::noun_select("Select commit Noun to mount", nouns.as_slice())
        .map_err(|_| "Aucun commit disponible pour le selecteur interactif")?
        .noun_prefix_len(4);

    prompt
        .write_to(output)
        .map_err(|_| "Erreur de rendu du prompt Noun")?;

    for &key in keys {
        if prompt.state() != PromptState::Pending {
            break;
        }
        prompt.handle_key(key);
        prompt
            .write_to(output)
            .map_err(|_| "Erreur de rafraichissement du prompt Noun")?;
    }

    let selected = match prompt.result() {
        Some(PromptResult::Noun(noun)) => Some(noun.clone()),
        _ => None,
    };

    if let Some(chosen_noun) = selected {
        let mounted = engine.mount(&chosen_noun, target_branch)?;
        Ok(Some(mounted))
    } else {
        Ok(None)
    }
}

/// Saisit interactivement un message de commit via [`wuu::Prompt::text`] et crée le commit scellé.
pub fn interactive_commit_prompt<D: BlockDevice, W: Write>(
    engine: &mut JiEngine<'_, D>,
    keys: &[PromptKey],
    timestamp: u64,
    budget: u32,
    output: &mut W,
) -> Result<Option<Noun>, &'static str> {
    let mut prompt = Prompt::text("Commit message");
    prompt
        .write_to(output)
        .map_err(|_| "Erreur de rendu du prompt texte")?;

    for &key in keys {
        if prompt.state() != PromptState::Pending {
            break;
        }
        prompt.handle_key(key);
        prompt
            .write_to(output)
            .map_err(|_| "Erreur de rafraichissement du prompt texte")?;
    }

    let mut msg_copy = StackString::<256>::new();
    let accepted = if let Some(PromptResult::Text(text)) = prompt.result() {
        let _ = output.write_str("");
        let _ = write!(&mut msg_copy, "{text}");
        !msg_copy.as_str().is_empty()
    } else {
        false
    };

    if accepted {
        let commit_noun = engine.commit(msg_copy.as_str(), timestamp, budget)?;
        Ok(Some(commit_noun))
    } else {
        Ok(None)
    }
}
