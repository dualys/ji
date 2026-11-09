use crate::engine::{JiEngine, format_noun_short};
use crate::tree::TRIE_PAYLOAD_CAPACITY;
use core::fmt::Write;
use heapless::Vec;
use jinshu::storage::BlockDevice;
use wuu::{
    Arg, ColorMode, Command, Diff, MessageLevel, StackString, Table, TableStyle, write_ko_custom,
    write_ok_custom, write_styled_message,
};

const STATUS_ARGS: [Arg<'static>; 1] = [Arg::new("short")
    .short('s')
    .long("short")
    .help("Affichage compact")];

const STAGE_ARGS: [Arg<'static>; 3] = [
    Arg::new("delta")
        .short('d')
        .long("delta")
        .help("Encode par recette Delta kpack vis-a-vis de la version parente"),
    Arg::new("name")
        .takes_value("NAME")
        .required()
        .help("Nom du fichier ou composant OS"),
    Arg::new("content")
        .takes_value("DATA")
        .required()
        .help("Contenu du fichier"),
];

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
        .help("Nom du fichier a comparer"),
];

const SUBCOMMANDS: [Command<'static>; 6] = [
    Command::new("status")
        .about("Affiche l'etat de la branche AWQ, du Plan et du VirtualOcean")
        .args(&STATUS_ARGS),
    Command::new("stage")
        .about("Ecrit ou patche (Delta kpack) un fichier en Copy-on-Write dans VirtualOcean")
        .args(&STAGE_ARGS),
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
];

const DIFF_COMMAND: Command<'static> = Command::new("diff")
    .about("Affiche le diff unifie d'un fichier entre deux commits")
    .args(&DIFF_ARGS);

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

    if argv[0] == "diff" {
        let sub = DIFF_COMMAND
            .parse(&argv[1..])
            .map_err(|_| "Arguments invalides pour `ji diff`")?;
        let prefix_a = sub.value("commit_a").ok_or("commit_a manquant")?;
        let prefix_b = sub.value("commit_b").ok_or("commit_b manquant")?;
        let file = sub.value("file").ok_or("file manquant")?;
        return render_diff(engine, prefix_a, prefix_b, file, color_mode, output);
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
            let name = sub.value("name").ok_or("Nom de fichier requis")?;
            let content = sub.value("content").ok_or("Contenu requis")?;
            let blob_noun = if sub.flag("delta") {
                engine.stage_file_delta(name, content.as_bytes())?
            } else {
                engine.stage_file(name, content.as_bytes())?
            };
            let short = format_noun_short(&blob_noun);
            let desc = wuu::format_stack!("staged {} ({})", name, short.as_str());
            write_ok_custom(output, desc.as_str(), "virtual", 60, color_mode)
                .map_err(|_| "Erreur d'affichage")?;
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
                    let _ = write_ko_custom(output, "mount refused by Sceau", reason, 60, color_mode);
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
        _ => Err("Sous-commande inconnue"),
    }
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
