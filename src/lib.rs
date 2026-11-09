#![cfg_attr(not(test), no_std)]

extern crate alloc;

pub use prisme;

pub mod cli;
pub mod commit;
pub mod device;
pub mod engine;
pub mod superblock;
pub mod tree;

#[cfg(test)]
mod tests {
    use crate::cli::execute_cli;
    use crate::device::MemBlockDevice;
    use crate::engine::{JiEngine, format_noun_short};
    use crate::superblock::{FIRST_DATA_LBA, JiSuperblock};
    use crate::tree::{blob_inode_and_extent, to_cubefs_btree_node};
    use cubefs::KpackReader;
    use kpack::Opcode;
    use noun::Noun;
    use plans::layer::{Capabilities, Layer};
    use plans::sceau::Verdict;
    use wuu::{ColorMode, StackString};

    #[test]
    fn test_superblock_roundtrip_and_corruption_detection() {
        let device = MemBlockDevice::new(32);
        let mut sb = JiSuperblock::new();
        sb.active_branch = 3;
        sb.commit_seq = 7;
        sb.head_lba = 24;
        sb.branch_heads[3] = Noun::of(b"branch-3-head");
        sb.write_to_device(&device);

        let loaded = JiSuperblock::read_from_device(&device).expect("Superbloc valide");
        assert_eq!(loaded.active_branch, 3);
        assert_eq!(loaded.commit_seq, 7);
        assert_eq!(loaded.head_lba, 24);
        assert_eq!(loaded.branch_heads[3], Noun::of(b"branch-3-head"));

        // Altération d'un octet sur le disque au LBA 0 -> détection BLAKE3 garantie
        let mut raw = device.read_block(0).unwrap();
        raw[16] ^= 0xFF;
        device.write_block(0, &raw).unwrap();
        assert!(JiSuperblock::read_from_device(&device).is_err());
    }

    #[test]
    fn test_init_stage_commit_and_time_travel_mount() {
        let device = MemBlockDevice::new(64);
        let mut engine = JiEngine::init(
            &device,
            device.total_lbas(),
            JiEngine::<MemBlockDevice>::default_kernel_layer(),
        )
        .expect("ji init doit reussir");

        // Vérifie que le superbloc a bien été écrit au LBA 0 avec le commit Genesis (#1)
        let disk_sb = JiSuperblock::read_from_device(&device).unwrap();
        assert_eq!(disk_sb.commit_seq, 1);
        assert!(disk_sb.head_lba > FIRST_DATA_LBA);

        // --- COMMIT #2 : État v1 de l'OS (noyau v1 + config v1 + couche graphique v1) ---
        engine
            .stage_file("kernel.sys", b"AMENTYS-KERNEL-V1.0-STABLE")
            .unwrap();
        engine
            .stage_file("boot.cfg", b"console=tty0 root=noun:v1")
            .unwrap();
        // Test de compression RLE automatique dans Kpack
        let rle_noun = engine.stage_file("splash.raw", &[0xAA; 512]).unwrap();
        let rle_node = engine.fetch_node_ref(&rle_noun).unwrap();
        assert_eq!(rle_node.opcode, Opcode::Rle as u8);

        let tui_layer_v1 = Layer::new(
            "wuu-tui",
            1,
            "Wuu Terminal UI v1",
            Noun::of(b"wuu-tui-v1"),
            Capabilities::ReadWrite,
        );
        engine.add_os_layer(tui_layer_v1).unwrap();

        let commit_v1 = engine
            .commit("feat(os): release Amentys v1.0", 100, 500)
            .expect("commit v1");

        // --- COMMIT #3 : État v2 de l'OS (patch Delta Kpack sur kernel.sys + nouvelle config + couche réseau) ---
        let delta_noun = engine
            .stage_file_delta("kernel.sys", b"AMENTYS-KERNEL-V2.0-EXPERIMENTAL")
            .expect("stage delta");
        let delta_node = engine.fetch_node_ref(&delta_noun).unwrap();
        assert_eq!(delta_node.opcode, Opcode::Delta as u8);

        engine
            .stage_file("boot.cfg", b"console=tty0 root=noun:v2 debug=1")
            .unwrap();
        let net_layer_v2 = Layer::new(
            "hamon-net",
            2,
            "Hamon Hypercube Network Stack",
            Noun::of(b"hamon-net-v2"),
            Capabilities::All,
        );
        engine.add_os_layer(net_layer_v2).unwrap();

        let commit_v2 = engine
            .commit("feat(os): upgrade to v2.0 with delta kernel", 200, 900)
            .expect("commit v2");

        // Vérification de la lecture sur le commit v2 (reconstruit via Delta Kpack !)
        let mut buf = [0u8; 128];
        let n = engine.read_working_file("kernel.sys", &mut buf).unwrap();
        assert_eq!(&buf[..n], b"AMENTYS-KERNEL-V2.0-EXPERIMENTAL");
        assert_eq!(engine.active_plan.get_layers().len(), 3);

        // --- TIME-TRAVEL : `ji mount <commit_v1>` ---
        // On vide même le CoreOcean pour prouver que `CoreOceanBuilder::deep_load` recharge tout depuis le disque NVMe !
        let v1_commit_node = engine.fetch_node_ref(&commit_v1).unwrap();
        engine.core_ocean.clear();
        engine
            .core_ocean
            .insert(commit_v1.clone(), v1_commit_node);

        let mounted_v1 = engine
            .mount(&commit_v1, None)
            .expect("ji mount commit_v1 doit reussir");
        assert_eq!(mounted_v1.verdict, Verdict::Accept);
        assert_eq!(mounted_v1.commit.seq, 2);

        // L'OS est revenu exactement à l'état du commit v1 :
        let n_k1 = engine.read_working_file("kernel.sys", &mut buf).unwrap();
        assert_eq!(&buf[..n_k1], b"AMENTYS-KERNEL-V1.0-STABLE");

        let n_c1 = engine.read_working_file("boot.cfg", &mut buf).unwrap();
        assert_eq!(&buf[..n_c1], b"console=tty0 root=noun:v1");

        let mut splash_buf = [0u8; 512];
        let n_s1 = engine
            .read_working_file("splash.raw", &mut splash_buf)
            .unwrap();
        assert_eq!(n_s1, 512);
        assert_eq!(splash_buf, [0xAA; 512]);

        // Les Layers du Plan de l'OS sont elles aussi revenues exactement à celles du commit v1 (2 couches au lieu de 3) !
        assert_eq!(engine.active_plan.get_layers().len(), 2);
        assert_eq!(
            engine.active_plan.effective_capabilities(),
            Capabilities::ReadWrite
        );

        // Et on peut remonter le commit v2 sur une autre branche AWQ (ex. branche 4) en parallèle !
        let mounted_v2_on_b4 = engine
            .mount(&commit_v2, Some(4))
            .expect("ji mount commit_v2 sur branche AWQ 4");
        assert_eq!(mounted_v2_on_b4.branch, 4);
        assert_eq!(mounted_v2_on_b4.verdict, Verdict::Accept);
        let n_k2 = engine.read_working_file("kernel.sys", &mut buf).unwrap();
        assert_eq!(&buf[..n_k2], b"AMENTYS-KERNEL-V2.0-EXPERIMENTAL");
        assert_eq!(engine.active_plan.get_layers().len(), 3);
    }

    #[test]
    fn test_awq_branch_merge_and_cubefs_projection() {
        let device = MemBlockDevice::new(64);
        let mut engine = JiEngine::init(
            &device,
            device.total_lbas(),
            JiEngine::<MemBlockDevice>::default_kernel_layer(),
        )
        .unwrap();

        // Branche 0 : fichier de base
        engine.stage_file("base.lys", b"plan = main").unwrap();
        engine.commit("base commit on awq/0", 10, 500).unwrap();

        // Passage sur la branche AWQ 1 : ajout d'un driver audio et d'une Layer audio
        engine.switch_branch(1).unwrap();
        engine.stage_file("audio.drv", b"AUDIO-DSP-FIRMWARE").unwrap();
        engine
            .add_os_layer(Layer::new(
                "audio-dsp",
                1,
                "Audio Layer",
                Noun::of(b"audio"),
                Capabilities::ReadExecute,
            ))
            .unwrap();
        engine.commit("feat(awq/1): add audio driver", 20, 500).unwrap();

        // Retour sur la branche AWQ 0 : ajout d'un driver GPU et d'une Layer GPU
        engine.switch_branch(0).unwrap();
        engine.stage_file("gpu.drv", b"GPU-SHADER-CORE").unwrap();
        engine
            .add_os_layer(Layer::new(
                "gpu-core",
                1,
                "GPU Layer",
                Noun::of(b"gpu"),
                Capabilities::ReadWrite,
            ))
            .unwrap();
        engine.commit("feat(awq/0): add gpu driver", 30, 500).unwrap();

        // Fusion de la branche 1 dans la branche 0
        let merge_commit_noun = engine
            .merge_branches(1, 0, "merge awq/1 into awq/0", 40, 1000)
            .expect("merge_branches doit reussir");
        let merge_commit = engine.load_commit(&merge_commit_noun).unwrap();
        assert!(!merge_commit.second_parent.is_null());

        // Les 3 fichiers ("base.lys", "audio.drv", "gpu.drv") sont présents dans l'arbre fusionné !
        let mut buf = [0u8; 64];
        let n1 = engine.read_working_file("base.lys", &mut buf).unwrap();
        assert_eq!(&buf[..n1], b"plan = main");
        let n2 = engine.read_working_file("audio.drv", &mut buf).unwrap();
        assert_eq!(&buf[..n2], b"AUDIO-DSP-FIRMWARE");
        let n3 = engine.read_working_file("gpu.drv", &mut buf).unwrap();
        assert_eq!(&buf[..n3], b"GPU-SHADER-CORE");

        // Les Layers des deux branches ont été fusionnées de manière déterministe via `Plan::merge_with`
        assert_eq!(engine.active_plan.get_layers().len(), 3);

        // Vérification de la projection CubeFS BTreeNode et KpackReader
        let root_dir = engine.fetch_node_ref(&engine.working_root).unwrap();
        let btree_node = to_cubefs_btree_node(&root_dir);
        assert_eq!(&btree_node.magic, b"BNOD");
        assert_eq!(btree_node.entry_count, 3);

        let files = engine.list_working_files().unwrap();
        let gpu_rec = files.iter().find(|r| r.name.as_str() == "gpu.drv").unwrap();
        let gpu_node = engine.fetch_node_ref(&gpu_rec.target).unwrap();
        let (inode, extent) = blob_inode_and_extent(&gpu_rec.target, &gpu_node);
        assert_eq!(inode.size, 15);

        let extents = [extent];
        let reader = KpackReader::new(&extents);
        let mut kpack_stream = [0u8; 32];
        kpack_stream[0] = gpu_node.opcode;
        let len_le = (gpu_node.param as u16).to_le_bytes();
        kpack_stream[1] = len_le[0];
        kpack_stream[2] = len_le[1];
        kpack_stream[3..18].copy_from_slice(&gpu_node.payload[..15]);

        let mut section = [0u8; 15];
        reader
            .read_into_section(0, &mut section, |_chunk_id| Ok(&kpack_stream[..18]))
            .unwrap();
        assert_eq!(&section, b"GPU-SHADER-CORE");
    }

    #[test]
    fn test_cli_commands_with_wuu() {
        let device = MemBlockDevice::new(64);
        let mut engine = JiEngine::init(
            &device,
            device.total_lbas(),
            JiEngine::<MemBlockDevice>::default_kernel_layer(),
        )
        .unwrap();

        let mut out = StackString::<2048>::new();

        // 1. `ji stage init.lys "boot = fast"`
        execute_cli(
            &mut engine,
            &["stage", "init.lys", "boot = fast"],
            ColorMode::Plain,
            &mut out,
        )
        .unwrap();
        assert!(out.as_str().contains("[ virtual ]"));

        // 2. `ji commit -m "first config"`
        out = StackString::<2048>::new();
        execute_cli(
            &mut engine,
            &["commit", "-m", "first config"],
            ColorMode::Plain,
            &mut out,
        )
        .unwrap();
        assert!(out.as_str().contains("[ sealed ]"));
        let c1_short = format_noun_short(&engine.superblock.branch_heads[0]);

        // 3. `ji stage -d init.lys "boot = safe"` + `ji commit -m "second config"`
        out = StackString::<2048>::new();
        execute_cli(
            &mut engine,
            &["stage", "-d", "init.lys", "boot = safe"],
            ColorMode::Plain,
            &mut out,
        )
        .unwrap();
        execute_cli(
            &mut engine,
            &["commit", "-m", "second config"],
            ColorMode::Plain,
            &mut out,
        )
        .unwrap();
        let c2_short = format_noun_short(&engine.superblock.branch_heads[0]);

        // 4. `ji diff <c1> <c2> init.lys`
        out = StackString::<2048>::new();
        execute_cli(
            &mut engine,
            &["diff", c1_short.as_str(), c2_short.as_str(), "init.lys"],
            ColorMode::Plain,
            &mut out,
        )
        .unwrap();
        assert!(out.as_str().contains("-boot = fast"));
        assert!(out.as_str().contains("+boot = safe"));

        // 5. `ji log`
        out = StackString::<2048>::new();
        execute_cli(&mut engine, &["log"], ColorMode::Plain, &mut out).unwrap();
        assert!(out.as_str().contains("second config"));
        assert!(out.as_str().contains("first config"));

        // 6. `ji mount <c1> -b 2`
        out = StackString::<2048>::new();
        execute_cli(
            &mut engine,
            &["mount", "-b", "2", c1_short.as_str()],
            ColorMode::Plain,
            &mut out,
        )
        .unwrap();
        assert!(out.as_str().contains("[ mounted ]"));
        assert_eq!(engine.superblock.active_branch, 2);

        let mut file_buf = [0u8; 32];
        let n = engine.read_working_file("init.lys", &mut file_buf).unwrap();
        assert_eq!(&file_buf[..n], b"boot = fast");

        // 7. `ji status`
        out = StackString::<2048>::new();
        execute_cli(&mut engine, &["status"], ColorMode::Plain, &mut out).unwrap();
        assert!(out.as_str().contains("branch=awq/2"));
        assert!(out.as_str().contains("init.lys"));
    }

    #[test]
    fn test_cold_reboot_open_and_layer_persistence_on_disk() {
        let device = MemBlockDevice::new(64);
        let commit_v1: Noun;
        let commit_v2: Noun;
        let head_lba_before_reboot: u64;

        // Phase 1 : Session initiale avant extinction de l'OS
        {
            let mut engine = JiEngine::init(
                &device,
                device.total_lbas(),
                JiEngine::<MemBlockDevice>::default_kernel_layer(),
            )
            .expect("init");

            engine
                .stage_file("kernel.sys", b"AMENTYS-KERNEL-V1.0")
                .unwrap();
            engine.stage_file("initrd.img", &[0x5A; 256]).unwrap();
            engine
                .add_os_layer(Layer::new(
                    "vfs-cubefs",
                    1,
                    "CubeFS Immutable Recipe Filesystem",
                    Noun::of(b"vfs-cubefs-v1"),
                    Capabilities::ReadWrite,
                ))
                .unwrap();
            commit_v1 = engine.commit("release v1.0", 100, 600).unwrap();

            engine
                .stage_file_delta("kernel.sys", b"AMENTYS-KERNEL-V2.0-COLD-BOOT")
                .unwrap();
            engine
                .add_os_layer(Layer::new(
                    "wuu-console",
                    2,
                    "Wuu Zero-Heap Console & TUI",
                    Noun::of(b"wuu-console-v2"),
                    Capabilities::All,
                ))
                .unwrap();
            commit_v2 = engine.commit("release v2.0", 200, 1200).unwrap();
            head_lba_before_reboot = engine.superblock.head_lba;
        }
        // Ici `engine` est intégralement détruit : plus aucun état en RAM (`core_ocean`, `disk_index`, `commit_layers`) !

        // Phase 2 : Redémarrage à froid (`JiEngine::open`) uniquement à partir du BlockDevice
        let mut reopened =
            JiEngine::open(&device, device.total_lbas()).expect("JiEngine::open au reboot");

        assert_eq!(reopened.superblock.commit_seq, 3);
        assert_eq!(reopened.superblock.head_lba, head_lba_before_reboot);
        assert_eq!(reopened.superblock.branch_heads[0], commit_v2);

        // Vérifie que les 3 Layers ont été intégralement désérialisées depuis le TrieNode sur disque
        let restored_layers = reopened.active_plan.get_layers();
        assert_eq!(restored_layers.len(), 3);
        assert_eq!(restored_layers[0].name, "amentys-core");
        assert_eq!(restored_layers[1].name, "vfs-cubefs");
        assert_eq!(
            restored_layers[1].description,
            "CubeFS Immutable Recipe Filesystem"
        );
        assert_eq!(restored_layers[1].capabilities, Capabilities::ReadWrite);
        assert_eq!(restored_layers[2].name, "wuu-console");
        assert_eq!(restored_layers[2].version, 2);
        assert_eq!(restored_layers[2].capabilities, Capabilities::All);

        // Vérifie que les fichiers (y compris Delta Kpack et RLE) sont lisibles après reboot
        let mut buf = [0u8; 256];
        let n_k = reopened.read_working_file("kernel.sys", &mut buf).unwrap();
        assert_eq!(&buf[..n_k], b"AMENTYS-KERNEL-V2.0-COLD-BOOT");

        let n_i = reopened.read_working_file("initrd.img", &mut buf).unwrap();
        assert_eq!(n_i, 256);
        assert_eq!(&buf[..n_i], &[0x5A; 256]);

        // Phase 3 : Time-Travel (`ji mount`) vers `commit_v1` après un reboot à froid
        let mounted_v1 = reopened
            .mount(&commit_v1, None)
            .expect("mount v1 apres reboot");
        assert_eq!(mounted_v1.verdict, Verdict::Accept);
        assert_eq!(reopened.active_plan.get_layers().len(), 2);
        assert_eq!(reopened.active_plan.get_layers()[1].name, "vfs-cubefs");

        let n_k1 = reopened.read_working_file("kernel.sys", &mut buf).unwrap();
        assert_eq!(&buf[..n_k1], b"AMENTYS-KERNEL-V1.0");

        // Phase 4 : Nouveau commit après reboot -> allocation LBA continue sans écrasement
        reopened.stage_file("post_reboot.cfg", b"ok=true").unwrap();
        let commit_v3 = reopened
            .commit("post-reboot commit", 300, 800)
            .expect("commit post-reboot");
        assert!(!commit_v3.is_null());
        assert!(reopened.superblock.head_lba > head_lba_before_reboot);

        // Phase 5 : Détection par Sceau::weigh d'une altération d'une Layer sur le disque
        let v3_lba = reopened.disk_index.get_lba(&commit_v3).unwrap();
        let mut raw_commit_block = device.read_block(v3_lba).unwrap();
        // On altère le dernier octet sérialisé du commit (dans la description de la dernière Layer)
        let param_len = u32::from_le_bytes([
            raw_commit_block[4],
            raw_commit_block[5],
            raw_commit_block[6],
            raw_commit_block[7],
        ]) as usize;
        let payload_offset = 8 + 512; // en-tête TrieNode (8) + 16 branches Noun (512)
        raw_commit_block[payload_offset + param_len - 1] ^= 0x01;
        device.write_block(v3_lba, &raw_commit_block).unwrap();

        reopened.core_ocean.remove(&commit_v3);
        let err = reopened.load_commit(&commit_v3).unwrap_err();
        assert_eq!(err, "noun mismatch");
    }
}
