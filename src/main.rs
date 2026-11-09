#![cfg_attr(all(not(test), target_os = "none"), no_main, no_std)]

use ji::cli::execute_cli;
#[cfg(not(target_os = "none"))]
use ji::device::FileBlockDevice;
use ji::device::MemBlockDevice;
use ji::engine::JiEngine;
use wuu::{ColorMode, StackString};

#[cfg(all(not(test), target_os = "none"))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {
        core::hint::spin_loop();
    }
}

#[cfg(not(target_os = "none"))]
fn main() {
    let raw_args: std::vec::Vec<std::string::String> = std::env::args().skip(1).collect();

    let (disk_path, cli_slice): (Option<&str>, &[std::string::String]) =
        if raw_args.len() >= 2 && raw_args[0] == "--disk" {
            (Some(raw_args[1].as_str()), &raw_args[2..])
        } else {
            (None, &raw_args[..])
        };

    let argv: std::vec::Vec<&str> = if cli_slice.is_empty() {
        std::vec!["status"]
    } else {
        cli_slice
            .iter()
            .map(std::string::String::as_str)
            .collect()
    };

    let mut out = StackString::<4096>::new();

    if let Some(path) = disk_path {
        let Ok(device) = FileBlockDevice::open_or_create(path, 128) else {
            return;
        };
        let mut engine = if let Ok(opened) = JiEngine::open(&device, device.total_lbas()) {
            opened
        } else if let Ok(initialized) = JiEngine::init(
            &device,
            device.total_lbas(),
            JiEngine::<FileBlockDevice>::default_kernel_layer(),
        ) {
            initialized
        } else {
            return;
        };
        let _ = execute_cli(&mut engine, &argv, ColorMode::Ansi, &mut out);
    } else {
        let device = MemBlockDevice::new(64);
        let Ok(mut engine) = JiEngine::init(
            &device,
            device.total_lbas(),
            JiEngine::<MemBlockDevice>::default_kernel_layer(),
        ) else {
            return;
        };
        let _ = execute_cli(&mut engine, &argv, ColorMode::Ansi, &mut out);
    }

    if !out.as_str().is_empty() {
        print!("{}", out.as_str());
    }
}