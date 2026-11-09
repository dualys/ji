#![cfg_attr(all(not(test), target_os = "none"), no_main, no_std)]

use ji::cli::execute_cli;
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
    let device = MemBlockDevice::new(64);
    let Ok(mut engine) = JiEngine::init(
        &device,
        device.total_lbas(),
        JiEngine::<MemBlockDevice>::default_kernel_layer(),
    ) else {
        return;
    };

    let mut out = StackString::<256>::new();
    let _ = execute_cli(&mut engine, &["status"], ColorMode::Plain, &mut out);
}