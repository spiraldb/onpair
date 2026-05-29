// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-core L2 cache size, used to pick the decode table layout.
//!
//! The fat decode table is `dict_tokens * 16` bytes; if it would exceed L2 the
//! decoder falls back to the half-size `entries` layout, which stays cache
//! resident (see [`crate::decompress::plan`]). Detected once from sysfs on
//! Linux, with a conservative default elsewhere.

use std::sync::OnceLock;

/// Conservative fallback when the L2 size can't be read.
const DEFAULT_L2: usize = 512 * 1024;

/// Per-core L2 cache size in bytes, detected once and cached.
pub(crate) fn l2_cache_bytes() -> usize {
    static L2: OnceLock<usize> = OnceLock::new();
    *L2.get_or_init(detect_l2)
}

#[cfg(target_os = "linux")]
fn detect_l2() -> usize {
    use std::fs;
    let base = "/sys/devices/system/cpu/cpu0/cache";
    for i in 0..16 {
        let dir = format!("{base}/index{i}");
        let Ok(level) = fs::read_to_string(format!("{dir}/level")) else {
            break; // no more cache indices
        };
        if level.trim() == "2"
            && let Some(size) = fs::read_to_string(format!("{dir}/size"))
                .ok()
                .and_then(|s| parse_size(&s))
        {
            return size;
        }
    }
    DEFAULT_L2
}

#[cfg(not(target_os = "linux"))]
fn detect_l2() -> usize {
    DEFAULT_L2
}

/// Parse a sysfs cache size string like `1024K` or `2M`.
#[cfg(target_os = "linux")]
fn parse_size(s: &str) -> Option<usize> {
    let s = s.trim();
    let (num, mult) = match s.as_bytes().last() {
        Some(b'K' | b'k') => (&s[..s.len() - 1], 1024),
        Some(b'M' | b'm') => (&s[..s.len() - 1], 1024 * 1024),
        _ => (s, 1),
    };
    num.trim().parse::<usize>().ok().map(|n| n * mult)
}
