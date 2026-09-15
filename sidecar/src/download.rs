//! `rwkv-router fetch` — download the routing model bundle from the Ai00-X
//! model repositories (ModelScope first for CN CDN, then hf-mirror, then HF).
//!
//! Files (remote `rwkv/` prefix, local `<dir>/`):
//! - `router-0.1B-int8.st` — resident 0.1B classifier backbone (int8)
//! - `vocab.json`          — tokenizer
//! - `router_head.json`    — pretrained self-evolving MLP head
//! - optional tier generation models: `rwkv7-3B-int8.st` / `rwkv7-7B-int8.st`
//!   (`--tier-models 3b|7b`), used as R1+/R2/R3 embedded backends.
//!
//! Every file is streamed to `<name>.part` and atomically renamed; an
//! existing file with the remote size is kept (idempotent re-runs).

use std::path::{Path, PathBuf};

use reqwest::Client;

/// Candidate hosts in static fallback order (MS first — CN CDN).
const HOSTS: [&str; 3] = [
    "https://modelscope.cn/models/cgisky/Ai00-X/resolve/master",
    "https://hf-mirror.com/cgisky/ai00-x/resolve/main",
    "https://huggingface.co/cgisky/ai00-x/resolve/main",
];
const REMOTE_PREFIX: &str = "rwkv";

const CLASSIFIER_FILES: [&str; 3] = ["router-0.1B-int8.st", "vocab.json", "router_head.json"];
const TIER_MODEL_3B: &str = "rwkv7-3B-int8.st";
const TIER_MODEL_7B: &str = "rwkv7-7B-int8.st";

/// Downloads the bundle into `dir`. `tier_models`: None | Some("3b"|"7b").
pub fn run(dir: &Path, tier_models: Option<&str>) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;

    let mut files: Vec<&str> = CLASSIFIER_FILES.to_vec();
    match tier_models.map(str::trim) {
        Some(t) if t.eq_ignore_ascii_case("3b") => files.push(TIER_MODEL_3B),
        Some(t) if t.eq_ignore_ascii_case("7b") => files.push(TIER_MODEL_7B),
        Some(other) if !other.is_empty() => {
            return Err(format!("--tier-models expects '3b' or '7b', got '{other}'"));
        }
        _ => {}
    }

    let client = Client::builder()
        // Some CDNs (ModelScope LFS, behind Aliyun WAF) reject header-less
        // clients on the signed redirect hop — always present a UA.
        .user_agent("rwkv-router/0.1 (+https://github.com/cgisky1980/rwkv-router)")
        .connect_timeout(std::time::Duration::from_secs(20))
        .build()
        .map_err(|e| format!("http client: {e}"))?;

    let runtime = tokio::runtime::Runtime::new().map_err(|e| format!("tokio runtime: {e}"))?;
    let mut failures = Vec::new();
    for file in files {
        let target = dir.join(file);
        // Idempotent re-runs: a non-empty completed file is kept as-is.
        // (Remote HEAD length checks are unreliable across mirrors —
        // ModelScope omits Content-Length entirely. Delete the file to
        // force a re-download.)
        if target.metadata().map(|m| m.len() > 0).unwrap_or(false) {
            log::info!("[fetch] {file}: already present, skip");
            continue;
        }
        match runtime.block_on(download_file(&client, file, &target)) {
            Ok(()) => {}
            Err(e) => {
                log::error!("[fetch] {file}: {e}");
                failures.push(format!("{file}: {e}"));
            }
        }
    }
    if failures.is_empty() {
        log::info!("[fetch] done — bundle ready in {}", dir.display());
        Ok(())
    } else {
        Err(format!(
            "some downloads failed:\n  - {}\nhint: check connectivity (or re-run; completed files are skipped)",
            failures.join("\n  - ")
        ))
    }
}

async fn download_file(client: &Client, file: &str, target: &Path) -> Result<(), String> {
    let mut last_err = String::from("no host succeeded");
    for host in HOSTS {
        let url = format!("{host}/{REMOTE_PREFIX}/{file}");
        let resp = match client.get(&url).send().await {
            Ok(r) if r.status().is_success() => r,
            Ok(r) => {
                last_err = format!("{} -> HTTP {}", host_key(host), r.status());
                continue;
            }
            Err(e) => {
                last_err = format!("{} -> {e}", host_key(host));
                continue;
            }
        };
        let total = resp.content_length().unwrap_or(0);
        let part = part_path(target);
        let (written, stream_err) = stream_to_file(resp, &part, total).await;
        // Success = complete byte count. Some CDNs (ModelScope resolve) drop
        // the connection uncleanly right at EOF and some hosts omit
        // Content-Length on HEAD, so the GET body's own length is the
        // authority: writing it fully counts as success even if the stream
        // then errors.
        let complete = (total > 0 && written == total) || (total == 0 && stream_err.is_none());
        if complete {
            if let Some(e) = stream_err {
                log::warn!(
                    "[fetch] {file}: stream ended uncleanly after {written} bytes (accepted): {e}"
                );
            }
            std::fs::rename(&part, target)
                .map_err(|e| format!("rename {}: {e}", part.display()))?;
            log::info!(
                "[fetch] {file}: ok ({:.1} MB via {})",
                written as f64 / 1e6,
                host_key(host)
            );
            return Ok(());
        }
        last_err = match stream_err {
            Some(e) => format!("{} -> {e}", host_key(host)),
            None => format!(
                "{} -> incomplete download ({written} bytes)",
                host_key(host)
            ),
        };
        let _ = std::fs::remove_file(&part);
    }
    Err(last_err)
}

/// Streams the body to `part`; returns (bytes written, stream error if any).
async fn stream_to_file(
    mut resp: reqwest::Response,
    part: &Path,
    total: u64,
) -> (u64, Option<String>) {
    let mut file = match std::fs::File::create(part) {
        Ok(f) => f,
        Err(e) => return (0, Some(format!("create {}: {e}", part.display()))),
    };
    let mut written: u64 = 0;
    let mut next_log = if total > 0 { total / 10 } else { u64::MAX };
    use std::io::Write;
    loop {
        match resp.chunk().await {
            Ok(Some(chunk)) => {
                if let Err(e) = file.write_all(&chunk) {
                    return (written, Some(format!("write at {written}: {e}")));
                }
                written += chunk.len() as u64;
                if total > 0 && written >= next_log {
                    log::info!(
                        "[fetch] {}: {:.0}%",
                        part.file_name().unwrap_or_default().to_string_lossy(),
                        100.0 * written as f64 / total as f64
                    );
                    next_log += total / 10;
                }
            }
            Ok(None) => return (written, None),
            Err(e) => return (written, Some(format!("read stream at {written}: {e}"))),
        }
    }
}

fn part_path(target: &Path) -> PathBuf {
    let mut name = target.file_name().unwrap_or_default().to_os_string();
    name.push(".part");
    target.with_file_name(name)
}

fn host_key(host: &str) -> String {
    if host.contains("modelscope") {
        "ms".into()
    } else if host.contains("hf-mirror") {
        "hf-mirror".into()
    } else {
        "hf".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn part_path_appends_suffix() {
        let p = part_path(Path::new("models/router-0.1B-int8.st"));
        assert_eq!(p, Path::new("models/router-0.1B-int8.st.part"));
    }

    #[test]
    fn host_key_mapping() {
        assert_eq!(host_key(HOSTS[0]), "ms");
        assert_eq!(host_key(HOSTS[1]), "hf-mirror");
        assert_eq!(host_key(HOSTS[2]), "hf");
    }
}
