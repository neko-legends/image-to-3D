// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  VENDORED FILE — DO NOT EDIT HERE.                                         ║
// ║  Canonical source: github.com/neko-legends/NekoLegendsAI-Shared           ║
// ║                    rust/neko_store.rs                                      ║
// ║  Edits here will be OVERWRITTEN. Fix the canonical copy, then re-vendor:   ║
// ║    python NekoLegendsAI-Shared/scripts/neko_suite_doctor.py --fix          ║
// ╚══════════════════════════════════════════════════════════════════════════╝
//! neko_store — shared local AI store resolver for Neko Legends / ForPublic apps.
//!
//! Vendored module: not every app uses every public method, so allow dead_code.
#![allow(dead_code)]
//!
//! This is a single-file, dependency-light module (only `serde_json`, which every
//! Neko app already depends on). Vendor it by copying into `src-tauri/src/` and
//! adding `mod neko_store;` to your `lib.rs`.
//!
//! # What problem this solves
//!
//! Every GPU app in the suite installs its own multi-GB PyTorch + CUDA wheels and
//! re-downloads the same Hugging Face models. This module gives every app a single
//! shared, *visible* store so that:
//!
//!   * Hugging Face models download **once** (shared `HF_HOME`).
//!   * PyTorch / wheel downloads are cached once (shared `PIP_CACHE_DIR`,
//!     `UV_CACHE_DIR`). With `uv`, packages are hardlinked into each app venv on
//!     the same volume, so the bytes exist once on disk.
//!
//! # Hard requirement: works standalone, no controller required
//!
//! The store is a **convention with discovery + fallback**, never a dependency.
//! A portable app with nothing else installed still works: it discovers (or
//! creates) the default shared folder, and if even that is not writable it falls
//! back to a fully app-local store. The Neko Legends Control Center, *if present*,
//! only points at and manages this store — it is never required.
//!
//! # Discovery order (first hit wins)
//!
//!   1. Per-app override env var `NEKO_<APP>_STORE` (e.g. `NEKO_ANGLEFORGE_STORE`).
//!   2. Global env var `NEKO_AI_HOME`.
//!   3. Pointer file `<home>/NekoLegendsAI/store.json` (key `"root"`).
//!   4. Default visible folder `<volume-of-app>/NekoLegendsAI`.
//!   5. App-local fallback (the path the caller passes as `app_local_fallback`).
//!
//! The resolved store layout is intentionally flat and human-readable:
//!
//! ```text
//! NekoLegendsAI/
//!   store.json         (marker / metadata, written on first use)
//!   models/            HF_HOME — models downloaded once
//!     huggingface/
//!   cache/
//!     pip/             PIP_CACHE_DIR
//!     uv/              UV_CACHE_DIR
//!     torch/           TORCH_HOME
//!   envs/              per-app venvs live here when the store is shared
//!     <app>/
//! ```

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Resolved, ready-to-use paths for a shared (or app-local) Neko AI store.
#[derive(Debug, Clone)]
pub struct NekoStore {
    /// Root of the store (e.g. `D:/NekoLegendsAI` or the app-local fallback).
    pub root: PathBuf,
    /// `true` when this is a shared store, `false` when we fell back to app-local.
    pub shared: bool,
    /// Per-app virtual environment directory inside the store.
    pub env_dir: PathBuf,
    /// Shared models directory (used as `HF_HOME`).
    pub models_dir: PathBuf,
    /// Shared pip download cache.
    pub pip_cache: PathBuf,
    /// Shared uv cache.
    pub uv_cache: PathBuf,
    /// Shared torch hub cache (`TORCH_HOME`).
    pub torch_home: PathBuf,
}

impl NekoStore {
    /// Slug-friendly env var name component: `AngleForge` -> `ANGLEFORGE`.
    fn app_env_token(app_slug: &str) -> String {
        app_slug
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .collect::<String>()
            .to_ascii_uppercase()
    }

    /// Resolve the store for an app, defaulting the shared-store volume to the
    /// app's own data dir volume.
    ///
    /// * `app_slug` — short app identifier, e.g. `"angleforge"`. Used for the
    ///   per-app override env var and the per-app venv subfolder.
    /// * `app_local_fallback` — the app's own private data dir to use if no
    ///   shared store can be created (keeps portable apps fully self-contained).
    pub fn resolve(app_slug: &str, app_local_fallback: &Path) -> NekoStore {
        Self::resolve_with_volume_hint(app_slug, app_local_fallback, None)
    }

    /// Resolve the store, optionally preferring a different volume for the
    /// default shared-store location.
    ///
    /// On Windows the Tauri app-data dir lives on `C:`, but portable apps are
    /// often installed on a big data drive (e.g. `D:`). Passing the executable's
    /// directory as `volume_hint` makes the default store land on that drive
    /// (`D:\NekoLegendsAI`) with zero configuration — models are huge and belong
    /// on the roomy drive next to the app. Discovery steps 1–3 (env vars and the
    /// pointer file) always override this hint.
    pub fn resolve_with_volume_hint(
        app_slug: &str,
        app_local_fallback: &Path,
        volume_hint: Option<&Path>,
    ) -> NekoStore {
        let token = Self::app_env_token(app_slug);

        // 1. Per-app override.
        let per_app_key = format!("NEKO_{token}_STORE");
        if let Some(root) = env_path(&per_app_key) {
            if let Some(store) = Self::try_build(&root, app_slug) {
                return store;
            }
        }

        // 2. Global env var.
        if let Some(root) = env_path("NEKO_AI_HOME") {
            if let Some(store) = Self::try_build(&root, app_slug) {
                return store;
            }
        }

        // 3. Pointer file under the user profile.
        if let Some(root) = read_pointer_root() {
            if let Some(store) = Self::try_build(&root, app_slug) {
                return store;
            }
        }

        // 4. Default visible folder on the same volume as the app. Prefer the
        //    caller's volume hint (e.g. the executable dir on a big data drive),
        //    falling back to the app-local data dir's volume.
        let volume_basis = volume_hint.unwrap_or(app_local_fallback);
        if let Some(root) = default_store_root(volume_basis) {
            if let Some(store) = Self::try_build(&root, app_slug) {
                // Best-effort: drop a pointer file so sibling apps discover it too.
                let _ = write_pointer_root(&root);
                return store;
            }
        }

        // 5. App-local fallback — always works, fully portable.
        Self::build_local(app_local_fallback, app_slug)
    }

    /// Attempt to create + validate a shared store rooted at `root`.
    /// Returns `None` if the root is not creatable/writable.
    fn try_build(root: &Path, app_slug: &str) -> Option<NekoStore> {
        if fs::create_dir_all(root).is_err() {
            return None;
        }
        if !dir_is_writable(root) {
            return None;
        }
        let store = Self::layout(root, app_slug, true);
        store.ensure_dirs().ok()?;
        store.write_marker();
        Some(store)
    }

    /// Build an app-local store (no shared dirs); always succeeds best-effort.
    fn build_local(app_local_fallback: &Path, app_slug: &str) -> NekoStore {
        let _ = fs::create_dir_all(app_local_fallback);
        let store = Self::layout(app_local_fallback, app_slug, false);
        let _ = store.ensure_dirs();
        store
    }

    /// Compute the path layout for a given root.
    fn layout(root: &Path, app_slug: &str, shared: bool) -> NekoStore {
        let root = root.to_path_buf();
        // In a shared store, venvs live under envs/<app>; in app-local mode we keep
        // the historical `python-env` location so existing installs are reused.
        let env_dir = if shared {
            root.join("envs").join(app_slug)
        } else {
            root.join("python-env")
        };
        let cache = root.join("cache");
        NekoStore {
            models_dir: root.join("models"),
            pip_cache: cache.join("pip"),
            uv_cache: cache.join("uv"),
            torch_home: cache.join("torch"),
            env_dir,
            root,
            shared,
        }
    }

    fn ensure_dirs(&self) -> std::io::Result<()> {
        for dir in [
            &self.models_dir,
            &self.pip_cache,
            &self.uv_cache,
            &self.torch_home,
        ] {
            fs::create_dir_all(dir)?;
        }
        if let Some(parent) = self.env_dir.parent() {
            fs::create_dir_all(parent)?;
        }
        Ok(())
    }

    fn write_marker(&self) {
        let marker = self.root.join("store.json");
        if marker.exists() {
            return;
        }
        let body = serde_json::json!({
            "schema": "neko-ai-store/v1",
            "root": self.root.display().to_string(),
            "created_by": "neko_store.rs",
        });
        let _ = fs::write(
            &marker,
            serde_json::to_string_pretty(&body).unwrap_or_default(),
        );
    }

    /// The Hugging Face home directory (`HF_HOME`) inside this store.
    pub fn hf_home(&self) -> PathBuf {
        self.models_dir.join("huggingface")
    }

    /// Apply all shared cache env vars to a child process `Command`.
    ///
    /// This routes pip, uv, Hugging Face, and torch caches at the shared store so
    /// downloads happen once and are reused across every app. Safe to call on any
    /// worker command (Python venv bootstrap, model download, inference).
    pub fn apply_env(&self, command: &mut Command) {
        let hf_home = self.hf_home();
        command
            .env("NEKO_AI_STORE_ROOT", &self.root)
            .env("NEKO_AI_STORE_SHARED", if self.shared { "1" } else { "0" })
            .env("HF_HOME", &hf_home)
            .env("HF_HUB_CACHE", hf_home.join("hub"))
            .env("HUGGINGFACE_HUB_CACHE", hf_home.join("hub"))
            .env("PIP_CACHE_DIR", &self.pip_cache)
            .env("UV_CACHE_DIR", &self.uv_cache)
            .env("TORCH_HOME", &self.torch_home);
    }
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

fn env_path(key: &str) -> Option<PathBuf> {
    match std::env::var(key) {
        Ok(value) if !value.trim().is_empty() => Some(PathBuf::from(value.trim())),
        _ => None,
    }
}

/// The default store root lives on the same volume as the app's data dir, so
/// that `uv` hardlink dedup works (hardlinks require same-volume cache + venv).
fn default_store_root(app_local_fallback: &Path) -> Option<PathBuf> {
    let base = volume_root(app_local_fallback)?;
    Some(base.join("NekoLegendsAI"))
}

#[cfg(target_os = "windows")]
fn volume_root(path: &Path) -> Option<PathBuf> {
    use std::path::Component;
    let mut comps = path.components();
    if let Some(Component::Prefix(prefix)) = comps.next() {
        // Reattach the RootDir so we get `D:\` not `D:`.
        let mut root = PathBuf::new();
        root.push(prefix.as_os_str());
        root.push(std::path::MAIN_SEPARATOR.to_string());
        return Some(root);
    }
    // No drive prefix (e.g. a bare relative path): fall back to the user home.
    home_dir()
}

#[cfg(not(target_os = "windows"))]
fn volume_root(_path: &Path) -> Option<PathBuf> {
    // On Unix we don't know the mount cheaply; use the user's home to stay visible
    // and writable rather than the filesystem root.
    home_dir()
}

fn home_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        env_path("USERPROFILE").or_else(|| {
            match (std::env::var("HOMEDRIVE"), std::env::var("HOMEPATH")) {
                (Ok(d), Ok(p)) if !d.is_empty() => Some(PathBuf::from(format!("{d}{p}"))),
                _ => None,
            }
        })
    }
    #[cfg(not(target_os = "windows"))]
    {
        env_path("HOME")
    }
}

fn pointer_path() -> Option<PathBuf> {
    Some(home_dir()?.join("NekoLegendsAI").join("store.json"))
}

fn read_pointer_root() -> Option<PathBuf> {
    let pointer = pointer_path()?;
    let text = fs::read_to_string(&pointer).ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    let root = value.get("root")?.as_str()?.trim().to_string();
    if root.is_empty() {
        return None;
    }
    Some(PathBuf::from(root))
}

fn write_pointer_root(root: &Path) -> std::io::Result<()> {
    let Some(pointer) = pointer_path() else {
        return Ok(());
    };
    if pointer.exists() {
        return Ok(());
    }
    if let Some(parent) = pointer.parent() {
        fs::create_dir_all(parent)?;
    }
    let body = serde_json::json!({
        "schema": "neko-ai-store-pointer/v1",
        "root": root.display().to_string(),
    });
    fs::write(
        &pointer,
        serde_json::to_string_pretty(&body).unwrap_or_default(),
    )
}

fn dir_is_writable(dir: &Path) -> bool {
    let probe = dir.join(".neko-write-probe");
    match fs::write(&probe, b"ok") {
        Ok(_) => {
            let _ = fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "neko-store-test-{tag}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        dir
    }

    #[test]
    fn app_env_token_is_uppercase_alnum() {
        assert_eq!(NekoStore::app_env_token("AngleForge"), "ANGLEFORGE");
        assert_eq!(
            NekoStore::app_env_token("image-to-hunyuan3d"),
            "IMAGETOHUNYUAN3D"
        );
    }

    #[test]
    fn per_app_override_wins() {
        let override_root = temp_dir("override");
        let fallback = temp_dir("fallback");
        std::env::set_var("NEKO_TESTAPP_STORE", &override_root);

        let store = NekoStore::resolve("testapp", &fallback);
        assert!(store.shared);
        assert!(store.root.starts_with(&override_root));
        assert!(store.models_dir.exists());
        assert!(store.pip_cache.exists());
        // env layout for shared store: envs/<app>
        assert!(store.env_dir.ends_with(Path::new("envs").join("testapp")));

        std::env::remove_var("NEKO_TESTAPP_STORE");
        let _ = fs::remove_dir_all(&override_root);
        let _ = fs::remove_dir_all(&fallback);
    }

    #[test]
    fn apply_env_sets_shared_caches() {
        let override_root = temp_dir("env");
        let fallback = temp_dir("env-fb");
        std::env::set_var("NEKO_ENVAPP_STORE", &override_root);
        let store = NekoStore::resolve("envapp", &fallback);

        let mut cmd = Command::new("python");
        store.apply_env(&mut cmd);
        let envs: std::collections::HashMap<String, String> = cmd
            .get_envs()
            .filter_map(|(k, v)| {
                Some((
                    k.to_string_lossy().into_owned(),
                    v?.to_string_lossy().into_owned(),
                ))
            })
            .collect();

        assert_eq!(
            envs.get("HF_HOME").map(String::as_str),
            Some(store.hf_home().display().to_string().as_str())
        );
        assert!(envs.contains_key("PIP_CACHE_DIR"));
        assert!(envs.contains_key("UV_CACHE_DIR"));
        assert!(envs.contains_key("TORCH_HOME"));
        assert_eq!(
            envs.get("NEKO_AI_STORE_SHARED").map(String::as_str),
            Some("1")
        );

        std::env::remove_var("NEKO_ENVAPP_STORE");
        let _ = fs::remove_dir_all(&override_root);
        let _ = fs::remove_dir_all(&fallback);
    }

    #[test]
    fn marker_file_written_once() {
        let root = temp_dir("marker");
        let store = NekoStore::layout(&root, "markerapp", true);
        store.ensure_dirs().unwrap();
        store.write_marker();
        let marker = root.join("store.json");
        assert!(marker.exists());
        let first = fs::read_to_string(&marker).unwrap();
        store.write_marker(); // should not overwrite
        let second = fs::read_to_string(&marker).unwrap();
        assert_eq!(first, second);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn env_var_overrides_volume_hint() {
        // When an override env var is set, the volume hint must be ignored.
        let override_root = temp_dir("vh-override");
        let fallback = temp_dir("vh-fallback");
        let hint = temp_dir("vh-hint");
        std::env::set_var("NEKO_VHAPP_STORE", &override_root);

        let store = NekoStore::resolve_with_volume_hint("vhapp", &fallback, Some(&hint));
        assert!(store.root.starts_with(&override_root));

        std::env::remove_var("NEKO_VHAPP_STORE");
        let _ = fs::remove_dir_all(&override_root);
        let _ = fs::remove_dir_all(&fallback);
    }
}
