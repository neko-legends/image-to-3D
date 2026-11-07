//! Environment cleanup for child processes started from an AppImage.

use std::process::Command;

/// Inside an AppImage the runtime points PYTHONHOME, LD_LIBRARY_PATH, PATH,
/// XDG_DATA_DIRS, GTK/GIO module paths, etc. into its mount. Those break other
/// programs (system/uv Python, nvidia-smi, the AutoRemesher AppImage), so drop
/// every variable that refers to `$APPDIR` (keeping the other entries of
/// `:`-separated lists) plus the AppImage bookkeeping variables. No-op outside
/// an AppImage and on other platforms.
pub fn strip_appimage_env(command: &mut Command) {
    #[cfg(target_os = "linux")]
    {
        let Some(appdir) = std::env::var_os("APPDIR") else {
            return;
        };
        let appdir = appdir.to_string_lossy().trim_end_matches('/').to_string();
        if appdir.is_empty() {
            return;
        }
        for key in ["APPDIR", "APPIMAGE", "ARGV0", "OWD", "PYTHONDONTWRITEBYTECODE"] {
            command.env_remove(key);
        }
        if std::env::var("GDK_BACKEND").as_deref() == Ok("x11") {
            command.env_remove("GDK_BACKEND");
        }
        for (key, value) in std::env::vars_os() {
            let text = value.to_string_lossy();
            if !text.contains(appdir.as_str()) {
                continue;
            }
            let kept: Vec<&str> = text
                .split(':')
                .filter(|entry| !entry.is_empty() && !entry.starts_with(appdir.as_str()))
                .collect();
            if text.contains(':') && !kept.is_empty() {
                command.env(&key, kept.join(":"));
            } else {
                command.env_remove(&key);
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = command;
}
