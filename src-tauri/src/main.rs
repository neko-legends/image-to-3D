#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    // WebKitGTK's DMA-BUF renderer crashes on Wayland with NVIDIA
    // ("Error 71 (Protocol error) dispatching to Wayland display").
    #[cfg(target_os = "linux")]
    if std::env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_none() {
        std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
    }

    // Inside an AppImage: the runtime's Python/Perl/Qt paths only break child
    // processes (e.g. system python), and its GTK hook forces GDK_BACKEND=x11,
    // which renders through XWayland (blurry or 2x). Prefer native Wayland.
    #[cfg(target_os = "linux")]
    if std::env::var_os("APPDIR").is_some() {
        for key in ["PYTHONHOME", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "PERLLIB", "QT_PLUGIN_PATH"] {
            std::env::remove_var(key);
        }
        if std::env::var_os("WAYLAND_DISPLAY").is_some()
            && std::env::var("GDK_BACKEND").as_deref() == Ok("x11")
        {
            std::env::set_var("GDK_BACKEND", "wayland,x11");
        }
    }

    image_to_3d_lib::run()
}
