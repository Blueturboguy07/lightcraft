//! Photo ▸ AI ▸ Describe Photo… and Settings ▸ AI (desktop, `crate::ai`): where AI is on, the
//! menu, the Info panel and Settings offer it; the dialog opens for the selection and sends nothing
//! until Describe; Apply writes through `photo.setMeta`. No request leaves this computer: the
//! end-to-end tests talk to a local stand-in for the model.
#![cfg(not(target_arch = "wasm32"))]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use serde_json::{Value, json};

use crate::ai::{Ai, AppInfo};
use crate::headless::Headless;
use crate::menubar::{MenuNode, menu_bar};
use crate::state::Dialog;
use crate::{LightcraftApp, Services};

const SETTLE: Duration = Duration::from_secs(120);
const T: Duration = Duration::from_secs(20);

fn plain_app() -> LightcraftApp {
    LightcraftApp::new(lightcraft_engine::Session::with_demo(), Services { png: None, ..Default::default() })
}

/// AI on, with a key of the person's own chosen (never used: nothing is sent in these tests), and
/// no settings file or publik credential of this machine's.
fn ai_app() -> LightcraftApp {
    let mut app = plain_app();
    let mut ai = Ai::new(AppInfo {
        slug: "lightcraft-tests",
        name: "LightCraft",
        version: "0.0.0",
        app_token: "",
        dialects: &["chat_completions"],
        config_dir: None,
    });
    ai.skip_first_run();
    ai.state.settings.provider = "own".into();
    ai.state.settings.own_key = "sk-test".into();
    ai.state.settings.setup_done = true;
    app.ai = Some(ai);
    app
}

fn ai_submenu(app: &LightcraftApp) -> Option<Vec<MenuNode>> {
    let photo = menu_bar(app).into_iter().find(|(title, _)| title == "Photo")?.1;
    photo.into_iter().find_map(|node| match node {
        MenuNode::Submenu { label, children } if label == "AI" => Some(children),
        _ => None,
    })
}

fn ids_of(nodes: &[MenuNode]) -> Vec<String> {
    nodes
        .iter()
        .filter_map(|node| match node {
            MenuNode::Item { id, .. } => Some(id.clone()),
            _ => None,
        })
        .collect()
}

fn select_first(app: &mut LightcraftApp, n: usize) -> Vec<u64> {
    let ids: Vec<u64> = app.session.visible_cloned().iter().take(n).map(|id| id.0).collect();
    app.run("library.select", json!({"ids": ids})).unwrap();
    ids
}

#[test]
fn the_menu_offers_ai_only_where_it_is_on() {
    let mut off = plain_app();
    select_first(&mut off, 1);
    assert!(ai_submenu(&off).is_none(), "no Photo ▸ AI without AI");
    assert!(off.run("dialog.aiDescribe", json!({})).is_err());
    assert!(off.ui.dialog.is_none());

    let on = ai_app();
    let items = ai_submenu(&on).expect("Photo ▸ AI");
    assert_eq!(ids_of(&items), vec!["dialog.aiDescribe", "dialog.aiSettings"]);
}

#[test]
fn describe_opens_for_the_selection_and_sends_nothing() {
    let mut app = ai_app();
    let ids = select_first(&mut app, 2);
    let r = app.run("dialog.aiDescribe", json!({})).unwrap();
    assert_eq!(r["photos"], 2);
    assert_eq!(app.ui.dialog, Some(Dialog::AiDescribe { ids: ids.clone() }));
    let mut h = Headless::new(app, [1400.0, 1000.0], 1.0);
    h.settle(SETTLE);
    let text = h.painted_text().join("\n");
    for want in ["Describe Photos", "2 photos. Each photo is one request.", "Describe 2 Photos", "Close"] {
        assert!(text.contains(want), "{want:?} not painted:\n{text}");
    }
    // nothing runs and nothing is in the activity stack until Describe is clicked
    assert!(h.app.ai.as_ref().is_some_and(|ai| !ai.describing()));
    assert!(h.app.session.activity.list().iter().all(|t| t.kind != "ai"));
    // Apply waits for results
    assert!(!crate::ai::can_apply(&h.app));
    let d = h.app.ui.dialog.clone().unwrap();
    assert!(crate::panels::dialogs::confirm_dialog(&mut h.app, &d).is_err());
}

#[test]
fn the_info_panel_opens_describe_photo() {
    let mut app = ai_app();
    let ids = select_first(&mut app, 1);
    app.run("panel.info", json!({})).unwrap();
    let mut h = Headless::new(app, [1400.0, 1000.0], 1.0);
    h.settle(SETTLE);
    let r = h.request("ui.clickWidget", json!({"id": "button:infoAiDescribe"}), Duration::from_secs(20));
    assert_eq!(r["ok"], true, "{r}");
    h.settle(SETTLE);
    assert_eq!(h.app.ui.dialog, Some(Dialog::AiDescribe { ids }));
}

#[test]
fn settings_has_an_ai_tab() {
    let mut app = ai_app();
    app.run("dialog.aiSettings", json!({})).unwrap();
    assert_eq!(app.ui.dialog, Some(Dialog::Settings { tab: "ai".into() }));
    assert!(crate::panels::settings::TABS.iter().any(|(id, label)| *id == "ai" && *label == "AI"));
    let mut h = Headless::new(app, [1400.0, 1000.0], 1.0);
    h.settle(SETTLE);
    let text = h.painted_text().join("\n");
    assert!(text.contains("Your own key"), "{text}");

    // without AI the tab says so
    let mut off = plain_app();
    off.run("app.settings", json!({"tab": "ai"})).unwrap();
    let mut h = Headless::new(off, [1400.0, 1000.0], 1.0);
    h.settle(SETTLE);
    assert!(h.painted_text().join("\n").contains("AI is off in this session"));
}

#[test]
fn the_setup_dialog_stays_closed_while_an_agent_drives() {
    let mut app = ai_app();
    let ctx = egui::Context::default();
    for _ in 0..6 {
        crate::ai::frame(&mut app, &ctx);
    }
    assert!(app.ui.dialog.is_none());
}

/// One request at a time across these tests: `craft-publik` names its curl files by process id and
/// clock, and two requests in the same microsecond (two tests side by side) would share them.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

/// A local stand-in for an OpenAI-compatible endpoint: it answers request n with `replies[n]`
/// (status, body) once the test says go, and keeps what each request carried.
struct FakeModel {
    url: String,
    bodies: Arc<Mutex<Vec<String>>>,
    go: mpsc::Sender<()>,
}

fn fake_model(replies: Vec<(u16, String)>) -> FakeModel {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let (go, wait) = mpsc::channel::<()>();
    let kept = bodies.clone();
    std::thread::spawn(move || {
        for (status, body) in replies {
            let Ok((mut stream, _)) = listener.accept() else { return };
            let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
            kept.lock().unwrap().push(request_body(&mut stream));
            if wait.recv().is_err() {
                return;
            }
            let reply = format!(
                "HTTP/1.1 {status} Test\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(reply.as_bytes());
        }
    });
    FakeModel { url, bodies, go }
}

/// The body of the HTTP request on `stream` (by its Content-Length).
fn request_body(stream: &mut TcpStream) -> String {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 65536];
    loop {
        let n = stream.read(&mut chunk).unwrap_or(0);
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
        let Some(end) = buf.windows(4).position(|w| w == b"\r\n\r\n") else { continue };
        let head = String::from_utf8_lossy(&buf[..end]).to_lowercase();
        let len = head.lines().find_map(|l| l.strip_prefix("content-length:")).and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(0);
        if buf.len() >= end + 4 + len {
            return String::from_utf8_lossy(&buf[end + 4..end + 4 + len]).into_owned();
        }
    }
    String::new()
}

fn answer(title: &str) -> String {
    let content = json!({"title": title, "caption": "A calm lake below the peaks.", "keywords": ["lake", "calm", "Lake"]}).to_string();
    json!({"choices": [{"message": {"role": "assistant", "content": content}}]}).to_string()
}

/// `ai_app` whose own key points at `model`.
fn app_for(model: &FakeModel) -> LightcraftApp {
    let mut app = ai_app();
    let settings = &mut app.ai.as_mut().unwrap().state.settings;
    settings.own_base_url = model.url.clone();
    settings.own_model = "test-model".into();
    app
}

fn describing(h: &Headless) -> bool {
    h.app.ai.as_ref().is_some_and(|ai| ai.describing())
}

#[test]
fn describe_sends_one_small_jpeg_per_photo_and_apply_writes_the_answers() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let model = fake_model(vec![(200, answer("Lake one")), (200, answer("Lake two"))]);
    let mut app = app_for(&model);
    let ids = select_first(&mut app, 2);
    app.run("dialog.aiDescribe", json!({})).unwrap();
    let mut h = Headless::new(app, [1400.0, 1000.0], 1.0);
    h.settle(SETTLE);
    let r = h.request("ui.clickWidget", json!({"id": "button:aiDescribe"}), T);
    assert_eq!(r["ok"], true, "{r}");
    // while it runs it holds a row in the activity stack, with ✕
    assert!(h.step_until(T, describing));
    let rows = h.app.session.activity.list();
    let row = rows.iter().find(|t| t.kind == "ai").expect("an activity row");
    assert_eq!((row.label.as_str(), row.total, row.cancellable), ("Describing photos", 2, true));
    model.go.send(()).unwrap();
    model.go.send(()).unwrap();
    assert!(h.step_until(SETTLE, |h| !describing(h)));
    assert!(h.app.session.activity.list().iter().all(|t| t.kind != "ai"), "the row goes with the run");
    h.settle(SETTLE);
    assert!(h.painted_text().join("\n").contains("Apply to 2 Photos"));

    // each request: the model asked for, and the photo as a JPEG of at most 1024 px, without metadata
    let bodies = model.bodies.lock().unwrap().clone();
    assert_eq!(bodies.len(), 2);
    for body in &bodies {
        let v: Value = serde_json::from_str(body).unwrap();
        assert_eq!(v["model"], "test-model");
        let url = v["messages"][1]["content"][1]["image_url"]["url"].as_str().unwrap();
        let jpeg = craft_publik::client::unbase64(url.strip_prefix("data:image/jpeg;base64,").unwrap()).unwrap();
        assert!(jpeg.starts_with(&[0xFF, 0xD8]), "a JPEG");
        assert!(!jpeg.windows(6).any(|w| w == b"Exif\0\0"), "no camera data, no location");
        let decoded = lightcraft_codecs::decode(&jpeg, Default::default()).unwrap();
        assert!(decoded.image.width.max(decoded.image.height) <= crate::ai::EDGE as usize);
    }

    // Apply writes each photo's own answer (keywords once, whatever their case)
    let d = h.app.ui.dialog.clone().unwrap();
    let r = crate::panels::dialogs::confirm_dialog(&mut h.app, &d).unwrap();
    assert_eq!(r["applied"], 2);
    for (id, title) in ids.iter().zip(["Lake one", "Lake two"]) {
        let m = &h.app.session.catalog.photo(lightcraft_catalog::PhotoId(*id)).unwrap().meta;
        assert_eq!((m.title.as_str(), m.caption.as_str()), (title, "A calm lake below the peaks."));
        assert_eq!(m.keywords.iter().filter(|k| k.eq_ignore_ascii_case("lake")).count(), 1, "{:?}", m.keywords);
    }
}

#[test]
fn a_failed_request_stops_the_batch_and_says_why() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let model = fake_model(vec![(500, json!({"error": {"message": "the model is down"}}).to_string()), (200, answer("never asked"))]);
    model.go.send(()).unwrap();
    model.go.send(()).unwrap();
    let mut app = app_for(&model);
    select_first(&mut app, 3);
    app.run("dialog.aiDescribe", json!({})).unwrap();
    let mut h = Headless::new(app, [1400.0, 1000.0], 1.0);
    h.settle(SETTLE);
    let r = h.request("ui.clickWidget", json!({"id": "button:aiDescribe"}), T);
    assert_eq!(r["ok"], true, "{r}");
    assert!(h.step_until(SETTLE, |h| !describing(h) && h.app.session.activity.list().is_empty()));
    h.settle(SETTLE);
    assert_eq!(model.bodies.lock().unwrap().len(), 1, "the batch stops at the first failure");
    assert!(!crate::ai::can_apply(&h.app));
    let text = h.painted_text().join("\n");
    assert!(text.contains("the model is down"), "the reason is shown:\n{text}");
    assert!(text.contains("Describe 3 Photos"), "the button is back:\n{text}");
}
