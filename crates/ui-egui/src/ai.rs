//! AI through publik API or the person's own key, desktop only. The setup flow, the credential
//! file and the HTTP calls live in the shared `craft-publik` crate (one per Craft app family).
//! This module is LightCraft's thin host: the setup dialog on the first launch, Settings ▸ AI, and
//! Photo ▸ AI ▸ Describe Photo… (also in the Info panel), which writes a title, a caption and
//! keywords for the selected photos.
//!
//! Nothing is sent, and nothing is charged, until the person clicks Describe. A request carries one
//! JPEG of the photo as LightCraft renders it (its edits applied, at most [`EDGE`] pixels on the
//! long edge, no metadata) and a fixed instruction: one request per photo, and a batch stops at the
//! first failure. The results are applied only by Apply, through `photo.setMeta`, the way the Info
//! panel writes them. AI masks, AI Denoise and everything else stay on this computer.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError, channel};

pub use craft_publik::AppInfo;
use craft_publik::client::{self, Fail, Reply};
use craft_publik::ui::{Action, AiState};
use lightcraft_catalog::PhotoId;
use lightcraft_engine::activity::{Cancel, TaskGuard};
use lightcraft_engine::export::{ExportFormat, ExportOptions, MetadataPolicy, Resize};
use serde_json::{Value, json};

use crate::LightcraftApp;
use crate::state::Dialog;
use crate::theme::Tokens;

const DESCRIBE_SYSTEM: &str = "You describe photos for LightCraft, a photo library. Look at the photo and reply with one JSON object and nothing else, in this form: {\"title\": \"...\", \"caption\": \"...\", \"keywords\": [\"...\"]}. The title has at most eight words. The caption has one or two sentences that say what the photo shows: the subject, the place, the light and the mood. The keywords are 5 to 12 short words or phrases, in lowercase, that a person would search for. Do not guess the names of people. Do not use markdown.";

/// The long edge of the JPEG a request carries, in pixels.
pub const EDGE: u32 = 1024;
/// The JPEG quality of that copy.
const QUALITY: u8 = 80;
/// The most photos one Describe sends. A larger selection is cut, and the dialog says so.
pub const MAX_PHOTOS: usize = 50;
const MAX_KEYWORDS: usize = 12;
const MAX_TITLE_CHARS: usize = 200;
const MAX_CAPTION_CHARS: usize = 2_000;
const MAX_KEYWORD_CHARS: usize = 60;
/// The activity stack's row while Describe runs.
const ACTIVITY_KIND: &str = "ai";
const ACTIVITY_LABEL: &str = "Describing photos";

pub struct Ai {
    pub state: AiState,
    frames: u32,
    first_run_checked: bool,
    /// The first launch asked for the setup flow; `frame` opens it when no other window shows.
    open_setup: bool,
    /// The dialog the setup flow goes back to when it ends: Settings ▸ AI or Describe Photo.
    /// `None` closes it.
    setup_return: Option<Dialog>,
    describe: Describe,
}

impl Ai {
    pub fn new(app: AppInfo) -> Self {
        Ai { state: AiState::new(app), frames: 0, first_run_checked: false, open_setup: false, setup_return: None, describe: Describe::default() }
    }

    /// Do not open the setup dialog on this launch (an agent drives the app over `--control`).
    /// Settings ▸ AI and Describe Photo still offer it.
    pub fn skip_first_run(&mut self) {
        self.first_run_checked = true;
    }

    /// A Describe is running (`ui.inspect`, tests).
    pub fn describing(&self) -> bool {
        self.describe.run.is_some()
    }
}

/// `PUBLIK_SELF_TEST=1`: provision for real and print one line (publikhq.com/skills/setup.md).
pub fn self_test(app: &AppInfo) -> String {
    craft_publik::self_test(app)
}

// ---- Describe Photo: state --------------------------------------------------------------------

/// Describe Photo's photos, their results and the request in flight.
#[derive(Default)]
struct Describe {
    items: Vec<Item>,
    error: Option<String>,
    run: Option<Run>,
}

/// One photo of a Describe and what the model wrote for it, as the person edits it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Item {
    id: u64,
    /// The photo's file name (shown, never sent).
    name: String,
    /// The model answered for this photo.
    described: bool,
    title: String,
    caption: String,
    /// Comma-separated, as typed into the Keywording box.
    keywords: String,
}

/// A Describe in flight: its worker's answers, its stop flag and its row in the activity stack.
struct Run {
    rx: Receiver<Step>,
    cancel: Arc<AtomicBool>,
    guard: TaskGuard,
    total: usize,
    done: usize,
}

/// What the worker reports for one photo.
enum Step {
    Described(usize, Reply),
    Failed(usize, Why),
}

enum Why {
    /// The JPEG could not be made (the original is missing, the file is damaged…).
    Render(String),
    /// The AI request failed (a 402, a retired key, the network…).
    Request(Fail),
}

impl Describe {
    /// Take the worker's answers. Called once a frame.
    fn poll(&mut self, state: &mut AiState) {
        let Some(mut run) = self.run.take() else { return };
        loop {
            match run.rx.try_recv() {
                Ok(Step::Described(index, reply)) => {
                    state.note_reply(&reply);
                    let d = parse_description(&reply.text);
                    if let Some(item) = self.items.get_mut(index) {
                        item.described = true;
                        item.title = d.title;
                        item.caption = d.caption;
                        item.keywords = d.keywords.join(", ");
                    }
                    run.done = run.done.saturating_add(1);
                    run.guard.progress(run.done as u64, run.total as u64);
                }
                Ok(Step::Failed(index, why)) => {
                    let name = self.items.get(index).map(|i| i.name.clone()).unwrap_or_default();
                    let message = match why {
                        Why::Render(e) => Some(e),
                        Why::Request(f) => {
                            state.note_fail(&f);
                            // a 402 or a retired key is the notice bar's (one message, one link)
                            state.notice.is_none().then_some(f.message)
                        }
                    };
                    if let Some(m) = message {
                        self.error = Some(if self.items.len() > 1 && !name.is_empty() { format!("{name}: {m}") } else { m });
                    }
                }
                Err(TryRecvError::Empty) => {
                    self.run = Some(run);
                    return;
                }
                // the worker is done (its row goes with `run`)
                Err(TryRecvError::Disconnected) => return,
            }
        }
    }

    /// The photos `ids` (at most [`MAX_PHOTOS`]) from `catalog`. A Describe that runs keeps its own.
    fn sync(&mut self, catalog: &lightcraft_catalog::Catalog, ids: &[u64]) {
        if self.run.is_some() {
            return;
        }
        let want: Vec<u64> = ids.iter().copied().filter(|id| catalog.photo(PhotoId(*id)).is_some()).take(MAX_PHOTOS).collect();
        if self.items.iter().map(|i| i.id).eq(want.iter().copied()) {
            return;
        }
        self.items = want
            .into_iter()
            .map(|id| Item { id, name: catalog.photo(PhotoId(id)).map(|p| p.file_name.clone()).unwrap_or_default(), ..Item::default() })
            .collect();
        self.error = None;
    }

    fn described(&self) -> usize {
        self.items.iter().filter(|i| i.described).count()
    }
}

// ---- once a frame -------------------------------------------------------------------------------

/// Once a frame, before drawing: take finished work, stop a Describe whose dialog closed, and open
/// the setup dialog on the first launch.
pub fn frame(app: &mut LightcraftApp, _ctx: &egui::Context) {
    // the setup dialog waits until nothing else asks for the person's attention
    let quiet = app.ui.dialog.is_none() && app.notices.is_empty() && app.library_problem.is_none() && app.quit_prompt.is_none();
    let describing = matches!(app.ui.dialog, Some(Dialog::AiDescribe { .. } | Dialog::AiSetup));
    let Some(ai) = app.ai.as_mut() else { return };
    ai.state.poll();
    ai.describe.poll(&mut ai.state);
    // closing the dialog stops a batch before its next photo (the request on its way still ends)
    if !describing && let Some(run) = &ai.describe.run {
        run.cancel.store(true, Ordering::Relaxed);
    }
    ai.frames = ai.frames.saturating_add(1);
    // the first launch asks once the window is up (a few frames in); "Later" ends that for good
    if !ai.first_run_checked && ai.frames > 3 {
        ai.first_run_checked = true;
        if ai.state.wants_first_run() || ai.state.owes_first_card() {
            ai.open_setup = true;
        }
    }
    if ai.open_setup && quiet {
        ai.open_setup = false;
        ai.setup_return = None;
        ai.state.begin();
        app.ui.dialog = Some(Dialog::AiSetup);
    }
}

fn off(ui: &mut egui::Ui, t: &Tokens) {
    ui.label(
        egui::RichText::new("AI is off in this session. LightCraft keeps no settings here (--memory or LIGHTCRAFT_NO_PREFS).").color(t.text_dim),
    );
}

fn open_url(app: &mut LightcraftApp, ctx: &egui::Context, url: &str) {
    if let Err(e) = crate::links::open(app, url) {
        app.toast_error(ctx, e);
    }
}

/// The setup flow in place of the current dialog: it goes back to `back` when it ends.
fn begin_setup(app: &mut LightcraftApp, back: Dialog) {
    if let Some(ai) = app.ai.as_mut() {
        ai.state.begin();
        ai.setup_return = Some(back);
        app.ui.dialog = Some(Dialog::AiSetup);
    }
}

// ---- the setup dialog ---------------------------------------------------------------------------

/// The AI setup dialog's body: the flow of `craft-publik` (publik API or your own key). When it
/// ends, the dialog it came from comes back (Settings ▸ AI, Describe Photo), or it closes.
pub fn setup_body(app: &mut LightcraftApp, ui: &mut egui::Ui) {
    let t = Tokens::get(ui.ctx());
    let Some(ai) = app.ai.as_mut() else {
        off(ui, &t);
        return;
    };
    match ai.state.setup_ui(ui) {
        Action::None => {}
        Action::OpenUrl(url) => open_url(app, ui.ctx(), &url),
        // (another dialog in place of this one, or none: `dialogs::show` keeps what is set here)
        Action::Close => app.ui.dialog = ai.setup_return.take(),
    }
}

// ---- Settings ▸ AI ------------------------------------------------------------------------------

/// Settings ▸ AI: what is in use, sign-in and key controls, and the way into the setup flow.
pub fn settings_tab(app: &mut LightcraftApp, ui: &mut egui::Ui, t: &Tokens) {
    ui.add_space(4.0);
    ui.label(egui::RichText::new("AI").font(t.semibold(12.5)).color(t.text));
    ui.label(
        egui::RichText::new(
            "Describe Photo (Photo > AI, and the Info panel) writes a title, a caption and keywords through publik API or your own key. AI masks, AI Denoise and everything else stay on this computer.",
        )
        .size(11.0)
        .color(t.text_dim),
    );
    ui.add_space(6.0);
    let Some(ai) = app.ai.as_mut() else {
        off(ui, t);
        return;
    };
    let mut open_setup = false;
    let action = ai.state.options_ui(ui, &mut open_setup);
    if open_setup {
        // (options_ui began the flow)
        ai.setup_return = Some(Dialog::Settings { tab: "ai".into() });
        app.ui.dialog = Some(Dialog::AiSetup);
    }
    if let Action::OpenUrl(url) = action {
        open_url(app, ui.ctx(), &url);
    }
}

// ---- Describe Photo -----------------------------------------------------------------------------

/// `dialog.aiDescribe {ids?}`: open Describe Photo for `ids`, else the selected photos (at most
/// [`MAX_PHOTOS`]). While a Describe runs, the dialog shows that one. Nothing is sent here.
pub fn open_describe(app: &mut LightcraftApp, p: &Value) -> Result<Value, String> {
    let Some(ai) = app.ai.as_ref() else {
        return Err("AI is off in this session: LightCraft keeps no settings here".into());
    };
    let ids: Vec<u64> = if ai.describe.run.is_some() {
        ai.describe.items.iter().map(|i| i.id).collect()
    } else {
        match p.get("ids").and_then(Value::as_array) {
            Some(a) => a.iter().filter_map(Value::as_u64).collect(),
            None => app.session.selected().into_iter().map(|id| id.0).collect(),
        }
    };
    let ids: Vec<u64> = ids.into_iter().filter(|id| app.session.catalog.photo(PhotoId(*id)).is_some()).collect();
    if ids.is_empty() {
        return Err("Select a photo first".into());
    }
    let n = ids.len().min(MAX_PHOTOS);
    app.ui.dialog = Some(Dialog::AiDescribe { ids });
    Ok(json!({"photos": n}))
}

/// The JPEG a request carries: the photo as rendered with its edits, sRGB, [`EDGE`] pixels on the
/// long edge at most, without metadata (no camera data, no location, no keywords).
fn jpeg_options() -> ExportOptions {
    ExportOptions {
        format: ExportFormat::Jpeg,
        quality: QUALITY,
        resize: Some(Resize::long_edge(EDGE)),
        metadata: MetadataPolicy::None,
        remove_location: true,
        ..ExportOptions::default()
    }
}

/// The request's text: the language the answer is in follows the interface's.
fn user_text(language: crate::i18n::Locale) -> String {
    let mut s = String::from("Describe this photo.");
    if language != crate::i18n::Locale::En {
        s.push_str(&format!(" Write the title, the caption and the keywords in {} (language code {}).", language.name(), language.code()));
    }
    s
}

/// Start describing the dialog's photos on a worker: render each one's JPEG, send it, report.
fn start(app: &mut LightcraftApp, ctx: &egui::Context) -> Result<(), String> {
    let (provider, ids, names) = {
        let Some(ai) = app.ai.as_ref() else { return Err("AI is off in this session".into()) };
        if ai.describe.run.is_some() {
            return Ok(());
        }
        let ids: Vec<PhotoId> = ai.describe.items.iter().map(|i| PhotoId(i.id)).collect();
        let names: Vec<String> = ai.describe.items.iter().map(|i| i.name.clone()).collect();
        (ai.state.provider(), ids, names)
    };
    if ids.is_empty() {
        return Err("Select a photo first".into());
    }
    // set up on this thread (cheap, needs the session); rendered on the worker
    let jobs = lightcraft_engine::export::prepare_batch(&mut app.session, &ids, &jpeg_options())?;
    let total = jobs.len();
    let user = user_text(crate::i18n::language());
    let cancel = Arc::new(AtomicBool::new(false));
    let guard = app.session.activity.start(ACTIVITY_KIND, ACTIVITY_LABEL, Cancel::Flag(cancel.clone()));
    guard.progress(0, total as u64);
    let row = guard.handle();
    let (tx, rx) = channel();
    let (stop, repaint) = (cancel.clone(), ctx.clone());
    let work = move || {
        for (index, job) in jobs.into_iter().enumerate() {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            if let Some(name) = names.get(index) {
                row.detail(name);
            }
            let step = match job.run() {
                Err(e) => Step::Failed(index, Why::Render(e)),
                Ok(jpeg) => match client::chat_with_image(&provider, DESCRIBE_SYSTEM, &user, &jpeg.bytes, "image/jpeg") {
                    Ok(reply) => Step::Described(index, reply),
                    Err(f) => Step::Failed(index, Why::Request(f)),
                },
            };
            let failed = matches!(step, Step::Failed(..));
            let sent = tx.send(step).is_ok();
            repaint.request_repaint();
            // one failure stops the batch: the next photos would most likely fail (and cost) too
            if failed || !sent {
                break;
            }
        }
        repaint.request_repaint();
    };
    std::thread::Builder::new().name("ai-describe".into()).spawn(work).map_err(|e| format!("Describe could not start: {e}"))?;
    if let Some(ai) = app.ai.as_mut() {
        for item in &mut ai.describe.items {
            item.described = false;
            item.title.clear();
            item.caption.clear();
            item.keywords.clear();
        }
        ai.describe.error = None;
        ai.describe.run = Some(Run { rx, cancel, guard, total, done: 0 });
    }
    Ok(())
}

/// The Describe Photo dialog's body. Apply (the dialog's confirm) writes the results.
pub fn describe_body(app: &mut LightcraftApp, ui: &mut egui::Ui, ids: &[u64]) {
    let t = Tokens::get(ui.ctx());
    let ctx = ui.ctx().clone();
    let Some(ai) = app.ai.as_mut() else {
        off(ui, &t);
        return;
    };
    ai.describe.sync(&app.session.catalog, ids);
    if !ai.state.provider().is_set_up() {
        ui.label("Describe Photo needs AI. Set it up first: publik API (no key to paste) or your own key.");
        ui.add_space(6.0);
        if crate::button::Button::new("Set Up AI…").primary().widget("button:aiSetUp").show(ui).clicked() {
            begin_setup(app, Dialog::AiDescribe { ids: ids.to_vec() });
        }
        return;
    }
    let n = ai.describe.items.len();
    if n == 0 {
        ui.label(egui::RichText::new("The photo is no longer in the library.").color(t.text_dim));
        return;
    }
    if n == 1 {
        let name = ai.describe.items.first().map(|i| i.name.clone()).unwrap_or_default();
        ui.label(egui::RichText::new(name).font(t.semibold(12.5)).color(t.text));
    } else {
        ui.label(egui::RichText::new(format!("{n} photos. Each photo is one request.")).color(t.text_label));
    }
    if ids.len() > MAX_PHOTOS {
        ui.label(egui::RichText::new(format!("Only the first {MAX_PHOTOS} of the {} selected photos are described.", ids.len())).color(t.text_dim));
    }
    ui.label(
        egui::RichText::new(format!(
            "LightCraft sends a JPEG of {} with your edits, at most {EDGE} pixels on the long edge and without its metadata. Nothing is sent until you click Describe.",
            if n == 1 { "the photo" } else { "each photo" }
        ))
        .size(11.0)
        .color(t.text_dim),
    );
    ui.add_space(4.0);
    let mut start_now = false;
    ui.horizontal(|ui| {
        if let Some(run) = &ai.describe.run {
            ui.spinner();
            let at = run.done.saturating_add(1).min(run.total);
            let text = if run.total > 1 { format!("Describing photo {at} of {}…", run.total) } else { "Describing the photo…".to_string() };
            ui.label(egui::RichText::new(text).color(t.text_label));
            if run.total > 1 && crate::button::Button::new("Stop").widget("button:aiStop").show(ui).clicked() {
                run.cancel.store(true, Ordering::Relaxed);
            }
        } else {
            let label = match (n, ai.describe.described() > 0) {
                (1, false) => "Describe".to_string(),
                (1, true) => "Describe Again".to_string(),
                (n, false) => format!("Describe {n} Photos"),
                (n, true) => format!("Describe {n} Photos Again"),
            };
            if crate::button::Button::new(&label).widget("button:aiDescribe").show(ui).clicked() {
                start_now = true;
            }
        }
    });
    if let Some(e) = &ai.describe.error {
        ui.add(egui::Label::new(egui::RichText::new(e).color(t.caution)).wrap());
    }
    if ai.describe.described() > 0 {
        ui.add_space(4.0);
        if n == 1 {
            if let Some(item) = ai.describe.items.first_mut() {
                item_fields(ui, &t, item, None);
            }
        } else {
            egui::ScrollArea::vertical()
                .id_salt("ai-describe-results")
                .max_height(crate::panels::dialogs::list_height(ui, 360.0))
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    for (i, item) in ai.describe.items.iter_mut().enumerate().filter(|(_, item)| item.described) {
                        ui.add_space(4.0);
                        ui.label(egui::RichText::new(&item.name).font(t.semibold(12.0)).color(t.text));
                        item_fields(ui, &t, item, Some(i));
                    }
                });
        }
    }
    ui.add_space(4.0);
    let bar = ai.state.result_bar(ui);
    ui.label(egui::RichText::new(ai.state.status_line()).size(11.0).color(t.text_dim));
    if let Action::OpenUrl(url) = bar {
        open_url(app, &ctx, &url);
    }
    if start_now
        && let Err(e) = start(app, &ctx)
        && let Some(ai) = app.ai.as_mut()
    {
        ai.describe.error = Some(e);
    }
}

/// One photo's title, caption and keywords, editable, and Copy. `index` names the fields of a
/// batch's photos apart.
fn item_fields(ui: &mut egui::Ui, t: &Tokens, item: &mut Item, index: Option<usize>) {
    let suffix = index.map(|i| format!("-{i}")).unwrap_or_default();
    let small = |ui: &mut egui::Ui, text: &str| ui.label(egui::RichText::new(text).size(11.5).color(t.text_dim));
    let rows = if index.is_some() { 1 } else { 3 };
    small(ui, "Title");
    crate::text_field::TextField::singleline(&format!("field:aiTitle{suffix}"), &mut item.title).label("Title").width(f32::INFINITY).show(ui);
    small(ui, "Caption");
    crate::text_field::TextField::multiline(&format!("field:aiCaption{suffix}"), &mut item.caption)
        .rows(rows.max(2))
        .label("Caption")
        .width(f32::INFINITY)
        .show(ui);
    small(ui, "Keywords");
    crate::text_field::TextField::multiline(&format!("field:aiKeywords{suffix}"), &mut item.keywords)
        .rows(rows.min(2))
        .label("Keywords")
        .hint("Separate keywords with commas")
        .width(f32::INFINITY)
        .show(ui);
    if crate::button::Button::new("Copy").small().widget(format!("button:aiCopy{suffix}")).show(ui).clicked() {
        ui.ctx().copy_text(copy_text(item));
    }
}

/// An item as plain text for the clipboard.
fn copy_text(item: &Item) -> String {
    format!("{}\n{}\nKeywords: {}", item.title.trim(), item.caption.trim(), split_keywords(&item.keywords).join(", "))
}

/// Apply is possible: results are in, and no Describe runs.
pub fn can_apply(app: &LightcraftApp) -> bool {
    app.ai.as_ref().is_some_and(|ai| ai.describe.run.is_none() && ai.describe.items.iter().any(|i| i.described && !is_blank(i)))
}

/// The dialog's confirm: "Apply to Photo", "Apply to 3 Photos".
pub fn apply_label(app: &LightcraftApp) -> String {
    let n = app.ai.as_ref().map(|ai| ai.describe.items.iter().filter(|i| i.described && !is_blank(i)).count()).unwrap_or(0);
    if n > 1 { format!("Apply to {n} Photos") } else { "Apply to Photo".to_string() }
}

fn is_blank(item: &Item) -> bool {
    item.title.trim().is_empty() && item.caption.trim().is_empty() && split_keywords(&item.keywords).is_empty()
}

/// The `photo.setMeta` params for one photo: its title and caption where not empty, its keywords
/// added (a new name goes inside the default parent, as typed into the Keywording box).
fn meta_params(catalog: &lightcraft_catalog::Catalog, item: &Item) -> Option<Value> {
    if is_blank(item) {
        return None;
    }
    let mut p = json!({"ids": [item.id]});
    if !item.title.trim().is_empty() {
        p["title"] = json!(item.title.trim());
    }
    if !item.caption.trim().is_empty() {
        p["caption"] = json!(item.caption.trim());
    }
    let keywords: Vec<String> = split_keywords(&item.keywords).iter().map(|k| catalog.typed_keyword(k)).filter(|k| !k.is_empty()).collect();
    if !keywords.is_empty() {
        p["addKeywords"] = json!(keywords);
    }
    Some(p)
}

/// Apply: write each described photo's title, caption and keywords (`photo.setMeta`, one undo step
/// per photo).
pub fn apply(app: &mut LightcraftApp) -> Result<Value, String> {
    let writes: Vec<Value> = {
        let Some(ai) = app.ai.as_ref() else { return Err("AI is off in this session".into()) };
        if ai.describe.run.is_some() {
            return Err("Wait until Describe is done".into());
        }
        ai.describe.items.iter().filter(|i| i.described).filter_map(|i| meta_params(&app.session.catalog, i)).collect()
    };
    if writes.is_empty() {
        return Err("Nothing to apply yet: click Describe first".into());
    }
    let mut applied = 0usize;
    for p in writes {
        app.run("photo.setMeta", p)?;
        applied = applied.saturating_add(1);
    }
    let ctx = app.tasks.repaint.clone().unwrap_or_default();
    app.toast(
        &ctx,
        if applied == 1 {
            "Applied the title, caption and keywords".to_string()
        } else {
            format!("Applied titles, captions and keywords to {applied} photos")
        },
    );
    Ok(json!({"applied": applied}))
}

/// The Info panel's way in (below the colour labels): Describe Photo for the selection.
pub fn info_button(app: &mut LightcraftApp, ui: &mut egui::Ui) {
    if app.ai.is_none() {
        return;
    }
    let r = crate::widgets::text_button(ui, "infoAiDescribe", "AI: Describe Photo…", false)
        .on_hover_text("Write a title, a caption and keywords with AI (publik API or your own key)");
    if r.clicked()
        && let Err(e) = app.run("dialog.aiDescribe", json!({}))
    {
        app.toast_error(ui.ctx(), e);
    }
}

// ---- the model's answer -------------------------------------------------------------------------

/// What the model wrote for a photo, cleaned up.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Description {
    title: String,
    caption: String,
    keywords: Vec<String>,
}

impl Description {
    fn is_empty(&self) -> bool {
        self.title.is_empty() && self.caption.is_empty() && self.keywords.is_empty()
    }
}

/// The answer as asked (one JSON object), else labelled lines (`Title: …`), else the first line as
/// the title, the second as the caption and the rest as keywords.
fn parse_description(text: &str) -> Description {
    let body: String = text.lines().filter(|l| !l.trim_start().starts_with("```")).collect::<Vec<_>>().join("\n");
    let raw = from_json(&body).unwrap_or_else(|| from_lines(&body));
    Description {
        title: clip(&one_line(raw.title.trim_matches(['"', '\'', '*'])), MAX_TITLE_CHARS),
        caption: clip(&one_line(&raw.caption), MAX_CAPTION_CHARS),
        keywords: clean_keywords(raw.keywords),
    }
}

fn from_json(text: &str) -> Option<Description> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    let v: Value = serde_json::from_str(text.get(start..=end)?).ok()?;
    let s = |keys: &[&str]| keys.iter().find_map(|k| v.get(*k).and_then(Value::as_str)).unwrap_or("").to_string();
    let keywords = match v.get("keywords").or_else(|| v.get("tags")) {
        Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).map(str::to_string).collect(),
        Some(Value::String(s)) => split_keywords(s),
        _ => Vec::new(),
    };
    let d = Description { title: s(&["title"]), caption: s(&["caption", "description"]), keywords };
    (!d.is_empty()).then_some(d)
}

fn from_lines(text: &str) -> Description {
    let mut d = Description::default();
    let mut loose: Vec<&str> = Vec::new();
    for line in text.lines().map(|l| l.trim().trim_start_matches(['-', '*', '#', '•']).trim()).filter(|l| !l.is_empty()) {
        let labelled = line
            .split_once(':')
            .map(|(label, rest)| (label.trim().trim_matches('*').trim().to_lowercase(), rest.trim().trim_matches(['*', '"']).trim()));
        match labelled {
            Some((label, rest)) if label == "title" => d.title = rest.to_string(),
            Some((label, rest)) if label == "caption" || label == "description" => d.caption = rest.to_string(),
            Some((label, rest)) if label == "keywords" || label == "tags" => d.keywords.extend(split_keywords(rest)),
            _ => loose.push(line),
        }
    }
    if d.is_empty() {
        let mut lines = loose.into_iter();
        d.title = lines.next().unwrap_or_default().to_string();
        d.caption = lines.next().unwrap_or_default().to_string();
        for line in lines {
            d.keywords.extend(split_keywords(line));
        }
    }
    d
}

/// Keywords from text: separated by commas, semicolons or lines.
fn split_keywords(text: &str) -> Vec<String> {
    text.split([',', ';', '\n']).map(str::trim).filter(|k| !k.is_empty()).map(str::to_string).collect()
}

/// At most [`MAX_KEYWORDS`] keywords, each once whatever its case. A `|` would make a keyword
/// hierarchy in LightCraft, so the model's keywords have none.
fn clean_keywords(raw: Vec<String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for k in raw {
        let k = one_line(&k.replace('|', " "));
        let k = k.trim().trim_start_matches('#').trim().trim_matches(['"', '\'', '.', '*']).trim().to_string();
        if k.is_empty() || k.chars().count() > MAX_KEYWORD_CHARS || out.iter().any(|x| x.to_lowercase() == k.to_lowercase()) {
            continue;
        }
        out.push(k);
        if out.len() >= MAX_KEYWORDS {
            break;
        }
    }
    out
}

/// Whitespace (line breaks, tabs, runs of spaces) as single spaces.
fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// At most `max` characters (never inside one).
fn clip(text: &str, max: usize) -> String {
    text.chars().take(max).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_answers_parse_and_are_cleaned() {
        let d = parse_description(
            "```json\n{\"title\": \"Harbour at dusk\", \"caption\": \"Boats rest in a calm\\nharbour.\", \"keywords\": [\"harbour\", \"Boats\", \"boats\", \"#dusk\", \"sea|coast\", \"\"]}\n```",
        );
        assert_eq!(d.title, "Harbour at dusk");
        assert_eq!(d.caption, "Boats rest in a calm harbour.");
        assert_eq!(d.keywords, vec!["harbour", "Boats", "dusk", "sea coast"]);
    }

    #[test]
    fn keywords_may_come_as_one_string() {
        let d = parse_description(r#"Here you go: {"title": "Fox", "description": "A red fox.", "keywords": "fox, animal; snow"}"#);
        assert_eq!((d.title.as_str(), d.caption.as_str()), ("Fox", "A red fox."));
        assert_eq!(d.keywords, vec!["fox", "animal", "snow"]);
    }

    #[test]
    fn labelled_lines_are_the_first_fallback() {
        let d = parse_description("**Title:** Old bridge\nCaption: A stone bridge over a river.\nKeywords: bridge, river, stone");
        assert_eq!(d.title, "Old bridge");
        assert_eq!(d.caption, "A stone bridge over a river.");
        assert_eq!(d.keywords, vec!["bridge", "river", "stone"]);
    }

    #[test]
    fn plain_lines_are_the_last_fallback() {
        let d = parse_description("Mountain lake\nA clear lake below snowy peaks.\nlake, mountains\nsnow");
        assert_eq!(d.title, "Mountain lake");
        assert_eq!(d.caption, "A clear lake below snowy peaks.");
        assert_eq!(d.keywords, vec!["lake", "mountains", "snow"]);
        assert_eq!(parse_description(""), Description::default());
    }

    #[test]
    fn answers_are_capped() {
        let many: Vec<String> = (0..40).map(|i| format!("k{i}")).collect();
        let text = json!({"title": "t".repeat(500), "caption": "c".repeat(5000), "keywords": many}).to_string();
        let d = parse_description(&text);
        assert_eq!(d.title.chars().count(), MAX_TITLE_CHARS);
        assert_eq!(d.caption.chars().count(), MAX_CAPTION_CHARS);
        assert_eq!(d.keywords.len(), MAX_KEYWORDS);
        assert!(clean_keywords(vec!["x".repeat(MAX_KEYWORD_CHARS + 1)]).is_empty());
    }

    #[test]
    fn apply_writes_only_what_is_there() {
        let catalog = lightcraft_catalog::Catalog::default();
        let item = |title: &str, caption: &str, keywords: &str| Item {
            id: 7,
            described: true,
            title: title.into(),
            caption: caption.into(),
            keywords: keywords.into(),
            ..Item::default()
        };
        assert_eq!(
            meta_params(&catalog, &item(" Fox ", "", "fox, , snow")),
            Some(json!({"ids": [7], "title": "Fox", "addKeywords": ["fox", "snow"]}))
        );
        assert_eq!(meta_params(&catalog, &item("", "A red fox.", "")), Some(json!({"ids": [7], "caption": "A red fox."})));
        assert_eq!(meta_params(&catalog, &item(" ", "", " , ")), None);
    }

    #[test]
    fn the_jpeg_carries_no_metadata_and_stays_small() {
        let o = jpeg_options();
        assert_eq!(o.format, ExportFormat::Jpeg);
        assert_eq!(o.metadata, MetadataPolicy::None);
        assert!(o.remove_location && !o.hdr && o.watermark.is_none());
        assert_eq!(o.resize.map(|r| (r.value, r.dont_enlarge)), Some((EDGE as f32, true)));
    }

    #[test]
    fn apply_writes_through_set_meta_and_undoes() {
        let mut app = LightcraftApp::new(lightcraft_engine::Session::with_demo(), Default::default());
        let mut ai =
            Ai::new(AppInfo { slug: "lightcraft-tests", name: "LightCraft", version: "0.0.0", app_token: "", dialects: &[], config_dir: None });
        ai.skip_first_run();
        app.ai = Some(ai);
        let id = app.session.visible_cloned()[0].0;
        app.run("library.select", json!({"ids": [id]})).unwrap();
        app.run("dialog.aiDescribe", json!({})).unwrap();
        let before = app.session.catalog.photo(PhotoId(id)).unwrap().meta.clone();
        let ai = app.ai.as_mut().unwrap();
        ai.describe.sync(&app.session.catalog, &[id]);
        let item = &mut ai.describe.items[0];
        (item.described, item.title, item.caption, item.keywords) = (true, " Harbour at dusk ".into(), "Boats rest.".into(), "harbour, dusk".into());
        assert!(can_apply(&app));
        assert_eq!(apply_label(&app), "Apply to Photo");
        let d = app.ui.dialog.clone().unwrap();
        let r = crate::panels::dialogs::confirm_dialog(&mut app, &d).unwrap();
        assert_eq!(r["applied"], 1);
        let m = &app.session.catalog.photo(PhotoId(id)).unwrap().meta;
        assert_eq!((m.title.as_str(), m.caption.as_str()), ("Harbour at dusk", "Boats rest."));
        assert!(["harbour", "dusk"].iter().all(|k| m.keywords.iter().any(|x| x == k)), "{:?}", m.keywords);
        // one Edit Info step, undone as one
        app.run("edit.undo", json!({})).unwrap();
        assert_eq!(app.session.catalog.photo(PhotoId(id)).unwrap().meta, before);
    }

    #[test]
    fn the_answer_follows_the_interface_language() {
        assert_eq!(user_text(crate::i18n::Locale::En), "Describe this photo.");
        assert!(user_text(crate::i18n::Locale::De).contains("(language code de)"));
    }
}
