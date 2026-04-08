use std::error::Error;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tuiq_core::save::SimulationSnapshot;
use tuiq_core::Simulation;
use tuiq_render::RenderTheme;
use uuid::Uuid;

const SAVE_FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaveEnvelope {
    pub format_version: u32,
    pub app_version: String,
    pub saved_at_unix_secs: u64,
    pub summary: SaveSummary,
    pub session: SavedSession,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaveSummary {
    pub day: u64,
    pub creature_count: usize,
    pub species_count: usize,
    pub producer_biomass: f32,
    pub consumer_biomass: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedSession {
    #[serde(default = "new_session_uuid")]
    pub session_id: Uuid,
    pub simulation: SimulationSnapshot,
    pub paused: bool,
    pub speed: f32,
    pub show_diagnostics: bool,
    pub theme: SavedTheme,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SavedTheme {
    Classic,
    Ocean,
    DeepSea,
    CoralReef,
    Brackish,
    RetroCrt,
    Blueprint,
    Frozen,
}

impl From<RenderTheme> for SavedTheme {
    fn from(value: RenderTheme) -> Self {
        match value {
            RenderTheme::Classic => SavedTheme::Classic,
            RenderTheme::Ocean => SavedTheme::Ocean,
            RenderTheme::DeepSea => SavedTheme::DeepSea,
            RenderTheme::CoralReef => SavedTheme::CoralReef,
            RenderTheme::Brackish => SavedTheme::Brackish,
            RenderTheme::RetroCrt => SavedTheme::RetroCrt,
            RenderTheme::Blueprint => SavedTheme::Blueprint,
            RenderTheme::Frozen => SavedTheme::Frozen,
        }
    }
}

impl From<SavedTheme> for RenderTheme {
    fn from(value: SavedTheme) -> Self {
        match value {
            SavedTheme::Classic => RenderTheme::Classic,
            SavedTheme::Ocean => RenderTheme::Ocean,
            SavedTheme::DeepSea => RenderTheme::DeepSea,
            SavedTheme::CoralReef => RenderTheme::CoralReef,
            SavedTheme::Brackish => RenderTheme::Brackish,
            SavedTheme::RetroCrt => RenderTheme::RetroCrt,
            SavedTheme::Blueprint => RenderTheme::Blueprint,
            SavedTheme::Frozen => RenderTheme::Frozen,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SaveListEntry {
    pub path: PathBuf,
    pub label: String,
    pub saved_at_unix_secs: u64,
    pub preview: Result<SavePreview, String>,
}

#[derive(Debug, Clone)]
pub struct SavePreview {
    pub app_version: String,
    pub summary: SaveSummary,
}

#[derive(Debug, Clone)]
pub struct SaveArtifacts {
    pub session_path: PathBuf,
    pub history_csv_path: PathBuf,
}

pub fn save_session(session: &crate::SessionState) -> Result<SaveArtifacts, Box<dyn Error>> {
    let saved_at_unix_secs = now_unix_secs();
    let stats = session.sim.stats();
    let producer_biomass = stats.producer_leaf_biomass
        + stats.producer_structural_biomass
        + stats.producer_belowground_reserve;
    let envelope = SaveEnvelope {
        format_version: SAVE_FORMAT_VERSION,
        app_version: env!("CARGO_PKG_VERSION").to_string(),
        saved_at_unix_secs,
        summary: SaveSummary {
            day: stats.elapsed_days + 1,
            creature_count: stats.creature_count,
            species_count: stats.species_count,
            producer_biomass,
            consumer_biomass: stats.consumer_biomass,
        },
        session: SavedSession {
            session_id: session.session_id,
            simulation: session.sim.snapshot(),
            paused: session.paused,
            speed: session.speed,
            show_diagnostics: session.show_diagnostics,
            theme: session.theme.into(),
        },
    };

    let dir = saves_dir()?;
    let path = dir.join(format!("tuiquarium_session_{}.json", session.session_id));
    let json = serde_json::to_string_pretty(&envelope)?;
    fs::write(&path, json)?;
    let history_csv_path = export_history_csv(session)?;
    Ok(SaveArtifacts {
        session_path: path,
        history_csv_path,
    })
}

pub fn load_session(path: &Path) -> Result<crate::SessionState, Box<dyn Error>> {
    let envelope = load_envelope(path)?;
    ensure_supported_version(&envelope)?;
    let sim = tuiq_core::AquariumSim::from_snapshot(&envelope.session.simulation)
        .map_err(|err| format!("save restore failed: {err}"))?;
    Ok(crate::SessionState {
        session_id: envelope.session.session_id,
        sim,
        renderer: tuiq_render::TuiRenderer::new(),
        paused: envelope.session.paused,
        speed: envelope.session.speed,
        show_diagnostics: envelope.session.show_diagnostics,
        show_help: false,
        theme: envelope.session.theme.into(),
        tick_rate: std::time::Duration::from_millis(50),
        last_tick: std::time::Instant::now(),
        accumulator: std::time::Duration::ZERO,
    })
}

pub fn export_history_csv(session: &crate::SessionState) -> Result<PathBuf, Box<dyn Error>> {
    let dir = analysis_dir()?;
    let path = dir.join(format!("tuiquarium_history_{}.csv", session.session_id));
    let mut file = fs::File::create(&path)?;
    writeln!(
        file,
        "day,producer_total_biomass,consumer_biomass,creature_count,juvenile_count,species_count,detritus_energy,rolling_producer_npp,rolling_consumer_intake,rolling_consumer_maintenance,dissolved_n,dissolved_p,phytoplankton_load,creature_births_delta,creature_deaths_delta,producer_births_delta,producer_deaths_delta"
    )?;
    for sample in session.sim.archived_daily_history() {
        writeln!(
            file,
            "{},{:.4},{:.4},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{},{},{}",
            sample.day,
            sample.producer_total_biomass,
            sample.consumer_biomass,
            sample.creature_count,
            sample.juvenile_count,
            sample.species_count,
            sample.detritus_energy,
            sample.rolling_producer_npp,
            sample.rolling_consumer_intake,
            sample.rolling_consumer_maintenance,
            sample.dissolved_n,
            sample.dissolved_p,
            sample.phytoplankton_load,
            sample.creature_births_delta,
            sample.creature_deaths_delta,
            sample.producer_births_delta,
            sample.producer_deaths_delta,
        )?;
    }
    Ok(path)
}

pub fn list_save_entries() -> Result<Vec<SaveListEntry>, Box<dyn Error>> {
    let dir = saves_dir()?;
    let mut files: Vec<_> = fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();

    files.sort_by_key(|path| {
        fs::metadata(path)
            .and_then(|meta| meta.modified())
            .unwrap_or(UNIX_EPOCH)
    });
    files.reverse();

    let mut entries = Vec::with_capacity(files.len());
    for path in files {
        let label = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown save")
            .to_string();
        let saved_at_unix_secs = fs::metadata(&path)
            .and_then(|meta| meta.modified())
            .ok()
            .and_then(system_time_to_unix_secs)
            .unwrap_or(0);
        let preview = load_envelope(&path)
            .and_then(|envelope| {
                ensure_supported_version(&envelope)?;
                Ok(SavePreview {
                    app_version: envelope.app_version,
                    summary: envelope.summary,
                })
            })
            .map_err(|err| err.to_string());
        entries.push(SaveListEntry {
            path,
            label,
            saved_at_unix_secs,
            preview,
        });
    }

    Ok(entries)
}

pub fn latest_valid_save(entries: &[SaveListEntry]) -> Option<&SaveListEntry> {
    entries.iter().find(|entry| entry.preview.is_ok())
}

pub fn saves_dir() -> Result<PathBuf, Box<dyn Error>> {
    let dir = app_home_dir()?.join("saves");
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub fn analysis_dir() -> Result<PathBuf, Box<dyn Error>> {
    let dir = app_home_dir()?.join("analysis");
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn load_envelope(path: &Path) -> Result<SaveEnvelope, Box<dyn Error>> {
    let json = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&json)?)
}

fn ensure_supported_version(envelope: &SaveEnvelope) -> Result<(), Box<dyn Error>> {
    match envelope.format_version {
        SAVE_FORMAT_VERSION => Ok(()),
        other => Err(format!("unsupported save format version {other}").into()),
    }
}

fn app_home_dir() -> Result<PathBuf, Box<dyn Error>> {
    let home = if cfg!(windows) {
        std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME"))?
    } else {
        std::env::var("HOME")?
    };
    let dir = PathBuf::from(home).join(".tuiquarium");
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn system_time_to_unix_secs(value: SystemTime) -> Option<u64> {
    value.duration_since(UNIX_EPOCH).ok().map(|duration| duration.as_secs())
}

fn new_session_uuid() -> Uuid {
    Uuid::new_v4()
}
