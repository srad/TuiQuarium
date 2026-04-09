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

#[derive(Debug, Clone)]
pub struct DeleteArtifacts {
    pub session_path: PathBuf,
    pub history_csv_path: Option<PathBuf>,
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
    let path = session_json_path(&dir, session.session_id);
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
    let path = session_history_csv_path(&dir, session.session_id);
    let mut file = fs::File::create(&path)?;
    writeln!(
        file,
        "day,producer_total_biomass,consumer_biomass,creature_count,juvenile_count,subadult_count,species_count,rooted_producer_count,rooted_producer_biomass,detritus_energy,mean_maturity_progress,mean_reserve_buffer,mean_reproductive_buffer,max_reproductive_buffer,mean_recent_assimilation,mean_brood_cooldown,reproduction_ready_count,rolling_pelagic_consumer_intake,rolling_producer_npp,rolling_consumer_intake,rolling_consumer_maintenance,dissolved_n,dissolved_p,phytoplankton_load,creature_births_delta,creature_deaths_delta,producer_births_delta,producer_deaths_delta"
    )?;
    for sample in session.sim.archived_daily_history() {
        writeln!(
            file,
            "{},{:.4},{:.4},{},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{},{},{}",
            sample.day,
            sample.producer_total_biomass,
            sample.consumer_biomass,
            sample.creature_count,
            sample.juvenile_count,
            sample.subadult_count,
            sample.species_count,
            sample.rooted_producer_count,
            sample.rooted_producer_biomass,
            sample.detritus_energy,
            sample.mean_maturity_progress,
            sample.mean_reserve_buffer,
            sample.mean_reproductive_buffer,
            sample.max_reproductive_buffer,
            sample.mean_recent_assimilation,
            sample.mean_brood_cooldown,
            sample.reproduction_ready_count,
            sample.rolling_pelagic_consumer_intake,
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

pub fn delete_save(path: &Path) -> Result<DeleteArtifacts, Box<dyn Error>> {
    let analysis = analysis_dir()?;
    delete_save_with_analysis_dir(path, &analysis)
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
        let saved_at_unix_secs = fs::metadata(&path)
            .and_then(|meta| meta.modified())
            .ok()
            .and_then(system_time_to_unix_secs)
            .unwrap_or(0);
        let preview_envelope = load_envelope(&path);
        let label = preview_envelope
            .as_ref()
            .ok()
            .map(|envelope| envelope.session.session_id.to_string())
            .or_else(|| {
                path.file_stem()
                    .and_then(|stem| stem.to_str())
                    .map(str::to_string)
            })
            .unwrap_or_else(|| "unknown save".to_string());
        let preview = preview_envelope
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
    value
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_secs())
}

fn new_session_uuid() -> Uuid {
    Uuid::new_v4()
}

fn session_json_filename(session_id: Uuid) -> String {
    format!("{session_id}.json")
}

fn session_history_csv_filename(session_id: Uuid) -> String {
    format!("{session_id}.csv")
}

fn session_json_path(dir: &Path, session_id: Uuid) -> PathBuf {
    dir.join(session_json_filename(session_id))
}

fn session_history_csv_path(dir: &Path, session_id: Uuid) -> PathBuf {
    dir.join(session_history_csv_filename(session_id))
}

fn session_id_from_path(path: &Path) -> Option<Uuid> {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .and_then(|stem| Uuid::parse_str(stem).ok())
}

fn session_id_for_save(path: &Path) -> Option<Uuid> {
    load_envelope(path)
        .ok()
        .map(|envelope| envelope.session.session_id)
        .or_else(|| session_id_from_path(path))
}

fn delete_save_with_analysis_dir(
    path: &Path,
    analysis_dir: &Path,
) -> Result<DeleteArtifacts, Box<dyn Error>> {
    let session_id = session_id_for_save(path);
    let session_path = path.to_path_buf();
    if path.exists() {
        fs::remove_file(path)?;
    }

    let history_csv_path = if let Some(session_id) = session_id {
        let candidate = session_history_csv_path(analysis_dir, session_id);
        if candidate.exists() {
            fs::remove_file(&candidate)?;
            Some(candidate)
        } else {
            None
        }
    } else {
        None
    };

    Ok(DeleteArtifacts {
        session_path,
        history_csv_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_test_dir(name: &str) -> PathBuf {
        let unique = format!("tuiquarium_session_persistence_{name}_{}", Uuid::new_v4());
        std::env::temp_dir().join(unique)
    }

    #[test]
    fn test_session_file_names_use_raw_session_ids() {
        let id = Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap();
        assert_eq!(
            session_json_filename(id),
            "123e4567-e89b-12d3-a456-426614174000.json"
        );
        assert_eq!(
            session_history_csv_filename(id),
            "123e4567-e89b-12d3-a456-426614174000.csv"
        );
    }

    #[test]
    fn test_delete_save_removes_matching_history_csv() {
        let root = temp_test_dir("delete");
        let saves = root.join("saves");
        let analysis = root.join("analysis");
        fs::create_dir_all(&saves).unwrap();
        fs::create_dir_all(&analysis).unwrap();

        let id = Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap();
        let session_path = session_json_path(&saves, id);
        let history_path = session_history_csv_path(&analysis, id);
        fs::write(&session_path, "{}").unwrap();
        fs::write(&history_path, "day\n").unwrap();

        let deleted = delete_save_with_analysis_dir(&session_path, &analysis).unwrap();
        assert_eq!(deleted.session_path, session_path);
        assert_eq!(deleted.history_csv_path, Some(history_path.clone()));
        assert!(!session_path.exists());
        assert!(!history_path.exists());

        fs::remove_dir_all(root).unwrap();
    }
}
