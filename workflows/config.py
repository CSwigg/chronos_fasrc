from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tomllib


CONFIG_ENV_VAR = "SUPERNOVAE_MAP_CONFIG"


@dataclass(frozen=True)
class InputPaths:
    cluster_catalog_csv: Path
    cluster_family_csv: Path | None
    member_catalog_csv: Path
    velocity_catalog_csv: Path
    chronos_ages_csv: Path
    lifetime_tracks_csv: Path
    dust_xyz_fits: Path
    extinction_healpix_fits: Path
    bayestar2019_h5: Path | None = None
    decaps_h5: Path | None = None
    gaia_neighborhood_cache_dir: Path | None = None
    hunt_selection_cache_dir: Path | None = None
    mccallum_ne_fits: Path | None = None
    parsec_isochrone_dir: Path | None = None
    mist_isochrone_dir: Path | None = None


@dataclass(frozen=True)
class SurveyPaths:
    apogee: Path | None = None
    galah: Path | None = None
    lamost: Path | None = None
    desi: Path | None = None
    ges: Path | None = None
    mwm: Path | None = None
    mwm_aspcap: Path | None = None
    rave: Path | None = None

    def as_dict(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for key, value in self.__dict__.items():
            if value is not None:
                out[key] = str(value)
        return out


@dataclass(frozen=True)
class OutputPaths:
    velocity_dir: Path
    chronos_dir: Path
    map_dir: Path
    paper_dir: Path
    visualizer_plots_dir: Path

    @property
    def paper_figures_dir(self) -> Path:
        return self.paper_dir / "Figures"


@dataclass(frozen=True)
class VelocityBuildSettings:
    surveys_to_use: tuple[str, ...]
    snr_filter_enabled: bool = True
    snr_min: float = 3.0


@dataclass(frozen=True)
class RuntimePaths:
    repo_root: Path
    config_path: Path
    inputs: InputPaths
    surveys: SurveyPaths
    outputs: OutputPaths
    velocity: VelocityBuildSettings


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_config_path() -> Path:
    return repo_root() / "configs" / "paths.toml"


def example_config_path() -> Path:
    return repo_root() / "configs" / "paths.example.toml"


def resolve_config_path(config_path: str | Path | None = None) -> Path:
    if config_path is not None:
        return Path(config_path).expanduser().resolve()

    repo_local = default_config_path()
    if repo_local.exists():
        return repo_local.resolve()

    env_path = os.environ.get(CONFIG_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser().resolve()

    raise FileNotFoundError(
        "No runtime config found. Pass --config, create configs/paths.toml, "
        f"or set {CONFIG_ENV_VAR}. Start from {example_config_path()}."
    )


def _resolve_path(raw: str, *, base: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def _read_section(data: dict, section: str) -> dict:
    value = data.get(section)
    if not isinstance(value, dict):
        raise KeyError(f"Missing [{section}] section in config file.")
    return value


def load_runtime_paths(config_path: str | Path | None = None) -> RuntimePaths:
    resolved_config = resolve_config_path(config_path)
    data = tomllib.loads(resolved_config.read_text(encoding="utf-8"))
    base = repo_root()

    inputs_raw = _read_section(data, "inputs")
    surveys_raw = _read_section(data, "surveys")
    outputs_raw = _read_section(data, "outputs")
    velocity_raw = data.get("velocity", {})
    if velocity_raw is None:
        velocity_raw = {}
    if not isinstance(velocity_raw, dict):
        raise TypeError("[velocity] section must be a table if present.")

    inputs = InputPaths(
        cluster_catalog_csv=_resolve_path(inputs_raw["cluster_catalog_csv"], base=base),
        cluster_family_csv=(
            _resolve_path(inputs_raw["cluster_family_csv"], base=base)
            if inputs_raw.get("cluster_family_csv")
            else None
        ),
        member_catalog_csv=_resolve_path(inputs_raw["member_catalog_csv"], base=base),
        velocity_catalog_csv=_resolve_path(inputs_raw["velocity_catalog_csv"], base=base),
        chronos_ages_csv=_resolve_path(inputs_raw["chronos_ages_csv"], base=base),
        lifetime_tracks_csv=_resolve_path(inputs_raw["lifetime_tracks_csv"], base=base),
        dust_xyz_fits=_resolve_path(inputs_raw["dust_xyz_fits"], base=base),
        extinction_healpix_fits=_resolve_path(inputs_raw["extinction_healpix_fits"], base=base),
        bayestar2019_h5=(
            _resolve_path(inputs_raw["bayestar2019_h5"], base=base)
            if inputs_raw.get("bayestar2019_h5")
            else None
        ),
        decaps_h5=(
            _resolve_path(inputs_raw["decaps_h5"], base=base)
            if inputs_raw.get("decaps_h5")
            else None
        ),
        gaia_neighborhood_cache_dir=(
            _resolve_path(inputs_raw["gaia_neighborhood_cache_dir"], base=base)
            if inputs_raw.get("gaia_neighborhood_cache_dir")
            else None
        ),
        hunt_selection_cache_dir=(
            _resolve_path(inputs_raw["hunt_selection_cache_dir"], base=base)
            if inputs_raw.get("hunt_selection_cache_dir")
            else None
        ),
        mccallum_ne_fits=(
            _resolve_path(inputs_raw["mccallum_ne_fits"], base=base)
            if inputs_raw.get("mccallum_ne_fits")
            else None
        ),
        parsec_isochrone_dir=(
            _resolve_path(inputs_raw["parsec_isochrone_dir"], base=base)
            if inputs_raw.get("parsec_isochrone_dir")
            else None
        ),
        mist_isochrone_dir=(
            _resolve_path(inputs_raw["mist_isochrone_dir"], base=base)
            if inputs_raw.get("mist_isochrone_dir")
            else None
        ),
    )
    surveys = SurveyPaths(
        apogee=_resolve_path(surveys_raw["apogee"], base=base) if surveys_raw.get("apogee") else None,
        galah=_resolve_path(surveys_raw["galah"], base=base) if surveys_raw.get("galah") else None,
        lamost=_resolve_path(surveys_raw["lamost"], base=base) if surveys_raw.get("lamost") else None,
        desi=_resolve_path(surveys_raw["desi"], base=base) if surveys_raw.get("desi") else None,
        ges=_resolve_path(surveys_raw["ges"], base=base) if surveys_raw.get("ges") else None,
        mwm=_resolve_path(surveys_raw["mwm"], base=base) if surveys_raw.get("mwm") else None,
        mwm_aspcap=(
            _resolve_path(surveys_raw["mwm_aspcap"], base=base) if surveys_raw.get("mwm_aspcap") else None
        ),
        rave=_resolve_path(surveys_raw["rave"], base=base) if surveys_raw.get("rave") else None,
    )
    outputs = OutputPaths(
        velocity_dir=_resolve_path(outputs_raw["velocity_dir"], base=base),
        chronos_dir=_resolve_path(outputs_raw["chronos_dir"], base=base),
        map_dir=_resolve_path(outputs_raw["map_dir"], base=base),
        paper_dir=_resolve_path(outputs_raw["paper_dir"], base=base),
        visualizer_plots_dir=_resolve_path(outputs_raw["visualizer_plots_dir"], base=base),
    )
    surveys_to_use_raw = velocity_raw.get("surveys_to_use")
    if surveys_to_use_raw is None:
        surveys_to_use = ("gaia", "apogee", "galah")
    else:
        if not isinstance(surveys_to_use_raw, list) or not all(isinstance(item, str) for item in surveys_to_use_raw):
            raise TypeError("velocity.surveys_to_use must be an array of strings.")
        surveys_to_use = tuple(surveys_to_use_raw)
    velocity = VelocityBuildSettings(
        surveys_to_use=surveys_to_use,
        snr_filter_enabled=bool(velocity_raw.get("snr_filter_enabled", True)),
        snr_min=float(velocity_raw.get("snr_min", 3.0)),
    )
    return RuntimePaths(
        repo_root=base,
        config_path=resolved_config,
        inputs=inputs,
        surveys=surveys,
        outputs=outputs,
        velocity=velocity,
    )


def ensure_output_dirs(paths: RuntimePaths) -> None:
    for directory in (
        paths.outputs.velocity_dir,
        paths.outputs.chronos_dir,
        paths.outputs.map_dir,
        paths.outputs.paper_dir,
        paths.outputs.paper_figures_dir,
        paths.outputs.visualizer_plots_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def require_existing(path: Path, *, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


__all__ = [
    "CONFIG_ENV_VAR",
    "InputPaths",
    "SurveyPaths",
    "OutputPaths",
    "VelocityBuildSettings",
    "RuntimePaths",
    "default_config_path",
    "example_config_path",
    "resolve_config_path",
    "load_runtime_paths",
    "ensure_output_dirs",
    "require_existing",
]
