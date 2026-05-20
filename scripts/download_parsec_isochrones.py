#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin


CMD_URL = "https://stev.oapd.inaf.it/cgi-bin/cmd_3.7"
DEFAULT_OUTPUT_DIR = Path("inputs/parsec_isochrones_hybrid_0p1myr_to13gyr")
DEFAULT_PHOTSYS_FILE = "YBC_tab_mag_odfnew/tab_mag_gaiaEDR3.dat"
DEFAULT_PHOTSYS_VERSION = "YBC"
DEFAULT_TRACK_PARSEC = "parsec_CAF09_v1.2S"
DEFAULT_TRACK_COLIBRI = "parsec_CAF09_v1.2S_S_LMC_08_web"
DEFAULT_MAX_ISOCHRONES_PER_REQUEST = 400


@dataclass(frozen=True)
class AgeSegment:
    name: str
    is_log_age: bool
    low: float
    high: float | None
    step: float
    count: int


@dataclass(frozen=True)
class MetallicityGroup:
    name: str
    low: float
    high: float
    step: float
    count: int


class CMDFormParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.fields: dict[str, Any] = {}
        self._select_name: str | None = None
        self._select_first_value: str | None = None
        self._select_selected_value: str | None = None

    @staticmethod
    def _attrs(attrs: list[tuple[str, str | None]]) -> dict[str, str]:
        return {key.lower(): value or "" for key, value in attrs}

    def _add_field(self, name: str, value: str) -> None:
        if name in self.fields:
            old_value = self.fields[name]
            if isinstance(old_value, list):
                old_value.append(value)
            else:
                self.fields[name] = [old_value, value]
        else:
            self.fields[name] = value

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = self._attrs(attrs)
        if tag == "select":
            self._select_name = attr.get("name")
            self._select_first_value = None
            self._select_selected_value = None
            return
        if tag == "option" and self._select_name is not None:
            value = attr.get("value", "")
            if self._select_first_value is None:
                self._select_first_value = value
            if "selected" in attr:
                self._select_selected_value = value
            return
        if tag != "input":
            return
        name = attr.get("name")
        if not name:
            return
        input_type = attr.get("type", "text").lower()
        value = attr.get("value", "")
        if input_type in {"text", "hidden"}:
            self._add_field(name, value)
        elif input_type == "radio":
            if "checked" in attr:
                self.fields[name] = value
        elif input_type == "checkbox":
            self.fields[name] = "1" if "checked" in attr else "0"

    def handle_endtag(self, tag: str) -> None:
        if tag != "select" or self._select_name is None:
            return
        self.fields[self._select_name] = self._select_selected_value or self._select_first_value or ""
        self._select_name = None
        self._select_first_value = None
        self._select_selected_value = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _format_float(value: float) -> str:
    return f"{value:.12g}"


def _format_myr_label(age_yr: float) -> str:
    label = f"{age_yr / 1.0e6:g}".replace(".", "p")
    return f"{label}myr"


def _format_dex_label(step: float) -> str:
    return f"{step:g}".replace(".", "p")


def _age_segments(
    *,
    min_age_yr: float,
    linear_start_yr: float,
    linear_end_yr: float,
    linear_step_yr: float,
    max_age_yr: float,
    log_step_dex: float,
) -> list[AgeSegment]:
    linear_count = int(math.floor((linear_end_yr - linear_start_yr) / linear_step_yr)) + 1
    log_low = math.log10(linear_end_yr) + log_step_dex
    log_high_requested = math.log10(max_age_yr)
    log_count = max(0, int(math.floor((log_high_requested - log_low) / log_step_dex)))
    segments: list[AgeSegment] = []
    if min_age_yr < linear_start_yr:
        segments.append(
            AgeSegment(
                name=f"age_{_format_myr_label(min_age_yr)}",
                is_log_age=False,
                low=float(min_age_yr),
                high=None,
                step=0.0,
                count=1,
            )
        )
    segments.append(
        AgeSegment(
            name=(
                "young_linear_"
                f"{_format_myr_label(linear_start_yr)}_to_{_format_myr_label(linear_end_yr)}_"
                f"{_format_myr_label(linear_step_yr)}"
            ),
            is_log_age=False,
            low=float(linear_start_yr),
            high=float(linear_end_yr),
            step=float(linear_step_yr),
            count=linear_count,
        )
    )
    if log_count > 0:
        segments.append(
            AgeSegment(
                name=f"old_log_after_{_format_myr_label(linear_end_yr)}_{_format_dex_label(log_step_dex)}dex",
                is_log_age=True,
                low=float(log_low),
                high=float(log_low + (log_count - 1) * log_step_dex),
                step=float(log_step_dex),
                count=log_count,
            )
        )
    if max_age_yr > linear_end_yr:
        segments.append(
            AgeSegment(
                name=f"age_{_format_myr_label(max_age_yr)}",
                is_log_age=False,
                low=float(max_age_yr),
                high=None,
                step=0.0,
                count=1,
            )
        )
    return segments


def _metallicity_groups() -> list[MetallicityGroup]:
    return [
        MetallicityGroup(f"mh_{_metallicity_label(value)}", value, value, 0.0, 1)
        for value in (-1.5, -1.3, -1.1, -0.9, -0.5, -0.3, -0.1, 0.1, 0.3)
    ]


def _metallicity_label(value: float) -> str:
    prefix = "m" if value < 0 else "p"
    return prefix + str(abs(value)).replace(".", "p")


def _group_from_values(values: list[float]) -> MetallicityGroup:
    low = float(values[0])
    high = float(values[-1])
    step = float(values[1] - values[0]) if len(values) > 1 else 0.0
    if len(values) == 1:
        name = f"mh_{_metallicity_label(low)}"
    else:
        name = f"mh_{_metallicity_label(low)}_to_{_metallicity_label(high)}"
    return MetallicityGroup(name=name, low=low, high=high, step=step, count=len(values))


def _request_plan(
    *,
    age_segments: list[AgeSegment],
    metallicity_groups: list[MetallicityGroup],
    max_isochrones_per_request: int,
) -> list[tuple[AgeSegment, MetallicityGroup]]:
    requests: list[tuple[AgeSegment, MetallicityGroup]] = []
    for segment in age_segments:
        for group in metallicity_groups:
            max_ages_per_request = max(1, max_isochrones_per_request // max(1, group.count))
            for chunk_index, age_chunk in enumerate(_split_age_segment(segment, max_ages_per_request), start=1):
                if age_chunk.count * group.count <= max_isochrones_per_request:
                    requests.append((age_chunk, group))
                    continue
                if group.step <= 0:
                    raise ValueError(
                        f"Cannot split metallicity group {group.name}; "
                        f"{age_chunk.count * group.count} isochrones exceeds {max_isochrones_per_request}."
                    )
                metals_per_request = max(1, max_isochrones_per_request // max(1, age_chunk.count))
                values = [round(group.low + index * group.step, 10) for index in range(group.count)]
                for start in range(0, len(values), metals_per_request):
                    requests.append((age_chunk, _group_from_values(values[start : start + metals_per_request])))
    return requests


def _split_age_segment(segment: AgeSegment, max_count: int) -> list[AgeSegment]:
    if segment.count <= max_count:
        return [segment]
    if segment.step <= 0:
        raise ValueError(f"Cannot split single-age segment {segment.name}.")
    chunks: list[AgeSegment] = []
    for chunk_number, start_index in enumerate(range(0, segment.count, max_count), start=1):
        count = min(max_count, segment.count - start_index)
        low = segment.low + start_index * segment.step
        high = low + (count - 1) * segment.step
        chunks.append(
            AgeSegment(
                name=f"{segment.name}_part{chunk_number:02d}",
                is_log_age=segment.is_log_age,
                low=float(low),
                high=float(high),
                step=segment.step,
                count=count,
            )
        )
    return chunks


def _base_cmd_params(session: Any) -> dict[str, Any]:
    response = session.get(CMD_URL, timeout=60)
    response.raise_for_status()
    parser = CMDFormParser()
    parser.feed(response.text)
    params = dict(parser.fields)
    params["submit_form"] = "Submit"
    return params


def _request_params(
    base_params: dict[str, Any],
    *,
    age_segment: AgeSegment,
    metallicity_group: MetallicityGroup,
    photsys_file: str,
    photsys_version: str,
    track_parsec: str,
    track_colibri: str,
) -> dict[str, Any]:
    params = dict(base_params)
    params.update(
        {
            "track_parsec": track_parsec,
            "track_colibri": track_colibri,
            "track_postagb": "no",
            "photsys_file": photsys_file,
            "photsys_version": photsys_version,
            "dust_sourceM": "dpmod60alox40",
            "dust_sourceC": "AMCSIC15",
            "extinction_av": "0.0",
            "extinction_coeff": "constant",
            "extinction_curve": "cardelli",
            "output_kind": "0",
            "output_gzip": "1",
            "isoc_ismetlog": "1",
            "isoc_metlow": _format_float(metallicity_group.low),
            "isoc_metupp": _format_float(metallicity_group.high),
            "isoc_dmet": _format_float(metallicity_group.step),
        }
    )
    if age_segment.is_log_age:
        params.update(
            {
                "isoc_isagelog": "1",
                "isoc_lagelow": _format_float(age_segment.low),
                "isoc_lageupp": _format_float(age_segment.high if age_segment.high is not None else age_segment.low),
                "isoc_dlage": _format_float(age_segment.step),
            }
        )
    else:
        params.update(
            {
                "isoc_isagelog": "0",
                "isoc_agelow": _format_float(age_segment.low),
                "isoc_ageupp": _format_float(age_segment.high if age_segment.high is not None else age_segment.low),
                "isoc_dage": _format_float(age_segment.step),
            }
        )
    return params


def _extract_output_url(response_text: str) -> str:
    matches = re.findall(r"""href=(?:"([^"]*output[^"]*)"|'([^']*output[^']*)'|([^\s>]*output[^\s>]*))""", response_text)
    if matches:
        href = next(part for part in matches[0] if part)
        return urljoin(CMD_URL, href)
    error_text = " ".join(re.findall(r"""<p[^>]*class=["']errorwarning["'][^>]*>(.*?)</p>""", response_text, flags=re.S))
    error_text = re.sub(r"<[^>]+>", " ", error_text)
    error_text = re.sub(r"\s+", " ", error_text).strip()
    if error_text:
        raise RuntimeError(f"CMD rejected request: {error_text}")
    raise RuntimeError("CMD did not return an output link or a parseable error message.")


def _request_with_retries(session: Any, method: str, url: str, *, retries: int, **kwargs: Any) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except Exception as exc:  # noqa: BLE001 - preserve original request exception after final retry.
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(60.0, 5.0 * attempt))
    raise RuntimeError(f"Request failed after {retries} attempts: {url}") from last_error


def _download_dat(session: Any, *, params: dict[str, Any], output_path: Path, retries: int) -> None:
    response = _request_with_retries(session, "POST", CMD_URL, data=params, timeout=180, retries=retries)
    output_url = _extract_output_url(response.text)
    data_response = _request_with_retries(session, "GET", output_url, timeout=600, retries=retries)
    payload = data_response.content
    if output_url.endswith(".gz") or payload.startswith(b"\x1f\x8b"):
        payload = gzip.decompress(payload)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_bytes(payload)
    tmp_path.replace(output_path)


def _actual_grid_counts(files: list[Path]) -> dict[str, Any]:
    ages: set[float] = set()
    metallicities: set[float] = set()
    row_count = 0
    for path in files:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    metallicities.add(round(float(parts[1]), 8))
                    ages.add(round(float(parts[2]), 8))
                    row_count += 1
                except ValueError:
                    continue
    age_myr = sorted(10.0**age / 1.0e6 for age in ages)
    return {
        "rows": row_count,
        "ages": len(ages),
        "metallicities": len(metallicities),
        "isochrones": len(ages) * len(metallicities),
        "age_min_myr": min(age_myr) if age_myr else None,
        "age_max_myr": max(age_myr) if age_myr else None,
        "metallicity_values": sorted(metallicities),
    }


def _build_manifest(
    *,
    output_dir: Path,
    age_segments: list[AgeSegment],
    metallicity_groups: list[MetallicityGroup],
    requests_plan: list[tuple[AgeSegment, MetallicityGroup]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    n_ages = sum(segment.count for segment in age_segments)
    n_metals = sum(group.count for group in metallicity_groups)
    files = [output_dir / f"parsec_cmd37_{segment.name}_{group.name}.dat" for segment, group in requests_plan]
    return {
        "cmd_url": CMD_URL,
        "output_dir": str(output_dir),
        "photometric_system": args.photsys_file,
        "photometric_system_version": args.photsys_version,
        "track_parsec": args.track_parsec,
        "track_colibri": args.track_colibri,
        "age_grid": {
            "min_age_yr": args.min_age_yr,
            "linear_anchor_age_yr": args.min_age_yr,
            "linear_start_yr": args.linear_start_yr,
            "linear_end_yr": args.linear_end_yr,
            "linear_step_yr": args.linear_step_yr,
            "log_start_after_yr": args.linear_end_yr,
            "log_step_dex": args.log_step_dex,
            "max_age_yr": args.max_age_yr,
            "segments": [segment.__dict__ for segment in age_segments],
        },
        "metallicity_grid": {
            "parameter": "[M/H]",
            "groups": [group.__dict__ for group in metallicity_groups],
        },
        "extinction_handling": {
            "mode": "runtime_continuous_in_chronos",
            "downloaded_cmd_extinction_av": 0.0,
            "chronos_av_range_mag": [args.av_min_mag, args.av_max_mag],
            "reference_av_step_mag": args.av_step_mag,
            "note": (
                "Chronos uses unreddened PARSEC tables, then applies A_V at runtime. "
                "A 0.1 mag precomputed CMD extinction grid would multiply files and memory by "
                "roughly the number of A_V bins and is not used by the current interpolator."
            ),
        },
        "counts": {
            "ages": n_ages,
            "metallicities": n_metals,
            "isochrones": n_ages * n_metals,
            "files": len(files),
            "max_isochrones_per_request": args.max_isochrones_per_request,
        },
        "files": [str(path.relative_to(output_dir)) for path in files],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the Chronos hybrid full-age PARSEC CMD 3.7 grid."
    )
    parser.add_argument("--output-dir", type=Path, default=repo_root() / DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-age-yr", type=float, default=1.0e5)
    parser.add_argument("--linear-start-yr", type=float, default=1.0e5)
    parser.add_argument("--linear-end-yr", type=float, default=2.0e8)
    parser.add_argument("--linear-step-yr", type=float, default=1.0e5)
    parser.add_argument("--max-age-yr", type=float, default=1.3e10)
    parser.add_argument("--log-step-dex", type=float, default=0.02)
    parser.add_argument("--av-min-mag", type=float, default=0.0)
    parser.add_argument("--av-max-mag", type=float, default=5.0)
    parser.add_argument("--av-step-mag", type=float, default=0.1)
    parser.add_argument("--photsys-file", default=DEFAULT_PHOTSYS_FILE)
    parser.add_argument("--photsys-version", default=DEFAULT_PHOTSYS_VERSION)
    parser.add_argument("--track-parsec", default=DEFAULT_TRACK_PARSEC)
    parser.add_argument("--track-colibri", default=DEFAULT_TRACK_COLIBRI)
    parser.add_argument("--max-isochrones-per-request", type=int, default=DEFAULT_MAX_ISOCHRONES_PER_REQUEST)
    parser.add_argument("--sleep-sec", type=float, default=10.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    age_segments = _age_segments(
        min_age_yr=args.min_age_yr,
        linear_start_yr=args.linear_start_yr,
        linear_end_yr=args.linear_end_yr,
        linear_step_yr=args.linear_step_yr,
        max_age_yr=args.max_age_yr,
        log_step_dex=args.log_step_dex,
    )
    metallicity_groups = _metallicity_groups()
    requests_plan = _request_plan(
        age_segments=age_segments,
        metallicity_groups=metallicity_groups,
        max_isochrones_per_request=args.max_isochrones_per_request,
    )
    manifest = _build_manifest(
        output_dir=output_dir,
        age_segments=age_segments,
        metallicity_groups=metallicity_groups,
        requests_plan=requests_plan,
        args=args,
    )

    if args.dry_run:
        print(json.dumps(manifest, indent=2), flush=True)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for path in [*output_dir.glob("*.dat"), output_dir / "manifest.json"]:
            if path.exists():
                path.unlink()
    import requests

    session = requests.Session()
    base_params = _base_cmd_params(session)
    completed: list[Path] = []
    completed_plan: list[tuple[AgeSegment, MetallicityGroup]] = []
    for segment, group in requests_plan:
        output_path = output_dir / f"parsec_cmd37_{segment.name}_{group.name}.dat"
        if output_path.exists() and not args.overwrite:
            print(f"[skip] {output_path}", flush=True)
            completed.append(output_path)
            completed_plan.append((segment, group))
            continue
        print(f"[download] {segment.name} | {group.name} -> {output_path.name}", flush=True)
        params = _request_params(
            base_params,
            age_segment=segment,
            metallicity_group=group,
            photsys_file=args.photsys_file,
            photsys_version=args.photsys_version,
            track_parsec=args.track_parsec,
            track_colibri=args.track_colibri,
        )
        _download_dat(session, params=params, output_path=output_path, retries=args.retries)
        completed.append(output_path)
        completed_plan.append((segment, group))
        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    final_manifest = _build_manifest(
        output_dir=output_dir,
        age_segments=age_segments,
        metallicity_groups=metallicity_groups,
        requests_plan=completed_plan,
        args=args,
    )
    final_manifest["actual_counts"] = _actual_grid_counts(completed)
    (output_dir / "manifest.json").write_text(json.dumps(final_manifest, indent=2) + "\n")
    print(json.dumps(final_manifest["counts"], indent=2), flush=True)
    print(f"PARSEC grid directory: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
