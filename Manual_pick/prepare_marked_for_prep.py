#!/usr/bin/env python3
"""Prepare a manually curated Marked directory for direct prep2 usage.

This script is intentionally a compatibility layer only:

1. Normalize mixed SAC/SACPZ naming styles into the legacy prep layout.
2. Copy the preferred manual pick from T3 to A, falling back to T4.
3. Generate ``*.info`` and ``wave_file_v1.dat`` for ``prep2.bash``.

It does not change waveform samples, resampling, filtering, or response
removal. Those scientific steps remain inside the original prep2 workflow.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import obspy

SAC_SUFFIXES = {".SAC", ".sac"}
PZ_SUFFIXES = {".sacpz", ".SACPZ"}
MAX_PREP2_NPTS = 90000
StationKey = tuple[str, str, str, str]


@dataclass(frozen=True)
class WaveformRecord:
    sac_path: Path
    net: str
    sta: str
    loc_logic: str
    cha: str
    loc_output: str
    pick_value: float | None
    pick_source: str | None
    starttime: obspy.UTCDateTime

    @property
    def station_key(self) -> StationKey:
        return (self.net, self.sta, self.loc_logic, self.cha)

    @property
    def canonical_id(self) -> str:
        return ".".join(self.station_key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a curated Marked directory into prep2-ready input."
    )
    parser.add_argument(
        "marked_dir",
        help="Directory containing manually curated SAC and SACPZ files.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory. Defaults to <Marked parent>/prep2_ready.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory after cleaning prep2-ready files.",
    )
    return parser.parse_args()


def coerce_header_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value <= -12344.0:
        return None
    return value


def normalize_loc(loc_value: str | None) -> tuple[str, str]:
    loc = (loc_value or "").strip()
    if not loc or loc in {"--", ".."}:
        return "--", ""
    return loc, loc


def build_station_key(net: str, sta: str, loc_raw: str, cha: str) -> StationKey | None:
    net = net.strip()
    sta = sta.strip()
    cha = cha.strip()
    if not net or not sta or not cha:
        return None
    loc_logic, _ = normalize_loc(loc_raw.strip())
    return net, sta, loc_logic, cha


def choose_pick(sac_header: object) -> tuple[float | None, str | None]:
    """Return the manual pick that legacy prep2 will consume through SAC A."""
    for name in ("t3", "t4"):
        value = coerce_header_float(getattr(sac_header, name, None))
        if value is not None:
            return value, name.upper()
    return None, None


def read_waveform_record(sac_path: Path) -> WaveformRecord:
    trace = obspy.read(str(sac_path), headonly=True)[0]
    sac = getattr(trace.stats, "sac", None)
    if sac is None:
        raise RuntimeError("SAC header not found")

    net = str(getattr(trace.stats, "network", "")).strip()
    sta = str(getattr(trace.stats, "station", "")).strip()
    cha = str(getattr(trace.stats, "channel", "")).strip()
    if not net or not sta or not cha:
        raise RuntimeError("Missing net/station/channel in SAC header")

    loc_logic, loc_header = normalize_loc(getattr(trace.stats, "location", ""))
    pick_value, pick_source = choose_pick(sac)

    return WaveformRecord(
        sac_path=sac_path,
        net=net,
        sta=sta,
        loc_logic=loc_logic,
        cha=cha,
        loc_output=loc_header,
        pick_value=pick_value,
        pick_source=pick_source,
        starttime=trace.stats.starttime,
    )


def parse_pz_key(path: Path) -> StationKey | None:
    name = path.name

    if name.startswith("SACPZ."):
        parts = name.split(".")
        if len(parts) < 5:
            return None
        return build_station_key(parts[1], parts[2], parts[3], parts[4])

    if path.suffix in PZ_SUFFIXES:
        stem = path.name[: -len(path.suffix)]
        prefix = stem.split("_", 1)[0]
        parts = prefix.split(".")
        if len(parts) < 4:
            return None
        return build_station_key(parts[0], parts[1], parts[2], parts[3])

    return None


def discover_sac_files(marked_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in marked_dir.iterdir()
        if path.is_file() and path.suffix in SAC_SUFFIXES
    )


def discover_pz_files(marked_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in marked_dir.iterdir()
        if path.is_file()
        and (path.name.startswith("SACPZ.") or path.suffix in PZ_SUFFIXES)
    )


def cleanup_output_dir(out_dir: Path) -> None:
    patterns = (
        "*.SAC",
        "SACPZ.*",
        "*.info",
        "wave_file_v1.dat",
        "wave_file_az.dat",
        "station.list",
        "station.amp",
        "station.inv",
        "epicenter.dat",
        "epicenter_b.dat",
        "structure.dat",
        ".sac.tmp*",
        ".max_amp_wave",
        ".workfile",
        ".tmp.txt",
        ".ppick.dat",
        "prepare_marked_report.txt",
    )
    for pattern in patterns:
        for path in out_dir.glob(pattern):
            if path.is_file():
                path.unlink()
    wave_obs = out_dir / "wave.obs"
    if wave_obs.is_dir():
        shutil.rmtree(wave_obs)


def ensure_output_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists():
        if not out_dir.is_dir():
            raise RuntimeError(f"Output path is not a directory: {out_dir}")
        if not overwrite:
            raise RuntimeError(
                f"Output directory already exists: {out_dir}. Use --overwrite to replace prep2-ready files."
            )
        cleanup_output_dir(out_dir)
        return
    out_dir.mkdir(parents=True, exist_ok=False)


def canonical_sac_name(record: WaveformRecord) -> str:
    loc = record.loc_output
    return (
        f"{record.net}.{record.sta}.{loc}.{record.cha}.M."
        f"{record.starttime.year:04d}.{record.starttime.julday:03d}."
        f"{record.starttime.hour:02d}{record.starttime.minute:02d}{record.starttime.second:02d}.SAC"
    )


def canonical_pz_name(record: WaveformRecord) -> str:
    loc = record.loc_output
    return f"SACPZ.{record.net}.{record.sta}.{loc}.{record.cha}"


def info_name(record: WaveformRecord) -> str:
    return f"{record.sta}.{record.loc_output}.{record.cha}.info"


def write_output_sac(record: WaveformRecord, out_path: Path) -> None:
    trace = obspy.read(str(record.sac_path))[0]
    sac = getattr(trace.stats, "sac", None)
    if sac is None:
        raise RuntimeError(f"SAC header not found during write: {record.sac_path.name}")

    # Legacy prep_sac2wave_v2 reads the P pick from SAC A, not from T3/T4.
    sac.a = float(record.pick_value)

    # Keep a conservative sample count for prep2. Tail trimming does not
    # resample, filter, or otherwise process the retained data.
    if trace.stats.npts > MAX_PREP2_NPTS:
        endtime = trace.stats.starttime + (MAX_PREP2_NPTS - 1) * trace.stats.delta
        trace.trim(trace.stats.starttime, endtime, nearest_sample=True, pad=False)
    trace.write(str(out_path), format="SAC")


def generate_info_file(pz_path: Path, info_path: Path) -> None:
    executable = shutil.which("prep_sacpz2info")
    if executable is None:
        raise RuntimeError("Missing executable in PATH: prep_sacpz2info")
    result = subprocess.run(
        [executable, pz_path.name, info_path.name],
        cwd=str(info_path.parent),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"prep_sacpz2info failed for {pz_path.name}: {result.stderr.strip() or result.stdout.strip()}"
        )


def build_response_index(pz_files: list[Path]) -> tuple[dict[StationKey, list[Path]], list[str]]:
    index: dict[StationKey, list[Path]] = {}
    report_lines: list[str] = []
    for path in pz_files:
        key = parse_pz_key(path)
        if key is None:
            report_lines.append(f"SKIP_PZ_PARSE|{path.name}|Unable to parse response filename")
            continue
        index.setdefault(key, []).append(path)
    return index, report_lines


def collect_records(marked_dir: Path) -> tuple[
    list[WaveformRecord],
    dict[StationKey, list[Path]],
    list[str],
    int,
]:
    report_lines: list[str] = []
    sac_files = discover_sac_files(marked_dir)
    if not sac_files:
        raise RuntimeError(f"No SAC files found in {marked_dir}")

    records: list[WaveformRecord] = []
    duplicates: dict[StationKey, list[Path]] = {}
    seen: dict[StationKey, Path] = {}
    for sac_path in sac_files:
        try:
            record = read_waveform_record(sac_path)
        except Exception as exc:
            report_lines.append(f"SKIP_SAC|{sac_path.name}|{exc}")
            continue
        if record.station_key in seen:
            duplicates.setdefault(record.station_key, [seen[record.station_key]]).append(sac_path)
            continue
        seen[record.station_key] = sac_path
        records.append(record)

    if duplicates:
        lines = []
        for key, paths in sorted(duplicates.items()):
            joined = ", ".join(sorted(path.name for path in paths))
            lines.append(f"{'.'.join(key)} -> {joined}")
        raise RuntimeError(
            "Duplicate waveforms found for the same NET.STA.LOC.CHA; please resolve them before conversion:\n"
            + "\n".join(lines)
        )

    pz_index, pz_report = build_response_index(discover_pz_files(marked_dir))
    report_lines.extend(pz_report)
    return records, pz_index, report_lines, len(sac_files)


def format_report(
    marked_dir: Path,
    out_dir: Path,
    total_sac: int,
    written: int,
    skipped: int,
    details: list[str],
) -> str:
    lines = [
        "# prepare_marked_for_prep report",
        f"input_marked_dir: {marked_dir}",
        f"output_dir: {out_dir}",
        f"total_sac_files: {total_sac}",
        f"written_records: {written}",
        f"skipped_records: {skipped}",
        "",
        "status|station|detail",
    ]
    lines.extend(details)
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    marked_dir = Path(args.marked_dir).resolve()
    if not marked_dir.exists() or not marked_dir.is_dir():
        raise RuntimeError(f"Marked directory not found: {marked_dir}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else marked_dir.parent / "prep2_ready"
    ensure_output_dir(out_dir, overwrite=args.overwrite)

    records, pz_index, report_lines, total_sac = collect_records(marked_dir)

    written_entries: list[tuple[str, str]] = []
    written = 0
    skipped = 0
    for record in sorted(records, key=lambda item: item.station_key):
        station_id = record.canonical_id
        if record.pick_value is None or record.pick_source is None:
            skipped += 1
            report_lines.append(f"SKIP|{station_id}|Missing valid pick in T3/T4")
            continue

        matched_pz = pz_index.get(record.station_key, [])
        if len(matched_pz) != 1:
            skipped += 1
            if not matched_pz:
                reason = "Missing unique response file"
            else:
                reason = "Ambiguous response files: " + ", ".join(path.name for path in matched_pz)
            report_lines.append(f"SKIP|{station_id}|{reason}")
            continue

        sac_name = canonical_sac_name(record)
        pz_name = canonical_pz_name(record)
        info_file = info_name(record)

        out_sac = out_dir / sac_name
        out_pz = out_dir / pz_name
        out_info = out_dir / info_file

        write_output_sac(record, out_sac)
        shutil.copy2(matched_pz[0], out_pz)
        generate_info_file(out_pz, out_info)

        written_entries.append((out_sac.name, out_info.name))
        written += 1
        report_lines.append(
            f"OK|{station_id}|pick={record.pick_source}:{record.pick_value:.6f}; "
            f"sac={out_sac.name}; pz={out_pz.name}; info={out_info.name}"
        )

    written_entries.sort(key=lambda item: item[1])
    wave_file = out_dir / "wave_file_v1.dat"
    with wave_file.open("w", encoding="utf-8") as fout:
        # prep2.bash reads only the first two columns: SAC filename and info filename.
        for sac_name, info_file in written_entries:
            fout.write(f"{sac_name} {info_file}\n")

    report_text = format_report(
        marked_dir=marked_dir,
        out_dir=out_dir,
        total_sac=total_sac,
        written=written,
        skipped=skipped,
        details=report_lines,
    )
    report_path = out_dir / "prepare_marked_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print("--- prepare_marked_for_prep ---")
    print(f"Input Marked dir : {marked_dir}")
    print(f"Output dir       : {out_dir}")
    print(f"SAC discovered   : {total_sac}")
    print(f"Records written  : {written}")
    print(f"Records skipped  : {skipped}")
    print(f"wave_file_v1.dat : {wave_file}")
    print(f"Report           : {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
