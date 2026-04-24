#!/usr/bin/env python3
"""Shared utilities for the manual P-onset picker.

The current production picker workflow is intentionally pure-manual: the user
decides the final onset, and the displayed CC is only a diagnostic tied to the
current picks on screen.

The retained CC path is the zero-lag weighted diagnostic used by the GUI. It
does not search lags or update picks automatically.
"""

from pathlib import Path

import numpy as np
import obspy
from obspy.io.sac.sacpz import attach_paz

from common_utils import (
    Config,
    build_prefilt,
    get_sac_file_mode,
    preprocess_trace,
    resolve_pz_for_sac_path,
)

KM_PER_DEG = 111.19
SAC_UNDEFINED = -12345.0


def list_sac_files(data_dir):
    """Return sorted SAC files under one directory."""
    data_dir = Path(data_dir)
    return sorted(list(data_dir.glob("*.SAC")) + list(data_dir.glob("*.sac")))


def list_station_codes(data_dir):
    """List station codes inferred from all SAC filenames in one directory."""
    stations = set()
    for path in list_sac_files(data_dir):
        try:
            _net, sta, _loc, _chan = _parse_sac_filename_fields(path.name)
        except ValueError:
            continue
        stations.add(sta)
    return sorted(stations)


def resolve_station_files(data_dir, station, require_pz=True):
    """Resolve one exact station+BHZ SAC file and the matching SACPZ file."""
    data_dir = Path(data_dir)
    sac_files = [
        path for path in list_sac_files(data_dir)
        if _match_station_channel_filename(path.name, station, "BHZ")
    ]
    if not sac_files:
        raise FileNotFoundError(f"Missing BHZ SAC for station {station} in {data_dir}")
    if len(sac_files) != 1:
        names = ", ".join(path.name for path in sac_files)
        raise RuntimeError(f"Ambiguous BHZ SAC for station {station}: {names}")

    sac_path = sac_files[0]
    net, _sta, loc, chan = _parse_sac_filename_fields(sac_path.name)
    pz_path = resolve_pz_for_sac_path(
        sac_path,
        net=net,
        sta=station,
        loc=loc,
        chan=chan,
        search_dirs=(data_dir,),
    )
    if pz_path is not None:
        pz_path = Path(pz_path)
    if require_pz and pz_path is None:
        raise FileNotFoundError(f"Missing matching BHZ SACPZ for station {station} in {data_dir}")

    return sac_path, pz_path


def _match_station_channel_filename(filename, station, channel):
    try:
        _net, sta, _loc, chan = _parse_sac_filename_fields(filename)
    except ValueError:
        return False
    return sta == str(station) and chan == str(channel)


def _parse_sac_filename_fields(filename):
    """Parse strict legacy/new SAC naming schemes into common header fields."""
    filename = str(filename)
    mode = get_sac_file_mode(filename)
    parts = filename.split(".")

    if mode == "legacy" and len(parts) >= 5 and parts[-1] == "SAC":
        return parts[0], parts[1], parts[2], parts[3]

    if mode == "modern" and len(parts) >= 5 and parts[-1] == "sac":
        chan = parts[3].split("_", 1)[0]
        if chan:
            return parts[0], parts[1], parts[2], chan

    raise ValueError(f"Unsupported SAC filename: {filename}")


def _coerce_header_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value <= -12344.0:
        return None
    return value


def read_optional_sac_header(sac, name):
    """Read one SAC header and map undefined sentinel values to None."""
    return _coerce_header_float(getattr(sac, name, None))


def set_optional_sac_header(sac, name, value):
    """Write one SAC header using the SAC undefined sentinel when value is None."""
    setattr(sac, name, SAC_UNDEFINED if value is None else float(value))


def read_sac_headers_from_path(sac_path):
    """Read one SAC file and return the trace plus common timing headers."""
    tr = obspy.read(str(sac_path))[0]
    sac = getattr(tr.stats, "sac", None)
    if sac is None:
        raise RuntimeError(f"SAC header not found in {sac_path}")

    headers = {
        "t1": read_optional_sac_header(sac, "t1"),
        "t3": read_optional_sac_header(sac, "t3"),
        "t4": read_optional_sac_header(sac, "t4"),
        "az": read_optional_sac_header(sac, "az"),
        "gcarc": read_optional_sac_header(sac, "gcarc"),
        "dist": read_optional_sac_header(sac, "dist"),
    }
    return {
        "trace": tr,
        "station": tr.stats.station,
        "b": float(getattr(sac, "b", 0.0)),
        "headers": headers,
        "dist_deg": get_station_distance_deg(headers),
    }


def read_processed_velocity_record(sac_path, pz_path):
    """Read one SAC trace and convert it to a processed velocity record.

    Processing deliberately happens on the whole trace before any local CC
    window is extracted:

    1. ``preprocess_trace`` removes simple record-wide trends and applies the
       standard taper used elsewhere in the project.
    2. ``build_prefilt`` prepares the response-removal pre-filter from the
       actual trace metadata and the matching ``SACPZ`` file.
    3. ``attach_paz(..., tovel=True)`` and ``simulate`` convert the waveform
       to ground velocity.

    Later CC functions still perform an additional *window-level*
    standardization step; that is not redundant with this whole-trace
    preprocessing because the local pick window can still have its own mean
    offset and variance.
    """
    info = read_sac_headers_from_path(sac_path)
    tr_work = preprocess_trace(info["trace"].copy())
    final_filt, _ = build_prefilt(tr_work, str(pz_path), mode=Config.PRE_FILT_MODE)
    attach_paz(tr_work, str(pz_path), tovel=True)
    tr_work.simulate(paz_remove=tr_work.stats.paz, pre_filt=final_filt)

    return {
        "station": info["station"],
        "vel": np.asarray(tr_work.data, dtype=float),
        "dt": float(tr_work.stats.delta),
        "b": info["b"],
        "headers": info["headers"],
        "dist_deg": info["dist_deg"],
        "trace": info["trace"],
    }


def load_station_record(data_dir, station, require_pz=True):
    """Load one station record, including processed velocity when possible."""
    sac_path, pz_path = resolve_station_files(data_dir, station, require_pz=require_pz)
    if require_pz:
        info = read_processed_velocity_record(sac_path, pz_path)
    else:
        info = read_sac_headers_from_path(sac_path)
        info["dt"] = float(info["trace"].stats.delta)
        info["vel"] = np.asarray(info["trace"].data, dtype=float)

    info["path"] = sac_path
    info["pz"] = pz_path
    info["az"] = info["headers"]["az"]
    return info


def calc_circular_az_diff_deg(angle, ref_angle):
    """Return the smallest absolute difference between two azimuths in degrees."""
    diff = (float(angle) - float(ref_angle) + 180.0) % 360.0 - 180.0
    return abs(diff)


def get_station_distance_deg(headers):
    """Return epicentral distance in degrees using gcarc first, then dist in km."""
    gcarc = headers["gcarc"]
    if gcarc is not None and np.isfinite(gcarc):
        return float(gcarc)

    dist_km = headers["dist"]
    if dist_km is not None and np.isfinite(dist_km):
        return float(dist_km) / KM_PER_DEG

    return None


def sort_station_names_by_az(records, stations=None, reverse=True):
    """Sort station names by azimuth for consistent plotting."""
    names = list(records) if stations is None else list(stations)
    return sorted(
        names,
        key=lambda sta: float("-inf") if records[sta]["az"] is None else records[sta]["az"],
        reverse=reverse,
    )


def slice_relative_window(data, dt, t0, pick, w_left, w_right):
    """Return one window around a pick without interpolation."""
    t_abs = t0 + np.arange(len(data)) * dt
    mask = (t_abs >= pick + w_left) & (t_abs <= pick + w_right)
    if not np.any(mask):
        return None, None
    return t_abs[mask] - pick, np.asarray(data[mask], dtype=float)


def build_left_taper(rel_t, taper_sec):
    """Return a cosine taper that only softens the left edge of the window.

    Only the left edge is tapered because the pre-P side is the most sensitive
    to interpolation and window-boundary artefacts. The right side is left
    untouched so the P-onset waveform shape remains directly comparable across
    displayed stations.
    """
    taper = np.ones_like(rel_t, dtype=float)
    if taper_sec <= 0 or len(rel_t) < 2:
        return taper

    total_span = float(rel_t[-1] - rel_t[0])
    if total_span <= 0:
        return taper

    taper_sec = min(float(taper_sec), total_span)
    if taper_sec <= 0:
        return taper

    left_end = rel_t[0] + taper_sec
    left = rel_t < left_end
    if np.any(left):
        phase = np.clip((rel_t[left] - rel_t[0]) / taper_sec, 0.0, 1.0)
        taper[left] *= 0.5 * (1.0 - np.cos(np.pi * phase))

    return taper


def build_time_weights(rel_t, p_decay):
    """Weight the CC score toward the onset by down-weighting post-P energy.

    The first-arriving pulse is usually the most reliable onset cue. Late
    post-P energy can differ between stations because of path and site effects,
    so the exponential decay suppresses its influence without fully discarding
    it.
    """
    weights = np.ones_like(rel_t, dtype=float)
    if p_decay > 0:
        post = rel_t > 0.0
        weights[post] = np.exp(-rel_t[post] / p_decay)
    return weights


def build_cc_context(dt, window_left, window_right, p_decay, taper_sec):
    """Build a correlation window together with its taper and time weights.

    The returned tuple is the common context used by the manual picker:
    - ``rel_t`` defines the relative time samples around the pick,
    - ``window_taper`` softly suppresses only the left edge when requested,
    - ``time_weights`` down-weights late post-P energy.
    """
    rel_t = np.arange(window_left, window_right + dt, dt)
    return rel_t, build_left_taper(rel_t, taper_sec), build_time_weights(rel_t, p_decay)


def extract_normalized_window(data, dt, t0, pick, rel_t, window_taper):
    """Interpolate and standardize one local waveform window for CC.

    This is a *window-level* standardization step and should not be confused
    with whole-trace preprocessing. The window is:
    - re-sampled onto ``rel_t`` around the current pick,
    - demeaned within that local window,
    - multiplied by the optional left-edge taper,
    - normalized by its own standard deviation.

    Returning ``None`` for tiny variance is an intentional quality-control
    guard: a near-flat window does not carry reliable alignment information
    even if the full trace itself was successfully preprocessed.
    """
    t_abs = t0 + np.arange(len(data)) * dt
    win = np.interp(pick + rel_t, t_abs, data, left=0.0, right=0.0)
    win = np.asarray(win, dtype=float)
    win -= np.mean(win)
    win *= window_taper
    sd = np.std(win)
    if sd < 1e-12:
        return None
    return win / sd


def calc_weighted_cc_at_pick(
    target,
    dt,
    t0,
    pick,
    ref,
    ref_dt,
    ref_t0,
    ref_pick,
    rel_t,
    window_taper,
    time_weights,
):
    """Return the weighted zero-lag CC for the current picks.

    This is the manual picker's live diagnostic metric. It uses the same
    window construction, taper, and post-P weighting used by the picker, but it
    does *not* search over lags. The value therefore answers:

    "How similar are the two waveforms at the picks currently shown on screen?"

    Because the numerator and denominator use the same time weights, the score
    remains amplitude-normalized across stations while still emphasizing the
    onset part of the window.
    """
    if len(rel_t) < 2:
        return 0.0

    win_target = extract_normalized_window(target, dt, t0, pick, rel_t, window_taper)
    win_ref = extract_normalized_window(ref, ref_dt, ref_t0, ref_pick, rel_t, window_taper)
    if win_target is None or win_ref is None:
        return 0.0

    w = np.asarray(time_weights, dtype=float)
    den = np.sqrt(np.sum(w * win_target * win_target) * np.sum(w * win_ref * win_ref))
    if den <= 1e-12:
        return 0.0
    return float(np.sum(w * win_target * win_ref) / den)
