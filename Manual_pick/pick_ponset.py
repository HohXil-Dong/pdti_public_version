#!/usr/bin/env python3
"""Pure manual P-onset picker based on ObsPy and Matplotlib.

This tool is intentionally conservative:
- only ordinary stations are editable,
- reference stations are defined by existing ``t3`` picks,
- the final manual result is written to SAC header ``t4``,
- the displayed ``cc`` is only a zero-lag diagnostic tied to the current picks.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import obspy

from ponset_utils import (
    build_cc_context,
    calc_circular_az_diff_deg,
    calc_weighted_cc_at_pick,
    list_station_codes,
    load_station_record,
    set_optional_sac_header,
    slice_relative_window,
    sort_station_names_by_az,
)


@dataclass(frozen=True)
class PickerConfig:
    """Central runtime configuration for the final manual picker."""

    window_left: float = -2.0
    window_right: float = 25.0
    cc_window_left: float = -1.0
    cc_window_right: float = 6.0
    p_decay: float = 1.0
    taper_sec: float = 0.0
    font_scale: float = 1.60
    nudge_sec: float = 0.02
    big_nudge_sec: float = 0.10
    save_max_az_diff_deg: float = 40.0


PICKER_CONFIG = PickerConfig()

BASE_FONT_SIZES = {
    "list": 8.0,
    "status": 8.0,
    "lane_label": 8.0,
    "panel_title": 10.0,
    "axis_label": 10.0,
    "tick": 7.0,
    "legend": 7.0,
    "suptitle": 12.0,
}

ACTIVE_PALETTE = [
    "#1f77b4",
    "#2ca02c",
    "#9467bd",
    "#17becf",
    "#8c564b",
    "#bcbd22",
    "#e377c2",
    "#4c78a8",
    "#54a24b",
    "#5f6db0",
    "#5aa39a",
    "#7a8b3a",
]
REFERENCE_COLOR = "#c62828"
SECONDARY_REFERENCE_COLOR = "#000000"
PRIMARY_TRACE_WIDTH = 3.0
SECONDARY_TRACE_WIDTH = PRIMARY_TRACE_WIDTH * 0.3
LIST_BG = "#fbfbfb"
LIST_HL = "#e8f1ff"
LIST_HL_SELECTED = "#d7ead9"
LIST_HL_FOCUSED = "#fff1bf"
LIST_HL_FOCUSED_SELECTED = "#d7ead9"
LIST_TEXT = "#1f1f1f"
LIST_TEXT_MUTED = "#666666"
LIST_TEXT_ACTIVE = "#0f3d67"
LIST_EDGE_FOCUSED = "#9a6a00"
FOCUS_PANELS = ("reference", "candidate", "saved")
EPS = 1e-6


def _float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_float(value, fmt="{:.3f}", none_text="NA"):
    value = _float_or_none(value)
    if value is None or not np.isfinite(value):
        return none_text
    return fmt.format(value)


def build_font_sizes(scale):
    """Scale the base font table once so the UI stays internally consistent."""
    scale = float(scale)
    return {name: value * scale for name, value in BASE_FONT_SIZES.items()}


def get_manual_initial_pick(headers):
    """Use existing editable-station output first, then fall back to theoretical t1."""
    if headers["t4"] is not None:
        return headers["t4"]
    return headers["t1"]


def resolve_reference_pool(records):
    """Return all stations that can serve as reference stations."""
    candidates = [sta for sta, rec in records.items() if rec["headers"]["t3"] is not None]
    if not candidates:
        raise RuntimeError("No station with a valid t3 pick was found; cannot define a reference station.")
    return sort_station_names_by_az(records, candidates, reverse=False)


def validate_args(parser, args):
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        parser.error(f"data_dir does not exist: {data_dir}")
    if not data_dir.is_dir():
        parser.error(f"data_dir is not a directory: {data_dir}")


class ManualPOnsetPicker:
    """Manual picker with keyboard-driven list control and direct SAC writes.

    The runtime state is intentionally grouped into four layers so later
    maintenance does not need to reverse-engineer which variables drive which
    part of the UI:

    - data layer: station records and the three pool lists
    - interaction layer: visible references, active editable stations, and the
      currently selected editable station
    - focus layer: which side panel owns the keyboard focus plus per-panel row
      indices and pages
    - view layer: the currently displayed time window and the x-window history
    """

    def __init__(self, args):
        self.data_dir = Path(args.data_dir)
        self.config = PICKER_CONFIG
        self.records = {}

        self.base_window_left = float(self.config.window_left)
        self.base_window_right = float(self.config.window_right)
        self.view_window_left = self.base_window_left
        self.view_window_right = self.base_window_right
        self.cc_window_left = self.config.cc_window_left
        self.cc_window_right = self.config.cc_window_right
        self.p_decay = self.config.p_decay
        self.taper_sec = self.config.taper_sec
        self.font_scale = self.config.font_scale
        self.font_sizes = build_font_sizes(self.font_scale)
        self.nudge_sec = self.config.nudge_sec
        self.big_nudge_sec = self.config.big_nudge_sec
        self.save_max_az_diff_deg = float(self.config.save_max_az_diff_deg)

        self.status_message = ""
        self.color_map = {}

        self.pending_x_pick = None
        self.view_window_history = []

        # Focus layer: keyboard navigation is panel-based and independent from
        # which editable station is currently being nudged.
        self.reference_page = 0
        self.candidate_page = 0
        self.saved_page = 0
        self.focused_panel = "reference"
        self.focused_reference_index = 0
        self.focused_candidate_index = 0
        self.focused_saved_index = 0

        self.fig: Any = None
        self.ax_reference: Any = None
        self.ax_candidates: Any = None
        self.ax_offset: Any = None
        self.ax_shared: Any = None
        self.ax_saved: Any = None
        self.ax_status: Any = None

        self._load_records()
        if not self.records:
            raise RuntimeError(f"No readable SAC+SACPZ station records found in {self.data_dir}")

        # Reference stations are fixed manual anchors supplied before this GUI:
        # their t3 picks are displayed and used for CC, but never edited here.
        self.reference_pool = resolve_reference_pool(self.records)
        self.reference_station = None
        self.visible_reference_stations = []
        self.candidate_pool = []
        self.saved_pool = []
        self._rebuild_editable_pools()

        # Interaction layer: editable stations come from either the unsaved or
        # the already-saved pool, but the waveform area treats them uniformly.
        self.active_stations = []
        self.selected_station = None
        self.focused_reference_index = 0 if self.reference_pool else -1
        self.focused_candidate_index = 0 if self.candidate_pool else -1
        self.focused_saved_index = 0 if self.saved_pool else -1
        self._sync_focus_page("reference")
        self._sync_focus_page("candidate")
        self._sync_focus_page("saved")

    def _load_records(self):
        missing_t1_stations = []
        for station in list_station_codes(self.data_dir):
            rec = load_station_record(self.data_dir, station, require_pz=True)
            # Editable stations resume from saved t4 when available; otherwise
            # the theoretical t1 remains the initial manual-picking seed.
            rec["pick_initial"] = get_manual_initial_pick(rec["headers"])
            rec["pick_current"] = rec["pick_initial"]
            rec["last_cc"] = None
            if rec["headers"]["t1"] is None:
                missing_t1_stations.append(station)
            self.records[station] = rec

        if missing_t1_stations:
            names = ", ".join(sort_station_names_by_az(self.records, missing_t1_stations, reverse=False))
            raise RuntimeError(f"Missing t1 header for station(s): {names}")

    def _rebuild_editable_pools(self):
        """Recompute editable ordinary-station pools from SAC t3/t4 state."""
        ordered = sort_station_names_by_az(self.records, reverse=False)
        self.candidate_pool = [
            sta
            for sta in ordered
            if sta not in self.reference_pool
            and self.records[sta]["headers"]["t3"] is None
            and self.records[sta]["headers"]["t4"] is None
        ]
        self.saved_pool = [
            sta
            for sta in ordered
            if sta not in self.reference_pool
            and self.records[sta]["headers"]["t3"] is None
            and self.records[sta]["headers"]["t4"] is not None
        ]

    def _panel_rows(self, panel):
        if panel == "reference":
            return self.reference_pool
        if panel == "candidate":
            return self.candidate_pool
        if panel == "saved":
            return self.saved_pool
        return []

    def _focused_index_attr(self, panel):
        if panel == "reference":
            return "focused_reference_index"
        if panel == "candidate":
            return "focused_candidate_index"
        return "focused_saved_index"

    def _get_focused_index(self, panel):
        return getattr(self, self._focused_index_attr(panel))

    def _set_focused_index(self, panel, index):
        setattr(self, self._focused_index_attr(panel), index)

    def _panel_page_attr(self, panel):
        if panel == "reference":
            return "reference_page"
        if panel == "candidate":
            return "candidate_page"
        return "saved_page"

    def _page_index_for_item(self, total, visible, item_index):
        if item_index < 0:
            return 0
        page_starts = self._page_starts(total, visible)
        if not page_starts:
            return 0
        for idx in range(len(page_starts) - 1, -1, -1):
            if page_starts[idx] <= item_index:
                return idx
        return 0

    def _sync_focus_page(self, panel):
        rows = self._panel_rows(panel)
        index = self._get_focused_index(panel)
        if not rows:
            self._set_focused_index(panel, -1)
            setattr(self, self._panel_page_attr(panel), 0)
            return
        index = max(0, min(index, len(rows) - 1))
        self._set_focused_index(panel, index)
        _top, _bottom, _row_height, visible = self._list_geometry(panel)
        page_index = self._page_index_for_item(len(rows), visible, index)
        setattr(self, self._panel_page_attr(panel), page_index)

    def _set_focused_panel(self, panel):
        if panel not in FOCUS_PANELS:
            return
        self.focused_panel = panel
        self._sync_focus_page(panel)

    def _move_focus(self, step):
        panel = self.focused_panel
        rows = self._panel_rows(panel)
        if not rows:
            self._set_status(f"No rows in {panel} panel.")
            return
        index = self._get_focused_index(panel)
        if index < 0:
            index = 0
        else:
            index = (index + step) % len(rows)
        self._set_focused_index(panel, index)
        self._sync_focus_page(panel)
        self._set_status(f"Focused {panel} -> {rows[index]}")

    def _focused_station(self, panel):
        rows = self._panel_rows(panel)
        index = self._get_focused_index(panel)
        if not rows or index < 0 or index >= len(rows):
            return None
        return rows[index]

    def _activate_focused_row(self):
        panel = self.focused_panel
        station = self._focused_station(panel)
        if station is None:
            self._set_status(f"No focused station in {panel} panel.")
            return
        if panel == "reference":
            self._toggle_visible_reference_station(station)
        else:
            self._toggle_active_station(station)

    def _apply_reference_cc_focus(self):
        station = self._focused_station("reference")
        if station is None:
            self._set_status("No focused reference station.")
            return
        self._select_reference_station(station)

    def _set_status(self, message):
        """Cache and echo the most recent user-facing status message."""
        self.status_message = message
        print(message)

    def _focus_canvas(self):
        if self.fig is None:
            return
        canvas = getattr(self.fig, "canvas", None)
        if canvas is None:
            return

        try:
            widget_getter = getattr(canvas, "get_tk_widget", None)
            if callable(widget_getter):
                widget = widget_getter()
                focus_set = getattr(widget, "focus_set", None)
                if callable(focus_set):
                    focus_set()
                    return
        except Exception:
            pass

        for attr_name in ("setFocus", "SetFocus", "grab_focus", "focus_set"):
            try:
                focus_fn = getattr(canvas, attr_name, None)
                if callable(focus_fn):
                    focus_fn()
                    return
            except Exception:
                pass

        manager = getattr(canvas, "manager", None)
        window = getattr(manager, "window", None) if manager is not None else None
        if window is None:
            return
        for attr_name in ("focus_force", "activateWindow", "raise_", "SetFocus", "grab_focus", "focus_set"):
            try:
                focus_fn = getattr(window, attr_name, None)
                if callable(focus_fn):
                    focus_fn()
            except Exception:
                pass

    def _assign_color(self, station):
        if station in self.color_map:
            return self.color_map[station]
        color = ACTIVE_PALETTE[len(self.color_map) % len(ACTIVE_PALETTE)]
        self.color_map[station] = color
        return color

    def _displayed_stations(self):
        """Return the plotted stations ordered so azimuth increases upward."""
        stations = list(dict.fromkeys(list(self.visible_reference_stations) + list(self.active_stations)))
        return sort_station_names_by_az(self.records, stations, reverse=True)

    def _is_dirty(self, station):
        rec = self.records[station]
        pick = rec["pick_current"]
        if pick is None:
            return False
        saved = rec["headers"]["t4"]
        if saved is not None:
            return abs(float(pick) - float(saved)) > EPS
        initial = rec["pick_initial"]
        if initial is None:
            return False
        return abs(float(pick) - float(initial)) > EPS

    def _select_reference_station(self, station):
        if station not in self.reference_pool:
            return
        if station not in self.visible_reference_stations:
            self._set_status(f"{station} is not visible. Check VIS first, then switch CC.")
            return
        if station == self.reference_station:
            self._set_status(f"{station} is already the current CC reference.")
            return
        self.reference_station = station
        self._set_status(f"CC reference station -> {station}")

    def _toggle_visible_reference_station(self, station):
        """Show or hide one reference waveform while enforcing CC-ref visibility."""
        if station not in self.reference_pool:
            return
        if station == self.reference_station and station in self.visible_reference_stations:
            self._set_status(f"{station} is the current CC reference and must remain visible.")
            return
        if station in self.visible_reference_stations:
            self.visible_reference_stations.remove(station)
            self._set_status(f"Removed reference {station} from waveform area.")
        else:
            self.visible_reference_stations.append(station)
            self.visible_reference_stations = sort_station_names_by_az(
                self.records, self.visible_reference_stations, reverse=False
            )
            if self.reference_station is None:
                self.reference_station = station
                self._set_status(f"Added reference {station} and set it as CC reference.")
            else:
                self._set_status(f"Added reference {station} to waveform area.")

    def _toggle_active_station(self, station):
        """Add or remove one editable station from the interaction zone."""
        if station not in self.candidate_pool and station not in self.saved_pool:
            return
        rec = self.records[station]
        if station in self.active_stations:
            self.active_stations.remove(station)
            self._normalize_selected_station(preferred=None if self.selected_station == station else self.selected_station)
            self._set_status(f"Removed {station} from interaction zone.")
        else:
            if rec["pick_current"] is None:
                self._set_status(f"{station}: missing both t4 and t1, cannot add to interaction zone.")
                return
            self.active_stations.append(station)
            self.active_stations = sort_station_names_by_az(self.records, self.active_stations, reverse=True)
            self._assign_color(station)
            self._normalize_selected_station(preferred=self.selected_station or station)
            self._set_status(f"Added {station} to interaction zone.")

    def _normalize_selected_station(self, preferred=None):
        """Keep ``selected_station`` valid after pool or active-set changes."""
        if not self.active_stations:
            self.selected_station = None
            return
        if preferred in self.active_stations:
            self.selected_station = preferred
            return
        if self.selected_station in self.active_stations:
            return
        self.selected_station = self.active_stations[0]

    def _cycle_selected(self, step):
        """Cycle the current editable station within the active candidate list."""
        if not self.active_stations:
            self.selected_station = None
            self._set_status("No active station selected.")
            return
        if self.selected_station not in self.active_stations:
            self.selected_station = self.active_stations[0]
        else:
            idx = self.active_stations.index(self.selected_station)
            self.selected_station = self.active_stations[(idx + step) % len(self.active_stations)]
        self._set_status(f"Selected editing station -> {self.selected_station}")

    def _window_dt_source(self):
        """Pick a stable sampling interval for derived relative time vectors."""
        if self.reference_station is not None:
            return self.records[self.reference_station]["dt"]
        if self.selected_station is not None:
            return self.records[self.selected_station]["dt"]
        if self.candidate_pool:
            return self.records[self.candidate_pool[0]]["dt"]
        if self.saved_pool:
            return self.records[self.saved_pool[0]]["dt"]
        if self.reference_pool:
            return self.records[self.reference_pool[0]]["dt"]
        return 0.01

    def _current_reference_pick(self):
        """Return the manual anchor pick of the current CC reference station."""
        if self.reference_station is None:
            return None
        return self.records[self.reference_station]["headers"]["t3"]

    def _relative_context(self):
        """Build the diagnostic CC window, taper, and time weights.

        The picker reuses the shared windowing, left-edge taper, and post-P
        weighting helpers from `ponset_utils.py`, but only evaluates the
        zero-lag score at the current manual picks.
        """
        return build_cc_context(
            self._window_dt_source(),
            self.cc_window_left,
            self.cc_window_right,
            self.p_decay,
            self.taper_sec,
        )

    def _refresh_cc_metrics(self):
        """Refresh zero-lag weighted CC values for the current picker state."""
        for station in self.reference_pool:
            self.records[station]["last_cc"] = 1.0 if station == self.reference_station else None

        ref = None if self.reference_station is None else self.records[self.reference_station]
        if ref is None:
            for station in self.active_stations:
                self.records[station]["last_cc"] = None
            return

        rel_t, window_taper, time_weights = self._relative_context()

        for station in self.active_stations:
            rec = self.records[station]
            if rec["pick_current"] is None:
                rec["last_cc"] = None
                continue
            cc = calc_weighted_cc_at_pick(
                rec["vel"],
                rec["dt"],
                rec["b"],
                rec["pick_current"],
                ref["vel"],
                ref["dt"],
                ref["b"],
                ref["headers"]["t3"],
                rel_t,
                window_taper,
                time_weights,
            )
            rec["last_cc"] = cc

    def _undo_view_window(self):
        if self.pending_x_pick is not None:
            self.pending_x_pick = None
            self._set_status("Cancelled pending x-window pick.")
            return
        if not self.view_window_history:
            self._set_status("No previous view window to restore.")
            return
        left, right = self.view_window_history.pop()
        self.view_window_left = left
        self.view_window_right = right
        self._set_status(f"Restored previous view window [{left:.3f}, {right:.3f}] s")

    def _record_or_apply_zoom(self, event):
        if event.inaxes not in (self.ax_offset, self.ax_shared):
            self._set_status("Move the mouse over Offset View or Shared-Y View before pressing x.")
            return
        if event.xdata is None or not np.isfinite(event.xdata):
            self._set_status("Mouse time position is invalid; cannot define x-window.")
            return

        x_value = float(event.xdata)
        if self.pending_x_pick is None:
            self.pending_x_pick = x_value
            self._set_status(f"Recorded first x-boundary at {x_value:.3f}s. Move mouse and press x again.")
            return

        xmin = min(self.pending_x_pick, x_value)
        xmax = max(self.pending_x_pick, x_value)
        ref_dt = self._window_dt_source()
        min_span = max(2.0 * ref_dt, 1e-3)
        if xmax - xmin < min_span:
            self.pending_x_pick = None
            self._set_status(
                f"x-window too narrow ({xmax - xmin:.4f}s); need at least {min_span:.4f}s."
            )
            return

        xmin = max(self.base_window_left, xmin)
        xmax = min(self.base_window_right, xmax)
        if xmax - xmin < min_span:
            self.pending_x_pick = None
            self._set_status("x-window collapsed after clipping to the base window.")
            return

        old_window = (self.view_window_left, self.view_window_right)
        new_window = (xmin, xmax)
        if abs(old_window[0] - new_window[0]) <= EPS and abs(old_window[1] - new_window[1]) <= EPS:
            self.pending_x_pick = None
            self._set_status("x-window unchanged.")
            return
        self.view_window_history.append(old_window)
        self.view_window_left = xmin
        self.view_window_right = xmax
        self.pending_x_pick = None
        self._set_status(f"Zoomed view window to [{xmin:.3f}, {xmax:.3f}] s")

    def _nudge_selected(self, delta):
        station = self.selected_station
        if station is None:
            self._set_status("No active station selected.")
            return
        rec = self.records[station]
        if rec["pick_current"] is None:
            self._set_status(f"{station}: missing current pick, cannot edit.")
            return
        rec["pick_current"] += float(delta)
        self._set_status(f"{station}: nudged by {delta:+.3f}s -> current={rec['pick_current']:.3f}s")

    def _reset_selected(self):
        station = self.selected_station
        if station is None:
            self._set_status("No active station selected.")
            return
        rec = self.records[station]
        reset_pick = rec["headers"]["t4"] if rec["headers"]["t4"] is not None else rec["headers"]["t1"]
        if reset_pick is None:
            self._set_status(f"{station}: missing both t4 and t1, cannot reset.")
            return
        rec["pick_current"] = reset_pick
        source = "saved t4" if rec["headers"]["t4"] is not None else "t1"
        self._set_status(f"{station}: reset to {source} {rec['pick_current']:.3f}s")

    def _save_selected_to_t4(self):
        """Persist the current ordinary-station pick to SAC ``t4``.

        ``kuser0`` stores the reference station used during this save, so later
        QC can reconstruct which t3 anchor constrained the manual decision.
        """
        station = self.selected_station
        if station is None:
            self._set_status("No active station selected; nothing saved.")
            return False
        if self.reference_station is None:
            self._set_status("No CC reference selected; choose a visible reference station first.")
            return False

        rec = self.records[station]
        if rec["pick_current"] is None:
            self._set_status(f"{station}: missing current pick, cannot save.")
            return False
        if not self._can_save_selected_to_t4():
            return False

        tr = obspy.read(str(rec["path"]))[0]
        set_optional_sac_header(tr.stats.sac, "t4", rec["pick_current"])
        tr.stats.sac.kuser0 = self.reference_station
        tr.write(str(rec["path"]), format="SAC")
        rec["headers"]["t4"] = rec["pick_current"]
        self._refresh_station_pools_after_save(station)
        self._set_status(f"Saved {station} -> t4={rec['pick_current']:.3f}s using CC reference {self.reference_station}")
        return True

    def _can_save_selected_to_t4(self):
        """Require near-reference azimuth before saving the editable t4.

        The CC check assumes a point-source-like first motion over a short
        post-P window. The azimuth gate avoids saving picks against references
        whose radiation pattern may differ too much for that assumption.
        """
        station = self.selected_station
        if station is None or self.reference_station is None:
            return False

        ref = self.records[self.reference_station]
        rec = self.records[station]
        ref_az = ref["az"]
        az = rec["az"]
        if ref_az is None or az is None or not np.isfinite(ref_az) or not np.isfinite(az):
            message = (
                f"Warning: cannot save {station} -> t4 because az is missing for "
                f"{self.reference_station} or {station}."
            )
            self._set_status(message)
            return False

        az_diff = calc_circular_az_diff_deg(az, ref_az)
        if az_diff <= self.save_max_az_diff_deg:
            return True

        message = (
            f"Warning: cannot save {station} -> t4 because az diff to reference "
            f"{self.reference_station} is {az_diff:.2f} deg, exceeds "
            f"{self.save_max_az_diff_deg:.2f} deg."
        )
        self._set_status(message)
        return False

    def _refresh_station_pools_after_save(self, station):
        """Refresh side-panel membership after writing ``t4`` to disk."""
        if station not in self.records or station in self.reference_pool:
            return

        old_candidate = list(self.candidate_pool)
        old_saved = list(self.saved_pool)
        self._rebuild_editable_pools()
        if old_candidate == self.candidate_pool and old_saved == self.saved_pool:
            return

        self._normalize_selected_station(preferred=station)
        self._sync_focus_page("candidate")
        self._sync_focus_page("saved")

    def _make_traces(self):
        """Assemble normalized display traces for the current waveform area."""
        traces = []
        for station in self._displayed_stations():
            rec = self.records[station]
            is_ref = station in self.visible_reference_stations
            pick = rec["headers"]["t3"] if is_ref else rec["pick_current"]
            if pick is None:
                continue
            t_rel, y_win = slice_relative_window(
                rec["vel"],
                rec["dt"],
                rec["b"],
                pick,
                self.view_window_left,
                self.view_window_right,
            )
            if t_rel is None or y_win is None or len(y_win) == 0:
                continue
            amp = np.max(np.abs(y_win))
            if amp < 1e-12:
                continue
            traces.append(
                {
                    "station": station,
                    "is_ref": is_ref,
                    "is_cc_ref": station == self.reference_station,
                    "t_rel": t_rel,
                    "y_norm": y_win / amp,
                    "cc": self.records[station]["last_cc"],
                }
            )
        return traces

    def _list_geometry(self, axis_kind):
        top = 0.90
        bottom = 0.04
        row_height = min(0.075, 0.024 + 0.010 * self.font_scale)
        if axis_kind == "saved":
            row_height *= 0.95
        visible = max(1, int((top - bottom) / row_height))
        return top, bottom, row_height, visible

    def _page_starts(self, total, visible):
        if total <= 0:
            return [0]
        page_size = max(1, visible)
        return list(range(0, total, page_size))

    def _format_dist(self, rec):
        return _format_float(rec["dist_deg"], "{:.2f}", "NA")

    def _format_az(self, rec):
        return _format_float(rec["az"], "{:.1f}", "NA")

    def _editable_row_text(self, station, rec, include_saved_t4):
        text = (
            f"{'[x]' if station in self.active_stations else '[ ]'} {station:8s} "
            f"az={self._format_az(rec):>5s} dist={self._format_dist(rec):>6s} "
        )
        if include_saved_t4:
            text += f"t4={_format_float(rec['headers']['t4']):>7s} "
        return text + f"pick={_format_float(rec['pick_current']):>7s}"

    def _editable_row_entry(self, station, panel, include_saved_t4):
        rec = self.records[station]
        active = station in self.active_stations
        selected = station == self.selected_station
        return {
            "station": station,
            "text": self._editable_row_text(station, rec, include_saved_t4),
            "active": active,
            "selected": selected,
            "bold": selected,
            "focused": self.focused_panel == panel and self._focused_station(panel) == station,
            "color": self._assign_color(station) if active else (LIST_TEXT_ACTIVE if include_saved_t4 else LIST_TEXT),
        }

    def _draw_list_rows(self, ax, title, rows, page_attr, panel_kind, help_text=None):
        ax.clear()
        ax.axis("off")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)

        top, _bottom, row_height, visible = self._list_geometry(panel_kind)
        page_starts = self._page_starts(len(rows), visible)
        page_index = max(0, min(getattr(self, page_attr), len(page_starts) - 1))
        setattr(self, page_attr, page_index)
        start = page_starts[page_index]
        shown = rows[start : start + visible]

        total_pages = len(page_starts)
        current_page = page_index + 1
        title_text = f"{title} ({len(rows)})  page {current_page}/{total_pages}"
        ax.text(
            0.0,
            0.99,
            title_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=self.font_sizes["panel_title"],
            fontweight="bold",
        )
        ax.text(
            0.0,
            0.94,
            help_text or "scroll: mouse wheel | pageup/pagedown",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=self.font_sizes["legend"],
            color=LIST_TEXT_MUTED,
        )

        for idx, item in enumerate(shown):
            station = item["station"]
            y_top = top - idx * row_height
            y_bottom = y_top - row_height

            color = item.get("color", LIST_TEXT)
            bbox = None
            is_active = item.get("active")
            is_selected = item.get("selected")
            is_focused = item.get("focused")
            if is_focused and is_selected:
                bbox = {
                    "facecolor": LIST_HL_FOCUSED_SELECTED,
                    "edgecolor": LIST_EDGE_FOCUSED,
                    "linewidth": 1.1,
                    "pad": 0.18,
                    "alpha": 0.98,
                }
            elif is_focused:
                bbox = {
                    "facecolor": LIST_HL_FOCUSED,
                    "edgecolor": LIST_EDGE_FOCUSED,
                    "linewidth": 1.1,
                    "pad": 0.18,
                    "alpha": 0.98,
                }
            elif is_selected:
                bbox = {"facecolor": LIST_HL_SELECTED, "edgecolor": "none", "pad": 0.15, "alpha": 0.98}
            elif is_active:
                bbox = {"facecolor": LIST_HL, "edgecolor": "none", "pad": 0.15, "alpha": 0.95}
            ax.text(
                0.0,
                y_top,
                item["text"],
                transform=ax.transAxes,
                va="top",
                ha="left",
                family="monospace",
                fontsize=self.font_sizes["list"],
                fontweight="bold" if item.get("bold") else "normal",
                color=color,
                bbox=bbox,
            )
        if not rows:
            ax.text(
                0.0,
                top,
                "No stations available.",
                transform=ax.transAxes,
                va="top",
                ha="left",
                family="monospace",
                fontsize=self.font_sizes["list"],
                color=LIST_TEXT_MUTED,
            )

    def _draw_reference_list(self):
        rows = []
        for station in self.reference_pool:
            rec = self.records[station]
            visible = station in self.visible_reference_stations
            is_cc_ref = station == self.reference_station
            focused = (
                self.focused_panel == "reference"
                and self._focused_station("reference") == station
            )
            rows.append(
                {
                    "station": station,
                    "text": (
                        f"{'[x]' if is_cc_ref else '[ ]'} {'[x]' if visible else '[ ]'} {station:8s} az={self._format_az(rec):>5s} "
                        f"dist={self._format_dist(rec):>6s} t3={_format_float(rec['headers']['t3']):>7s}"
                    ),
                    "active": visible,
                    "bold": is_cc_ref,
                    "color": REFERENCE_COLOR if is_cc_ref else LIST_TEXT,
                    "selected": is_cc_ref,
                    "focused": focused,
                }
            )
        self._draw_list_rows(
            self.ax_reference,
            "Reference Stations (t3)",
            rows,
            "reference_page",
            "reference",
            help_text="1/2/3 focus | up/down move | enter=CC | space=VIS",
        )

    def _draw_candidate_list(self):
        rows = [self._editable_row_entry(station, "candidate", include_saved_t4=False) for station in self.candidate_pool]
        self._draw_list_rows(
            self.ax_candidates,
            "Candidate Stations",
            rows,
            "candidate_page",
            "candidate",
            help_text="1/2/3 focus | up/down move | space toggle | [ ] edit",
        )

    def _draw_saved_t4_list(self):
        rows = [self._editable_row_entry(station, "saved", include_saved_t4=True) for station in self.saved_pool]
        self._draw_list_rows(
            self.ax_saved,
            "Saved t4 (Editable Stations)",
            rows,
            "saved_page",
            "saved",
            help_text="1/2/3 focus | up/down move | space toggle | [ ] edit",
        )

    def _trace_style(self, item):
        """Use one consistent style rule for both waveform panels."""
        station = item["station"]
        if item["is_ref"] and item["is_cc_ref"]:
            return REFERENCE_COLOR, PRIMARY_TRACE_WIDTH
        if item["is_ref"]:
            return SECONDARY_REFERENCE_COLOR, SECONDARY_TRACE_WIDTH
        if station == self.selected_station:
            return self._assign_color(station), PRIMARY_TRACE_WIDTH
        return self._assign_color(station), SECONDARY_TRACE_WIDTH

    def _draw_offset_panel(self, traces):
        self.ax_offset.clear()
        lane_spacing = 1.45
        amp_scale = 1.65
        y_ticks = []
        y_labels = []

        for i, item in enumerate(traces):
            y0 = (len(traces) - 1 - i) * lane_spacing
            station = item["station"]
            color, lw = self._trace_style(item)
            self.ax_offset.plot(item["t_rel"], item["y_norm"] * amp_scale + y0, color=color, lw=lw, alpha=0.98)

            flags = []
            if item["is_ref"]:
                flags.append("REF")
            if item["is_cc_ref"]:
                flags.append("CCREF")
            if station == self.selected_station:
                flags.append("SEL")
            if item["cc"] is not None and not item["is_ref"]:
                flags.append(f"CC={item['cc']:.3f}")
            label = station if not flags else f"{station} [{' '.join(flags)}]"
            bbox = {"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.18}
            self.ax_offset.text(
                self.view_window_left + 0.02 * max(1.0, self.view_window_right - self.view_window_left),
                y0 + 0.38 * lane_spacing,
                label,
                fontsize=self.font_sizes["lane_label"],
                fontweight="bold" if (item["is_ref"] or station == self.selected_station) else "normal",
                color=color,
                ha="left",
                va="top",
                bbox=bbox,
            )

            y_ticks.append(y0)
            y_labels.append(station)

        self.ax_offset.axvline(0.0, color="0.45", lw=0.9, ls="--")
        self.ax_offset.set_title(
            f"Offset View | CC Ref {self.reference_station or 'none'} | [{self.view_window_left:.2f}, {self.view_window_right:.2f}] s",
            fontsize=self.font_sizes["panel_title"],
        )
        self.ax_offset.set_xlim(self.view_window_left, self.view_window_right)
        self.ax_offset.set_ylim(-lane_spacing, max(lane_spacing, len(traces) * lane_spacing))
        self.ax_offset.set_yticks(y_ticks)
        self.ax_offset.set_yticklabels(y_labels, fontsize=self.font_sizes["tick"])
        self.ax_offset.set_xlabel("Time Relative To Pick (s)", fontsize=self.font_sizes["axis_label"])
        self.ax_offset.tick_params(axis="x", labelsize=self.font_sizes["tick"])
        self.ax_offset.grid(True, axis="x", which="major", ls=":", alpha=0.5)
        self.ax_offset.grid(True, axis="y", ls=":", alpha=0.2)
        if self.pending_x_pick is not None:
            self.ax_offset.axvline(self.pending_x_pick, color="0.25", lw=1.3, ls=":")

    def _draw_shared_panel(self, traces):
        self.ax_shared.clear()
        plotted_count = 0
        for item in traces:
            station = item["station"]
            color, lw = self._trace_style(item)
            self.ax_shared.plot(item["t_rel"], item["y_norm"], color=color, lw=lw, alpha=0.95, label=station)
            plotted_count += 1

        self.ax_shared.axvline(0.0, color="0.45", lw=0.9, ls="--")
        self.ax_shared.set_title("Shared-Y View", fontsize=self.font_sizes["panel_title"])
        self.ax_shared.set_xlim(self.view_window_left, self.view_window_right)
        self.ax_shared.set_ylim(-1.5, 1.5)
        self.ax_shared.set_xlabel("Time Relative To Pick (s)", fontsize=self.font_sizes["axis_label"])
        self.ax_shared.set_ylabel("Normalized Amplitude", fontsize=self.font_sizes["axis_label"])
        self.ax_shared.tick_params(axis="both", labelsize=self.font_sizes["tick"])
        self.ax_shared.grid(True, axis="x", which="major", ls=":", alpha=0.5)
        self.ax_shared.grid(True, axis="y", ls=":", alpha=0.2)
        if plotted_count > 0:
            self.ax_shared.legend(
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                fontsize=self.font_sizes["legend"],
                frameon=True,
            )
        if self.pending_x_pick is not None:
            self.ax_shared.axvline(self.pending_x_pick, color="0.25", lw=1.3, ls=":")

    def _draw_status_panel(self):
        self.ax_status.clear()
        self.ax_status.axis("off")
        lines = [
            f"CC Ref: {self.reference_station or 'none'}",
            f"Visible Refs: {len(self.visible_reference_stations)}",
            f"Active Editable: {len(self.active_stations)}",
            f"Focused Panel: {self.focused_panel}",
            f"Base Window: [{self.base_window_left:.2f}, {self.base_window_right:.2f}] s",
            f"View Window: [{self.view_window_left:.2f}, {self.view_window_right:.2f}] s",
            f"Zoom Depth: {len(self.view_window_history)}",
            f"x Pending: {'none' if self.pending_x_pick is None else f'{self.pending_x_pick:.3f}s'}",
            "",
            "Controls:",
            "  1 / 2 / 3: focus reference / candidate / saved-t4",
            "  up/down: move focused station",
            "  space: toggle visible or editable station active",
            "  enter: switch CC within focused ref station",
            "  [: previous active station",
            "  ]: next active station",
            "  mouse wheel on list: scroll",
            "  left/right: small nudge",
            "  ,/. : large nudge",
            "  pageup/pagedown: page focused list",
            "  r: reset selected station",
            "  s: save selected station to SAC t4",
            "  x then x: zoom view window",
            "  o: restore previous view window",
            "",
            f"Diag CC Window: [{self.cc_window_left:.2f}, {self.cc_window_right:.2f}] s",
            f"Save Az Limit: {self.save_max_az_diff_deg:.2f} deg",
            "",
        ]

        lines.extend(self._selected_status_lines())
        focused_station = self._focused_station(self.focused_panel)
        lines.extend(["", f"Focused Row: {focused_station or 'none'}"])

        if self.status_message:
            lines.extend(["", f"Status: {self.status_message}"])

        self.ax_status.text(
            0.0,
            1.0,
            "\n".join(lines),
            va="top",
            ha="left",
            family="monospace",
            fontsize=self.font_sizes["status"],
        )

    def _selected_status_lines(self):
        """Build the right-panel summary for the current editable station.

        Naming follows the picker workflow:
        - ``t1`` is the theoretical pick from the SAC header,
        - ``t4`` is the value already saved to disk,
        - ``Initial`` is what the picker adopted when it started, i.e. ``t4``
          if present, otherwise ``t1``,
        - ``Current`` is the in-memory value now shown in the waveform view.
        """
        if self.selected_station is None:
            return ["Selected: none"]

        rec = self.records[self.selected_station]
        ref_pick = self._current_reference_pick()
        delta_ref = None if ref_pick is None or rec["pick_current"] is None else rec["pick_current"] - ref_pick
        delta_init = (
            None
            if rec["pick_initial"] is None or rec["pick_current"] is None
            else rec["pick_current"] - rec["pick_initial"]
        )
        return [
            f"Selected: {self.selected_station}",
            f"t1: {_format_float(rec['headers']['t1'])}",
            f"t4: {_format_float(rec['headers']['t4'])}",
            f"Initial: {_format_float(rec['pick_initial'])}",
            f"Current: {_format_float(rec['pick_current'])}",
            f"d_ref: {_format_float(delta_ref, '{:+.3f}')}",
            f"d_init: {_format_float(delta_init, '{:+.3f}')}",
            f"cc: {_format_float(rec['last_cc'], '{:.3f}')}",
            f"Modified: {'yes' if self._is_dirty(self.selected_station) else 'no'}",
        ]

    def draw(self):
        if (
            self.fig is None
            or self.ax_reference is None
            or self.ax_candidates is None
            or self.ax_offset is None
            or self.ax_shared is None
            or self.ax_saved is None
            or self.ax_status is None
        ):
            return
        self._refresh_cc_metrics()
        traces = self._make_traces()
        self._draw_reference_list()
        self._draw_candidate_list()
        self._draw_saved_t4_list()
        self._draw_offset_panel(traces)
        self._draw_shared_panel(traces)
        self._draw_status_panel()
        self.fig.suptitle(
            "Pure Manual P-Onset Picker: reference/candidate/saved-t4 panels, waveforms, and status",
            fontsize=self.font_sizes["suptitle"],
        )
        self.fig.canvas.draw_idle()
        self._focus_canvas()

    def _scroll_panel(self, panel, direction):
        rows = self._panel_rows(panel)
        _top, _bottom, _row_height, visible = self._list_geometry(panel)
        total_pages = len(self._page_starts(len(rows), visible))
        page_attr = self._panel_page_attr(panel)
        current = getattr(self, page_attr)
        updated = max(0, min(total_pages - 1, current + direction))
        if updated != current:
            setattr(self, page_attr, updated)
            if rows:
                self._set_focused_index(panel, self._page_starts(len(rows), visible)[updated])
            return True
        return False

    def _scroll_focused_panel(self, direction):
        panel = self.focused_panel
        return self._scroll_panel(panel, direction)

    def on_scroll(self, event):
        moved = False
        if event.inaxes == self.ax_reference:
            direction = -1 if event.button == "up" else 1
            moved = self._scroll_panel("reference", direction)
        elif event.inaxes == self.ax_candidates:
            direction = -1 if event.button == "up" else 1
            moved = self._scroll_panel("candidate", direction)
        elif event.inaxes == self.ax_saved:
            direction = -1 if event.button == "up" else 1
            moved = self._scroll_panel("saved", direction)
        if moved:
            self.draw()

    def on_key(self, event):
        key = event.key
        if key == "1":
            self._set_focused_panel("reference")
            self._set_status("Focused panel -> reference")
        elif key == "2":
            self._set_focused_panel("candidate")
            self._set_status("Focused panel -> candidate")
        elif key == "3":
            self._set_focused_panel("saved")
            self._set_status("Focused panel -> saved")
        elif key == "up":
            self._move_focus(-1)
        elif key == "down":
            self._move_focus(1)
        elif key == "left":
            self._nudge_selected(-self.nudge_sec)
        elif key == "right":
            self._nudge_selected(self.nudge_sec)
        elif key == ",":
            self._nudge_selected(-self.big_nudge_sec)
        elif key == ".":
            self._nudge_selected(self.big_nudge_sec)
        elif key == "[":
            self._cycle_selected(-1)
        elif key == "]":
            self._cycle_selected(1)
        elif key in {" ", "space"}:
            self._activate_focused_row()
        elif key == "enter":
            if self.focused_panel == "reference":
                self._apply_reference_cc_focus()
            else:
                self._set_status("Enter only switches CC in the reference panel.")
        elif key == "r":
            self._reset_selected()
        elif key == "s":
            self._save_selected_to_t4()
        elif key == "x":
            self._record_or_apply_zoom(event)
        elif key == "o":
            self._undo_view_window()
        elif key == "pageup":
            if not self._scroll_focused_panel(-1):
                return
        elif key == "pagedown":
            if not self._scroll_focused_panel(1):
                return
        else:
            return
        self.draw()

    def on_resize(self, _event):
        self.draw()

    def run(self):
        # Import pyplot lazily so ``--help`` and other non-GUI entry paths do
        # not initialize Matplotlib backends unnecessarily.
        import matplotlib.pyplot as plt

        plt.rcParams["keymap.save"] = []

        self.fig = plt.figure(figsize=(18.2, 11.8))
        gs = self.fig.add_gridspec(
            2,
            3,
            width_ratios=[1.15, 1.95, 1.05],
            height_ratios=[1.0, 1.0],
        )
        left = gs[:, 0].subgridspec(2, 1, height_ratios=[0.58, 1.0])
        center = gs[:, 1].subgridspec(2, 1, height_ratios=[1.0, 0.98])
        right = gs[:, 2].subgridspec(2, 1, height_ratios=[0.65, 1.0])

        self.ax_reference = self.fig.add_subplot(left[0, 0], facecolor=LIST_BG)
        self.ax_candidates = self.fig.add_subplot(left[1, 0], facecolor=LIST_BG)
        self.ax_offset = self.fig.add_subplot(center[0, 0])
        self.ax_shared = self.fig.add_subplot(center[1, 0])
        self.ax_saved = self.fig.add_subplot(right[0, 0], facecolor=LIST_BG)
        self.ax_status = self.fig.add_subplot(right[1, 0])
        self.fig.subplots_adjust(left=0.04, right=0.955, bottom=0.055, top=0.93, wspace=0.28, hspace=0.22)

        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("resize_event", self.on_resize)
        self.draw()
        plt.show()


def build_parser():
    parser = argparse.ArgumentParser(description="Pure manual P-onset picker.")
    parser.add_argument("data_dir", help="Directory containing SAC and SACPZ files.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    app = ManualPOnsetPicker(args)
    app.run()


if __name__ == "__main__":
    main()
