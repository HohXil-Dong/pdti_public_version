# `pick_ponset.py` User Guide

`pick_ponset.py` is the manual P-onset picker in the `script/pick` workflow. It uses stations with existing `t3` picks as reference anchors, lets the operator adjust ordinary-station P picks manually, and saves the final ordinary-station pick to SAC `t4`.

Core rules:

- Reference stations are defined by existing `t3`; this tool does not edit `t3`.
- Only ordinary stations are editable, and their saved result is written to `t4`.
- On save, the current reference station name is written to `kuser0` for later QC.
- The displayed `cc` is a zero-lag cross-correlation diagnostic for the current picks; it never moves picks automatically.
- Cross-correlation is intended for short post-P windows between stations with similar azimuths. It assumes point-source-like first-motion similarity and ignores finite-rupture-scale complexity.

## Launch

```bash
python pick_ponset.py /path/to/Marked
```

The input directory must contain each station's `BHZ` SAC file and matching SACPZ file. At startup, the picker reads the traces and removes instrument response to velocity for plotting and `cc` diagnostics.

Required SAC information:

- Every station must have a valid `t1`, used as the theoretical P pick or ordinary-station seed.
- At least one station must have a valid `t3`, used as a reference station.
- `az` is used for sorting and the save-time azimuth gate.
- `gcarc` is preferred for displayed epicentral distance; if missing, `dist` in km is converted to degrees.

## SAC Header Convention

- `t1`: theoretical P arrival, usually written by an earlier processing step.
- `t3`: manual P anchor for reference stations. Stations with `t3` enter the reference list and are not editable here.
- `t4`: final manual P pick for ordinary stations, written when pressing `s`.
- `kuser0`: reference station name used when the `t4` pick was saved.

Ordinary stations initialize from existing `t4` when available, otherwise from `t1`. Therefore `Initial` is not always equal to `t1`.

## Interface

The window has three columns:

- Upper-left `Reference Stations (t3)`: all stations with `t3`. `CC` marks the current correlation reference; `VIS` controls whether the waveform is visible.
- Lower-left `Candidate Stations`: ordinary stations without `t3` or `t4`.
- Middle `Offset View` / `Shared-Y View`: visible reference stations and active ordinary stations. Time is relative to each station's current pick, so 0 s is the pick.
- Upper-right `Saved t4 (Editable Stations)`: ordinary stations without `t3` but with existing `t4`; these can be reloaded for checking or editing.
- Lower-right status panel: reference state, view window, controls, and selected-station `t1/t4/Initial/Current/d_ref/d_init/cc/Modified`.

## Controls

| Key / Mouse | Action |
| --- | --- |
| `1` / `2` / `3` | Focus reference / candidate / saved list |
| `Up` / `Down` | Move focus within the current list |
| `PageUp` / `PageDown` | Page the current list |
| `Space` | Main row action: toggle reference `VIS`, or add/remove an ordinary station from the active area |
| `Enter` | In the reference list, set the focused reference as `CC Ref` |
| `[` / `]` | Cycle the selected editable station among active ordinary stations |
| `Left` / `Right` | Small pick nudge for the selected editable station |
| `,` / `.` | Large pick nudge for the selected editable station |
| `r` | Reset the selected editable station to saved `t4`, or to `t1` if no `t4` exists |
| `s` | Save the selected editable station to SAC `t4` and write `kuser0` |
| `x` then `x` | With the mouse over a waveform panel, define a new display time window |
| `o` | Restore the previous display time window |
| Mouse wheel | Page the list under the mouse |

## `cc` Diagnostic

`cc` compares the current ordinary-station `pick_current` against the current reference station's `t3` in a short P-relative window. The calculation:

1. preprocesses the full trace and removes instrument response to velocity;
2. interpolates a fixed relative window onto a common time axis;
3. demeans, optionally applies a left-edge taper, and standardizes each local window;
4. down-weights post-P energy with an exponential decay;
5. computes a weighted zero-lag correlation coefficient.

`cc` does not search over lag, does not update `pick_current`, and does not change when only the display window is zoomed. It answers only: "At the picks currently shown, how similar are these two P-window waveforms?"

## Saving And QC

Before saving with `s`, the picker requires the selected ordinary station and `CC Ref` to differ in azimuth by no more than `PickerConfig.save_max_az_diff_deg`. This avoids saving picks against references whose first-motion pattern may be too different for the short-window comparison.

Recommended workflow:

1. Choose a reliable `t3` reference, press `Space` to show it, then `Enter` to make it `CC Ref`.
2. Add several nearby-azimuth ordinary stations from the candidate or saved list.
3. Judge the pick using the overlay, `d_ref`, `d_init`, and `cc`.
4. Nudge the pick manually and press `s` when satisfied.
5. Revisit `Saved t4` stations after picking to check continuity relative to reference `t3` and whether `kuser0` is reasonable.

Later, `gen_inv_final.py` reads final P picks in `t3 -> t4 -> t1` priority order.

## After Picking: Direct `prep2`

If `Marked/` has already been manually curated and P picks are finalized, and the next step should use the legacy `prep2.bash` workflow, run:

```bash
python prepare_marked_for_prep.py /path/to/Marked --out-dir /path/to/prep2_ready
cd /path/to/prep2_ready
prep2.bash wave_file_v1.dat LAT LON DEP
```

`prepare_marked_for_prep.py` is a format-compatibility step, not a new scientific processing step:

- accepts both `*.SAC + SACPZ.*` and `*.sac + *.sacpz` naming styles;
- writes legacy-prep-compatible `*.SAC`, `SACPZ.*`, and `*.info` files;
- copies the manual P pick into SAC `A` using `t3 -> t4` priority, because `prep_sac2wave_v2` reads `A`;
- writes a two-column `wave_file_v1.dat` for `prep2.bash`;
- writes `prepare_marked_report.txt` with success records and skipped-record reasons; inputs longer than `50000` samples are conservatively tail-trimmed, but trim counts are not reported.

It does not resample, filter, remove response, or select additional stations. Records without `t3/t4`, without one unique matching SACPZ, or with duplicate `NET.STA.LOC.CHA` SAC files are not written to `wave_file_v1.dat`.
