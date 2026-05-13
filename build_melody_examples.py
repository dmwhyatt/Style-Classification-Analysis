"""
Build melody example assets for the factor network webapp.

Adapted from my old version of this at https://github.com/dmwhyatt/essen_new

Inputs (project root):
  - essen_china_europe_features.csv   (melody_id + numeric feature columns)
  - factor_scores_for_logreg.csv      (melody_id + F1..FN scores, from factor_logistic.R)
  - docs/network_data.json            (variable + factor nodes used in the webapp)

For each variable AND each factor in the network we pick the N_HIGH highest-scoring
and N_LOW lowest-scoring melodies and render each as a piano-roll PNG plus a
synthesized WAV. Renders are deduplicated by MIDI
basename, so each unique melody is rendered at most once and referenced from every
manifest entry that uses it.

Outputs (docs/melody_examples/):
  - <melody_stub>.png                  one per unique melody actually used
  - <melody_stub>.wav                  one per unique melody (when synthesis works)
  - manifest.json                      keyed by both variable id (e.g.
                                       "timing.average_note_duration") and factor id
                                       (e.g. "F1")

Run from project root with the venv active:
    python build_melody_examples.py
"""

import argparse
import json
import os
import re
import sys
import types
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

if "fluidsynth" not in sys.modules:
    _stub = types.ModuleType("fluidsynth")
    _stub.Synth = None
    sys.modules["fluidsynth"] = _stub
try:
    import pretty_midi
    HAS_PRETTY_MIDI = True
except (ImportError, OSError):
    HAS_PRETTY_MIDI = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except (ImportError, OSError):
    HAS_MATPLOTLIB = False


PROJECT_ROOT = Path(__file__).resolve().parent
DOCS = PROJECT_ROOT / "docs"
EXAMPLES_DIR = DOCS / "melody_examples"
CSV_PATH = PROJECT_ROOT / "essen_china_europe_features.csv"
FACTOR_SCORES_CSV = PROJECT_ROOT / "factor_scores_for_logreg.csv"
NETWORK_JSON = DOCS / "network_data.json"

N_HIGH = 3
N_LOW = 3
EXCERPT_NOTES = 24
AUDIO_SAMPLE_RATE = 44100
AUDIO_MAX_SECONDS = 10.0

VAR_ID_TO_CSV_COLUMN = {
    "interval.absolute_interval_range": "pitch_interval.absolute_interval_range",
    "interval.amount_of_arpeggiation": "pitch_interval.amount_of_arpeggiation",
    "interval.average_interval_span_by_melodic_arcs": "pitch_interval.average_interval_span_by_melodic_arcs",
    "interval.average_length_of_melodic_arcs": "pitch_interval.average_length_of_melodic_arcs",
    "interval.chromatic_motion": "pitch_interval.chromatic_motion",
    "interval.direction_of_melodic_motion": "pitch_interval.direction_of_melodic_motion",
    "interval.distance_between_most_prevalent_melodic_intervals": "pitch_interval.distance_between_most_prevalent_melodic_intervals",
    "interval.interval_direction_mean": "pitch_interval.interval_direction_mean",
    "interval.interval_direction_std": "pitch_interval.interval_direction_std",
    "interval.interval_entropy": "complexity.interval_entropy",
    "interval.mean_absolute_interval": "pitch_interval.mean_absolute_interval",
    "interval.melodic_large_intervals": "pitch_interval.melodic_large_intervals",
    "interval.melodic_octaves": "pitch_interval.melodic_octaves",
    "interval.melodic_perfect_fifths": "pitch_interval.melodic_perfect_fifths",
    "interval.melodic_sevenths": "pitch_interval.melodic_sevenths",
    "interval.melodic_sixths": "pitch_interval.melodic_sixths",
    "interval.melodic_thirds": "pitch_interval.melodic_thirds",
    "interval.minor_major_third_ratio": "pitch_interval.minor_major_third_ratio",
    "interval.modal_interval": "pitch_interval.modal_interval",
    "interval.number_of_common_melodic_intervals": "pitch_interval.number_of_common_melodic_intervals",
    "interval.prevalence_of_most_common_melodic_interval": "pitch_interval.prevalence_of_most_common_melodic_interval",
    "interval.relative_prevalence_of_most_common_melodic_intervals": "pitch_interval.relative_prevalence_of_most_common_melodic_intervals",
    "interval.standard_deviation_absolute_interval": "pitch_interval.standard_deviation_absolute_interval",
    "timing.duration_entropy": "complexity.duration_entropy",
    "contour.mean_melodic_accent": "expectation.mean_melodic_accent",
    "contour.melodic_accent_std": "expectation.melodic_accent_std",
    "absolute_pitch.pitch_entropy": "complexity.pitch_entropy",
    "idyom.mean_information_content_pitch_ltm": "idyom.pitch_ltm_mean_information_content",
}


def _csv_column_for_var(var_id: str, df_columns) -> Optional[str]:
    if var_id in df_columns:
        return var_id
    alt = VAR_ID_TO_CSV_COLUMN.get(var_id)
    return alt if alt and alt in df_columns else None


def _get_essen_corpus_lookup() -> Optional[Dict[str, Path]]:
    try:
        from melody_features.corpus import get_corpus_files
        paths = get_corpus_files("essen")
        return {Path(p).name: Path(p) for p in paths}
    except Exception:
        return None


def get_midi_path(
    melody_id: str,
    midi_dir: Optional[Path] = None,
    corpus_lookup: Optional[Dict[str, Path]] = None,
) -> Optional[Path]:
    if not melody_id or not isinstance(melody_id, str):
        return None
    melody_id = melody_id.strip()
    if not melody_id or melody_id.lower() == "nan":
        return None
    p = Path(melody_id)
    if p.is_absolute() and p.exists() and p.suffix.lower() in (".mid", ".midi"):
        return p
    basename = p.name
    if not basename.lower().endswith((".mid", ".midi")):
        return None
    if corpus_lookup and basename in corpus_lookup:
        return corpus_lookup[basename]
    for dir_candidate in (midi_dir, os.environ.get("MELODY_EXAMPLES_MIDI_DIR")):
        if not dir_candidate:
            continue
        d = Path(dir_candidate)
        if d.is_dir():
            candidate = d / basename
            if candidate.exists():
                return candidate
    try:
        import melody_features
        pkg = Path(melody_features.__file__).parent
        candidate = pkg / "corpora" / "essen_folksong_collection" / basename
        if candidate.exists():
            return candidate
    except Exception:
        pass
    for base in (PROJECT_ROOT, DOCS):
        c = base / basename
        if c.exists():
            return c
    return None


# Piano-key helpers (MIDI 60 = C4 in scientific pitch notation)
_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_WHITE_KEY_PCS = (0, 2, 4, 5, 7, 9, 11)   # C, D, E, F, G, A, B
_BLACK_KEY_PCS = (1, 3, 6, 8, 10)         # C#, D#, F#, G#, A#


def _pitch_to_name(midi: int) -> str:
    return f"{_NOTE_NAMES[midi % 12]}{(midi // 12) - 1}"


def _piano_roll_yticks(pitch_min: int, pitch_max: int):
    """Pick y-ticks for the pitch axis. Prefer one tick per white key; if the range
    is large enough that that would crowd the axis, fall back to C/F only."""
    white = [p for p in range(pitch_min, pitch_max + 1) if (p % 12) in _WHITE_KEY_PCS]
    if len(white) <= 14:
        return white, [_pitch_to_name(p) for p in white]
    sparse = [p for p in range(pitch_min, pitch_max + 1) if (p % 12) in (0, 5)]
    return sparse, [_pitch_to_name(p) for p in sparse]


def render_piano_roll(midi_path: Path, out_path: Path, max_notes: int = EXCERPT_NOTES) -> None:
    if not HAS_PRETTY_MIDI or not HAS_MATPLOTLIB:
        raise RuntimeError("pretty_midi and matplotlib are required for piano-roll PNGs.")
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    notes = []
    for inst in midi.instruments:
        if inst.notes:
            notes = sorted(inst.notes, key=lambda n: (n.start, n.pitch))[:max_notes]
            break
    if not notes:
        raise RuntimeError("No notes in excerpt")

    pitch_min = int(min(n.pitch for n in notes)) - 2
    pitch_max = int(max(n.pitch for n in notes)) + 2
    time_max = max(n.end for n in notes) + 0.1

    fig, ax = plt.subplots(figsize=(6, 2.5))

    # Subtle gray bands on black-key rows so the keyboard layout is visible at a glance
    for pitch in range(pitch_min, pitch_max + 1):
        if (pitch % 12) in _BLACK_KEY_PCS:
            ax.axhspan(pitch - 0.5, pitch + 0.5, color="#000000", alpha=0.07, zorder=0)
        if (pitch % 12) == 0:  # thin separator below each C
            ax.axhline(pitch - 0.5, color="#888", linewidth=0.5, alpha=0.35, zorder=1)

    for n in notes:
        ax.barh(n.pitch, n.end - n.start, left=n.start, height=0.8,
                color="#667eea", edgecolor="#4a5fc1", zorder=2)

    ticks, tick_labels = _piano_roll_yticks(pitch_min, pitch_max)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)
    ax.set_ylabel("Pitch")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(pitch_min, pitch_max)
    ax.set_xlim(0, time_max)

    # Melody name (MIDI basename without extension) as a left-aligned title
    ax.set_title(midi_path.stem, fontsize=10, loc="left", pad=4, color="#444")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def midi_to_audio(
    midi_path: Path,
    wav_path: Path,
    max_seconds: float = AUDIO_MAX_SECONDS,
    excerpt_notes: int = EXCERPT_NOTES,
) -> bool:
    if not HAS_PRETTY_MIDI or not HAS_SOUNDFILE:
        return False
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        if not midi.instruments:
            return False
        first_inst = None
        for inst in midi.instruments:
            if inst.notes:
                first_inst = inst
                break
        if first_inst is None:
            return False
        notes = sorted(first_inst.notes, key=lambda n: (n.start, n.pitch))[:excerpt_notes]
        if not notes:
            return False
        _times, tempi = midi.get_tempo_changes()
        initial_tempo = float(tempi[0]) if len(tempi) > 0 else 120.0
        excerpt_midi = pretty_midi.PrettyMIDI(initial_tempo=initial_tempo)
        new_inst = pretty_midi.Instrument(
            program=first_inst.program,
            is_drum=first_inst.is_drum,
            name=first_inst.name or "Melody",
        )
        for n in notes:
            new_inst.notes.append(
                pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.start, end=n.end)
            )
        excerpt_midi.instruments.append(new_inst)
        audio = excerpt_midi.synthesize(fs=AUDIO_SAMPLE_RATE)
        end_time = min(max(n.end for n in notes), max_seconds)
        max_samples = int(end_time * AUDIO_SAMPLE_RATE)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        sf.write(str(wav_path), audio, AUDIO_SAMPLE_RATE)
        return True
    except Exception:
        return False


def _safe_stub(name: str) -> str:
    """Filename-safe stub for a melody basename or arbitrary id."""
    stem = Path(name).stem if name.lower().endswith((".mid", ".midi")) else name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem)


class RenderCache:
    """Render each unique melody at most once; reuse PNG/WAV across manifest entries."""

    def __init__(
        self,
        examples_dir: Path,
        want_audio: bool,
        force_png: bool = False,
        force_wav: bool = False,
    ):
        self.examples_dir = examples_dir
        self.want_audio = want_audio
        self.force_png = force_png
        self.force_wav = force_wav
        # midi basename -> (relative_png, relative_wav_or_None)
        self._cache: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        self._png_fail: set[str] = set()

    def render(self, midi_path: Path) -> Tuple[Optional[str], Optional[str]]:
        key = midi_path.name
        if key in self._cache:
            return self._cache[key]
        if key in self._png_fail:
            return (None, None)
        stub = _safe_stub(key)
        png_rel: Optional[str] = None
        wav_rel: Optional[str] = None
        png_path = self.examples_dir / f"{stub}.png"
        wav_path = self.examples_dir / f"{stub}.wav"
        if self.force_png or not png_path.exists():
            try:
                render_piano_roll(midi_path, png_path)
            except Exception as e:
                print(f"  Skip PNG for {key}: {e}")
                self._png_fail.add(key)
                return (None, None)
        png_rel = f"melody_examples/{stub}.png"
        if self.want_audio:
            if self.force_wav or not wav_path.exists():
                if midi_to_audio(midi_path, wav_path):
                    wav_rel = f"melody_examples/{stub}.wav"
            else:
                wav_rel = f"melody_examples/{stub}.wav"
        self._cache[key] = (png_rel, wav_rel)
        return self._cache[key]


def _build_entries_from_sorted(
    sorted_df: pd.DataFrame,
    value_col: str,
    n: int,
    descending: bool,
    midi_dir: Optional[Path],
    corpus_lookup: Optional[Dict[str, Path]],
    cache: RenderCache,
) -> list:
    """Walk `sorted_df` (descending=True for "high", False for "low") and emit up to n
    manifest entries with successfully-rendered PNGs."""
    rows_iter = sorted_df.iterrows() if descending else sorted_df.iloc[::-1].iterrows()
    entries: list = []
    rank = 0
    for _, row in rows_iter:
        if rank >= n:
            break
        melody_id = row.get("melody_id")
        if pd.isna(melody_id):
            continue
        melody_id = str(melody_id).strip()
        midi_path = get_midi_path(melody_id, midi_dir, corpus_lookup)
        if midi_path is None:
            continue
        png_rel, wav_rel = cache.render(midi_path)
        if png_rel is None:
            continue
        rank += 1
        try:
            value = float(row[value_col])
        except (TypeError, ValueError):
            continue
        entry = {
            "rank": rank,
            "melody_id": melody_id,
            "value": round(value, 6),
            "png": png_rel,
        }
        if wav_rel:
            entry["audio"] = wav_rel
        entries.append(entry)
    return entries


def _process_one_column(
    label: str,
    df: pd.DataFrame,
    value_col: str,
    midi_dir: Optional[Path],
    corpus_lookup: Optional[Dict[str, Path]],
    cache: RenderCache,
) -> Optional[Tuple[dict, int, int]]:
    col = pd.to_numeric(df[value_col], errors="coerce")
    valid = col.notna()
    if valid.sum() < N_HIGH + N_LOW:
        return None
    sub = df.loc[valid, ["melody_id", value_col]].copy()
    sub = sub.astype({value_col: float})
    sorted_df = sub.sort_values(value_col, ascending=False)
    # Walk a generous head/tail buffer so unresolved melodies don't shrink the set
    buffer = max(3 * N_HIGH, 20)
    high_list = _build_entries_from_sorted(
        sorted_df.head(buffer), value_col, N_HIGH, True, midi_dir, corpus_lookup, cache,
    )
    low_list = _build_entries_from_sorted(
        sorted_df.tail(buffer), value_col, N_LOW, False, midi_dir, corpus_lookup, cache,
    )
    return ({"label": label, "high": high_list, "low": low_list}, len(high_list), len(low_list))


def main():
    parser = argparse.ArgumentParser(
        description="Build melody example PNGs/WAVs and manifest for the factor network webapp."
    )
    parser.add_argument(
        "--midi-dir", type=Path, default=None,
        help="Directory containing essen_folksong_collection MIDI files. Required when "
             "CSV melody_id paths don't exist on this machine.",
    )
    parser.add_argument(
        "--limit-features", type=int, default=None,
        help="Process at most this many feature/variable nodes (useful for smoke tests).",
    )
    parser.add_argument(
        "--skip-features", action="store_true",
        help="Skip per-feature examples; only process factors.",
    )
    parser.add_argument(
        "--skip-factors", action="store_true",
        help="Skip per-factor examples; only process features.",
    )
    parser.add_argument(
        "--wav-only", action="store_true",
        help="Regenerate only WAV files from existing manifest (no PNG re-render).",
    )
    parser.add_argument(
        "--force-png", action="store_true",
        help="Re-render every PNG even if it already exists on disk.",
    )
    parser.add_argument(
        "--force-wav", action="store_true",
        help="Re-synthesize every WAV even if it already exists on disk.",
    )
    args = parser.parse_args()

    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    if args.wav_only:
        manifest_path = EXAMPLES_DIR / "manifest.json"
        if not manifest_path.exists():
            print("No manifest.json found. Run the full build first.")
            sys.exit(1)
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        if not HAS_PRETTY_MIDI or not HAS_SOUNDFILE:
            print("WAV generation requires pretty_midi and soundfile. pip install soundfile")
            sys.exit(1)
        corpus_lookup = _get_essen_corpus_lookup()
        total, written = 0, 0
        for _key, data in manifest.items():
            for which in ("high", "low"):
                for entry in data.get(which) or []:
                    total += 1
                    melody_id = entry.get("melody_id")
                    if not melody_id:
                        continue
                    midi_path = get_midi_path(str(melody_id).strip(), args.midi_dir, corpus_lookup)
                    if midi_path is None:
                        continue
                    stub = _safe_stub(midi_path.name)
                    wav_path = EXAMPLES_DIR / f"{stub}.wav"
                    if wav_path.exists() or midi_to_audio(midi_path, wav_path):
                        entry["audio"] = f"melody_examples/{stub}.wav"
                        written += 1
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"WAV: {written}/{total} written; manifest updated -> {manifest_path}")
        sys.exit(0)

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Feature CSV not found: {CSV_PATH}")
    if not NETWORK_JSON.exists():
        raise FileNotFoundError(
            f"{NETWORK_JSON} not found. Run `Rscript factor_logistic.R` first."
        )
    if not HAS_PRETTY_MIDI or not HAS_MATPLOTLIB:
        print("ERROR: Piano-roll PNGs require pretty_midi and matplotlib (see requirements.txt).")
        sys.exit(1)
    if not HAS_SOUNDFILE:
        print("[warn] WAV output disabled (install soundfile: pip install soundfile)")
    if not HAS_PRETTY_MIDI:
        print("[warn] WAV output disabled (pretty_midi required for synthesis)")

    print("Loading inputs...")
    df = pd.read_csv(CSV_PATH)
    if "melody_id" not in df.columns:
        raise ValueError("CSV must have a melody_id column")
    print(f"  {CSV_PATH.name}: {len(df)} rows, {len(df.columns)} columns")

    factor_df = None
    if not args.skip_factors:
        if FACTOR_SCORES_CSV.exists():
            factor_df = pd.read_csv(FACTOR_SCORES_CSV)
            print(f"  {FACTOR_SCORES_CSV.name}: {len(factor_df)} rows")
        else:
            print(f"[warn] {FACTOR_SCORES_CSV} not found — factor examples skipped. "
                  "Re-run `Rscript factor_logistic.R` to generate factor scores.")

    with open(NETWORK_JSON, "r") as f:
        network = json.load(f)
    variable_nodes = [n for n in network["nodes"] if n.get("type") == "variable"]
    factor_nodes = [n for n in network["nodes"] if n.get("type") == "factor"]
    variable_labels = {n["id"]: n.get("name", n["id"]) for n in variable_nodes}
    factor_labels = {n["id"]: n.get("name", n["id"]) for n in factor_nodes}
    print(f"  network_data.json: {len(variable_nodes)} variables, {len(factor_nodes)} factors")

    corpus_lookup = _get_essen_corpus_lookup()
    if corpus_lookup:
        print("Using melody_features.get_corpus_files('essen') for MIDI resolution.")

    # Fail fast on MIDI resolution
    sample_id = str(df["melody_id"].dropna().iloc[0]).strip()
    if not get_midi_path(sample_id, args.midi_dir, corpus_lookup):
        print("ERROR: Cannot resolve MIDI path for a sample melody_id from the CSV.")
        print(f"  Sample: {sample_id[:120]}")
        print("Pass --midi-dir /path/to/essen_folksong_collection or "
              "set MELODY_EXAMPLES_MIDI_DIR.")
        sys.exit(1)

    want_audio = HAS_PRETTY_MIDI and HAS_SOUNDFILE
    cache = RenderCache(
        EXAMPLES_DIR,
        want_audio=want_audio,
        force_png=args.force_png,
        force_wav=args.force_wav,
    )

    manifest: dict = {}

    # ---- Features ----
    if not args.skip_features:
        var_ids = [v["id"] for v in variable_nodes]
        if args.limit_features:
            var_ids = var_ids[: args.limit_features]
        print(f"\nProcessing {len(var_ids)} feature(s)...")
        for i, var_id in enumerate(var_ids, 1):
            csv_col = _csv_column_for_var(var_id, df.columns)
            if csv_col is None:
                print(f"  [{i}/{len(var_ids)}] {var_id}: SKIP (no CSV column)")
                continue
            result = _process_one_column(
                label=variable_labels.get(var_id, var_id),
                df=df,
                value_col=csv_col,
                midi_dir=args.midi_dir,
                corpus_lookup=corpus_lookup,
                cache=cache,
            )
            if result is None:
                print(f"  [{i}/{len(var_ids)}] {var_id}: SKIP (insufficient data)")
                continue
            entry, n_hi, n_lo = result
            manifest[var_id] = entry
            print(f"  [{i}/{len(var_ids)}] {var_id}: {n_hi} high, {n_lo} low")

    # ---- Factors ----
    if not args.skip_factors and factor_df is not None and factor_nodes:
        print(f"\nProcessing {len(factor_nodes)} factor(s)...")
        if "melody_id" not in factor_df.columns:
            print("  [warn] factor_scores_for_logreg.csv missing melody_id column; "
                  "factor examples skipped.")
        else:
            for f in factor_nodes:
                fid = f["id"]
                if fid not in factor_df.columns:
                    print(f"  {fid}: SKIP (column not in factor scores CSV)")
                    continue
                result = _process_one_column(
                    label=factor_labels.get(fid, fid),
                    df=factor_df,
                    value_col=fid,
                    midi_dir=args.midi_dir,
                    corpus_lookup=corpus_lookup,
                    cache=cache,
                )
                if result is None:
                    print(f"  {fid}: SKIP (insufficient data)")
                    continue
                entry, n_hi, n_lo = result
                manifest[fid] = entry
                print(f"  {fid} ({factor_labels.get(fid, fid)}): {n_hi} high, {n_lo} low")

    manifest_path = EXAMPLES_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    n_features = sum(1 for k in manifest if not k.startswith("F") or "." in k)
    n_factors = sum(1 for k in manifest if k.startswith("F") and "." not in k)
    n_rendered = len({p for k, v in manifest.items()
                      for which in ("high", "low")
                      for e in v.get(which, [])
                      for p in [e.get("png")] if p})
    print(f"\nManifest: {manifest_path}")
    print(f"  feature entries: {n_features}")
    print(f"  factor  entries: {n_factors}")
    print(f"  unique PNGs:     {n_rendered}")


if __name__ == "__main__":
    main()
