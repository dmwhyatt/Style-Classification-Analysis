"""
Build melody example manifest for the factor network webapp.

Adapted from my old version of this at https://github.com/dmwhyatt/essen_new

Inputs:
  - essen_china_europe_features.csv           (melody_id + numeric feature columns, project root)
  - outputs/data/factor_scores_for_logreg.csv (melody_id + F1..FN scores, from factor_logistic.R)
  - docs/network_data.json                    (variable + factor nodes used in the webapp)

For each variable AND each factor in the network we pick the N_HIGH highest-scoring
and N_LOW lowest-scoring melodies and copy each unique MIDI into docs/melody_examples/midi/.
The webapp renders full piano rolls interactively with WaveRoll
(https://github.com/crescent-stdio/wave-roll).

Outputs (docs/melody_examples/):
  - midi/<melody_stub>.mid             one per unique melody actually used
  - manifest.json                      keyed by variable id and factor id (F1..FN)

Run from project root with the venv active:
    python build_melody_examples.py
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from helpers import output_paths as OP

PROJECT_ROOT = Path(__file__).resolve().parent
DOCS = PROJECT_ROOT / "docs"
EXAMPLES_DIR = DOCS / "melody_examples"
MIDI_DIR = EXAMPLES_DIR / "midi"
CSV_PATH = Path(OP.FEATURES_CSV)
FACTOR_SCORES_CSV = PROJECT_ROOT / "outputs" / "data" / "factor_scores_for_logreg.csv"
NETWORK_JSON = DOCS / "network_data.json"

N_HIGH = 3
N_LOW = 3

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
    for base in (PROJECT_ROOT, DOCS, MIDI_DIR):
        c = base / basename
        if c.exists():
            return c
    return None


def _safe_stub(name: str) -> str:
    """Filename-safe stub for a melody basename or arbitrary id."""
    stem = Path(name).stem if name.lower().endswith((".mid", ".midi")) else name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem)


class MidiCopyCache:
    """Copy each unique melody MIDI at most once; reuse paths across manifest entries."""

    def __init__(self, midi_dir: Path, force: bool = False):
        self.midi_dir = midi_dir
        self.midi_dir.mkdir(parents=True, exist_ok=True)
        self.force = force
        self._cache: Dict[str, str] = {}
        self._fail: set[str] = set()

    def copy(self, midi_path: Path) -> Optional[str]:
        key = midi_path.name
        if key in self._cache:
            return self._cache[key]
        if key in self._fail:
            return None
        stub = _safe_stub(key)
        dest = self.midi_dir / f"{stub}.mid"
        try:
            if self.force or not dest.exists():
                shutil.copy2(midi_path, dest)
        except OSError as e:
            print(f"  Skip MIDI copy for {key}: {e}")
            self._fail.add(key)
            return None
        rel = f"melody_examples/midi/{stub}.mid"
        self._cache[key] = rel
        return rel


def _build_entries_from_sorted(
    sorted_df: pd.DataFrame,
    value_col: str,
    n: int,
    descending: bool,
    midi_dir: Optional[Path],
    corpus_lookup: Optional[Dict[str, Path]],
    cache: MidiCopyCache,
) -> list:
    """Walk sorted_df and emit up to n manifest entries with copied MIDI paths."""
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
        midi_rel = cache.copy(midi_path)
        if midi_rel is None:
            continue
        rank += 1
        try:
            value = float(row[value_col])
        except (TypeError, ValueError):
            continue
        entries.append({
            "rank": rank,
            "melody_id": midi_path.name,
            "value": round(value, 6),
            "midi": midi_rel,
        })
    return entries


def _process_one_column(
    label: str,
    df: pd.DataFrame,
    value_col: str,
    midi_dir: Optional[Path],
    corpus_lookup: Optional[Dict[str, Path]],
    cache: MidiCopyCache,
) -> Optional[Tuple[dict, int, int]]:
    col = pd.to_numeric(df[value_col], errors="coerce")
    valid = col.notna()
    if valid.sum() < N_HIGH + N_LOW:
        return None
    sub = df.loc[valid, ["melody_id", value_col]].copy()
    sub = sub.astype({value_col: float})
    sorted_df = sub.sort_values(value_col, ascending=False)
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
        description="Build melody example MIDI assets and manifest for the factor network webapp."
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
        "--force-midi", action="store_true",
        help="Re-copy every MIDI even if it already exists on disk.",
    )
    args = parser.parse_args()

    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Feature CSV not found: {CSV_PATH}")
    if not NETWORK_JSON.exists():
        raise FileNotFoundError(
            f"{NETWORK_JSON} not found. Run `Rscript factor_logistic.R` first."
        )

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

    sample_id = str(df["melody_id"].dropna().iloc[0]).strip()
    if not get_midi_path(sample_id, args.midi_dir, corpus_lookup):
        print("ERROR: Cannot resolve MIDI path for a sample melody_id from the CSV.")
        print(f"  Sample: {sample_id[:120]}")
        print("Pass --midi-dir /path/to/essen_folksong_collection or "
              "set MELODY_EXAMPLES_MIDI_DIR.")
        sys.exit(1)

    cache = MidiCopyCache(MIDI_DIR, force=args.force_midi)
    manifest: dict = {}

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
    n_unique_midi = len(cache._cache)
    print(f"\nManifest: {manifest_path}")
    print(f"  feature entries: {n_features}")
    print(f"  factor  entries: {n_factors}")
    print(f"  unique MIDIs:    {n_unique_midi}")


if __name__ == "__main__":
    main()
