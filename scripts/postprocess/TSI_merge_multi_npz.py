#!/usr/bin/env python3
"""
merge_npz_and_analyze.py

Find .npz files under folder_1 .. folder_10 (or any globs), merge arrays by key,
save merged .npz, and run simple analyses.

Merging rules (default):
  - If a key exists in multiple files and all values are numpy arrays with
    compatible shapes along axis `concat_axis`, they are concatenated.
  - If key looks like a histogram ('hist' or 'counts' in the key name) -> sum them.
  - If shapes are incompatible, items are stored as a Python list for that key.
  - If an item is scalar (0-d numpy), they are stacked into a 1D array.
"""
from pathlib import Path
import numpy as np
import argparse
from collections import defaultdict
import math

def find_npz_files(base_dir: Path, pattern="folder_*/tsi_probml_fullinputs.npz"):
    return sorted(base_dir.glob(pattern))

def parse_exclude_indices(value: str):
    """Parse a comma-separated list of zero-based file indices."""
    if value is None or value.strip() == "":
        return []
    indices = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            idx = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid excluded index {item!r}; expected comma-separated integers"
            ) from exc
        if idx < 0:
            raise argparse.ArgumentTypeError(
                f"Invalid excluded index {idx}; indices must be non-negative"
            )
        indices.append(idx)
    return indices

def filter_files_by_indices(file_paths, exclude_indices=None, verbose=False):
    """Remove explicitly requested zero-based positions from a sorted file list."""
    if not exclude_indices:
        return list(file_paths)

    files = list(file_paths)
    unique_indices = sorted(set(exclude_indices))
    out_of_range = [idx for idx in unique_indices if idx >= len(files)]
    if out_of_range:
        raise ValueError(
            "Excluded file indices out of range: "
            f"{out_of_range}; found {len(files)} input files"
        )

    excluded = {idx: files[idx] for idx in unique_indices}
    if verbose:
        print("Excluding files by zero-based index:")
        for idx, path in excluded.items():
            print(f"  {idx}: {path}")

    return [path for idx, path in enumerate(files) if idx not in excluded]

def load_npz(path: Path, allow_pickle=True):
    return np.load(path, allow_pickle=allow_pickle)

def is_hist_key(key: str):
    key_l = key.lower()
    return ("hist" in key_l) or ("count" in key_l) or ("bins" in key_l and "hist" in key_l)

def try_concat(arrays, axis=0):
    """
    Try to concatenate arrays in arrays list along axis; return (ok, result).
    ok=False if concatenation fails due to incompatible shapes/dtypes.
    """
    try:
        # Convert scalars to 1D arrays to avoid errors
        norm = []
        for a in arrays:
            if np.isscalar(a) or (isinstance(a, np.ndarray) and a.ndim == 0):
                norm.append(np.atleast_1d(a))
            else:
                norm.append(a)
        return True, np.concatenate(norm, axis=axis)
    except Exception:
        return False, None

def merge_npz_files(file_paths, concat_axis=0, allow_pickle=True, verbose=False):
    """
    Merge multiple .npz files.
    Returns dict: key -> merged numpy array OR list (if incompatible).
    """
    per_key = defaultdict(list)
    types = {}  # track types seen per key

    for p in file_paths:
        if verbose:
            print("Loading", p)
        data = load_npz(p, allow_pickle=allow_pickle)
        for key in data.files:
            val = data[key]
            per_key[key].append(val)
            types.setdefault(key, set()).add(type(val))

    merged = {}
    for key, items in per_key.items():
        # If histogram-like key, attempt to sum
        if is_hist_key(key):
            # try elementwise sum if shapes match
            shapes = [np.shape(it) for it in items]
            if all(sh == shapes[0] for sh in shapes):
                try:
                    merged[key] = np.sum([np.asarray(it) for it in items], axis=0)
                    continue
                except Exception:
                    pass
            # fallback: store list
            merged[key] = items
            continue

        # If all are numpy arrays and shapes compatible -> concat
        if all(isinstance(it, np.ndarray) for it in items):
            ok, res = try_concat(items, axis=concat_axis)
            if ok:
                merged[key] = res
            else:
                # try stacking along new axis
                try:
                    merged[key] = np.stack(items, axis=0)
                except Exception:
                    merged[key] = items  # fallback to list
        else:
            # Mixed types or non-ndarray: collect as list
            merged[key] = items

    return merged

def save_merged_npz(merged_dict, out_path: Path):
    # np.savez requires mapping of str -> array-like
    kwargs = {}
    for k, v in merged_dict.items():
        if isinstance(v, list):
            # try to convert list of arrays/scalars to array if possible
            try:
                kwargs[k] = np.asarray(v)
            except Exception:
                # fallback: save as object array (requires allow_pickle when loading)
                kwargs[k] = np.array(v, dtype=object)
        else:
            kwargs[k] = v
    np.savez(out_path, **kwargs)
    return out_path

def summary_stats_for_array(arr):
    """Return mean/std/min/max/count for numeric arrays; handle NaNs."""
    a = np.asarray(arr)
    if a.size == 0:
        return {}
    # flatten numeric values only
    if np.issubdtype(a.dtype, np.number):
        valid = a[np.isfinite(a)]
        if valid.size == 0:
            return {"count": a.size}
        return {
            "count": int(a.size),
            "mean": float(np.mean(valid)),
            "std": float(np.std(valid, ddof=0)),
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
        }
    else:
        # non numeric (object dtype) -> basic info
        return {"count": a.size, "dtype": str(a.dtype)}

def analyze_merged(merged, report_keys=None, print_summary=True):
    """
    Simple analysis:
      - For each key, compute summary stats if numeric
      - Return a dict of stats keyed by array-name
    """
    stats = {}
    keys = report_keys if report_keys is not None else sorted(merged.keys())
    for k in keys:
        v = merged[k]
        if isinstance(v, list):
            # try to coerce to array
            try:
                arr = np.asarray(v)
            except Exception:
                arr = np.array(v, dtype=object)
        else:
            arr = v
        stats[k] = summary_stats_for_array(arr)
        if print_summary:
            print(f"Key: {k}")
            for kk, vv in stats[k].items():
                print(f"  {kk}: {vv}")
            print()
    return stats

def main(base_dir=".", glob_pattern="folder_*/*.npz", out_name="merged_results.npz",
         concat_axis=0, allow_pickle=True, verbose=True, exclude_indices=None):
    base = Path(base_dir)
    files = find_npz_files(base, glob_pattern)
    print(f"number of files = {len(files)}")
    if len(files) == 0:
        raise SystemExit(f"No .npz files found with pattern {glob_pattern} under {base.resolve()}")

    try:
        files = filter_files_by_indices(files, exclude_indices=exclude_indices, verbose=verbose)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    if len(files) == 0:
        raise SystemExit("No .npz files remain after applying --exclude-indices")
    if exclude_indices:
        print(f"number of files after exclusions = {len(files)}")

    if verbose:
        print(f"Found {len(files)} files. First few:\n", "\n".join(str(p) for p in files[:10]))

    merged = merge_npz_files(files, concat_axis=concat_axis, allow_pickle=allow_pickle, verbose=verbose)

    out_path = Path(out_name)
    save_merged_npz(merged, out_path)
    if verbose:
        print("Saved merged file to", out_path)

    stats = analyze_merged(merged)

    return merged, stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge .npz files from many folders and analyze results.")
    parser.add_argument("--base-dir", default=".", help="Parent directory containing folder_* directories")
    parser.add_argument("--pattern", default="folder_*/tsi_probml_fullinputs.npz", help="Glob pattern to find .npz files")
    parser.add_argument("--out", default="merged_results.npz", help="Output merged .npz filename")
    parser.add_argument("--axis", type=int, default=0, help="Axis to concatenate arrays along when possible")
    parser.add_argument("--no-allow-pickle", dest="allow_pickle", action="store_false", help="Disable allow_pickle when opening npz files")
    parser.set_defaults(allow_pickle=True)
    parser.add_argument(
        "--exclude-indices",
        type=parse_exclude_indices,
        default=[],
        metavar="I,J,K",
        help=(
            "Comma-separated zero-based positions in the sorted input file list "
            "to exclude before merging (default: none)"
        ),
    )
    
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    main(base_dir=args.base_dir, glob_pattern=args.pattern, out_name=args.out,
         concat_axis=args.axis, allow_pickle=args.allow_pickle,
         verbose=not args.quiet, exclude_indices=args.exclude_indices)
