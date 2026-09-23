"""Print the key/shape/dtype tree of an HDF5 file -- run before writing a
loader for a new PDEBench format (e.g. NS_Incom, whose layout isn't confirmed).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Print the key/shape/dtype structure of an HDF5 file."
    )
    parser.add_argument("file", type=Path, help="Path to an .h5/.hdf5 file.")
    parser.add_argument(
        "--max-keys",
        type=int,
        default=40,
        help="Stop after printing this many entries (some files have thousands of groups).",
    )
    args = parser.parse_args(argv)

    if not args.file.exists():
        print(f"No such file: {args.file}", file=sys.stderr)
        raise SystemExit(1)

    with h5py.File(args.file, "r") as f:
        top_level = list(f.keys())
        print(f"{args.file}")
        truncated_top = top_level[: args.max_keys]
        suffix = " ..." if len(top_level) > args.max_keys else ""
        print(f"  top-level keys ({len(top_level)}): {truncated_top}{suffix}")

        count = 0

        def visitor(name: str, obj: object) -> None:
            nonlocal count
            if count >= args.max_keys:
                return
            count += 1
            if isinstance(obj, h5py.Dataset):
                print(f"  {name:<40} shape={obj.shape} dtype={obj.dtype}")
            else:
                print(f"  {name}/")

        f.visititems(visitor)
        if count >= args.max_keys:
            print(
                f"  ... truncated at {args.max_keys} entries, pass --max-keys to see more"
            )


if __name__ == "__main__":
    main()
