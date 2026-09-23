"""CLI to download benchmark datasets into user-chosen, easily-deletable folders."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

DEFAULT_DEST = Path("data/raw")
DEFAULT_CACHE_DIR = Path("data/.cache")


@dataclass(frozen=True)
class Dataset:
    name: str
    url: str
    archive_name: str
    description: str


DATASETS: dict[str, Dataset] = {
    "airfrans": Dataset(
        name="airfrans",
        url="https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip",
        archive_name="airfrans.zip",
        description=(
            "2D airfoil RANS CFD simulations (NeurIPS 2022) for "
            "irregular-geometry operator learning."
        ),
    ),
}


def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with (
            open(dest, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar,
        ):
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                bar.update(len(chunk))


def extract_archive(archive: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest_dir)
    elif archive.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive) as tf:
            tf.extractall(dest_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive}")


def _list_datasets() -> None:
    print("Available datasets:")
    for key, spec in sorted(DATASETS.items()):
        print(f"  {key:<12} {spec.description}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download benchmark datasets for the FNO project."
    )
    parser.add_argument(
        "dataset", nargs="?", choices=sorted(DATASETS), help="Dataset to download."
    )
    parser.add_argument(
        "--list", action="store_true", help="List available datasets and exit."
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Root folder where extracted datasets are placed (default: {DEFAULT_DEST}).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Folder for downloaded archives before extraction (default: {DEFAULT_CACHE_DIR}).",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded archive in --cache-dir instead of deleting it after extraction.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite even if the destination already exists.",
    )
    args = parser.parse_args(argv)

    if args.list or not args.dataset:
        _list_datasets()
        return

    spec = DATASETS[args.dataset]
    dataset_dest = args.dest / spec.name
    archive_path = args.cache_dir / spec.archive_name

    if dataset_dest.exists() and any(dataset_dest.iterdir()) and not args.force:
        print(
            f"'{spec.name}' already present at {dataset_dest} (use --force to re-download)."
        )
        return
    if args.force and dataset_dest.exists():
        shutil.rmtree(dataset_dest)

    print(f"Downloading '{spec.name}' from {spec.url}")
    download_file(spec.url, archive_path)

    print(f"Extracting to {dataset_dest}")
    extract_archive(archive_path, dataset_dest)

    if not args.keep_archive:
        archive_path.unlink(missing_ok=True)

    print("\nDone.")
    print(f"  Dataset folder : {dataset_dest.resolve()}")
    print(f"  Cache folder   : {args.cache_dir.resolve()}")
    print(f"To remove this dataset later : rm -rf {dataset_dest}")
    print(f"To remove all downloaded data: rm -rf {args.dest} {args.cache_dir}")


if __name__ == "__main__":
    main()
