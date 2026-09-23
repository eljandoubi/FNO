"""CLI to download benchmark datasets into user-chosen, easily-deletable folders."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

DEFAULT_DATA_ROOT = Path("data")


@dataclass(frozen=True)
class DatasetFile:
    url: str
    filename: str
    md5: str | None = None


@dataclass(frozen=True)
class Dataset:
    name: str
    description: str
    files: dict[str, DatasetFile]
    default_variant: str
    extract: bool = True


DATASETS: dict[str, Dataset] = {
    "airfrans": Dataset(
        name="airfrans",
        description=(
            "2D airfoil RANS CFD simulations (NeurIPS 2022) for "
            "irregular-geometry operator learning (~10 GB)."
        ),
        files={
            "default": DatasetFile(
                url="https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip",
                filename="airfrans.zip",
            ),
        },
        default_variant="default",
        extract=True,
    ),
    # Single representative files from PDEBench (github.com/pdebench/PDEBench),
    # fetched directly from its DaRUS hosting -- not the whole multi-GB/TB PDE split.
    "pdebench-burgers1d": Dataset(
        name="pdebench-burgers1d",
        description="PDEBench 1D Burgers' equation -- pick one or more viscosities.",
        files={
            "nu0.001": DatasetFile(
                url="https://darus.uni-stuttgart.de/api/access/datafile/268190",
                filename="1D_Burgers_Sols_Nu0.001.hdf5",
                md5="44cb784d5a07aa2b1c864cabdcf625f9",
            ),
            "nu0.002": DatasetFile(
                url="https://darus.uni-stuttgart.de/api/access/datafile/268193",
                filename="1D_Burgers_Sols_Nu0.002.hdf5",
                md5="edf1cd13622d151dfde3e4e5af7a95b4",
            ),
            "nu0.004": DatasetFile(
                url="https://darus.uni-stuttgart.de/api/access/datafile/268191",
                filename="1D_Burgers_Sols_Nu0.004.hdf5",
                md5="435e1fecb8a64a0b4563bb8d09f81c33",
            ),
            "nu0.01": DatasetFile(
                url="https://darus.uni-stuttgart.de/api/access/datafile/281363",
                filename="1D_Burgers_Sols_Nu0.01.hdf5",
                md5="e6d9a4f62baf9a29121a816b919e2770",
            ),
            "nu0.02": DatasetFile(
                url="https://darus.uni-stuttgart.de/api/access/datafile/268189",
                filename="1D_Burgers_Sols_Nu0.02.hdf5",
                md5="7c8c717a3a7818145877baa57106b090",
            ),
            "nu0.04": DatasetFile(
                url="https://darus.uni-stuttgart.de/api/access/datafile/281362",
                filename="1D_Burgers_Sols_Nu0.04.hdf5",
                md5="16f4c7afaf8c16238be157e54c9297c7",
            ),
            "nu0.1": DatasetFile(
                url="https://darus.uni-stuttgart.de/api/access/datafile/268185",
                filename="1D_Burgers_Sols_Nu0.1.hdf5",
                md5="660ba1008d3843bf4e28d2895eb607ce",
            ),
            "nu0.2": DatasetFile(
                url="https://darus.uni-stuttgart.de/api/access/datafile/268187",
                filename="1D_Burgers_Sols_Nu0.2.hdf5",
                md5="01d9333254dff2f3cad090b61f5cf695",
            ),
            "nu0.4": DatasetFile(
                url="https://darus.uni-stuttgart.de/api/access/datafile/268192",
                filename="1D_Burgers_Sols_Nu0.4.hdf5",
                md5="58327a6107e8f9defb5075677cc42533",
            ),
            "nu1.0": DatasetFile(
                url="https://darus.uni-stuttgart.de/api/access/datafile/281365",
                filename="1D_Burgers_Sols_Nu1.0.hdf5",
                md5="9021eba35332d127306f11ef84c1a60f",
            ),
            "nu2.0": DatasetFile(
                url="https://darus.uni-stuttgart.de/api/access/datafile/281364",
                filename="1D_Burgers_Sols_Nu2.0.hdf5",
                md5="70fe0c24d9313e70f6059e5212bdb3d7",
            ),
            "nu4.0": DatasetFile(
                url="https://darus.uni-stuttgart.de/api/access/datafile/268188",
                filename="1D_Burgers_Sols_Nu4.0.hdf5",
                md5="8af7f70166c00ffad377c80fa477d81f",
            ),
        },
        default_variant="nu0.01",
        extract=False,
    ),
    "pdebench-darcy2d": Dataset(
        name="pdebench-darcy2d",
        description="PDEBench 2D Darcy Flow -- pick one or more diffusion coefficients (beta).",
        files={
            "beta0.01": DatasetFile(
                url="https://darus.uni-stuttgart.de/api/access/datafile/133217",
                filename="2D_DarcyFlow_beta0.01_Train.hdf5",
                md5="d05c287d4c0b7d3178b0097084238251",
            ),
            "beta0.1": DatasetFile(
                url="https://darus.uni-stuttgart.de/api/access/datafile/133218",
                filename="2D_DarcyFlow_beta0.1_Train.hdf5",
                md5="294f9a03a4aa16b0e386469ca8b471be",
            ),
            "beta1.0": DatasetFile(
                url="https://darus.uni-stuttgart.de/api/access/datafile/133219",
                filename="2D_DarcyFlow_beta1.0_Train.hdf5",
                md5="81694ed31306ff2e5f6b76349b0b4389",
            ),
            "beta10.0": DatasetFile(
                url="https://darus.uni-stuttgart.de/api/access/datafile/133220",
                filename="2D_DarcyFlow_beta10.0_Train.hdf5",
                md5="a7f23cf8011fc211b180828af39b7d1a",
            ),
            "beta100.0": DatasetFile(
                url="https://darus.uni-stuttgart.de/api/access/datafile/133221",
                filename="2D_DarcyFlow_beta100.0_Train.hdf5",
                md5="7c09f3b1bb097737d3fd52ffb0c6f1c8",
            ),
        },
        default_variant="beta1.0",
        extract=False,
    ),
    "pdebench-navierstokes2d": Dataset(
        name="pdebench-navierstokes2d",
        description=(
            "PDEBench 2D incompressible Navier-Stokes -- pick one or more shards "
            "(~8 GB each; only 5 of the ~275 total shards are registered here)."
        ),
        files={
            "shard0": DatasetFile(
                url="https://darus.uni-stuttgart.de/api/access/datafile/133280",
                filename="ns_incom_inhom_2d_512-0.h5",
                md5="54109d46f9c957317bd670ddb2068ac0",
            ),
            "shard1": DatasetFile(
                url="https://darus.uni-stuttgart.de/api/access/datafile/136439",
                filename="ns_incom_inhom_2d_512-1.h5",
                md5="e280fd3208fccb8ad5c1ff46c4796864",
            ),
            "shard2": DatasetFile(
                url="https://darus.uni-stuttgart.de/api/access/datafile/133721",
                filename="ns_incom_inhom_2d_512-2.h5",
                md5="1d7a2aac41a410bea6c887f624273d20",
            ),
            "shard3": DatasetFile(
                url="https://darus.uni-stuttgart.de/api/access/datafile/136466",
                filename="ns_incom_inhom_2d_512-3.h5",
                md5="e9c92a19854e96c918d3a46d39994be4",
            ),
            "shard4": DatasetFile(
                url="https://darus.uni-stuttgart.de/api/access/datafile/166289",
                filename="ns_incom_inhom_2d_512-4.h5",
                md5="306a56ac14ea5686921cbb3f09c7dcb6",
            ),
        },
        default_variant="shard0",
        extract=False,
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


def _md5sum(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _list_datasets() -> None:
    print("Available datasets:")
    for key, spec in sorted(DATASETS.items()):
        print(f"  {key:<24} {spec.description}")
        if len(spec.files) > 1:
            variants = ", ".join(sorted(spec.files))
            print(
                f"      variants (--variant): {variants}  [default: {spec.default_variant}]"
            )


def _download_variant(
    file_spec: DatasetFile,
    dataset_dest: Path,
    cache_dir: Path,
    *,
    extract: bool,
    keep_archive: bool,
    force: bool,
) -> None:
    already_present = (
        dataset_dest.exists() and any(dataset_dest.iterdir())
        if extract
        else (dataset_dest / file_spec.filename).exists()
    )
    if already_present and not force:
        print(
            f"  already present, skipping: {file_spec.filename} (use --force to re-download)"
        )
        return
    if extract and force and dataset_dest.exists():
        shutil.rmtree(dataset_dest)

    archive_path = cache_dir / file_spec.filename
    print(f"Downloading {file_spec.filename} from {file_spec.url}")
    download_file(file_spec.url, archive_path)

    if file_spec.md5:
        print("  verifying checksum...")
        actual_md5 = _md5sum(archive_path)
        if actual_md5 != file_spec.md5:
            archive_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for {file_spec.filename}: "
                f"expected {file_spec.md5}, got {actual_md5}"
            )

    if extract:
        print(f"  extracting to {dataset_dest}")
        extract_archive(archive_path, dataset_dest)
        if not keep_archive:
            archive_path.unlink(missing_ok=True)
    else:
        dataset_dest.mkdir(parents=True, exist_ok=True)
        shutil.move(str(archive_path), str(dataset_dest / file_spec.filename))


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
        "--variant",
        action="append",
        metavar="KEY",
        help=(
            "Select a specific file to download (repeatable, e.g. --variant nu0.01 "
            "--variant nu0.1). Defaults to the dataset's default variant. Use "
            "--list to see available variants per dataset."
        ),
    )
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="Download every registered variant for the chosen dataset.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=(
            "Single root folder for everything this command creates "
            f"(raw datasets + download cache), so it can all be deleted at once "
            f"(default: {DEFAULT_DATA_ROOT})."
        ),
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded archive under <data-root>/.cache instead of deleting it after extraction.",
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

    if args.all_variants:
        selected = sorted(spec.files)
    elif args.variant:
        unknown = set(args.variant) - spec.files.keys()
        if unknown:
            raise SystemExit(
                f"Unknown variant(s) for '{spec.name}': {', '.join(sorted(unknown))}. "
                f"Available: {', '.join(sorted(spec.files))}"
            )
        selected = args.variant
    else:
        selected = [spec.default_variant]

    dataset_dest = args.data_root / "raw" / spec.name
    cache_dir = args.data_root / ".cache"

    for variant_key in selected:
        _download_variant(
            spec.files[variant_key],
            dataset_dest,
            cache_dir,
            extract=spec.extract,
            keep_archive=args.keep_archive,
            force=args.force,
        )

    print("\nDone.")
    print(f"  Dataset folder: {dataset_dest.resolve()}")
    print(f"  Cache folder  : {cache_dir.resolve()}")
    print(f"To remove this dataset later: rm -rf {dataset_dest}")
    print(f"To remove everything this command downloaded: rm -rf {args.data_root}")


if __name__ == "__main__":
    main()
