"""Unit tests for fno.data.download (no real network calls)."""

import hashlib
import zipfile

import pytest

from fno.data import download


def test_all_default_variants_exist():
    for key, spec in download.DATASETS.items():
        assert spec.default_variant in spec.files, f"{key}: missing default variant"


def test_all_datasets_have_at_least_one_variant():
    for key, spec in download.DATASETS.items():
        assert len(spec.files) >= 1, f"{key}: no file variants registered"


def test_filenames_unique_within_dataset():
    for key, spec in download.DATASETS.items():
        filenames = [f.filename for f in spec.files.values()]
        assert len(filenames) == len(set(filenames)), f"{key}: duplicate filenames"


def test_urls_unique_across_registry():
    urls = [
        file_spec.url
        for spec in download.DATASETS.values()
        for file_spec in spec.files.values()
    ]
    assert len(urls) == len(set(urls)), "duplicate URLs found across datasets"


def test_md5sum(tmp_path):
    content = b"hello fno"
    path = tmp_path / "sample.bin"
    path.write_bytes(content)
    assert download._md5sum(path) == hashlib.md5(content).hexdigest()


def test_extract_archive_zip(tmp_path):
    archive = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("inner.txt", "contents")
    dest = tmp_path / "out"
    download.extract_archive(archive, dest)
    assert (dest / "inner.txt").read_text() == "contents"


def test_extract_archive_unsupported_format(tmp_path):
    archive = tmp_path / "sample.rar"
    archive.write_bytes(b"not a real archive")
    with pytest.raises(ValueError):
        download.extract_archive(archive, tmp_path / "out")


def test_list_command_does_not_require_dataset(capsys):
    download.main(["--list"])
    captured = capsys.readouterr()
    assert "airfrans" in captured.out
    assert "pdebench-burgers1d" in captured.out


def test_unknown_variant_exits_without_downloading(monkeypatch):
    called = False

    def fake_download_file(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(download, "download_file", fake_download_file)
    with pytest.raises(SystemExit):
        download.main(["pdebench-burgers1d", "--variant", "bogus-variant"])
    assert called is False


def test_invalid_dataset_choice_exits():
    with pytest.raises(SystemExit):
        download.main(["not-a-real-dataset"])


def test_download_variant_skips_when_already_present(tmp_path, monkeypatch):
    called = False

    def fake_download_file(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(download, "download_file", fake_download_file)

    dataset_dest = tmp_path / "raw" / "pdebench-darcy2d"
    dataset_dest.mkdir(parents=True)
    file_spec = download.DATASETS["pdebench-darcy2d"].files["beta1.0"]
    (dataset_dest / file_spec.filename).write_bytes(b"already here")

    download._download_variant(
        file_spec,
        dataset_dest,
        tmp_path / ".cache",
        extract=False,
        keep_archive=False,
        force=False,
    )
    assert called is False


def test_download_variant_checksum_mismatch_raises(tmp_path, monkeypatch):
    file_spec = download.DatasetFile(
        url="https://example.invalid/file", filename="data.bin", md5="0" * 32
    )

    def fake_download_file(_url, dest, chunk_size=1 << 20):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"wrong content")

    monkeypatch.setattr(download, "download_file", fake_download_file)

    dataset_dest = tmp_path / "raw" / "fake"
    cache_dir = tmp_path / ".cache"

    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        download._download_variant(
            file_spec,
            dataset_dest,
            cache_dir,
            extract=False,
            keep_archive=False,
            force=False,
        )
    assert not (cache_dir / "data.bin").exists()


def test_expand_variant_selectors_handles_ranges_and_lists():
    assert download._expand_variant_selectors(["shard0-shard3"]) == [
        "shard0",
        "shard1",
        "shard2",
        "shard3",
    ]
    assert download._expand_variant_selectors(["shard0,shard5,shard9"]) == [
        "shard0",
        "shard5",
        "shard9",
    ]
    assert download._expand_variant_selectors(["shard0-shard2", "shard10"]) == [
        "shard0",
        "shard1",
        "shard2",
        "shard10",
    ]
    assert download._expand_variant_selectors(["nu0.01", "nu0.1"]) == [
        "nu0.01",
        "nu0.1",
    ]


def test_expand_variant_selectors_rejects_backwards_range():
    with pytest.raises(SystemExit):
        download._expand_variant_selectors(["shard9-shard0"])


class _FakeDataverseResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _fake_ns_incom_payload(n_shards: int = 5) -> dict:
    return {
        "data": {
            "latestVersion": {
                "files": [
                    {
                        "dataFile": {
                            "id": 100 + i,
                            "filename": f"ns_incom_inhom_2d_512-{i}.h5",
                            "md5": None,
                        }
                    }
                    for i in range(n_shards)
                ]
                + [
                    # a non-NS_incom file that must be filtered out
                    {
                        "dataFile": {
                            "id": 999999,
                            "filename": "2D_DarcyFlow_beta0.01_Train.hdf5",
                            "md5": "unrelated",
                        }
                    }
                ]
            }
        }
    }


def test_fetch_ns_incom_shard_index_parses_and_filters(tmp_path, monkeypatch):
    monkeypatch.setattr(
        download.requests,
        "get",
        lambda url, timeout: _FakeDataverseResponse(_fake_ns_incom_payload(3)),
    )

    shards = download._fetch_ns_incom_shard_index(tmp_path / ".cache")

    assert set(shards) == {"shard0", "shard1", "shard2"}
    assert shards["shard0"].filename == "ns_incom_inhom_2d_512-0.h5"
    assert shards["shard0"].url.endswith("/100")


def test_fetch_ns_incom_shard_index_uses_cache(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        return _FakeDataverseResponse(_fake_ns_incom_payload(2))

    monkeypatch.setattr(download.requests, "get", fake_get)

    cache_dir = tmp_path / ".cache"
    download._fetch_ns_incom_shard_index(cache_dir)
    download._fetch_ns_incom_shard_index(cache_dir)

    assert len(calls) == 1
    assert (cache_dir / "ns_incom_shard_index.json").exists()


def test_main_navierstokes_variant_range_downloads_expected_shards(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        download.requests,
        "get",
        lambda url, timeout: _FakeDataverseResponse(_fake_ns_incom_payload(5)),
    )

    def fake_download_file(_url, dest, chunk_size=1 << 20):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")

    monkeypatch.setattr(download, "download_file", fake_download_file)

    download.main(
        [
            "pdebench-navierstokes2d",
            "--variant",
            "shard1-shard3",
            "--data-root",
            str(tmp_path),
        ]
    )

    dest_dir = tmp_path / "raw" / "pdebench-navierstokes2d"
    assert sorted(p.name for p in dest_dir.iterdir()) == [
        "ns_incom_inhom_2d_512-1.h5",
        "ns_incom_inhom_2d_512-2.h5",
        "ns_incom_inhom_2d_512-3.h5",
    ]

