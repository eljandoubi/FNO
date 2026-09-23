"""Unit tests for fno.data.inspect_h5."""

import h5py
import numpy as np
import pytest

from fno.data import inspect_h5


def test_prints_flat_dataset(tmp_path, capsys):
    path = tmp_path / "flat.h5"
    with h5py.File(path, "w") as f:
        f["tensor"] = np.zeros((2, 3), dtype=np.float32)
        f["x-coordinate"] = np.zeros(3, dtype=np.float32)

    inspect_h5.main([str(path)])
    out = capsys.readouterr().out
    assert "tensor" in out
    assert "shape=(2, 3)" in out
    assert "x-coordinate" in out


def test_prints_nested_groups(tmp_path, capsys):
    path = tmp_path / "grouped.h5"
    with h5py.File(path, "w") as f:
        grp = f.create_group("0000")
        grp["data"] = np.zeros((4, 4), dtype=np.float32)
        grp2 = f.create_group("0001")
        grp2["data"] = np.zeros((4, 4), dtype=np.float32)

    inspect_h5.main([str(path)])
    out = capsys.readouterr().out
    assert "0000/" in out
    assert "0000/data" in out
    assert "0001/data" in out


def test_max_keys_truncates_output(tmp_path, capsys):
    path = tmp_path / "many.h5"
    with h5py.File(path, "w") as f:
        for i in range(10):
            f[f"field_{i}"] = np.zeros(2, dtype=np.float32)

    inspect_h5.main([str(path), "--max-keys", "3"])
    out = capsys.readouterr().out
    assert "truncated at 3 entries" in out


def test_missing_file_exits(tmp_path):
    with pytest.raises(SystemExit):
        inspect_h5.main([str(tmp_path / "does-not-exist.h5")])
