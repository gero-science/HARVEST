"""Batch reuse must depend on what a batch contains, not on its index.

Keying resume on the index alone meant that adding a patent silently reused a
stale batch and dropped the new work from the exported table -- a wrong answer,
never an error.
"""

import json
import os

import pandas as pd
import pytest

from bindingdb_export.artifacts import get_patent_directories
from bindingdb_export.processing import reusable_batches, write_batch_manifest

BATCH_SIZE = 200


def make_batch(batches_dir, index, patents, *, manifest=True, corrupt=False):
    """Write a batch Parquet plus, by default, its membership manifest."""
    batch_file = os.path.join(batches_dir, f"batch_{index:04d}.parquet")
    if corrupt:
        with open(batch_file, "w") as f:
            f.write("not a parquet file")
    else:
        pd.DataFrame({"patent_number": list(patents)}).to_parquet(batch_file)
    if manifest:
        write_batch_manifest(batch_file, list(patents), BATCH_SIZE)
    return batch_file


def test_unchanged_patent_set_reuses_the_batch(tmp_path):
    d = str(tmp_path)
    make_batch(d, 0, ["/r/US1", "/r/US2"])

    reused, covered = reusable_batches(d, ["/r/US1", "/r/US2"], force=False)

    assert reused == {0}
    assert covered == {"/r/US1", "/r/US2"}


def test_added_patent_keeps_the_batch_and_leaves_the_new_one_uncovered(tmp_path):
    """The regression: the new patent must still be exported."""
    d = str(tmp_path)
    make_batch(d, 0, ["/r/US1", "/r/US2"])

    requested = ["/r/US1", "/r/US2", "/r/US3"]
    reused, covered = reusable_batches(d, requested, force=False)

    assert reused == {0}
    assert [p for p in requested if p not in covered] == ["/r/US3"]


def test_removed_patent_discards_the_batch(tmp_path):
    d = str(tmp_path)
    batch_file = make_batch(d, 0, ["/r/US1", "/r/US2"])

    reused, covered = reusable_batches(d, ["/r/US1"], force=False)

    assert reused == set()
    assert covered == set()
    assert not os.path.exists(batch_file), "stale batch must be deleted"


def test_batch_without_manifest_is_not_trusted(tmp_path):
    """Batches written before manifests existed cannot prove their contents."""
    d = str(tmp_path)
    batch_file = make_batch(d, 0, ["/r/US1"], manifest=False)

    reused, covered = reusable_batches(d, ["/r/US1"], force=False)

    assert reused == set()
    assert covered == set()
    assert not os.path.exists(batch_file)


def test_corrupt_batch_is_deleted(tmp_path):
    d = str(tmp_path)
    batch_file = make_batch(d, 0, ["/r/US1"], corrupt=True)

    reused, covered = reusable_batches(d, ["/r/US1"], force=False)

    assert reused == set()
    assert not os.path.exists(batch_file)


def test_force_ignores_every_existing_batch(tmp_path):
    d = str(tmp_path)
    make_batch(d, 0, ["/r/US1"])

    assert reusable_batches(d, ["/r/US1"], force=True) == (set(), set())


def test_new_batches_are_numbered_after_the_reused_ones(tmp_path):
    """Reused and fresh batches must coexist so the merge picks up both."""
    d = str(tmp_path)
    make_batch(d, 0, ["/r/US1"])
    make_batch(d, 1, ["/r/US2"])

    reused, covered = reusable_batches(d, ["/r/US1", "/r/US2", "/r/US3"], force=False)

    assert reused == {0, 1}
    assert max(reused, default=-1) + 1 == 2


def test_missing_batches_dir_is_not_an_error(tmp_path):
    assert reusable_batches(str(tmp_path / "absent"), ["/r/US1"], force=False) == (set(), set())


def test_get_patent_directories_is_sorted(tmp_path):
    """Batching slices this list, so its order must not depend on the filesystem."""
    for name in ("US3", "US1", "US2"):
        d = tmp_path / name
        d.mkdir()
        (d / f"{name}_resolved.json").write_text("[]")

    found = get_patent_directories(str(tmp_path))

    assert found == sorted(found)
    assert [os.path.basename(p) for p in found] == ["US1", "US2", "US3"]


def test_manifest_records_membership_and_batch_size(tmp_path):
    batch_file = make_batch(str(tmp_path), 0, ["/r/US2", "/r/US1"])

    payload = json.loads((tmp_path / "batch_0000.json").read_text())

    assert payload["patents"] == ["/r/US1", "/r/US2"], "stored sorted for stable diffs"
    assert payload["batch_size"] == BATCH_SIZE
    assert os.path.exists(batch_file)
