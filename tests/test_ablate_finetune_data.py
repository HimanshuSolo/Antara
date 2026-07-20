from pathlib import Path

from src.deep.ablate_finetune_data import materialize_subset


def test_materialize_subset_symlinks_selected_dirs(tmp_path):
    pool_dir = tmp_path / "pool"
    triplet_a = pool_dir / "triplet_a"
    triplet_b = pool_dir / "triplet_b"
    triplet_a.mkdir(parents=True)
    triplet_b.mkdir(parents=True)
    (triplet_a / "t.png").write_bytes(b"fake-png-bytes")

    subset_dir = tmp_path / "subset"
    materialize_subset([triplet_a], subset_dir)

    assert (subset_dir / "triplet_a").is_dir()
    assert (subset_dir / "triplet_a" / "t.png").read_bytes() == b"fake-png-bytes"
    assert not (subset_dir / "triplet_b").exists()


def test_materialize_subset_is_idempotent(tmp_path):
    pool_dir = tmp_path / "pool"
    triplet_a = pool_dir / "triplet_a"
    triplet_a.mkdir(parents=True)

    subset_dir = tmp_path / "subset"
    materialize_subset([triplet_a], subset_dir)
    materialize_subset([triplet_a], subset_dir)  # must not raise on rerun

    assert (subset_dir / "triplet_a").is_dir()
