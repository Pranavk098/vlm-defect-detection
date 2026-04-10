"""Tests for vlm_defect.data — create_dataset(), load_split(), collate_fn()."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from vlm_defect.data import MVTecDataset, collate_fn, create_dataset, load_split


# ---------------------------------------------------------------------------
# create_dataset()
# ---------------------------------------------------------------------------

class TestCreateDataset:
    def test_creates_both_json_files(self, fake_mvtec, tmp_path):
        train = tmp_path / "train.json"
        test  = tmp_path / "test.json"
        create_dataset(fake_mvtec, train, test)
        assert train.exists()
        assert test.exists()

    def test_train_json_only_contains_good_images(self, fake_mvtec, tmp_path):
        train = tmp_path / "train.json"
        test  = tmp_path / "test.json"
        create_dataset(fake_mvtec, train, test)

        records = json.loads(train.read_text())
        # All training labels must be "No." — no anomaly-positive samples
        assert len(records) > 0
        for r in records:
            assert r["conversations"][1]["value"] == "No."
            assert "train/good" in r["image"]

    def test_train_json_has_no_test_images(self, fake_mvtec, tmp_path):
        train = tmp_path / "train.json"
        test  = tmp_path / "test.json"
        create_dataset(fake_mvtec, train, test)

        records = json.loads(train.read_text())
        for r in records:
            assert "/test/" not in r["image"]

    def test_test_json_contains_good_and_defect(self, fake_mvtec, tmp_path):
        train = tmp_path / "train.json"
        test  = tmp_path / "test.json"
        create_dataset(fake_mvtec, train, test)

        records = json.loads(test.read_text())
        labels = [r["conversations"][1]["value"] for r in records]
        assert any(v == "No." for v in labels),        "test split must include good (No.) samples"
        assert any(v.startswith("Yes") for v in labels), "test split must include defective samples"

    def test_test_json_defect_label_includes_class_name(self, fake_mvtec, tmp_path):
        train = tmp_path / "train.json"
        test  = tmp_path / "test.json"
        create_dataset(fake_mvtec, train, test)

        records = json.loads(test.read_text())
        defect_labels = [
            r["conversations"][1]["value"]
            for r in records
            if r["conversations"][1]["value"].startswith("Yes")
        ]
        assert all("broken_large" in lbl for lbl in defect_labels)

    def test_record_schema(self, fake_json_files):
        train_json, test_json = fake_json_files
        for path in (train_json, test_json):
            records = json.loads(path.read_text())
            for r in records:
                assert "id"            in r
                assert "image"         in r
                assert "conversations" in r
                assert len(r["conversations"]) == 2
                assert r["conversations"][0]["from"] == "human"
                assert r["conversations"][1]["from"] == "gpt"

    def test_each_record_has_unique_id(self, fake_json_files):
        train_json, test_json = fake_json_files
        all_ids = []
        for path in (train_json, test_json):
            all_ids.extend(r["id"] for r in json.loads(path.read_text()))
        assert len(all_ids) == len(set(all_ids)), "every record must have a unique id"

    def test_missing_root_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Dataset root not found"):
            create_dataset(
                tmp_path / "does_not_exist",
                tmp_path / "train.json",
                tmp_path / "test.json",
            )

    def test_sample_counts_match_directory(self, fake_mvtec, tmp_path):
        """2 train/good → 2 train records; 1 test/good + 1 broken_large → 2 test records."""
        train = tmp_path / "train.json"
        test  = tmp_path / "test.json"
        create_dataset(fake_mvtec, train, test)

        assert len(json.loads(train.read_text())) == 2
        assert len(json.loads(test.read_text()))  == 2


# ---------------------------------------------------------------------------
# load_split()
# ---------------------------------------------------------------------------

class TestLoadSplit:
    def test_reproducibility_same_seed(self, fake_json_files, mock_processor):
        path, _ = fake_json_files
        tr1, va1 = load_split(path, Path("."), mock_processor, seed=42)
        tr2, va2 = load_split(path, Path("."), mock_processor, seed=42)
        assert [r["id"] for r in tr1.data] == [r["id"] for r in tr2.data]
        assert [r["id"] for r in va1.data] == [r["id"] for r in va2.data]

    def test_different_seeds_produce_different_order(self, tmp_path, mock_processor):
        """Different seeds must produce a different shuffle.

        Two records give only 2 possible orderings so seeds can collide
        trivially.  This test writes 10 synthetic records inline so that
        the probability of a collision between seed=0 and seed=1 is 1/10!.
        """
        import uuid as _uuid, json as _json

        records = [
            {
                "id": str(_uuid.uuid4()),
                "image": f"bottle/train/good/{i:03d}.png",
                "conversations": [
                    {"from": "human", "value": "<image>\nAny anomaly?"},
                    {"from": "gpt",   "value": "No."},
                ],
            }
            for i in range(10)
        ]
        json_path = tmp_path / "big.json"
        json_path.write_text(_json.dumps(records))

        tr1, va1 = load_split(json_path, Path("."), mock_processor, seed=0)
        tr2, va2 = load_split(json_path, Path("."), mock_processor, seed=1)

        ids1 = [r["id"] for r in tr1.data] + [r["id"] for r in va1.data]
        ids2 = [r["id"] for r in tr2.data] + [r["id"] for r in va2.data]
        assert ids1 != ids2, "different seeds must produce a different shuffle"

    def test_no_overlap_between_train_and_val(self, fake_json_files, mock_processor):
        path, _ = fake_json_files
        train_ds, val_ds = load_split(path, Path("."), mock_processor, seed=42)
        train_ids = {r["id"] for r in train_ds.data}
        val_ids   = {r["id"] for r in val_ds.data}
        assert train_ids.isdisjoint(val_ids), "train and val must not share records"

    def test_val_fraction_respected(self, fake_json_files, mock_processor):
        path, _ = fake_json_files
        records_total = len(json.loads(path.read_text()))
        train_ds, val_ds = load_split(
            path, Path("."), mock_processor, val_fraction=0.5, seed=42
        )
        assert len(train_ds) + len(val_ds) == records_total

    def test_val_minimum_one_sample(self, fake_json_files, mock_processor):
        """Even with val_fraction near zero, at least 1 sample must be held out."""
        path, _ = fake_json_files
        _, val_ds = load_split(
            path, Path("."), mock_processor, val_fraction=0.01, seed=42
        )
        assert len(val_ds) >= 1

    def test_returns_mvtec_dataset_instances(self, fake_json_files, mock_processor):
        path, _ = fake_json_files
        train_ds, val_ds = load_split(path, Path("."), mock_processor)
        assert isinstance(train_ds, MVTecDataset)
        assert isinstance(val_ds,   MVTecDataset)


# ---------------------------------------------------------------------------
# collate_fn()
# ---------------------------------------------------------------------------

class TestCollateFn:
    def _make_batch(self, seq_lens: list[int]) -> list[dict]:
        return [
            {
                "input_ids":      torch.zeros(n, dtype=torch.long),
                "attention_mask": torch.ones(n,  dtype=torch.long),
                "pixel_values":   torch.zeros(3, 32, 32),
                "labels":         torch.full((n,), -100, dtype=torch.long),
            }
            for n in seq_lens
        ]

    def test_pads_input_ids_to_max_length(self):
        batch = self._make_batch([4, 7])
        out = collate_fn(batch)
        assert out["input_ids"].shape == (2, 7)

    def test_input_ids_padded_with_zeros(self):
        batch = self._make_batch([3, 6])
        out = collate_fn(batch)
        # The short sequence (len=3) should be padded with 0 on the right
        assert out["input_ids"][0, 3:].eq(0).all()

    def test_labels_padded_with_minus_100(self):
        batch = self._make_batch([3, 6])
        out = collate_fn(batch)
        assert out["labels"][0, 3:].eq(-100).all()

    def test_attention_mask_padded_with_zeros(self):
        batch = self._make_batch([3, 6])
        out = collate_fn(batch)
        assert out["attention_mask"][0, 3:].eq(0).all()

    def test_pixel_values_stacked_correctly(self):
        batch = self._make_batch([5, 5])
        out = collate_fn(batch)
        assert out["pixel_values"].shape == (2, 3, 32, 32)

    def test_output_keys(self):
        batch = self._make_batch([4])
        out = collate_fn(batch)
        assert set(out.keys()) == {"input_ids", "attention_mask", "pixel_values", "labels"}
