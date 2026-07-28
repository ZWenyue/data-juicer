# -*- coding: utf-8 -*-
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data_juicer._au.ops.mapper.robot_lerobot_parquet_loader_mapper import (  # noqa: E402
    RobotLeRobotParquetLoaderMapper,
)
from data_juicer.utils.constant import Fields  # noqa: E402

REAL_DATASET_DIR = (
    "/mnt/r/DATA/Galaxea-Open-World-Dataset/lerobot"
    "/Connect_Router_Cables_20250625_002"
)
REAL_DATA_AVAILABLE = os.path.isdir(os.path.join(REAL_DATASET_DIR, "data", "chunk-000"))


class TestLoaderSynthetic(unittest.TestCase):
    def test_skip_if_present(self):
        op = RobotLeRobotParquetLoaderMapper()
        sample = {"states": [[0.0] * 16], "actions": [[0.0] * 16], "parquet_path": "/nope"}
        out = op.process_single(sample)
        self.assertEqual(out["states"], [[0.0] * 16])


@unittest.skipUnless(REAL_DATA_AVAILABLE, "Real dataset not found")
class TestLoaderReal(unittest.TestCase):
    def test_loads_16dim(self):
        import glob

        pf = sorted(
            glob.glob(os.path.join(REAL_DATASET_DIR, "data", "chunk-000", "episode_*.parquet"))
        )[0]
        op = RobotLeRobotParquetLoaderMapper(skip_if_present=False)
        sample = {
            "id": "ep0",
            "parquet_path": pf,
            Fields.stats: {},
            Fields.meta: {},
        }
        out = op.process_single(sample)
        self.assertEqual(len(out["states"][0]), 16)
        self.assertEqual(len(out["actions"][0]), 16)
        self.assertEqual(out["num_frames"], len(out["states"]))


class TestPointerConvert(unittest.TestCase):
    @unittest.skipUnless(REAL_DATA_AVAILABLE, "Real dataset not found")
    def test_pointer_one_task(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "filter"))
        # tests_au/ops/mapper -> sibling filter
        filter_dir = Path(__file__).resolve().parents[1] / "filter"
        sys.path.insert(0, str(filter_dir))
        from convert_lerobot_episodes import convert_pointer_root

        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "ptr.jsonl")
            n = convert_pointer_root(REAL_DATASET_DIR, out)
            self.assertGreater(n, 0)
            # pointer file should be tiny vs materialize
            self.assertLess(os.path.getsize(out), 2_000_000)
            import json

            row = json.loads(open(out).readline())
            self.assertIn("parquet_path", row)
            self.assertNotIn("states", row)
            self.assertTrue(os.path.isfile(row["parquet_path"]))


if __name__ == "__main__":
    unittest.main()
