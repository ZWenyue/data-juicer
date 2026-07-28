# -*- coding: utf-8 -*-
import glob
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_lerobot_episodes import (
    ARM_SLOT_LEFT,
    ARM_SLOT_RIGHT,
    convert,
    episode_arrays_from_df,
    pack_decomposed_to_16,
)

REAL_DATASET_DIR = (
    "/mnt/r/DATA/Galaxea-Open-World-Dataset/lerobot"
    "/Connect_Router_Cables_20250625_002"
)
REAL_DATA_AVAILABLE = os.path.isdir(
    os.path.join(REAL_DATASET_DIR, "data", "chunk-000")
)


class TestPackDecomposed(unittest.TestCase):
    def test_pack_shapes_and_slots(self):
        T = 5
        left_arm = np.arange(T * 6, dtype=float).reshape(T, 6)
        right_arm = np.arange(100, 100 + T * 6, dtype=float).reshape(T, 6)
        left_g = np.full((T,), 0.3)
        right_g = np.full((T,), 0.7)
        out = pack_decomposed_to_16(left_arm, left_g, right_arm, right_g)
        self.assertEqual(out.shape, (T, 16))
        np.testing.assert_array_equal(out[:, ARM_SLOT_LEFT], left_arm)
        np.testing.assert_array_equal(out[:, ARM_SLOT_RIGHT], right_arm)
        self.assertTrue(np.allclose(out[:, 7], 0.3))
        self.assertTrue(np.allclose(out[:, 15], 0.7))
        self.assertTrue(np.allclose(out[:, 1], 0.0))
        self.assertTrue(np.allclose(out[:, 9], 0.0))

    def test_pack_accepts_column_vector_gripper(self):
        T = 3
        left_arm = np.zeros((T, 6))
        right_arm = np.zeros((T, 6))
        out = pack_decomposed_to_16(
            left_arm, np.ones((T, 1)) * 0.5, right_arm, np.zeros((T, 1))
        )
        self.assertEqual(out.shape, (T, 16))
        self.assertTrue(np.allclose(out[:, 7], 0.5))


@unittest.skipUnless(REAL_DATA_AVAILABLE, "Real dataset not found")
class TestConvertReal(unittest.TestCase):
    def test_episode_arrays_from_one_parquet(self):
        # Prefer NOT full-dataset convert (too slow). Only one parquet via episode_arrays_from_df.
        pf = sorted(
            glob.glob(
                os.path.join(
                    REAL_DATASET_DIR, "data", "chunk-000", "episode_*.parquet"
                )
            )
        )[0]
        df = pq.read_table(pf).to_pandas()
        states, actions = episode_arrays_from_df(df)
        self.assertEqual(states.ndim, 2)
        self.assertEqual(states.shape[1], 16)
        self.assertEqual(actions.shape[1], 16)
        self.assertEqual(states.shape[0], len(df))


@unittest.skipUnless(REAL_DATA_AVAILABLE, "Real dataset not found")
class TestPercentilesDecomposed(unittest.TestCase):
    def test_load_all_frames_decomposed(self):
        from compute_embodiment_percentiles import load_all_frames

        states, actions = load_all_frames(REAL_DATASET_DIR, max_files=3)
        self.assertEqual(states.shape[1], 16)
        self.assertEqual(actions.shape[1], 16)
        self.assertGreater(states.shape[0], 100)


if __name__ == "__main__":
    unittest.main()
