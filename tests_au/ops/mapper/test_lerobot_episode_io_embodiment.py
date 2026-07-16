# -*- coding: utf-8 -*-
"""Tests for embodiment-aware episode loading and video_key resolution."""

import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data_juicer._au.utils.lerobot_episode_io import (  # noqa: E402
    CLEAN_SIGNAL_DIM,
    load_episode_arrays,
    resolve_video_key,
)

SIM_R1PRO_DATASET_DIR = (
    "/mnt/r/DATA/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/press/"
    "sim_behavior_r1_pro.task-0000_turning_on_radio"
)
SIM_R1PRO_AVAILABLE = os.path.isdir(os.path.join(SIM_R1PRO_DATASET_DIR, "data", "chunk-000"))

GALAXEA_LITE_DIR = "/mnt/r/DATA/tst/Galaxea-Open-World-Dataset/Connect_Router_Cables_20250625_002"
GALAXEA_LITE_AVAILABLE = os.path.isdir(os.path.join(GALAXEA_LITE_DIR, "data", "chunk-000"))


@unittest.skipUnless(SIM_R1PRO_AVAILABLE, "GR00T sim R1 Pro dataset not found")
class TestLoadSimBehaviorR1Pro(unittest.TestCase):
    def test_embodiment_packs_to_16_not_256(self):
        import glob

        pf = sorted(
            glob.glob(os.path.join(SIM_R1PRO_DATASET_DIR, "data", "chunk-000", "episode_*.parquet"))
        )[0]
        # without embodiment → raw packed dims
        raw_s, raw_a = load_episode_arrays(pf)
        self.assertEqual(raw_s.shape[1], 256)
        self.assertEqual(raw_a.shape[1], 23)

        s, a = load_episode_arrays(pf, embodiment="sim_behavior_r1_pro")
        self.assertEqual(s.shape[1], CLEAN_SIGNAL_DIM)
        self.assertEqual(a.shape[1], CLEAN_SIGNAL_DIM)
        self.assertEqual(s.shape[0], raw_s.shape[0])
        # left arm slice [158:165] identity into slots 0:7
        np.testing.assert_allclose(s[:, 0:7], raw_s[:, 158:165])
        np.testing.assert_allclose(a[:, 0:7], raw_a[:, 7:14])
        np.testing.assert_allclose(s[:, 7], raw_s[:, 193])
        np.testing.assert_allclose(a[:, 7], raw_a[:, 14])

    def test_resolve_head_video_key(self):
        key = resolve_video_key(SIM_R1PRO_DATASET_DIR, preferred="observation.images.head_rgb")
        self.assertIsNotNone(key)
        self.assertIn("head", key.lower())
        self.assertTrue(
            os.path.isdir(os.path.join(SIM_R1PRO_DATASET_DIR, "videos", "chunk-000", key))
        )


@unittest.skipUnless(GALAXEA_LITE_AVAILABLE, "Galaxea lite dataset not found")
class TestLoadGalaxeaLiteStill16(unittest.TestCase):
    def test_decomposed_with_embodiment(self):
        import glob

        pf = sorted(
            glob.glob(os.path.join(GALAXEA_LITE_DIR, "data", "chunk-000", "episode_*.parquet"))
        )[0]
        s0, a0 = load_episode_arrays(pf)
        s1, a1 = load_episode_arrays(pf, embodiment="galaxea_r1_lite")
        self.assertEqual(s0.shape[1], 16)
        self.assertEqual(s1.shape[1], 16)
        np.testing.assert_allclose(s0, s1)
        np.testing.assert_allclose(a0, a1)


if __name__ == "__main__":
    unittest.main()
