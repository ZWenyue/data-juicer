# -*- coding: utf-8 -*-
"""Unit tests for 80-dim unified state packing and mapper."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data_juicer._au.ops.mapper.robot_unified_state_mapper import (  # noqa: E402
    RobotUnifiedStateMapper,
)
from data_juicer._au.utils.embodiment_layout import (  # noqa: E402
    CANONICAL_JOINT_NAMES,
    LEFT_BASE,
    OFF_EEF,
    OFF_GRIPPER,
    OFF_JOINT,
    RIGHT_BASE,
    SHARED_BASE,
    UNIFIED_DIM,
    build_dim_mask,
    load_embodiment_config,
    pack_episode_to_80,
    quat_wxyz_to_rot6d,
    remap_joints_to_canonical,
)
from data_juicer.utils.constant import Fields  # noqa: E402

REAL_DATASET_DIR = "/mnt/r/DATA/tst/Galaxea-Open-World-Dataset/Connect_Router_Cables_20250625_002"
REAL_DATA_AVAILABLE = os.path.isdir(os.path.join(REAL_DATASET_DIR, "data", "chunk-000"))


class TestRemapJoints(unittest.TestCase):
    def test_canonical_order_and_null_pad(self):
        # source order deliberately shuffled vs canonical
        # src dims: [elbow, shoulder_pitch, wrist_pitch, ...]
        joint_map = {
            "shoulder_pitch": 1,
            "shoulder_roll": None,
            "shoulder_yaw": None,
            "elbow_pitch": 0,
            "forearm_roll": None,
            "wrist_pitch": 2,
            "wrist_roll": None,
        }
        src = np.array([[10.0, 1.0, 5.0]], dtype=float)  # elbow, shoulder_pitch, wrist_pitch
        out, occ = remap_joints_to_canonical(src, joint_map)
        self.assertEqual(out.shape, (1, 7))
        np.testing.assert_allclose(out[0, 0], 1.0)  # shoulder_pitch
        np.testing.assert_allclose(out[0, 3], 10.0)  # elbow
        np.testing.assert_allclose(out[0, 5], 5.0)  # wrist_pitch
        self.assertEqual(occ.tolist(), [1, 0, 0, 1, 0, 1, 0])

    def test_joint_sign(self):
        jm = {n: (i if i < 3 else None) for i, n in enumerate(CANONICAL_JOINT_NAMES)}
        src = np.ones((2, 3))
        signs = [-1, 1, -1, 1, 1, 1, 1]
        out, occ = remap_joints_to_canonical(src, jm, signs)
        np.testing.assert_allclose(out[:, 0], -1)
        np.testing.assert_allclose(out[:, 1], 1)
        np.testing.assert_allclose(out[:, 2], -1)


class TestQuatRot6d(unittest.TestCase):
    def test_identity_quat(self):
        # wxyz identity
        q = np.array([[1.0, 0.0, 0.0, 0.0]])
        r6 = quat_wxyz_to_rot6d(q)
        np.testing.assert_allclose(r6[0], [1, 0, 0, 0, 1, 0], atol=1e-6)


class TestGalaxeaConfig(unittest.TestCase):
    def test_load_by_name(self):
        cfg = load_embodiment_config("galaxea_r1_lite")
        self.assertEqual(cfg["name"], "galaxea_r1_lite")
        mask = build_dim_mask(cfg)
        self.assertEqual(mask.shape, (UNIFIED_DIM,))
        # R1 Lite: shoulder_roll (slot 1) padded; wrist_roll (slot 6) active
        self.assertEqual(mask[LEFT_BASE + OFF_JOINT + 1], 0.0)
        self.assertEqual(mask[RIGHT_BASE + OFF_JOINT + 1], 0.0)
        self.assertEqual(mask[LEFT_BASE + OFF_JOINT + 6], 1.0)
        for i in (0, 2, 3, 4, 5, 6):
            self.assertEqual(mask[LEFT_BASE + OFF_JOINT + i], 1.0)
        # gripper + eef
        self.assertEqual(mask[LEFT_BASE + OFF_GRIPPER], 1.0)
        self.assertTrue(np.all(mask[LEFT_BASE + OFF_EEF : LEFT_BASE + OFF_EEF + 9] == 1))
        # hand empty
        self.assertTrue(np.all(mask[LEFT_BASE + 17 : LEFT_BASE + 29] == 0))
        # chassis + torso shared
        self.assertTrue(np.all(mask[SHARED_BASE : SHARED_BASE + 7] == 1))
        self.assertTrue(np.all(mask[SHARED_BASE + 7 :] == 0))
        self.assertEqual(int(mask.sum()), 39)


@unittest.skipUnless(REAL_DATA_AVAILABLE, "Real dataset not found")
class TestPackReal(unittest.TestCase):
    def test_pack_episode_shapes_and_mask(self):
        import glob

        import pyarrow.parquet as pq

        pf = sorted(
            glob.glob(os.path.join(REAL_DATASET_DIR, "data", "chunk-000", "episode_*.parquet"))
        )[0]
        df = pq.read_table(pf).to_pandas()
        cfg = load_embodiment_config("galaxea_r1_lite")
        states, actions, mask = pack_episode_to_80(df, cfg)
        T = len(df)
        self.assertEqual(states.shape, (T, 80))
        self.assertEqual(actions.shape, (T, 80))
        self.assertEqual(mask.shape, (T, 80))
        self.assertTrue(np.all((mask == 0) | (mask == 1)))
        # shoulder_roll pad slot stays zero
        self.assertTrue(np.allclose(states[:, LEFT_BASE + 1], 0))
        self.assertTrue(np.allclose(mask[:, LEFT_BASE + 1], 0))
        # remapped joints nonzero somewhere (skip pad slot 1)
        active = [LEFT_BASE + i for i in (0, 2, 3, 4, 5, 6)]
        self.assertFalse(np.allclose(states[:, active], 0))
        # gripper mask on; values may be near zero depending on episode
        self.assertEqual(mask[0, LEFT_BASE + OFF_GRIPPER], 1.0)
        # action has no EEF fill → EEF dims may be 0 but mask still 1
        self.assertTrue(np.allclose(actions[:, LEFT_BASE + OFF_EEF : LEFT_BASE + OFF_EEF + 9], 0))
        self.assertTrue(np.all(mask[:, LEFT_BASE + OFF_EEF : LEFT_BASE + OFF_EEF + 9] == 1))


@unittest.skipUnless(REAL_DATA_AVAILABLE, "Real dataset not found")
class TestMapperReal(unittest.TestCase):
    def test_mapper_writes_keys(self):
        import glob

        pf = sorted(
            glob.glob(os.path.join(REAL_DATASET_DIR, "data", "chunk-000", "episode_*.parquet"))
        )[0]
        op = RobotUnifiedStateMapper(embodiment="galaxea_r1_lite", skip_if_present=False)
        sample = {
            "id": "ep0",
            "parquet_path": pf,
            Fields.stats: {},
            Fields.meta: {},
        }
        out = op.process_single(sample)
        self.assertEqual(len(out["unified_states"][0]), 80)
        self.assertEqual(len(out["unified_actions"][0]), 80)
        self.assertEqual(len(out["unified_dim_mask"][0]), 80)
        self.assertEqual(out["num_frames"], len(out["unified_states"]))
        meta = json.loads(out[Fields.meta]["unified_dim_occupancy"])
        self.assertEqual(meta["dim"], 80)
        self.assertEqual(meta["num_active"], 39)

    def test_skip_if_present(self):
        op = RobotUnifiedStateMapper(skip_if_present=True)
        sample = {
            "unified_states": [[0.0] * 80],
            "unified_actions": [[0.0] * 80],
            "unified_dim_mask": [[0.0] * 80],
            "parquet_path": "/nope",
        }
        out = op.process_single(sample)
        self.assertEqual(out["unified_states"], [[0.0] * 80])


class TestCustomYaml(unittest.TestCase):
    def test_custom_config_path(self):
        cfg = {
            "name": "toy_arm",
            "arms": {
                "left": {
                    "joint_map": {n: (i if i < 2 else None) for i, n in enumerate(CANONICAL_JOINT_NAMES)},
                    "source": {
                        "state_column": "observation.state.left_arm",
                        "action_column": "action.left_arm",
                    },
                    "gripper": None,
                    "eef": None,
                    "hand": None,
                }
            },
            "shared": {},
        }
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "toy_arm.yaml")
            with open(path, "w") as f:
                yaml.safe_dump(cfg, f)
            loaded = load_embodiment_config(path)
            mask = build_dim_mask(loaded)
            self.assertEqual(int(mask.sum()), 2)
            self.assertEqual(mask[0], 1)
            self.assertEqual(mask[1], 1)
            self.assertEqual(mask[2], 0)


if __name__ == "__main__":
    unittest.main()
