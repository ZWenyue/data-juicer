# -*- coding: utf-8 -*-
import json
import os
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from data_juicer._au.ops.mapper.robot_base_frame_alignment_mapper import (
    RobotBaseFrameAlignmentMapper,
)
from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

REAL_DATASET_DIR = "/mnt/r/DATA/tst/Galaxea-Open-World-Dataset" "/Connect_Router_Cables_20250625_002"
REAL_DATA_AVAILABLE = os.path.isdir(os.path.join(REAL_DATASET_DIR, "data", "chunk-000"))

RZ90 = Rotation.from_euler("z", 90, degrees=True).as_matrix()  # maps +x->+y, +y->-x


def _matrix_to_rot6d(mat):
    return np.concatenate([mat[:, :, 0], mat[:, :, 1]], axis=1)


def _sample(states=None, actions=None, robot_type="synthetic_arm"):
    s = {"id": "ep", "robot_type": robot_type, Fields.meta: {}}
    if states is not None:
        s["states"] = states
    if actions is not None:
        s["actions"] = actions
    return s


class RobotBaseFrameAlignmentMapperTest(DataJuicerTestCaseBase):

    # ---- Test 1: euler absolute pose, preset z_90 ----
    def test_euler_absolute_z90(self):
        # pos=[1,0,0], euler(xyz)=identity
        states = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        op = RobotBaseFrameAlignmentMapper(
            pose_layout=[{"target": "state", "pos": [0, 3], "rot": [3, 6], "rot_type": "euler", "is_delta": False}],
            correction_source="preset",
            preset_name="z_90",
        )
        out = op.process_single(_sample(states=states))
        pos = np.array(out["states"][0][:3])
        euler = np.array(out["states"][0][3:6])
        np.testing.assert_allclose(pos, [0.0, 1.0, 0.0], atol=1e-8)
        np.testing.assert_allclose(euler, [0.0, 0.0, 90.0], atol=1e-6)

    # ---- Test 2: quat absolute pose, param mode (euler correction) ----
    def test_quat_absolute_param(self):
        states = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]  # pos + identity quat xyzw
        op = RobotBaseFrameAlignmentMapper(
            pose_layout=[{"target": "state", "pos": [0, 3], "rot": [3, 7], "rot_type": "quat", "is_delta": False}],
            correction_source="param",
            correction_type="euler",
            correction_value=[0.0, 0.0, 90.0],
            degrees=True,
        )
        out = op.process_single(_sample(states=states))
        pos = np.array(out["states"][0][:3])
        quat = np.array(out["states"][0][3:7])
        np.testing.assert_allclose(pos, [0.0, 1.0, 0.0], atol=1e-8)
        # z+90 quat = [0,0,sin45,cos45]
        np.testing.assert_allclose(quat, [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)], atol=1e-6)

    # ---- Test 3: rot6d absolute pose validity + forward mapping ----
    def test_rot6d_absolute(self):
        # identity rotation -> rot6d = [1,0,0, 0,1,0]
        states = [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]
        op = RobotBaseFrameAlignmentMapper(
            pose_layout=[{"target": "state", "pos": [0, 3], "rot": [3, 9], "rot_type": "rot6d", "is_delta": False}],
            correction_source="preset",
            preset_name="z_90",
        )
        out = op.process_single(_sample(states=states))
        r6 = np.array(out["states"][0][3:9])
        # R' = Rz90 -> col0=[0,1,0], col1=[-1,0,0]
        np.testing.assert_allclose(r6, [0.0, 1.0, 0.0, -1.0, 0.0, 0.0], atol=1e-6)
        # columns orthonormal
        c0, c1 = r6[:3], r6[3:6]
        self.assertAlmostEqual(np.linalg.norm(c0), 1.0, places=6)
        self.assertAlmostEqual(np.linalg.norm(c1), 1.0, places=6)
        self.assertAlmostEqual(float(c0 @ c1), 0.0, places=6)

    # ---- Test 4: matrix absolute pose ----
    def test_matrix_absolute(self):
        eye = np.eye(3).reshape(-1).tolist()
        states = [[0.0, 1.0, 0.0] + eye]  # pos=[0,1,0], R=I (9)
        op = RobotBaseFrameAlignmentMapper(
            pose_layout=[{"target": "state", "pos": [0, 3], "rot": [3, 12], "rot_type": "matrix", "is_delta": False}],
            correction_source="preset",
            preset_name="z_90",
        )
        out = op.process_single(_sample(states=states))
        pos = np.array(out["states"][0][:3])
        M = np.array(out["states"][0][3:12]).reshape(3, 3)
        np.testing.assert_allclose(pos, [-1.0, 0.0, 0.0], atol=1e-8)
        np.testing.assert_allclose(M, RZ90, atol=1e-8)

    # ---- Test 5: delta rotvec similarity transform + delta translation ----
    def test_delta_rotvec_similarity(self):
        theta = 0.3
        # delta rotvec about +x axis; delta trans along +x
        actions = [[theta, 0.0, 0.0, 1.0, 0.0, 0.0]]
        op = RobotBaseFrameAlignmentMapper(
            pose_layout=[{"target": "action", "rot": [0, 3], "pos": [3, 6], "rot_type": "rotvec", "is_delta": True}],
            correction_source="preset",
            preset_name="z_90",
        )
        out = op.process_single(_sample(actions=actions))
        rotvec = np.array(out["actions"][0][:3])
        trans = np.array(out["actions"][0][3:6])
        # similarity: axis x -> Rz90·x = +y, angle preserved
        np.testing.assert_allclose(rotvec, [0.0, theta, 0.0], atol=1e-6)
        # delta translation rotated: +x -> +y
        np.testing.assert_allclose(trans, [0.0, 1.0, 0.0], atol=1e-8)

    # ---- Test 6: round-trip (apply then inverse restores original) ----
    def test_round_trip(self):
        rng = np.random.RandomState(0)
        states = np.column_stack(
            [
                rng.randn(20, 3),  # pos
                _matrix_to_rot6d(Rotation.random(20, random_state=1).as_matrix()),  # rot6d
            ]
        ).tolist()
        layout = [{"target": "state", "pos": [0, 3], "rot": [3, 9], "rot_type": "rot6d", "is_delta": False}]
        fwd = RobotBaseFrameAlignmentMapper(pose_layout=layout, correction_source="preset", preset_name="z_90")
        inv = RobotBaseFrameAlignmentMapper(pose_layout=layout, correction_source="preset", preset_name="z_-90")
        s1 = fwd.process_single(_sample(states=[r[:] for r in states]))
        s2 = inv.process_single(_sample(states=s1["states"]))
        np.testing.assert_allclose(np.array(s2["states"]), np.array(states), atol=1e-6)

    # ---- Test 7: stats_json per-embodiment selection ----
    def test_stats_json_embodiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "corr.json")
            with open(path, "w") as f:
                json.dump(
                    {
                        "armA": {"type": "euler", "value": [0, 0, 90], "degrees": True},
                        "armB": {"type": "euler", "value": [0, 0, 0], "degrees": True},
                    },
                    f,
                )
            layout = [{"target": "state", "pos": [0, 3], "rot": [3, 6], "rot_type": "euler", "is_delta": False}]
            op = RobotBaseFrameAlignmentMapper(
                pose_layout=layout,
                correction_source="stats_json",
                correction_stats_path=path,
                embodiment_field="robot_type",
            )
            a = op.process_single(_sample(states=[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], robot_type="armA"))
            b = op.process_single(_sample(states=[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], robot_type="armB"))
            np.testing.assert_allclose(a["states"][0][:3], [0.0, 1.0, 0.0], atol=1e-8)  # rotated
            np.testing.assert_allclose(b["states"][0][:3], [1.0, 0.0, 0.0], atol=1e-8)  # identity

    # ---- Test 8: stats_json unknown embodiment -> identity pass-through ----
    def test_stats_json_unknown_embodiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "corr.json")
            with open(path, "w") as f:
                json.dump({"armA": {"type": "euler", "value": [0, 0, 90]}}, f)
            op = RobotBaseFrameAlignmentMapper(
                pose_layout=[{"target": "state", "pos": [0, 3], "rot": [3, 6], "rot_type": "euler", "is_delta": False}],
                correction_source="stats_json",
                correction_stats_path=path,
                embodiment_field="robot_type",
            )
            states = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
            out = op.process_single(_sample(states=[r[:] for r in states], robot_type="unknown"))
            np.testing.assert_allclose(np.array(out["states"]), np.array(states), atol=1e-12)

    # ---- Test 9: pass-through when no pose_layout ----
    def test_pass_through_no_layout(self):
        states = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        op = RobotBaseFrameAlignmentMapper(correction_source="preset", preset_name="z_90")
        out = op.process_single(_sample(states=[r[:] for r in states]))
        np.testing.assert_allclose(np.array(out["states"]), np.array(states), atol=1e-12)
        # no report written on full pass-through
        self.assertNotIn("base_frame_alignment_report", out.get(Fields.meta, {}))

    # ---- Test 10: pass-through when slice out of bounds (joint-space data) ----
    def test_pass_through_out_of_bounds(self):
        # joint-space style: only 4 dims, but layout expects rot up to dim 9
        states = [[0.1, 0.2, 0.3, 0.4]] * 5
        op = RobotBaseFrameAlignmentMapper(
            pose_layout=[{"target": "state", "pos": [0, 3], "rot": [3, 9], "rot_type": "rot6d", "is_delta": False}],
            correction_source="preset",
            preset_name="z_90",
        )
        out = op.process_single(_sample(states=[r[:] for r in states]))
        np.testing.assert_allclose(np.array(out["states"]), np.array(states), atol=1e-12)
        report = json.loads(out[Fields.meta]["base_frame_alignment_report"])
        self.assertFalse(report["changed"])
        self.assertEqual(report["num_blocks"], 0)

    # ---- Test 11: state + action both transformed in one pass ----
    def test_state_and_action(self):
        states = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]  # pos + euler
        actions = [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]  # delta rotvec + delta trans
        op = RobotBaseFrameAlignmentMapper(
            pose_layout=[
                {"target": "state", "pos": [0, 3], "rot": [3, 6], "rot_type": "euler", "is_delta": False},
                {"target": "action", "rot": [0, 3], "pos": [3, 6], "rot_type": "rotvec", "is_delta": True},
            ],
            correction_source="preset",
            preset_name="z_90",
        )
        out = op.process_single(_sample(states=states, actions=actions))
        np.testing.assert_allclose(out["states"][0][:3], [0.0, 1.0, 0.0], atol=1e-8)
        np.testing.assert_allclose(out["actions"][0][3:6], [0.0, 1.0, 0.0], atol=1e-8)
        report = json.loads(out[Fields.meta]["base_frame_alignment_report"])
        self.assertTrue(report["changed"])
        self.assertEqual(report["num_blocks"], 2)

    # ---- Test 12: quat wxyz ordering ----
    def test_quat_wxyz(self):
        # identity quat in wxyz = [1,0,0,0]
        states = [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
        op = RobotBaseFrameAlignmentMapper(
            pose_layout=[{"target": "state", "pos": [0, 3], "rot": [3, 7], "rot_type": "quat", "is_delta": False}],
            correction_source="preset",
            preset_name="z_90",
            quat_order="wxyz",
        )
        out = op.process_single(_sample(states=states))
        quat = np.array(out["states"][0][3:7])
        # z+90 in wxyz = [cos45, 0, 0, sin45]
        np.testing.assert_allclose(quat, [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)], atol=1e-6)

    # ---- Test 13: pipeline via dataset.map ----
    def test_pipeline_map(self):
        ds_list = [
            {
                "id": "a",
                "robot_type": "synthetic_arm",
                "states": [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                "actions": [[0.0, 0.0, 0.0]],
            },
            {
                "id": "b",
                "robot_type": "synthetic_arm",
                "states": [[0.0, 2.0, 0.0, 0.0, 0.0, 0.0]],
                "actions": [[0.0, 0.0, 0.0]],
            },
        ]
        dataset = Dataset.from_list(ds_list)
        op = RobotBaseFrameAlignmentMapper(
            pose_layout=[{"target": "state", "pos": [0, 3], "rot": [3, 6], "rot_type": "euler", "is_delta": False}],
            correction_source="preset",
            preset_name="z_90",
        )
        dataset = dataset.map(op.process)
        rows = dataset.to_list()
        for r in rows:
            self.assertIn("base_frame_alignment_report", r[Fields.meta])
        by_id = {r["id"]: r for r in rows}
        np.testing.assert_allclose(by_id["a"]["states"][0][:3], [0.0, 1.0, 0.0], atol=1e-8)
        np.testing.assert_allclose(by_id["b"]["states"][0][:3], [-2.0, 0.0, 0.0], atol=1e-8)

    # ---- Test 14: validation errors ----
    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            RobotBaseFrameAlignmentMapper(signal_source="invalid")
        with self.assertRaises(ValueError):
            RobotBaseFrameAlignmentMapper(correction_source="invalid")
        with self.assertRaises(ValueError):
            RobotBaseFrameAlignmentMapper(correction_source="param")  # missing value
        with self.assertRaises(ValueError):
            RobotBaseFrameAlignmentMapper(correction_source="preset", preset_name="bogus")
        with self.assertRaises(ValueError):
            RobotBaseFrameAlignmentMapper(correction_source="stats_json")  # missing path
        with self.assertRaises(ValueError):
            RobotBaseFrameAlignmentMapper(
                pose_layout=[{"target": "state", "rot": [3, 5], "rot_type": "euler"}],
                correction_source="preset",
                preset_name="z_90",
            )  # euler needs 3 dims, got 2

    # ---- Test 15: real Galaxea data — joint-space, no layout -> pass-through ----
    @unittest.skipUnless(REAL_DATA_AVAILABLE, "Real dataset not found")
    def test_real_data_pass_through(self):
        import pyarrow.parquet as pq

        pf = os.path.join(REAL_DATASET_DIR, "data", "chunk-000", "episode_000000.parquet")
        df = pq.read_table(pf).to_pandas()
        states = df["observation.state"].tolist()
        actions = df["action"].tolist()
        # No pose_layout -> Stage 5 must be a no-op on joint-space data.
        op = RobotBaseFrameAlignmentMapper(correction_source="preset", preset_name="z_90")
        out = op.process_single(
            _sample(states=[list(s) for s in states], actions=[list(a) for a in actions], robot_type="galaxea_r1_lite")
        )
        np.testing.assert_allclose(np.array(out["states"]), np.array(states), atol=1e-9)
        np.testing.assert_allclose(np.array(out["actions"]), np.array(actions), atol=1e-9)


if __name__ == "__main__":
    unittest.main()
