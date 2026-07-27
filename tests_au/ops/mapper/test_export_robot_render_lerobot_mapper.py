# -*- coding: utf-8 -*-
"""Unit tests for dual-arm LeRobot export concat (state16 / action14)."""

from __future__ import annotations

import unittest

from data_juicer._au.ops.mapper.export_robot_render_lerobot_mapper import (
    ACTION_DIM_DUAL,
    ACTION_DIM_SINGLE,
    STATE_DIM_DUAL,
    STATE_DIM_SINGLE,
    extract_single_or_dual_trajectory,
    merge_dual_arm_trajectories,
)


class TestDualArmExportConcat(unittest.TestCase):
    def test_merge_right_left_16_14(self):
        right = {
            "valid_frame_ids": [0, 2, 3],
            "states": [[1.0] * 8, [2.0] * 8, [3.0] * 8],
            "actions": [[10.0] * 7, [20.0] * 7, [30.0] * 7],
        }
        left = {
            "valid_frame_ids": [1, 2, 3],
            "states": [[4.0] * 8, [5.0] * 8, [6.0] * 8],
            "actions": [[40.0] * 7, [50.0] * 7, [60.0] * 7],
        }
        states, actions, fids = merge_dual_arm_trajectories(right, left)
        self.assertEqual(fids, [2, 3])
        self.assertEqual(len(states[0]), STATE_DIM_DUAL)
        self.assertEqual(len(actions[0]), ACTION_DIM_DUAL)
        self.assertEqual(states[0][:8], [2.0] * 8)
        self.assertEqual(states[0][8:], [5.0] * 8)
        self.assertEqual(actions[0][:7], [20.0] * 7)
        self.assertEqual(actions[0][7:], [50.0] * 7)

    def test_extract_dual(self):
        data = {
            "right": {
                "valid_frame_ids": [0, 1],
                "states": [[1] * STATE_DIM_SINGLE, [2] * STATE_DIM_SINGLE],
                "actions": [[3] * ACTION_DIM_SINGLE, [4] * ACTION_DIM_SINGLE],
            },
            "left": {
                "valid_frame_ids": [0, 1],
                "states": [[5] * STATE_DIM_SINGLE, [6] * STATE_DIM_SINGLE],
                "actions": [[7] * ACTION_DIM_SINGLE, [8] * ACTION_DIM_SINGLE],
            },
        }
        states, actions, fids, is_dual = extract_single_or_dual_trajectory(data)
        self.assertTrue(is_dual)
        self.assertEqual(fids, [0, 1])
        self.assertEqual(len(states[0]), STATE_DIM_DUAL)
        self.assertEqual(len(actions[0]), ACTION_DIM_DUAL)

    def test_extract_single_fallback(self):
        data = {
            "right": {
                "valid_frame_ids": [0],
                "states": [[1] * STATE_DIM_SINGLE],
                "actions": [[2] * ACTION_DIM_SINGLE],
            },
            "left": {},
        }
        states, actions, fids, is_dual = extract_single_or_dual_trajectory(data)
        self.assertFalse(is_dual)
        self.assertEqual(len(states[0]), STATE_DIM_SINGLE)
        self.assertEqual(len(actions[0]), ACTION_DIM_SINGLE)


if __name__ == "__main__":
    unittest.main()
