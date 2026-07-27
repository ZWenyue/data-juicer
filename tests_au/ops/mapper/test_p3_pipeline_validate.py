# -*- coding: utf-8
"""Tests for P3 pipeline validation helpers."""

import unittest

from data_juicer._au.utils.hand_to_robot.pipeline_validate import (
    compare_action_invariance,
    snapshot_hand_actions,
)
from data_juicer.utils.constant import Fields


class TestPipelineValidate(unittest.TestCase):
    def test_action_invariance_ok(self):
        sample = {
            Fields.meta: {
                "hand_action_tags": [
                    {"right": {"states": [[1, 2, 3, 0, 0, 0, 0, 1]], "actions": [[0, 0, 0, 0, 0, 0, 1]]}}
                ]
            }
        }
        before = snapshot_hand_actions(sample)
        after = snapshot_hand_actions(sample)
        ok, rep = compare_action_invariance(before, after)
        self.assertTrue(ok)
        self.assertEqual(rep["max_abs_diff"], 0.0)

    def test_action_invariance_fail(self):
        before = [{"right": {"states": [[1.0, 0, 0, 0, 0, 0, 0, 1]]}}]
        after = [{"right": {"states": [[2.0, 0, 0, 0, 0, 0, 0, 1]]}}]
        ok, rep = compare_action_invariance(before, after)
        self.assertFalse(ok)
        self.assertGreater(rep["max_abs_diff"], 0)


if __name__ == "__main__":
    unittest.main()
