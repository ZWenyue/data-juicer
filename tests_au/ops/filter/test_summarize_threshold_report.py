# -*- coding: utf-8 -*-
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_threshold_report import (  # noqa: E402
    load_stats_jsonl,
    suggest_thresholds,
    write_report,
)


def _write_stats(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


class TestSummarize(unittest.TestCase):
    def test_suggest_stage2_picks_in_band(self):
        rows = []
        for i, da in enumerate([0.50, 0.55, 0.62, 0.64, 0.66, 0.68, 0.70, 0.80, 0.90, 0.95]):
            rows.append(
                {
                    "id": f"ep{i}",
                    "__dj__stats__": {
                        "sudden_change_flagged_ratio": 0.01 * i,
                        "sudden_change_max_run": i,
                        "state_action_min_da": da,
                        "state_action_mean_da": da,
                        "extreme_value_flagged_ratio": 0.08,
                    },
                }
            )
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "stats.jsonl")
            _write_stats(p, rows)
            stats = load_stats_jsonl(p)
            sug = suggest_thresholds(stats)
            self.assertIn("da_threshold", sug)
            self.assertIn(sug["da_threshold"], (0.60, 0.65, 0.70))
            self.assertIn("max_flagged_ratio", sug)
            self.assertIn("alpha", sug)
            md = os.path.join(td, "threshold_report.md")
            yml = os.path.join(td, "clean_recipe_suggested.yaml")
            write_report(stats, sug, md, yml, probe_alpha=0.1)
            text = open(md).read()
            self.assertIn("Stage 1", text)
            self.assertIn("Stage 2", text)
            self.assertIn("Stage 3", text)
            self.assertIn("建议", text)
            self.assertTrue(os.path.isfile(yml))
            ytxt = open(yml).read()
            self.assertIn("robot_sudden_change_filter", ytxt)
            self.assertIn("robot_state_action_alignment_filter", ytxt)
            self.assertIn("robot_extreme_value_filter", ytxt)

    def test_missing_keys_raise(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "bad.jsonl")
            _write_stats(p, [{"id": "x", "__dj__stats__": {"sudden_change_flagged_ratio": 0.1}}])
            with self.assertRaises(ValueError):
                load_stats_jsonl(p)


if __name__ == "__main__":
    unittest.main()
