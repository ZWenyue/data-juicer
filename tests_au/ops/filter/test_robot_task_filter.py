"""Tests for task-directory-level robot task filtering."""

import json
import tempfile
import unittest
from pathlib import Path

from data_juicer._au.ops.filter.robot_task_filter import RobotTaskFilter
from data_juicer._au.pipeline.filter_robot_tasks import filter_task_root
from data_juicer.utils.constant import Fields


def _sample(name, labels):
    return {
        "task_name": name,
        "task_labels": labels,
        Fields.stats: {},
        Fields.meta: {},
    }


def _make_task(root: Path, name: str, labels):
    task = root / name
    (task / "data").mkdir(parents=True)
    (task / "meta").mkdir()
    (task / "meta" / "info.json").write_text("{}", encoding="utf-8")
    with (task / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as stream:
        for index, label in enumerate(labels):
            stream.write(json.dumps({"task_index": index, "task": label}, ensure_ascii=False) + "\n")
    return task


class RobotTaskFilterTest(unittest.TestCase):
    def _run(self, op, sample):
        op.compute_stats_single(sample)
        return op.process_single(sample)

    def test_open_door_skill_matches_english_directory_name(self):
        op = RobotTaskFilter(skill="open_door")
        sample = _sample("Open_The_Door_20250802", [])

        self.assertTrue(self._run(op, sample))
        self.assertTrue(sample[Fields.stats]["robot_task_filter_keep"])

    def test_open_door_skill_allows_words_between_action_and_door(self):
        op = RobotTaskFilter(skill="open_door")
        samples = [
            _sample("Open_And_Close_The_Door_20250802", []),
            _sample("Open_The_Bedroom_Door_To_Enter20250701", []),
        ]

        for sample in samples:
            with self.subTest(sample=sample):
                self.assertTrue(self._run(op, sample))

    def test_open_door_skill_matches_bilingual_task_label(self):
        op = RobotTaskFilter(skill="open_door", text_fields=("task_name", "task_labels"))
        sample = _sample(
            "indoor_manipulation_001",
            ["用左手打开门@Open the bedroom door with the left hand"],
        )

        self.assertTrue(self._run(op, sample))
        report = json.loads(sample[Fields.meta]["robot_task_filter_report"])
        self.assertIn("开门", report["include_matches"])

    def test_subtask_opening_a_door_keeps_complete_task(self):
        op = RobotTaskFilter(skill="open_door")
        sample = _sample(
            "Connect_Router_Cables_20250625_002",
            [
                "连接路由器网线@Connect the router cable",
                "打开冰箱门@Open the refrigerator door",
            ],
        )

        self.assertTrue(self._run(op, sample))

    def test_task_name_only_can_ignore_subtask_labels(self):
        op = RobotTaskFilter(skill="open_door", text_fields=("task_name",))
        sample = _sample(
            "Connect_Router_Cables_20250625_002",
            ["打开冰箱门@Open the refrigerator door"],
        )

        self.assertFalse(self._run(op, sample))

    def test_custom_keyword_and_exclusion(self):
        op = RobotTaskFilter(
            include_keywords=["router cable"],
            exclude_keywords=["unqualified"],
            text_fields=("task_name", "task_labels"),
        )
        good = _sample("router_task", ["Connect the router cable"])
        bad = _sample(
            "router_task_bad",
            ["Connect the router cable", "unqualified"],
        )

        self.assertTrue(self._run(op, good))
        self.assertFalse(self._run(op, bad))

    def test_filter_task_root_materializes_only_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source"
            output = base / "output"
            source.mkdir()
            matching = _make_task(
                source,
                "Open_And_Close_The_Door_20250802_012",
                ["开门@Open the door"],
            )
            _make_task(
                source,
                "Connect_Router_Cables_20250625_002",
                ["连接网线@Connect the router cable"],
            )

            result = filter_task_root(
                source,
                output,
                skill="open_door",
                mode="symlink",
            )

            self.assertEqual(result["scanned_tasks"], 2)
            self.assertEqual(result["selected_tasks"], 1)
            selected = output / matching.name
            self.assertTrue(selected.is_symlink())
            self.assertEqual(selected.resolve(), matching.resolve())
            self.assertTrue((output / "task_filter_results.jsonl").is_file())
            summary = json.loads((output / "task_filter_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["selected_task_names"], [matching.name])


if __name__ == "__main__":
    unittest.main()
