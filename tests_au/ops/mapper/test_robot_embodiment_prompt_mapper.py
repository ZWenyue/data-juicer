# -*- coding: utf-8 -*-
"""Unit tests for embodiment prompt field resolution and mapper."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data_juicer._au.ops.mapper.robot_embodiment_prompt_mapper import (  # noqa: E402
    RobotEmbodimentPromptMapper,
)
from data_juicer._au.utils.embodiment_layout import load_embodiment_config  # noqa: E402
from data_juicer._au.utils.embodiment_prompt import (  # noqa: E402
    build_prompt_fields_for_episode,
    discretize_speed,
    format_prompt_fields_for_debug,
    pick_instruction_lang,
    resolve_camera_view_direction,
    resolve_instruction,
    resolve_prompt_fields,
    write_episodes_jsonl_with_prompt_fields,
)
from data_juicer.utils.constant import Fields  # noqa: E402

REAL_DATASET_DIR = "/mnt/r/DATA/tst/Galaxea-Open-World-Dataset/Connect_Router_Cables_20250625_002"
REAL_DATA_AVAILABLE = os.path.isdir(os.path.join(REAL_DATASET_DIR, "data", "chunk-000"))


class TestInstructionHelpers(unittest.TestCase):
    def test_pick_instruction_lang(self):
        raw = "中文指令@English instruction"
        self.assertEqual(pick_instruction_lang(raw, "en"), "English instruction")
        self.assertEqual(pick_instruction_lang(raw, "zh"), "中文指令")
        self.assertEqual(pick_instruction_lang("only english", "en"), "only english")

    def test_resolve_instruction_skips_quality_labels(self):
        tasks = ["qualified", "null", "拿起杯子@Pick up the cup", "unqualified"]
        self.assertEqual(resolve_instruction(tasks=tasks, lang="en"), "Pick up the cup")

    def test_resolve_instruction_fallback_task_index(self):
        m = {0: "a@Alpha", 1: "b@Beta"}
        self.assertEqual(
            resolve_instruction(tasks=["qualified"], tasks_map=m, task_index=1, lang="en"),
            "Beta",
        )


class TestCameraView(unittest.TestCase):
    def test_galaxea_yaml_mapping(self):
        cfg = load_embodiment_config("galaxea_r1_lite")
        self.assertEqual(
            resolve_camera_view_direction(cfg, "observation.images.head_rgb"),
            "opposite side",
        )
        self.assertEqual(
            resolve_camera_view_direction(cfg, "observation.images.left_wrist_rgb"),
            "arm side",
        )
        self.assertEqual(resolve_camera_view_direction(cfg, None), "opposite side")


class TestDiscretizeSpeed(unittest.TestCase):
    def test_bins(self):
        self.assertEqual(discretize_speed(1), 500)
        self.assertEqual(discretize_speed(500), 500)
        self.assertEqual(discretize_speed(501), 1000)
        self.assertEqual(discretize_speed(8278), 8500)


class TestPromptFields(unittest.TestCase):
    def test_resolve_prompt_fields_includes_speed(self):
        fields = resolve_prompt_fields(
            embodiment="galaxea_r1_lite",
            instruction="Pick up the cup",
            length=8278,
            fps=15,
            camera_view_direction="opposite side",
        )
        self.assertEqual(fields["length"], 8278)
        self.assertEqual(fields["speed"], 8500)
        self.assertIn("speed:", format_prompt_fields_for_debug(fields))

    def test_build_prefers_episode_length(self):
        cfg = load_embodiment_config("galaxea_r1_lite")
        fields = build_prompt_fields_for_episode(
            cfg,
            {"tasks": ["拿起@Pick"], "length": 100},
            fps=15,
            length=999,
            video_key="observation.images.head_rgb",
        )
        self.assertEqual(fields["length"], 100)
        self.assertEqual(fields["speed"], 500)
        self.assertEqual(fields["instruction"], "Pick")
        self.assertEqual(fields["fps"], 15)


class TestWriteEpisodes(unittest.TestCase):
    def test_enrich_episodes_jsonl(self):
        cfg = load_embodiment_config("galaxea_r1_lite")
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "episodes.jsonl"
            dst = Path(td) / "out.jsonl"
            src.write_text(
                json.dumps(
                    {
                        "episode_index": 0,
                        "length": 120,
                        "tasks": ["qualified", "拿起@Pick up"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
                + json.dumps({"episode_index": 1, "length": 50, "tasks": ["x@Y"]})
                + "\n",
                encoding="utf-8",
            )
            n = write_episodes_jsonl_with_prompt_fields(
                src,
                dst,
                cfg=cfg,
                fps=15,
                tasks_map={},
                keep_episode_indices={0},
                video_key="observation.images.head_rgb",
            )
            self.assertEqual(n, 1)
            row = json.loads(dst.read_text(encoding="utf-8").strip())
            pf = row["prompt_fields"]
            self.assertEqual(pf["embodiment"], "galaxea_r1_lite")
            self.assertEqual(pf["instruction"], "Pick up")
            self.assertEqual(pf["length"], 120)
            self.assertEqual(pf["speed"], 500)
            self.assertEqual(pf["fps"], 15)
            self.assertEqual(pf["camera_view_direction"], "opposite side")


@unittest.skipUnless(REAL_DATA_AVAILABLE, "Galaxea dataset not mounted")
class TestMapperRealData(unittest.TestCase):
    def test_mapper_writes_meta(self):
        parquet = sorted(
            Path(REAL_DATASET_DIR).glob("data/chunk-*/episode_*.parquet")
        )[0]
        op = RobotEmbodimentPromptMapper(
            embodiment="galaxea_r1_lite",
            skip_if_present=False,
            write_debug_text=True,
        )
        sample = {"id": "ep0", "parquet_path": str(parquet), "text": "ep0"}
        out = op.process_single(sample)
        meta = out[Fields.meta]
        fields = json.loads(meta["prompt_fields"])
        self.assertEqual(fields["embodiment"], "galaxea_r1_lite")
        self.assertTrue(fields["instruction"])
        self.assertIsInstance(fields["length"], int)
        self.assertGreater(fields["length"], 0)
        self.assertEqual(fields["speed"], discretize_speed(fields["length"]))
        self.assertEqual(fields["fps"], 15)
        self.assertIn(fields["camera_view_direction"], ("arm side", "opposite side"))
        self.assertIn("embodiment:", meta["embodiment_prompt_debug"])


if __name__ == "__main__":
    unittest.main()
