# -*- coding: utf-8 -*-
"""Attach structured embodiment-prompt fields from LeRobot meta onto each sample."""

from __future__ import annotations

import json
from typing import Optional

import pyarrow.parquet as pq
from loguru import logger

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.utils.constant import Fields

from ...utils.embodiment_layout import load_embodiment_config, resolve_parquet_path
from ...utils.embodiment_prompt import (
    build_prompt_fields_for_episode,
    format_prompt_fields_for_debug,
    load_episodes_by_index,
    load_info_fps,
    load_tasks_map,
    task_root_from_parquet,
)
from ...utils.lerobot_episode_io import resolve_video_key

OP_NAME = "robot_embodiment_prompt_mapper"


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class RobotEmbodimentPromptMapper(Mapper):
    """Resolve raw embodiment-prompt fields and write them into ``Fields.meta``.

    Does **not** discretize speed or apply prompt dropout — those belong in
    training. Writes:

    - ``Fields.meta[meta_field]``: JSON dict with embodiment / instruction /
      length / fps / camera_view_direction
    - optional debug string under ``Fields.meta[debug_field]``
    """

    def __init__(
        self,
        embodiment: str = "galaxea_r1_lite",
        embodiment_config: Optional[str] = None,
        parquet_field: str = "parquet_path",
        task_root_field: str = "task_root",
        meta_field: str = "prompt_fields",
        debug_field: str = "embodiment_prompt_debug",
        write_debug_text: bool = True,
        skip_if_present: bool = True,
        instruction_lang: str = "en",
        video_key: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.embodiment = embodiment
        self.embodiment_config = embodiment_config
        self.parquet_field = parquet_field
        self.task_root_field = task_root_field
        self.meta_field = meta_field
        self.debug_field = debug_field
        self.write_debug_text = write_debug_text
        self.skip_if_present = skip_if_present
        self.instruction_lang = instruction_lang
        self.video_key = video_key
        cfg_ref = embodiment_config or embodiment
        self._cfg = load_embodiment_config(cfg_ref)
        # Cache meta lookups per task root
        self._meta_cache = {}
        logger.info(
            f"[{OP_NAME}] loaded embodiment={self._cfg.get('name', cfg_ref)} "
            f"from {self._cfg.get('_config_path')}"
        )

    def _meta_bundle(self, task_root: str):
        if task_root not in self._meta_cache:
            meta_dir = f"{task_root}/meta"
            self._meta_cache[task_root] = {
                "tasks_map": load_tasks_map(meta_dir),
                "episodes": load_episodes_by_index(meta_dir),
                "fps": load_info_fps(meta_dir),
                "video_key": resolve_video_key(task_root, preferred=self.video_key),
            }
        return self._meta_cache[task_root]

    def process_single(self, sample):
        if Fields.meta not in sample or sample[Fields.meta] is None:
            sample[Fields.meta] = {}
        meta = sample[Fields.meta]
        if self.skip_if_present and self.meta_field in meta and meta[self.meta_field]:
            return sample

        ppath = resolve_parquet_path(sample, self.parquet_field)
        if sample.get(self.task_root_field):
            task_root = str(sample[self.task_root_field])
        else:
            task_root = str(task_root_from_parquet(ppath))

        bundle = self._meta_bundle(task_root)

        # episode_index: sample → parquet column → stem
        ep_idx = sample.get("episode_index")
        task_index = sample.get("task_index")
        length = sample.get("num_frames") or sample.get("length")

        if ep_idx is None or task_index is None or length is None:
            schema_names = set(pq.read_schema(ppath).names)
            cols = [c for c in ("episode_index", "task_index") if c in schema_names]
            if cols:
                df = pq.read_table(ppath, columns=cols).to_pandas()
            else:
                df = pq.read_table(ppath).to_pandas()
            if ep_idx is None and "episode_index" in df.columns and len(df):
                ep_idx = int(df["episode_index"].iloc[0])
            if task_index is None and "task_index" in df.columns and len(df):
                task_index = int(df["task_index"].iloc[0])
            if length is None:
                # cheap length: metadata num_rows when possible
                try:
                    length = int(pq.ParquetFile(ppath).metadata.num_rows)
                except Exception:
                    length = int(len(df)) if cols else int(len(pq.read_table(ppath)))

        if ep_idx is None:
            # episode_000012.parquet
            stem = str(ppath).rsplit("/", 1)[-1]
            try:
                ep_idx = int(stem.split("_")[-1].split(".")[0])
            except ValueError as e:
                raise ValueError(f"[{OP_NAME}] cannot resolve episode_index for {ppath}") from e

        ep_row = bundle["episodes"].get(int(ep_idx), {})
        fields = build_prompt_fields_for_episode(
            self._cfg,
            ep_row,
            tasks_map=bundle["tasks_map"],
            fps=bundle["fps"],
            length=int(length) if length is not None else None,
            task_index=int(task_index) if task_index is not None else None,
            video_key=bundle["video_key"] or self.video_key,
            instruction_lang=self.instruction_lang,
        )

        meta[self.meta_field] = json.dumps(fields, ensure_ascii=False)
        if self.write_debug_text:
            meta[self.debug_field] = format_prompt_fields_for_debug(fields)
        sample.setdefault("embodiment", fields.get("embodiment", self.embodiment))
        return sample
