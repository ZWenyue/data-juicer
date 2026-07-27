# -*- coding: utf-8 -*-
"""LeRobot export that encodes robot-view frames into episode videos.

The stock ``export_to_lerobot_mapper`` copies the original ego video in
whole-video mode. For hand→robot pipelines the observation video must come
from ``robot_render_frames`` (or any configured ``frame_field``).
"""

from __future__ import annotations

import os
import uuid

from loguru import logger

from data_juicer.ops.base_op import OPERATORS
from data_juicer.ops.mapper.export_to_lerobot_mapper import ExportToLeRobotMapper
from data_juicer.utils.constant import Fields

OP_NAME = "export_robot_render_lerobot_mapper"


@OPERATORS.register_module(OP_NAME)
class ExportRobotRenderLeRobotMapper(ExportToLeRobotMapper):
    """Whole-video export with MP4 built from rendered frame paths."""

    def __init__(
        self,
        encode_video_from_frames: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.encode_video_from_frames = bool(encode_video_from_frames)

    def _process_whole_video(self, sample):
        action_data_list = sample[Fields.meta].get(self.hand_action_field, [])
        if not action_data_list:
            logger.warning("No hand action data found, skipping export.")
            return sample

        task_desc = sample.get(self.text_key, "")
        if not task_desc:
            task_desc = "manipulate object with robot arm"

        video_sources = sample.get(self.video_key, [])
        exported_episodes = []

        for video_idx, video_action_data in enumerate(action_data_list):
            if "states" in video_action_data:
                action_data = video_action_data
            else:
                action_data = {}
                for ht in ["right", "left"]:
                    hand_entry = video_action_data.get(ht, {})
                    if hand_entry.get("states", []):
                        action_data = hand_entry
                        break

            states = action_data.get("states", [])
            actions = action_data.get("actions", [])
            valid_frame_ids = action_data.get("valid_frame_ids", None)

            if len(states) < 2:
                continue

            ep_uuid = uuid.uuid4().hex
            parquet_path, num_frames = self._stage_parquet(states, actions, ep_uuid, valid_frame_ids)

            video_dst = None
            if self.encode_video_from_frames:
                all_frames = self._get_frame_paths(sample)
                if valid_frame_ids:
                    frame_paths = [
                        all_frames[fid]
                        for fid in valid_frame_ids
                        if isinstance(fid, int) and 0 <= fid < len(all_frames) and all_frames[fid]
                    ]
                else:
                    frame_paths = [p for p in all_frames if p]
                if frame_paths:
                    staging_mp4 = os.path.join(self.staging_video_dir, f"{ep_uuid}.mp4")
                    video_dst = self._encode_frames_to_video(frame_paths, staging_mp4, self.fps)

            if video_dst is None and video_idx < len(video_sources):
                video_dst = self._stage_video(video_sources[video_idx], ep_uuid)

            self._stage_episode_meta(ep_uuid, num_frames, task_desc, video_dst)
            exported_episodes.append(
                {
                    "uuid": ep_uuid,
                    "parquet_path": parquet_path,
                    "video_path": video_dst,
                    "num_frames": num_frames,
                    "encoded_from_frames": bool(
                        self.encode_video_from_frames and video_dst and str(video_dst).endswith(".mp4")
                    ),
                }
            )

        sample[Fields.meta]["lerobot_export"] = exported_episodes
        return sample
