# -*- coding: utf-8 -*-
"""LeRobot export that encodes robot-view frames into episode videos.

The stock ``export_to_lerobot_mapper`` copies the original ego video in
whole-video mode. For hand→robot pipelines the observation video must come
from ``robot_render_frames`` (or any configured ``frame_field``).

Dual-arm (``hand_type=both`` / both sides present) exports concatenated
trajectories:

- ``observation.state``: right(8) || left(8) → shape 16
- ``action``: right(7) || left(7) → shape 14
"""

from __future__ import annotations

import json
import os
import uuid
from typing import List, Optional, Sequence, Tuple

from loguru import logger

from data_juicer.ops.base_op import OPERATORS
from data_juicer.ops.mapper.export_to_lerobot_mapper import DEFAULT_CHUNKS_SIZE, ExportToLeRobotMapper
from data_juicer.utils.constant import Fields
from data_juicer.utils.lazy_loader import LazyLoader

pa = LazyLoader("pyarrow", "pyarrow")
pd = LazyLoader("pandas", "pandas")

OP_NAME = "export_robot_render_lerobot_mapper"

STATE_DIM_SINGLE = 8
ACTION_DIM_SINGLE = 7
STATE_DIM_DUAL = 16
ACTION_DIM_DUAL = 14


def _as_vec(row: Sequence[float], dim: int) -> List[float]:
    arr = list(row) if row is not None else []
    if len(arr) >= dim:
        return [float(x) for x in arr[:dim]]
    return [float(x) for x in arr] + [0.0] * (dim - len(arr))


def merge_dual_arm_trajectories(
    right: dict,
    left: dict,
    state_dim: int = STATE_DIM_SINGLE,
    action_dim: int = ACTION_DIM_SINGLE,
) -> Tuple[List[List[float]], List[List[float]], List[int]]:
    """Intersect frame ids and concat right||left → (states16, actions14, frame_ids)."""
    r_ids = [int(x) for x in (right.get("valid_frame_ids") or [])]
    l_ids = [int(x) for x in (left.get("valid_frame_ids") or [])]
    r_states = right.get("states") or []
    l_states = left.get("states") or []
    r_actions = right.get("actions") or []
    l_actions = left.get("actions") or []

    r_map = {fid: i for i, fid in enumerate(r_ids)}
    l_map = {fid: i for i, fid in enumerate(l_ids)}
    common = sorted(set(r_map) & set(l_map))

    states_out: List[List[float]] = []
    actions_out: List[List[float]] = []
    for fid in common:
        ri, li = r_map[fid], l_map[fid]
        rs = _as_vec(r_states[ri] if ri < len(r_states) else [], state_dim)
        ls = _as_vec(l_states[li] if li < len(l_states) else [], state_dim)
        ra = _as_vec(r_actions[ri] if ri < len(r_actions) else [0.0] * action_dim, action_dim)
        la = _as_vec(l_actions[li] if li < len(l_actions) else [0.0] * action_dim, action_dim)
        states_out.append(rs + ls)
        actions_out.append(ra + la)
    return states_out, actions_out, common


def extract_single_or_dual_trajectory(video_action_data: dict) -> Tuple[List, List, Optional[List[int]], bool]:
    """Return (states, actions, valid_frame_ids, is_dual)."""
    if "states" in video_action_data:
        states = video_action_data.get("states") or []
        actions = video_action_data.get("actions") or []
        if actions and len(actions) < len(states):
            actions = list(actions) + [[0.0] * ACTION_DIM_SINGLE] * (len(states) - len(actions))
        return states, actions, video_action_data.get("valid_frame_ids"), False

    right = video_action_data.get("right") or {}
    left = video_action_data.get("left") or {}
    r_ok = bool(right.get("states"))
    l_ok = bool(left.get("states"))

    if r_ok and l_ok:
        states, actions, fids = merge_dual_arm_trajectories(right, left)
        return states, actions, fids, True

    hand = right if r_ok else left if l_ok else {}
    states = hand.get("states") or []
    actions = hand.get("actions") or []
    if actions and len(actions) < len(states):
        actions = list(actions) + [[0.0] * ACTION_DIM_SINGLE] * (len(states) - len(actions))
    return states, actions, hand.get("valid_frame_ids"), False


@OPERATORS.register_module(OP_NAME)
class ExportRobotRenderLeRobotMapper(ExportToLeRobotMapper):
    """Whole-video export with MP4 built from rendered frame paths."""

    def __init__(
        self,
        encode_video_from_frames: bool = True,
        dual_arm_order: str = "right_left",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.encode_video_from_frames = bool(encode_video_from_frames)
        if dual_arm_order != "right_left":
            raise ValueError("Only dual_arm_order='right_left' is supported (state16/action14)")
        self.dual_arm_order = dual_arm_order
        self._last_export_dual = False

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
        any_dual = False

        for video_idx, video_action_data in enumerate(action_data_list):
            states, actions, valid_frame_ids, is_dual = extract_single_or_dual_trajectory(video_action_data)
            any_dual = any_dual or is_dual

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

            state_dim = STATE_DIM_DUAL if is_dual else STATE_DIM_SINGLE
            action_dim = ACTION_DIM_DUAL if is_dual else ACTION_DIM_SINGLE
            self._stage_episode_meta_ext(
                ep_uuid,
                num_frames,
                task_desc,
                video_dst,
                state_dim=state_dim,
                action_dim=action_dim,
                dual_arm=is_dual,
            )
            exported_episodes.append(
                {
                    "uuid": ep_uuid,
                    "parquet_path": parquet_path,
                    "video_path": video_dst,
                    "num_frames": num_frames,
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                    "dual_arm": is_dual,
                    "encoded_from_frames": bool(
                        self.encode_video_from_frames and video_dst and str(video_dst).endswith(".mp4")
                    ),
                }
            )

        self._last_export_dual = any_dual
        sample[Fields.meta]["lerobot_export"] = exported_episodes
        return sample

    def _stage_episode_meta_ext(
        self,
        ep_uuid,
        num_frames,
        task_desc,
        video_path,
        state_dim: int,
        action_dim: int,
        dual_arm: bool,
    ):
        meta_path = os.path.join(self.staging_meta_dir, f"{ep_uuid}.jsonl")
        video_ext = ".mp4"
        if video_path and isinstance(video_path, str):
            video_ext = os.path.splitext(video_path)[1] or ".mp4"
        entry = {
            "uuid": ep_uuid,
            "length": num_frames,
            "task": task_desc,
            "video_ext": video_ext,
            "state_dim": int(state_dim),
            "action_dim": int(action_dim),
            "dual_arm": bool(dual_arm),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @staticmethod
    def _write_modality_json(meta_dir, dual_arm: bool = False):
        if dual_arm:
            modality = {
                "state": {
                    "right_x": {"start": 0, "end": 1},
                    "right_y": {"start": 1, "end": 2},
                    "right_z": {"start": 2, "end": 3},
                    "right_roll": {"start": 3, "end": 4},
                    "right_pitch": {"start": 4, "end": 5},
                    "right_yaw": {"start": 5, "end": 6},
                    "right_pad": {"start": 6, "end": 7},
                    "right_gripper": {"start": 7, "end": 8},
                    "left_x": {"start": 8, "end": 9},
                    "left_y": {"start": 9, "end": 10},
                    "left_z": {"start": 10, "end": 11},
                    "left_roll": {"start": 11, "end": 12},
                    "left_pitch": {"start": 12, "end": 13},
                    "left_yaw": {"start": 13, "end": 14},
                    "left_pad": {"start": 14, "end": 15},
                    "left_gripper": {"start": 15, "end": 16},
                },
                "action": {
                    "right_x": {"start": 0, "end": 1},
                    "right_y": {"start": 1, "end": 2},
                    "right_z": {"start": 2, "end": 3},
                    "right_roll": {"start": 3, "end": 4},
                    "right_pitch": {"start": 4, "end": 5},
                    "right_yaw": {"start": 5, "end": 6},
                    "right_gripper": {"start": 6, "end": 7},
                    "left_x": {"start": 7, "end": 8},
                    "left_y": {"start": 8, "end": 9},
                    "left_z": {"start": 9, "end": 10},
                    "left_roll": {"start": 10, "end": 11},
                    "left_pitch": {"start": 11, "end": 12},
                    "left_yaw": {"start": 12, "end": 13},
                    "left_gripper": {"start": 13, "end": 14},
                },
                "video": {"primary_image": {"original_key": "observation.images.image"}},
                "annotation": {"human.action.task_description": {"original_key": "task_index"}},
            }
        else:
            modality = {
                "state": {
                    "x": {"start": 0, "end": 1},
                    "y": {"start": 1, "end": 2},
                    "z": {"start": 2, "end": 3},
                    "roll": {"start": 3, "end": 4},
                    "pitch": {"start": 4, "end": 5},
                    "yaw": {"start": 5, "end": 6},
                    "pad": {"start": 6, "end": 7},
                    "gripper": {"start": 7, "end": 8},
                },
                "action": {
                    "x": {"start": 0, "end": 1},
                    "y": {"start": 1, "end": 2},
                    "z": {"start": 2, "end": 3},
                    "roll": {"start": 3, "end": 4},
                    "pitch": {"start": 4, "end": 5},
                    "yaw": {"start": 5, "end": 6},
                    "gripper": {"start": 6, "end": 7},
                },
                "video": {"primary_image": {"original_key": "observation.images.image"}},
                "annotation": {"human.action.task_description": {"original_key": "task_index"}},
            }
        path = os.path.join(meta_dir, "modality.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(modality, f, indent=4)

    @staticmethod
    def finalize_dataset(
        output_dir,
        fps=10,
        robot_type="r1_lite_ego_retarget",
        chunks_size=DEFAULT_CHUNKS_SIZE,
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
    ):
        """Finalize staging → LeRobot v2; supports single (8/7) or dual (16/14)."""
        staging_dir = os.path.join(output_dir, "staging")
        staging_data = os.path.join(staging_dir, "data")
        staging_video = os.path.join(staging_dir, "videos")
        staging_meta = os.path.join(staging_dir, "meta")
        meta_dir = os.path.join(output_dir, "meta")
        os.makedirs(meta_dir, exist_ok=True)

        episodes = []
        if os.path.exists(staging_meta):
            for fname in sorted(os.listdir(staging_meta)):
                if not fname.endswith(".jsonl"):
                    continue
                fpath = os.path.join(staging_meta, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            episodes.append(json.loads(line))

        if not episodes:
            logger.warning("No staged episodes found. Nothing to finalize.")
            return

        if state_dim is None:
            state_dim = int(episodes[0].get("state_dim") or STATE_DIM_SINGLE)
        if action_dim is None:
            action_dim = int(episodes[0].get("action_dim") or ACTION_DIM_SINGLE)
        dual_arm = bool(episodes[0].get("dual_arm")) or (state_dim == STATE_DIM_DUAL)

        ExportRobotRenderLeRobotMapper._write_modality_json(meta_dir, dual_arm=dual_arm)

        episodes.sort(key=lambda e: e["uuid"])
        task_to_index = {}
        global_frame_offset = 0
        for ep_idx, ep in enumerate(episodes):
            ep["episode_index"] = ep_idx
            ep["global_frame_offset"] = global_frame_offset
            global_frame_offset += ep["length"]
            task = ep["task"]
            if task not in task_to_index:
                task_to_index[task] = len(task_to_index)
            ep["task_index"] = task_to_index[task]

        total_episodes = len(episodes)
        total_frames = global_frame_offset
        total_chunks = max(1, (total_episodes + chunks_size - 1) // chunks_size)

        for chunk_idx in range(total_chunks):
            chunk_name = f"chunk-{chunk_idx:03d}"
            os.makedirs(os.path.join(output_dir, "data", chunk_name), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "videos", chunk_name, "observation.images.image"), exist_ok=True)

        total_videos = 0
        for ep in episodes:
            ep_uuid = ep["uuid"]
            ep_idx = ep["episode_index"]
            chunk_name = f"chunk-{ep_idx // chunks_size:03d}"

            src_parquet = os.path.join(staging_data, f"{ep_uuid}.parquet")
            if os.path.exists(src_parquet):
                table = pa.parquet.read_table(src_parquet)
                df = table.to_pandas()
                df["episode_index"] = ep_idx
                df["task_index"] = ep["task_index"]
                df["index"] = ep["global_frame_offset"] + df["frame_index"].values
                dst_parquet = os.path.join(output_dir, "data", chunk_name, f"episode_{ep_idx:06d}.parquet")
                pa.parquet.write_table(pa.Table.from_pandas(df), dst_parquet)

            video_ext = ep.get("video_ext", ".mp4")
            src_video = os.path.join(staging_video, f"{ep_uuid}{video_ext}")
            if os.path.exists(src_video):
                import shutil

                dst_video = os.path.join(
                    output_dir,
                    "videos",
                    chunk_name,
                    "observation.images.image",
                    f"episode_{ep_idx:06d}{video_ext}",
                )
                shutil.move(src_video, dst_video)
                total_videos += 1

        with open(os.path.join(meta_dir, "episodes.jsonl"), "w", encoding="utf-8") as f:
            for ep in episodes:
                f.write(
                    json.dumps(
                        {"episode_index": ep["episode_index"], "length": ep["length"], "task": ep["task"]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        with open(os.path.join(meta_dir, "tasks.jsonl"), "w", encoding="utf-8") as f:
            for task, idx in sorted(task_to_index.items(), key=lambda x: x[1]):
                f.write(json.dumps({"task_index": idx, "task": task}, ensure_ascii=False) + "\n")

        video_info = ExportToLeRobotMapper._probe_video_resolution(os.path.join(output_dir, "videos"))
        features = {
            "observation.state": {"dtype": "float32", "shape": [int(state_dim)]},
            "action": {"dtype": "float32", "shape": [int(action_dim)]},
            "observation.images.image": {
                "dtype": "video",
                "shape": [video_info["height"], video_info["width"], video_info["channels"]],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": video_info["height"],
                    "video.width": video_info["width"],
                    "video.channels": video_info["channels"],
                    "video.codec": video_info["codec"],
                    "video.pix_fmt": video_info["pix_fmt"],
                    "video.is_depth_map": False,
                    "video.fps": fps,
                    "has_audio": False,
                },
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }
        info = {
            "codebase_version": "v2.0",
            "robot_type": robot_type,
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": len(task_to_index),
            "total_videos": total_videos,
            "total_chunks": total_chunks,
            "chunks_size": chunks_size,
            "fps": fps,
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features,
            "dual_arm": dual_arm,
            "state_layout": "right(8)||left(8)" if dual_arm else "single(8)",
            "action_layout": "right(7)||left(7)" if dual_arm else "single(7)",
        }
        with open(os.path.join(meta_dir, "info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

        # Cleanup staging
        import shutil

        if os.path.isdir(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)
        logger.info(
            f"Finalized LeRobot dataset at {output_dir}: "
            f"{total_episodes} episodes, state_dim={state_dim}, action_dim={action_dim}"
        )
