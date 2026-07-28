# -*- coding: utf-8 -*-
"""Deterministic caption stub for hand→robot pipeline smoke / P3 acceptance.

Production captioning should use ``VideoActionCaptioningMapper`` (VLM) with
``frame_field: robot_render_frames``. This stub avoids GPU/API dependencies
while exercising Render → Caption → Export wiring.
"""

from __future__ import annotations

from typing import Optional

from data_juicer.ops.base_op import OPERATORS, Mapper
from data_juicer.utils.constant import Fields, MetaKeys


OP_NAME = "video_hand_action_caption_stub_mapper"


@OPERATORS.register_module(OP_NAME)
class VideoHandActionCaptionStubMapper(Mapper):
    """Write a placeholder task string from hand side (no VLM)."""

    def __init__(
        self,
        hand_type: str = "right",
        frame_field: str = "robot_render_frames",
        tag_field_name: str = "hand_action_caption",
        action_template: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if hand_type not in ("left", "right", "both"):
            raise ValueError(f"hand_type must be left/right/both, got {hand_type}")
        self.hand_type = hand_type
        self.frame_field = frame_field
        self.tag_field_name = tag_field_name
        self.action_template = action_template or (
            "Manipulate the object with the {hand_type} robot arm."
        )

    def process_single(self, sample, rank=None):
        meta = sample.setdefault(Fields.meta, {})
        if self.tag_field_name in meta:
            return sample

        frame_data = meta.get(self.frame_field, []) or sample.get(self.frame_field, [])
        n_frames = 0
        if isinstance(frame_data, list) and frame_data:
            clip = frame_data[0] if isinstance(frame_data[0], list) else frame_data
            n_frames = len(clip)

        if self.hand_type == "both":
            right_action = self.action_template.format(hand_type="right")
            left_action = self.action_template.format(hand_type="left")
            meta[self.tag_field_name] = {
                "right": {"think": f"stub caption for right hand ({n_frames} robot-view frames).", "action": right_action},
                "left": {"think": f"stub caption for left hand ({n_frames} robot-view frames).", "action": left_action},
                "source": "stub",
                "num_frames": n_frames,
            }
            sample[self.text_key] = f"right hand: {right_action}; left hand: {left_action}"
        else:
            action = self.action_template.format(hand_type=self.hand_type)
            meta[self.tag_field_name] = {
                "think": f"stub caption for {self.hand_type} hand ({n_frames} robot-view frames).",
                "action": action,
                "source": "stub",
                "num_frames": n_frames,
            }
            sample[self.text_key] = action
        return sample
