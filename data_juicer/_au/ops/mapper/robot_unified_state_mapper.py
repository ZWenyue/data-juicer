# -*- coding: utf-8 -*-
"""Map heterogeneous LeRobot episode columns into 80-dim unified state/action + action dim mask."""

from __future__ import annotations

import json

import pyarrow.parquet as pq
from loguru import logger

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.utils.constant import Fields

# Relative import: custom_operator_paths loads this package as `_au`.
from ...utils.embodiment_layout import (
    UNIFIED_DIM,
    load_embodiment_config,
    pack_episode_to_80,
    resolve_parquet_path,
)

OP_NAME = "robot_unified_state_mapper"


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class RobotUnifiedStateMapper(Mapper):
    """Pack embodiment-specific columns into unified ``(T, 80)`` vectors + action mask.

    Reads ``parquet_path``, applies the embodiment YAML (canonical 7-joint map,
    EE pose quat→6D, shared chassis/torso slots), and writes top-level sample keys
    so exporters persist vectors and **action** occupancy mask together (for
    ``loss × mask``). State-only dims (e.g. EEF pose) are packed into
    ``unified_states`` but stay 0 in the mask.

    Registered as a tagging OP so ``dj-analyze`` runs it before Filter stats when
    placed early in the recipe (typically right after the parquet loader).
    """

    def __init__(
        self,
        embodiment: str = "galaxea_r1_lite",
        embodiment_config: str = None,
        parquet_field: str = "parquet_path",
        state_key: str = "unified_states",
        action_key: str = "unified_actions",
        mask_key: str = "unified_dim_mask",
        skip_if_present: bool = True,
        write_meta_occupancy: bool = True,
        meta_occupancy_field: str = "unified_dim_occupancy",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.embodiment = embodiment
        self.embodiment_config = embodiment_config
        self.parquet_field = parquet_field
        self.state_key = state_key
        self.action_key = action_key
        self.mask_key = mask_key
        self.skip_if_present = skip_if_present
        self.write_meta_occupancy = write_meta_occupancy
        self.meta_occupancy_field = meta_occupancy_field
        cfg_ref = embodiment_config or embodiment
        self._cfg = load_embodiment_config(cfg_ref)
        logger.info(
            f"[{OP_NAME}] loaded embodiment={self._cfg.get('name', cfg_ref)} "
            f"from {self._cfg.get('_config_path')}"
        )

    def process_single(self, sample):
        if (
            self.skip_if_present
            and self.state_key in sample
            and self.action_key in sample
            and self.mask_key in sample
        ):
            return sample

        ppath = resolve_parquet_path(sample, self.parquet_field)
        df = pq.read_table(ppath).to_pandas()
        states, actions, dim_mask = pack_episode_to_80(df, self._cfg)

        if states.shape[1] != UNIFIED_DIM or actions.shape[1] != UNIFIED_DIM:
            raise ValueError(
                f"Expected last dim {UNIFIED_DIM}, got states={states.shape} actions={actions.shape}"
            )
        if dim_mask.shape != actions.shape:
            raise ValueError(f"action_dim_mask shape {dim_mask.shape} != actions {actions.shape}")

        sample[self.state_key] = states.tolist()
        sample[self.action_key] = actions.tolist()
        sample[self.mask_key] = dim_mask.tolist()
        sample["num_frames"] = int(states.shape[0])
        sample.setdefault("embodiment", self._cfg.get("name", self.embodiment))

        if self.write_meta_occupancy:
            if Fields.meta not in sample or sample[Fields.meta] is None:
                sample[Fields.meta] = {}
            # compact (80,) audit copy — action occupancy for loss masking
            occ = dim_mask[0].astype(int).tolist() if len(dim_mask) else []
            sample[Fields.meta][self.meta_occupancy_field] = json.dumps(
                {
                    "dim": UNIFIED_DIM,
                    "occupancy": occ,
                    "num_active": int(sum(occ)),
                    "mask_kind": "action",
                    "embodiment": self._cfg.get("name", self.embodiment),
                    "config_path": self._cfg.get("_config_path"),
                }
            )
        return sample
