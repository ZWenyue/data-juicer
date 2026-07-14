# -*- coding: utf-8 -*-
"""Load LeRobot episode parquet into top-level states/actions for downstream OPs."""

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper

# Relative import: custom_operator_paths loads this package as `_au`, not
# `data_juicer._au`. Absolute `data_juicer._au...` would re-exec `__init__`
# and double-register Operators.
from ...utils.lerobot_episode_io import load_episode_arrays

OP_NAME = "robot_lerobot_parquet_loader_mapper"


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class RobotLeRobotParquetLoaderMapper(Mapper):
    """Read ``parquet_path`` and materialize ``states`` / ``actions`` on the sample.

    Supports unified ``observation.state``/``action`` columns and Galaxea decomposed
    arm+gripper columns packed into the 16-dim layout used by Stage 1/2/3 filters.

    Registered as a tagging OP so ``dj-analyze`` runs it before Filter stats.
    """

    def __init__(
        self,
        parquet_field: str = "parquet_path",
        state_key: str = "states",
        action_key: str = "actions",
        skip_if_present: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.parquet_field = parquet_field
        self.state_key = state_key
        self.action_key = action_key
        self.skip_if_present = skip_if_present

    def process_single(self, sample):
        if self.skip_if_present and self.state_key in sample and self.action_key in sample:
            return sample
        ppath = sample.get(self.parquet_field)
        if not ppath:
            raise ValueError(f"Sample missing '{self.parquet_field}' for parquet loading")
        states, actions = load_episode_arrays(ppath)
        sample[self.state_key] = states.tolist()
        sample[self.action_key] = actions.tolist()
        sample["num_frames"] = int(states.shape[0])
        return sample
