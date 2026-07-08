# data_juicer/_au/ops/mapper/robot_base_frame_alignment_mapper.py
# -*- coding: utf-8 -*-
"""Qwen-RobotManip Stage 5: Base Frame and End-Effector Orientation Alignment —— data-juicer 自定义 Mapper。

不同采集环境因传感器摆放 / 标定约定不同，末端执行器（EEF）位姿被记录在不同的
世界系里。本算子对每个 dataset / embodiment 施加一个「规范化旋转校正」$R_{corr}$，
把所有 EEF 位姿统一到规范系——使 **+x 轴恒指向机器人前向**，让跨机型的统一
state-action 表示几何一致。

与 Stage 1/2/3（Filter：删 / 标记）不同，本阶段是「数据修正」的 **Mapper**：不
删数据，而是就地改写 state / action 向量中的 EEF 位姿切片。

变换规则（逐帧向量化）：

    绝对位姿 (p, R):  p' = R_corr · p          R' = R_corr · R
    delta 位姿 (Δt, ΔR):  Δt' = R_corr · Δt     ΔR' = R_corr · ΔR · R_corrᵀ

支持的旋转表示：euler / quat(xyzw|wxyz) / rotvec / matrix(3x3) / rot6d
（Zhou et al. 6D 连续旋转，取旋转矩阵前两列，Gram-Schmidt 反解）。

校正来源三选一：
    - param     : 直接给定 correction_type + correction_value
    - preset    : 预置命名旋转（identity / z_90 / z_-90 / z_180 / x_90 / ...）
    - stats_json: 按 embodiment 从 JSON 查表（企业化：跨 dataset 复用一份配置）

安全 pass-through：未配置 pose_layout、或切片越界（如关节空间 Galaxea 数据）→
样本原样返回，保证与老算子在同一 recipe 中级联时不破坏关节数据。
"""

import json

import numpy as np
from loguru import logger

from data_juicer.ops.base_op import OPERATORS, Mapper
from data_juicer.utils.constant import Fields, MetaKeys

OP_NAME = "robot_base_frame_alignment_mapper"

_VALID_ROT_TYPES = ("euler", "quat", "rotvec", "matrix", "rot6d")

# 预置旋转：以 (x, y, z) 内旋欧拉角（度）表示，from_euler("xyz", ...)
_PRESETS = {
    "identity": (0.0, 0.0, 0.0),
    "z_90": (0.0, 0.0, 90.0),
    "z_-90": (0.0, 0.0, -90.0),
    "z_180": (0.0, 0.0, 180.0),
    "x_90": (90.0, 0.0, 0.0),
    "x_-90": (-90.0, 0.0, 0.0),
    "y_90": (0.0, 90.0, 0.0),
    "y_-90": (0.0, -90.0, 0.0),
}


@OPERATORS.register_module(OP_NAME)
class RobotBaseFrameAlignmentMapper(Mapper):
    """Apply per-dataset rotation correction to EEF poses so that +x = forward.

    对 state / action 向量内的 EEF 位姿切片施加旋转校正 $R_{corr}$，统一世界系朝向。
    绝对位姿左乘 $R_{corr}$；delta 位姿的旋转做相似变换、平移做旋转。无 pose_layout
    或切片越界时安全 pass-through。
    """

    def __init__(
        self,
        # ---- 信号来源 ----
        signal_source: str = "top_level",
        top_level_state_key: str = "states",
        top_level_action_key: str = "actions",
        hand_action_field: str = MetaKeys.hand_action_tags,
        states_key: str = "states",
        actions_key: str = "actions",
        # ---- EEF 位姿布局：向量内的位姿切片描述 ----
        pose_layout: list = None,
        # ---- 校正来源 ----
        correction_source: str = "param",
        correction_type: str = "euler",
        correction_value: list = None,
        preset_name: str = None,
        correction_stats_path: str = None,
        embodiment: str = None,
        embodiment_field: str = "robot_type",
        # ---- 旋转表示细节 ----
        degrees: bool = True,
        euler_seq: str = "xyz",
        quat_order: str = "xyzw",
        # ---- 报告 ----
        report_field: str = "base_frame_alignment_report",
        *args,
        **kwargs,
    ):
        """
        :param signal_source: 信号来源，{"top_level","hand_action_tags"}。
        :param top_level_state_key/top_level_action_key: top_level 时样本顶层键名。
        :param hand_action_field: hand_action_tags 时 meta 中的字段名。
        :param states_key/actions_key: hand_action_tags 每个 hand dict 内的键名。
        :param pose_layout: EEF 位姿块描述列表；每项为 dict：
            {"target": "state"|"action",
             "pos":  [i, j]  (可选，长度须为 3 的平移/位置切片),
             "rot":  [i, j]  (旋转切片；长度须匹配 rot_type 维度),
             "rot_type": "euler"|"quat"|"rotvec"|"matrix"|"rot6d",
             "is_delta": bool  (delta 位姿走相似变换)}。
            为 None / 空时整体 pass-through。
        :param correction_source: 校正来源，{"param","preset","stats_json"}。
        :param correction_type: param 模式下 correction_value 的旋转表示。
        :param correction_value: param 模式下的旋转值（欧拉角 / xyzw 四元数 /
            旋转向量 / 3x3 矩阵）。
        :param preset_name: preset 模式下的预置名（identity/z_90/z_-90/z_180/...）。
        :param correction_stats_path: stats_json 模式下的 JSON 路径；结构
            {embodiment: {"type": rot_type, "value": [...], "degrees": bool?,
            "euler_seq": str?, "quat_order": str?}}。
        :param embodiment: 显式 embodiment 名；None 时读 sample[embodiment_field]。
        :param embodiment_field: 样本中 embodiment 字段名（默认 "robot_type"）。
        :param degrees: euler 校正 / 预置角度是否为度（True）。
        :param euler_seq: euler 表示的旋转序（如 "xyz"）。
        :param quat_order: 四元数分量顺序，{"xyzw"(scipy 默认),"wxyz"}。
        :param report_field: 明细报告写入 meta 的字段名。
        """
        super().__init__(*args, **kwargs)

        if signal_source not in ("top_level", "hand_action_tags"):
            raise ValueError(f"Invalid signal_source: {signal_source}")
        if correction_source not in ("param", "preset", "stats_json"):
            raise ValueError(f"Invalid correction_source: {correction_source}")
        if correction_type not in ("euler", "quat", "rotvec", "matrix"):
            raise ValueError(f"Invalid correction_type: {correction_type}")
        if quat_order not in ("xyzw", "wxyz"):
            raise ValueError(f"Invalid quat_order: {quat_order}")
        if correction_source == "param" and correction_value is None:
            raise ValueError("param mode requires correction_value.")
        if correction_source == "preset" and preset_name is None:
            raise ValueError("preset mode requires preset_name.")
        if correction_source == "preset" and preset_name not in _PRESETS:
            raise ValueError(f"Unknown preset_name: {preset_name}. Valid: {sorted(_PRESETS)}")
        if correction_source == "stats_json" and correction_stats_path is None:
            raise ValueError("stats_json mode requires correction_stats_path.")

        self.signal_source = signal_source
        self.top_level_state_key = top_level_state_key
        self.top_level_action_key = top_level_action_key
        self.hand_action_field = hand_action_field
        self.states_key = states_key
        self.actions_key = actions_key

        self.pose_layout = [self._normalize_spec(s) for s in pose_layout] if pose_layout else []

        self.correction_source = correction_source
        self.correction_type = correction_type
        self.correction_value = correction_value
        self.preset_name = preset_name
        self.correction_stats_path = correction_stats_path
        self.embodiment = embodiment
        self.embodiment_field = embodiment_field

        self.degrees = bool(degrees)
        self.euler_seq = euler_seq
        self.quat_order = quat_order

        self.report_field = report_field

        # 旋转校正缓存：param -> "__param__"，stats_json -> embodiment 名
        self._corr_cache = {}

    # ------------------------------------------------------------------ #
    # 布局与校验
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_spec(spec: dict) -> dict:
        """规整单个 pose block spec，校验 rot_type 与切片长度。"""
        if not isinstance(spec, dict):
            raise ValueError(f"pose_layout entry must be dict, got {type(spec)}")
        target = spec.get("target", "state")
        if target not in ("state", "action"):
            raise ValueError(f"pose_layout target must be state|action, got {target}")
        rot_type = spec.get("rot_type", "quat")
        if rot_type not in _VALID_ROT_TYPES:
            raise ValueError(f"Invalid rot_type: {rot_type}. Valid: {_VALID_ROT_TYPES}")
        out = {
            "target": target,
            "pos": list(spec["pos"]) if spec.get("pos") is not None else None,
            "rot": list(spec["rot"]) if spec.get("rot") is not None else None,
            "rot_type": rot_type,
            "is_delta": bool(spec.get("is_delta", False)),
        }
        if out["pos"] is not None and (out["pos"][1] - out["pos"][0]) != 3:
            raise ValueError(f"pos slice must span 3 dims, got {out['pos']}")
        if out["rot"] is not None:
            need = {"euler": 3, "quat": 4, "rotvec": 3, "matrix": 9, "rot6d": 6}[rot_type]
            if (out["rot"][1] - out["rot"][0]) != need:
                raise ValueError(f"rot slice for {rot_type} must span {need} dims, got {out['rot']}")
        return out

    @staticmethod
    def _spec_in_bounds(spec: dict, dim: int) -> bool:
        """判断该 spec 的所有切片是否落在 [0, dim) 内。"""
        for key in ("pos", "rot"):
            sl = spec.get(key)
            if sl is not None and not (0 <= sl[0] < sl[1] <= dim):
                return False
        return True

    # ------------------------------------------------------------------ #
    # 旋转校正解析（param / preset / stats_json）
    # ------------------------------------------------------------------ #
    def _reorder_quat_to_xyzw(self, q: np.ndarray, order: str) -> np.ndarray:
        """把四元数从给定顺序统一到 scipy 需要的 xyzw。"""
        q = np.asarray(q, dtype=np.float64)
        if order == "wxyz":
            # (w,x,y,z) -> (x,y,z,w)
            return q[..., [1, 2, 3, 0]]
        return q

    def _reorder_quat_from_xyzw(self, q: np.ndarray, order: str) -> np.ndarray:
        """把 scipy 输出的 xyzw 四元数写回给定顺序。"""
        q = np.asarray(q, dtype=np.float64)
        if order == "wxyz":
            # (x,y,z,w) -> (w,x,y,z)
            return q[..., [3, 0, 1, 2]]
        return q

    def _rotation_from_spec(self, rot_type, value, degrees, euler_seq, quat_order):
        """把一个「单个」旋转描述转成 scipy Rotation。"""
        from scipy.spatial.transform import Rotation

        arr = np.asarray(value, dtype=np.float64)
        if rot_type == "euler":
            return Rotation.from_euler(euler_seq, arr, degrees=degrees)
        if rot_type == "quat":
            return Rotation.from_quat(self._reorder_quat_to_xyzw(arr, quat_order))
        if rot_type == "rotvec":
            return Rotation.from_rotvec(arr)
        if rot_type == "matrix":
            return Rotation.from_matrix(arr.reshape(3, 3))
        raise ValueError(f"Unsupported correction rot_type: {rot_type}")

    def _resolve_correction(self, sample):
        """按来源解析出本样本适用的 R_corr（scipy Rotation），带缓存。无则 None。"""
        from scipy.spatial.transform import Rotation

        if self.correction_source == "preset":
            key = f"__preset__:{self.preset_name}"
            if key not in self._corr_cache:
                self._corr_cache[key] = Rotation.from_euler("xyz", _PRESETS[self.preset_name], degrees=True)
            return self._corr_cache[key]

        if self.correction_source == "param":
            if "__param__" not in self._corr_cache:
                self._corr_cache["__param__"] = self._rotation_from_spec(
                    self.correction_type,
                    self.correction_value,
                    self.degrees,
                    self.euler_seq,
                    self.quat_order,
                )
            return self._corr_cache["__param__"]

        # stats_json：按 embodiment 查表
        emb = self.embodiment if self.embodiment is not None else sample.get(self.embodiment_field, "default")
        if emb in self._corr_cache:
            return self._corr_cache[emb]
        with open(self.correction_stats_path) as f:
            table = json.load(f)
        if emb not in table:
            logger.warning(f"Embodiment '{emb}' not in {self.correction_stats_path}; skip correction (identity).")
            self._corr_cache[emb] = None
            return None
        entry = table[emb]
        rot = self._rotation_from_spec(
            entry.get("type", "euler"),
            entry["value"],
            bool(entry.get("degrees", True)),
            entry.get("euler_seq", "xyz"),
            entry.get("quat_order", "xyzw"),
        )
        self._corr_cache[emb] = rot
        return rot

    # ------------------------------------------------------------------ #
    # 旋转表示 <-> 旋转矩阵 (T,3,3)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
        """6D 连续旋转 (T,6) -> 旋转矩阵 (T,3,3)。r6 = [col0(3), col1(3)]，Gram-Schmidt。"""
        r6 = np.asarray(r6, dtype=np.float64)
        a1 = r6[:, 0:3]
        a2 = r6[:, 3:6]
        b1 = a1 / (np.linalg.norm(a1, axis=1, keepdims=True) + 1e-12)
        proj = np.sum(b1 * a2, axis=1, keepdims=True) * b1
        b2 = a2 - proj
        b2 = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-12)
        b3 = np.cross(b1, b2)
        return np.stack([b1, b2, b3], axis=-1)  # 按列拼 -> (T,3,3)

    @staticmethod
    def _matrix_to_rot6d(mat: np.ndarray) -> np.ndarray:
        """旋转矩阵 (T,3,3) -> 6D 连续旋转 (T,6)，取前两列。"""
        return np.concatenate([mat[:, :, 0], mat[:, :, 1]], axis=1)

    def _rot_from_repr(self, block: np.ndarray, rot_type: str) -> np.ndarray:
        """任意旋转表示切片 (T,k) -> 旋转矩阵 (T,3,3)。"""
        from scipy.spatial.transform import Rotation

        block = np.asarray(block, dtype=np.float64)
        if rot_type == "euler":
            return Rotation.from_euler(self.euler_seq, block, degrees=self.degrees).as_matrix()
        if rot_type == "quat":
            return Rotation.from_quat(self._reorder_quat_to_xyzw(block, self.quat_order)).as_matrix()
        if rot_type == "rotvec":
            return Rotation.from_rotvec(block).as_matrix()
        if rot_type == "matrix":
            return block.reshape(-1, 3, 3)
        if rot_type == "rot6d":
            return self._rot6d_to_matrix(block)
        raise ValueError(f"Unsupported rot_type: {rot_type}")

    def _rot_to_repr(self, mat: np.ndarray, rot_type: str) -> np.ndarray:
        """旋转矩阵 (T,3,3) -> 指定旋转表示切片 (T,k)。"""
        from scipy.spatial.transform import Rotation

        if rot_type == "euler":
            return Rotation.from_matrix(mat).as_euler(self.euler_seq, degrees=self.degrees)
        if rot_type == "quat":
            q = Rotation.from_matrix(mat).as_quat()
            return self._reorder_quat_from_xyzw(q, self.quat_order)
        if rot_type == "rotvec":
            return Rotation.from_matrix(mat).as_rotvec()
        if rot_type == "matrix":
            return mat.reshape(-1, 9)
        if rot_type == "rot6d":
            return self._matrix_to_rot6d(mat)
        raise ValueError(f"Unsupported rot_type: {rot_type}")

    # ------------------------------------------------------------------ #
    # 对单个位姿块施加校正
    # ------------------------------------------------------------------ #
    def _apply_to_block(self, mat2d: np.ndarray, spec: dict, Rcorr) -> np.ndarray:
        """就地校正 (T,D) 向量中的一个 EEF 位姿块，返回新数组。"""
        out = mat2d.copy()
        Rc = np.asarray(Rcorr.as_matrix(), dtype=np.float64)  # (3,3)

        # 位置 / 平移：p' = R_corr · p （绝对与 delta 一致）
        if spec["pos"] is not None:
            i, j = spec["pos"]
            p = out[:, i:j]  # (T,3)
            out[:, i:j] = p @ Rc.T

        # 旋转：绝对左乘；delta 相似变换
        if spec["rot"] is not None:
            i, j = spec["rot"]
            M = self._rot_from_repr(out[:, i:j], spec["rot_type"])  # (T,3,3)
            if spec["is_delta"]:
                tmp = np.einsum("ab,tbc->tac", Rc, M)  # R_corr · ΔR
                Mn = np.einsum("tac,dc->tad", tmp, Rc)  # · R_corrᵀ
            else:
                Mn = np.einsum("ab,tbc->tac", Rc, M)  # R_corr · R
            out[:, i:j] = self._rot_to_repr(Mn, spec["rot_type"])
        return out

    # ------------------------------------------------------------------ #
    # 目标数组抽取 / 写回
    # ------------------------------------------------------------------ #
    def _apply_specs_to_array(self, arr_raw, target: str, Rcorr, transformed: list):
        """对一个 (T,D) 数组施加所有 target 匹配的 specs；越界则整块 pass-through。"""
        specs = [s for s in self.pose_layout if s["target"] == target]
        if not specs or arr_raw is None:
            return arr_raw, False
        mat = np.asarray(arr_raw, dtype=np.float64)
        if mat.ndim != 2:
            return arr_raw, False
        dim = mat.shape[1]
        # 任一 spec 越界 -> 该数组整体 pass-through（安全，兼容关节空间数据）
        for spec in specs:
            if not self._spec_in_bounds(spec, dim):
                return arr_raw, False
        for spec in specs:
            mat = self._apply_to_block(mat, spec, Rcorr)
            transformed.append(
                {
                    "target": target,
                    "rot": spec["rot"],
                    "pos": spec["pos"],
                    "rot_type": spec["rot_type"],
                    "is_delta": spec["is_delta"],
                }
            )
        return mat.tolist(), True

    # ------------------------------------------------------------------ #
    # Mapper 接口
    # ------------------------------------------------------------------ #
    def process_single(self, sample):
        # 无布局 -> 全量 pass-through
        if not self.pose_layout:
            return sample

        Rcorr = self._resolve_correction(sample)
        if Rcorr is None:
            return sample

        transformed = []
        changed = False

        if self.signal_source == "top_level":
            for target, key in (("state", self.top_level_state_key), ("action", self.top_level_action_key)):
                new_arr, ok = self._apply_specs_to_array(sample.get(key), target, Rcorr, transformed)
                if ok:
                    sample[key] = new_arr
                    changed = True
        else:  # hand_action_tags
            meta = sample.get(Fields.meta, {}) or {}
            tags = meta.get(self.hand_action_field)
            if isinstance(tags, str):
                tags = json.loads(tags)
            if tags:
                for hand in tags:
                    if not isinstance(hand, dict):
                        continue
                    for target, key in (("state", self.states_key), ("action", self.actions_key)):
                        new_arr, ok = self._apply_specs_to_array(hand.get(key), target, Rcorr, transformed)
                        if ok:
                            hand[key] = new_arr
                            changed = True
                meta[self.hand_action_field] = tags
                sample[Fields.meta] = meta

        # 写 meta 报告（JSON 字符串，规避 Arrow 嵌套 schema 冲突）
        emb = self.embodiment if self.embodiment is not None else sample.get(self.embodiment_field, "unknown")
        meta = sample.setdefault(Fields.meta, {})
        meta[self.report_field] = json.dumps(
            {
                "embodiment": emb,
                "correction_source": self.correction_source,
                "changed": bool(changed),
                "num_blocks": len(transformed),
                "blocks": transformed,
                "R_corr_matrix": np.asarray(Rcorr.as_matrix(), dtype=np.float64).round(8).tolist(),
            },
            ensure_ascii=False,
        )
        return sample
