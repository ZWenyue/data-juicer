# -*- coding: utf-8 -*-
"""Unit tests for R1 Lite arm MJCF asset builder."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_juicer._au.tools.build_r1_arm_mjcf import (
    DEFAULT_MESH_DIR,
    DEFAULT_URDF,
    VISUAL_GEOM_GROUP,
    build_all,
    resolve_visual_mesh,
)

MUJOCO_AVAILABLE = False
try:
    import mujoco  # noqa: F401

    MUJOCO_AVAILABLE = True
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parents[3]
HAS_ASSETS = DEFAULT_URDF.is_file() and DEFAULT_MESH_DIR.is_dir()


@unittest.skipUnless(HAS_ASSETS, "URDF/meshes not available")
class TestR1ArmAssetBuilder(unittest.TestCase):
    def test_resolve_visual_mesh_prefers_stl(self):
        path = resolve_visual_mesh(DEFAULT_MESH_DIR, "right_arm_link1")
        self.assertTrue(path.name.lower().endswith(".stl"))
        self.assertTrue(path.is_file())

    def test_build_manifest_and_static_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "generated"
            smoke_dir = Path(tmp) / "smoke"
            manifest = build_all(
                urdf_path=DEFAULT_URDF,
                mesh_dir=DEFAULT_MESH_DIR,
                out_dir=out_dir,
                sides=("left", "right"),
                smoke_dir=smoke_dir if MUJOCO_AVAILABLE else None,
            )
            self.assertEqual(manifest["schema_version"], 1)
            for side in ("left", "right"):
                xml_path = Path(manifest["sides"][side]["mjcf"])
                self.assertTrue(xml_path.is_file())
                self.assertTrue((out_dir / "intermediate" / f"r1_lite_arm_{side}.urdf").is_file())
                links = manifest["sides"][side]["links"]
                self.assertIn(f"{side}_arm_link1", links)
                self.assertIn(f"{side}_gripper_link", links)

            if MUJOCO_AVAILABLE:
                for side in ("left", "right"):
                    smoke = manifest["sides"][side]["smoke"]
                    self.assertIsNotNone(smoke)
                    self.assertGreater(smoke["mask_pixels"], 0)
                    self.assertGreater(smoke["rgb_nonzero"], 0)
                    # Load model and verify visual geom group.
                    model = mujoco.MjModel.from_xml_path(manifest["sides"][side]["mjcf"])
                    self.assertTrue(np.any(model.geom_group == VISUAL_GEOM_GROUP))
                    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_site")
                    self.assertGreaterEqual(site_id, 0)

            # Manifest JSON is readable.
            man_path = out_dir / "asset_manifest.json"
            payload = json.loads(man_path.read_text())
            self.assertIn("sides", payload)


if __name__ == "__main__":
    unittest.main()
