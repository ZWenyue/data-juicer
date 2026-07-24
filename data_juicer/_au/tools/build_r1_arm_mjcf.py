# -*- coding: utf-8 -*-
"""Build left/right R1 Lite arm MJCF assets from URDF + STL meshes.

Generates:
  - b/d/urdf/generated/intermediate/r1_lite_arm_{side}.urdf
  - b/d/urdf/generated/r1_lite_arm_{side}.xml
  - b/d/urdf/generated/asset_manifest.json
  - optional smoke RGB/mask/depth images under --smoke-dir

P0 policy:
  - STL files are used as visual meshes (URDF historically referenced .obj).
  - Collision uses auto capsule proxies (not dense STL).
  - D405 wrist camera body is omitted (no mesh in current asset pack).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Arm kinematic chain (parent of base is a free mocap/world root in MJCF).
ARM_JOINT_ORDER = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
VISUAL_GEOM_GROUP = 1
COLLISION_GEOM_GROUP = 3

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_URDF = DEFAULT_REPO_ROOT / "b" / "d" / "urdf" / "r1_lite.urdf"
DEFAULT_MESH_DIR = DEFAULT_REPO_ROOT / "b" / "d" / "urdf" / "meshes"
DEFAULT_OUT_DIR = DEFAULT_REPO_ROOT / "b" / "d" / "urdf" / "generated"


@dataclass(frozen=True)
class JointSpec:
    name: str
    jtype: str
    parent: str
    child: str
    xyz: Tuple[float, float, float]
    rpy: Tuple[float, float, float]
    axis: Tuple[float, float, float]
    lower: Optional[float]
    upper: Optional[float]
    effort: Optional[float]
    velocity: Optional[float]


def _parse_xyz(text: Optional[str]) -> Tuple[float, float, float]:
    if not text:
        return (0.0, 0.0, 0.0)
    vals = [float(x) for x in text.split()]
    if len(vals) != 3:
        raise ValueError(f"Invalid xyz/rpy/axis '{text}'")
    return (vals[0], vals[1], vals[2])


def _fmt_vec(vec: Sequence[float]) -> str:
    return " ".join(f"{float(v):.8g}" for v in vec)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_urdf_joints(urdf_path: Path) -> Dict[str, JointSpec]:
    root = ET.parse(urdf_path).getroot()
    specs: Dict[str, JointSpec] = {}
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        specs[joint.get("name")] = JointSpec(
            name=joint.get("name"),
            jtype=joint.get("type"),
            parent=joint.find("parent").get("link"),
            child=joint.find("child").get("link"),
            xyz=_parse_xyz(origin.get("xyz") if origin is not None else None),
            rpy=_parse_xyz(origin.get("rpy") if origin is not None else None),
            axis=_parse_xyz(axis.get("xyz") if axis is not None else "0 0 1"),
            lower=float(limit.get("lower")) if limit is not None and limit.get("lower") is not None else None,
            upper=float(limit.get("upper")) if limit is not None and limit.get("upper") is not None else None,
            effort=float(limit.get("effort")) if limit is not None and limit.get("effort") is not None else None,
            velocity=float(limit.get("velocity")) if limit is not None and limit.get("velocity") is not None else None,
        )
    return specs


def _arm_links(side: str) -> List[str]:
    return [
        f"{side}_arm_base_link",
        f"{side}_arm_link1",
        f"{side}_arm_link2",
        f"{side}_arm_link3",
        f"{side}_arm_link4",
        f"{side}_arm_link5",
        f"{side}_arm_link6",
        f"{side}_gripper_link",
        f"{side}_gripper_finger_link1",
        f"{side}_gripper_finger_link2",
    ]


def _arm_joint_names(side: str) -> List[str]:
    return [
        f"{side}_arm_joint1",
        f"{side}_arm_joint2",
        f"{side}_arm_joint3",
        f"{side}_arm_joint4",
        f"{side}_arm_joint5",
        f"{side}_arm_joint6",
        f"{side}_gripper_joint",
        f"{side}_gripper_finger_joint1",
        f"{side}_gripper_finger_joint2",
    ]


def resolve_visual_mesh(mesh_dir: Path, link_name: str) -> Path:
    """Map URDF visual *.obj references onto available *.STL assets."""
    candidates = [
        mesh_dir / f"{link_name}.STL",
        mesh_dir / f"{link_name}.stl",
        mesh_dir / f"{link_name}.obj",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No visual mesh for link '{link_name}' under {mesh_dir}")


def _capsule_from_aabb(mins: np.ndarray, maxs: np.ndarray) -> Tuple[float, Tuple[float, float, float]]:
    """Approximate an AABB with a capsule along the longest axis.

    Returns (radius, fromto) where fromto is absolute in the link frame
    (MuJoCo forbids combining pos= with fromto=).
    """
    extents = maxs - mins
    axis = int(np.argmax(extents))
    radius = 0.5 * float(np.max(np.delete(extents, axis)))
    radius = max(radius, 1e-3)
    length = max(float(extents[axis]) - 2.0 * radius, 1e-3)
    center = 0.5 * (mins + maxs)
    half = 0.5 * length
    start = center.copy()
    end = center.copy()
    start[axis] -= half
    end[axis] += half
    fromto = (float(start[0]), float(start[1]), float(start[2]), float(end[0]), float(end[1]), float(end[2]))
    return radius, fromto


def _mesh_aabb(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read ASCII/binary STL bounds without heavy deps."""
    raw = path.read_bytes()
    # Binary STL: 80-byte header + uint32 triangle count.
    if len(raw) >= 84:
        n_tri = int.from_bytes(raw[80:84], "little", signed=False)
        expected = 84 + n_tri * 50
        if expected == len(raw) and n_tri > 0:
            verts = []
            offset = 84
            for _ in range(n_tri):
                # normal(12) + 3*vertex(36) + attr(2)
                for i in range(3):
                    base = offset + 12 + i * 12
                    verts.append(np.frombuffer(raw[base : base + 12], dtype="<f4"))
                offset += 50
            arr = np.vstack(verts)
            return arr.min(axis=0), arr.max(axis=0)

    # ASCII fallback.
    text = raw.decode("utf-8", errors="ignore")
    verts = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("vertex"):
            parts = line.split()
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not verts:
        raise ValueError(f"Failed to parse STL AABB: {path}")
    arr = np.asarray(verts, dtype=np.float64)
    return arr.min(axis=0), arr.max(axis=0)


def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        for child in elem:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = indent


def build_intermediate_urdf(
    side: str,
    joints: Dict[str, JointSpec],
    mesh_dir: Path,
    out_path: Path,
) -> Dict[str, dict]:
    """Write a trimmed URDF that remaps visuals to available STL files."""
    robot = ET.Element("robot", name=f"r1_lite_arm_{side}")
    mujoco = ET.SubElement(robot, "mujoco")
    ET.SubElement(mujoco, "compiler", meshdir="../meshes", discardvisual="false")

    link_manifest: Dict[str, dict] = {}
    for link in _arm_links(side):
        mesh_path = resolve_visual_mesh(mesh_dir, link)
        link_el = ET.SubElement(robot, "link", name=link)
        visual = ET.SubElement(link_el, "visual")
        ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
        geom = ET.SubElement(visual, "geometry")
        ET.SubElement(geom, "mesh", filename=f"meshes/{mesh_path.name}")
        mins, maxs = _mesh_aabb(mesh_path)
        radius, fromto = _capsule_from_aabb(mins, maxs)
        collision = ET.SubElement(link_el, "collision")
        ET.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
        cgeom = ET.SubElement(collision, "geometry")
        # Keep URDF collision as box approx for documentation; MJCF uses capsule.
        size = 0.5 * (maxs - mins)
        ET.SubElement(cgeom, "box", size=_fmt_vec(size))
        link_manifest[link] = {
            "visual": str(mesh_path.as_posix()),
            "collision": "primitive:capsule",
            "scale": [1.0, 1.0, 1.0],
            "aabb_min": mins.tolist(),
            "aabb_max": maxs.tolist(),
            "capsule_radius": radius,
            "capsule_fromto_local": list(fromto),
            "sha256": _sha256_file(mesh_path),
        }

    for jname in _arm_joint_names(side):
        spec = joints[jname]
        j_el = ET.SubElement(robot, "joint", name=spec.name, type=spec.jtype)
        ET.SubElement(j_el, "origin", xyz=_fmt_vec(spec.xyz), rpy=_fmt_vec(spec.rpy))
        ET.SubElement(j_el, "parent", link=spec.parent)
        ET.SubElement(j_el, "child", link=spec.child)
        if spec.jtype != "fixed":
            ET.SubElement(j_el, "axis", xyz=_fmt_vec(spec.axis))
            attrs = {}
            if spec.lower is not None:
                attrs["lower"] = f"{spec.lower:.8g}"
            if spec.upper is not None:
                attrs["upper"] = f"{spec.upper:.8g}"
            if spec.effort is not None:
                attrs["effort"] = f"{spec.effort:.8g}"
            if spec.velocity is not None:
                attrs["velocity"] = f"{spec.velocity:.8g}"
            if attrs:
                ET.SubElement(j_el, "limit", **attrs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _indent_xml(robot)
    ET.ElementTree(robot).write(out_path, encoding="utf-8", xml_declaration=True)
    return link_manifest


def _fill_link_geoms(body: ET.Element, side: str, link_name: str, link_manifest: Dict[str, dict]) -> None:
    ET.SubElement(
        body,
        "geom",
        name=f"{link_name}_visual",
        type="mesh",
        mesh=link_name,
        group=str(VISUAL_GEOM_GROUP),
        contype="0",
        conaffinity="0",
        rgba="0.72 0.74 0.78 1",
    )
    meta = link_manifest[link_name]
    ET.SubElement(
        body,
        "geom",
        name=f"{link_name}_collision",
        type="capsule",
        group=str(COLLISION_GEOM_GROUP),
        contype="0",
        conaffinity="0",
        size=f"{meta['capsule_radius']:.8g}",
        fromto=_fmt_vec(meta["capsule_fromto_local"]),
        rgba="0.2 0.8 0.2 0.25",
    )


def _add_child_joint_body(
    parent_body: ET.Element,
    side: str,
    spec: JointSpec,
    child_joints: Dict[str, List[JointSpec]],
    link_manifest: Dict[str, dict],
) -> None:
    child_body = ET.SubElement(parent_body, "body", name=spec.child, pos=_fmt_vec(spec.xyz))
    if any(abs(v) > 1e-12 for v in spec.rpy):
        child_body.set("euler", _fmt_vec(spec.rpy))

    if spec.jtype == "revolute":
        jattrs = {"name": spec.name, "type": "hinge", "axis": _fmt_vec(spec.axis)}
        if spec.lower is not None and spec.upper is not None:
            jattrs["range"] = f"{spec.lower:.8g} {spec.upper:.8g}"
        ET.SubElement(child_body, "joint", **jattrs)
    elif spec.jtype == "prismatic":
        jattrs = {"name": spec.name, "type": "slide", "axis": _fmt_vec(spec.axis)}
        if spec.lower is not None and spec.upper is not None:
            jattrs["range"] = f"{spec.lower:.8g} {spec.upper:.8g}"
        ET.SubElement(child_body, "joint", **jattrs)
    elif spec.jtype != "fixed":
        raise ValueError(f"Unsupported joint type: {spec.jtype}")

    _fill_link_geoms(child_body, side, spec.child, link_manifest)
    if spec.child == f"{side}_gripper_link":
        ET.SubElement(child_body, "site", name="gripper_site", pos="0 0 0", size="0.008", rgba="1 0 0 1")
    for grand in child_joints.get(spec.child, []):
        _add_child_joint_body(child_body, side, grand, child_joints, link_manifest)


def build_mjcf(
    side: str,
    joints: Dict[str, JointSpec],
    mesh_dir: Path,
    out_xml: Path,
    link_manifest: Dict[str, dict],
) -> None:
    child_joints: Dict[str, List[JointSpec]] = {}
    for jname in _arm_joint_names(side):
        spec = joints[jname]
        child_joints.setdefault(spec.parent, []).append(spec)

    mesh_dir = mesh_dir.resolve()
    out_xml = out_xml.resolve()
    try:
        meshdir_attr = os.path.relpath(mesh_dir, start=out_xml.parent)
    except ValueError:
        meshdir_attr = str(mesh_dir)

    root = ET.Element("mujoco", model=f"r1_lite_arm_{side}")
    ET.SubElement(
        root,
        "compiler",
        angle="radian",
        meshdir=meshdir_attr,
        autolimits="true",
        eulerseq="xyz",
    )
    ET.SubElement(root, "option", timestep="0.002", gravity="0 0 0")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="720")
    ET.SubElement(root, "default")

    asset = ET.SubElement(root, "asset")
    for link in _arm_links(side):
        mesh_path = resolve_visual_mesh(mesh_dir, link)
        ET.SubElement(asset, "mesh", name=link, file=mesh_path.name)

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(
        world,
        "camera",
        name="ego_cam",
        pos="0 0 0",
        xyaxes="1 0 0 0 -1 0",
        fovy="60",
    )
    # Soft ambient light for smoke renders.
    ET.SubElement(world, "light", name="top", pos="0 0 1.5", dir="0 0 -1", diffuse="0.8 0.8 0.8")

    anchor = ET.SubElement(world, "body", name="robot_anchor", mocap="true")
    base_name = f"{side}_arm_base_link"
    base_body = ET.SubElement(anchor, "body", name=base_name)
    _fill_link_geoms(base_body, side, base_name, link_manifest)
    for spec in child_joints.get(base_name, []):
        _add_child_joint_body(base_body, side, spec, child_joints, link_manifest)

    out_xml.parent.mkdir(parents=True, exist_ok=True)
    _indent_xml(root)
    ET.ElementTree(root).write(out_xml, encoding="utf-8", xml_declaration=True)


def smoke_render(xml_path: Path, smoke_dir: Path, side: str, width: int = 640, height: int = 480) -> dict:
    import mujoco

    os.environ.setdefault("MUJOCO_GL", "egl")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # Mid-range arm pose.
    q_arm = np.array([0.0, 1.2, -1.5, 0.0, 0.0, 0.0], dtype=np.float64)
    finger = 0.025
    # qpos layout: 6 arm hinges + 2 fingers (mocap does not consume qpos).
    assert model.nq >= 8, f"Unexpected nq={model.nq}"
    data.qpos[:6] = q_arm
    data.qpos[6] = finger
    data.qpos[7] = -finger

    # Place base slightly below/side of camera in MuJoCo camera frame.
    # OpenCV camera_to_base_ref for right: [0, -0.3, -0.2] (x,y,z) with y rightward-down?
    # Design cam space: right arm [0, +0.3, -0.2]. After OpenCV->MuJoCo (180deg X):
    # (x, y, z)_cv -> (x, -y, -z)_mj
    if side == "right":
        t_cv = np.array([0.0, 0.3, -0.2])
    else:
        t_cv = np.array([0.0, -0.3, -0.2])
    t_mj = np.array([t_cv[0], -t_cv[1], -t_cv[2]])
    data.mocap_pos[0] = t_mj
    data.mocap_quat[0] = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera="ego_cam")
    rgb = renderer.render().copy()

    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera="ego_cam")
    seg = renderer.render()
    renderer.disable_segmentation_rendering()
    geom_ids = seg[:, :, 0]
    visual_ids = np.flatnonzero(model.geom_group == VISUAL_GEOM_GROUP)
    mask = np.isin(geom_ids, visual_ids)

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="ego_cam")
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    renderer.close()

    smoke_dir.mkdir(parents=True, exist_ok=True)
    try:
        import cv2

        cv2.imwrite(str(smoke_dir / f"{side}_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(smoke_dir / f"{side}_mask.png"), (mask.astype(np.uint8) * 255))
        depth_vis = depth.copy()
        finite = np.isfinite(depth_vis) & (depth_vis > 0) & (depth_vis < 10)
        if finite.any():
            d = depth_vis[finite]
            depth_vis = np.clip((depth_vis - d.min()) / max(d.max() - d.min(), 1e-6), 0, 1)
        cv2.imwrite(str(smoke_dir / f"{side}_depth.png"), (depth_vis * 255).astype(np.uint8))
    except Exception:
        np.save(smoke_dir / f"{side}_rgb.npy", rgb)
        np.save(smoke_dir / f"{side}_mask.npy", mask)
        np.save(smoke_dir / f"{side}_depth.npy", depth)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_site")
    site_pos = data.site_xpos[site_id].copy()
    return {
        "rgb_nonzero": int(np.count_nonzero(rgb)),
        "mask_pixels": int(mask.sum()),
        "depth_finite": int(np.isfinite(depth).sum()),
        "gripper_site_pos": site_pos.tolist(),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "ngeom": int(model.ngeom),
    }


def build_side(
    side: str,
    urdf_path: Path,
    mesh_dir: Path,
    out_dir: Path,
    smoke_dir: Optional[Path] = None,
) -> dict:
    joints = _load_urdf_joints(urdf_path)
    missing = [n for n in _arm_joint_names(side) if n not in joints]
    if missing:
        raise KeyError(f"URDF missing joints for {side}: {missing}")

    intermediate = out_dir / "intermediate" / f"r1_lite_arm_{side}.urdf"
    xml_path = out_dir / f"r1_lite_arm_{side}.xml"
    link_manifest = build_intermediate_urdf(side, joints, mesh_dir, intermediate)
    build_mjcf(side, joints, mesh_dir, xml_path, link_manifest)

    smoke = None
    if smoke_dir is not None:
        smoke = smoke_render(xml_path, smoke_dir, side)

    return {
        "side": side,
        "intermediate_urdf": str(intermediate),
        "mjcf": str(xml_path),
        "mjcf_sha256": _sha256_file(xml_path),
        "links": link_manifest,
        "smoke": smoke,
    }


def build_all(
    urdf_path: Path = DEFAULT_URDF,
    mesh_dir: Path = DEFAULT_MESH_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    sides: Sequence[str] = ("left", "right"),
    smoke_dir: Optional[Path] = None,
) -> dict:
    results = {}
    for side in sides:
        results[side] = build_side(side, urdf_path, mesh_dir, out_dir, smoke_dir=smoke_dir)

    manifest = {
        "schema_version": 1,
        "robot": "r1_lite",
        "source_urdf": str(urdf_path),
        "mesh_dir": str(mesh_dir),
        "visual_geom_group": VISUAL_GEOM_GROUP,
        "collision_geom_group": COLLISION_GEOM_GROUP,
        "sides": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = _sha256_file(manifest_path)
    # Rewrite with hash of content without circular hash field.
    payload = {k: v for k, v in manifest.items() if k not in ("manifest_path", "manifest_sha256")}
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    manifest["manifest_sha256"] = _sha256_file(manifest_path)
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--mesh-dir", type=Path, default=DEFAULT_MESH_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sides", nargs="+", default=["left", "right"], choices=["left", "right"])
    parser.add_argument("--smoke-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    manifest = build_all(
        urdf_path=args.urdf,
        mesh_dir=args.mesh_dir,
        out_dir=args.out_dir,
        sides=args.sides,
        smoke_dir=args.smoke_dir,
    )
    print(json.dumps({k: manifest[k] for k in ("schema_version", "robot", "manifest_sha256") if k in manifest}, indent=2))
    for side in args.sides:
        smoke = manifest["sides"][side].get("smoke")
        print(f"[{side}] mjcf={manifest['sides'][side]['mjcf']}")
        if smoke:
            print(f"  smoke mask_pixels={smoke['mask_pixels']} gripper_site={smoke['gripper_site_pos']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
