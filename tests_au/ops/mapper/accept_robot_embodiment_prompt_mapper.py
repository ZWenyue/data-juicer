# -*- coding: utf-8 -*-
"""Acceptance checks for robot_embodiment_prompt_mapper + export prompt_fields."""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path


REQUIRED_KEYS = (
    "embodiment",
    "instruction",
    "length",
    "speed",
    "fps",
    "camera_view_direction",
)


def _load_result_rows(export_glob: str):
    files = sorted(glob.glob(export_glob))
    if not files:
        raise FileNotFoundError(f"No export matching {export_glob}")
    rows = []
    for p in files:
        if "stats" in Path(p).name:
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if rows:
            return p, rows
    raise RuntimeError(f"Empty export under {export_glob}")


def verify_dj_export(export_glob: str) -> None:
    path, rows = _load_result_rows(export_glob)
    errors = []
    for r in rows:
        ep = r.get("id", "?")
        meta = r.get("__dj__meta__") or {}
        raw = meta.get("prompt_fields")
        if not raw:
            errors.append(f"{ep}: missing meta.prompt_fields")
            continue
        fields = json.loads(raw) if isinstance(raw, str) else raw
        for k in REQUIRED_KEYS:
            if k not in fields:
                errors.append(f"{ep}: missing prompt_fields.{k}")
        if fields.get("length") is not None and int(fields["length"]) <= 0:
            errors.append(f"{ep}: length must be positive, got {fields.get('length')}")
        if fields.get("speed") is not None and fields.get("length") is not None:
            length = int(fields["length"])
            expected = int(math.ceil(length / 500) * 500)
            if int(fields["speed"]) != expected:
                errors.append(
                    f"{ep}: speed={fields['speed']} != ceil(length/500)*500={expected}"
                )
        if not fields.get("instruction"):
            errors.append(f"{ep}: empty instruction")
        view = fields.get("camera_view_direction")
        if view not in ("arm side", "opposite side"):
            errors.append(f"{ep}: bad camera_view_direction={view!r}")
        dbg = meta.get("embodiment_prompt_debug") or ""
        if "embodiment:" not in dbg:
            errors.append(f"{ep}: missing embodiment_prompt_debug")

    if errors:
        for e in errors:
            print("  FAIL:", e, file=sys.stderr)
        raise SystemExit(1)
    print(f"DJ export ok: {len(rows)} samples from {path}")
    print("sample prompt_fields:", json.loads(rows[0]["__dj__meta__"]["prompt_fields"]))


def verify_episodes_jsonl(episodes_path: str) -> None:
    path = Path(episodes_path)
    if not path.is_file():
        raise FileNotFoundError(episodes_path)
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert rows, f"empty {path}"
    errors = []
    for row in rows:
        ep = row.get("episode_index", "?")
        fields = row.get("prompt_fields")
        if not isinstance(fields, dict):
            errors.append(f"ep{ep}: missing prompt_fields dict")
            continue
        for k in REQUIRED_KEYS:
            if k not in fields:
                errors.append(f"ep{ep}: missing {k}")
        if fields.get("speed") is not None and fields.get("length") is not None:
            length = int(fields["length"])
            expected = int(math.ceil(length / 500) * 500)
            if int(fields["speed"]) != expected:
                errors.append(
                    f"ep{ep}: speed={fields['speed']} != expected {expected}"
                )
    if errors:
        for e in errors:
            print("  FAIL:", e, file=sys.stderr)
        raise SystemExit(1)
    print(f"episodes.jsonl ok: {len(rows)} rows with prompt_fields ({path})")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dj-export-glob",
        default="tests_au/ops/mapper/outputs/accept_prompt_result.jsonl*",
    )
    p.add_argument("--episodes-jsonl", default=None)
    args = p.parse_args(argv)
    verify_dj_export(args.dj_export_glob)
    if args.episodes_jsonl:
        verify_episodes_jsonl(args.episodes_jsonl)
    print("ACCEPTANCE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
