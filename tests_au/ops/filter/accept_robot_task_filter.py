#!/usr/bin/env python3
"""Acceptance check for task-level filtering on a real LeRobot dataset."""

import argparse
import json
from pathlib import Path

from data_juicer._au.pipeline.filter_robot_tasks import filter_task_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = filter_task_root(
        args.dataset,
        args.output,
        include_keywords=["router cable", "路由器", "router"],
        mode="symlink",
    )
    if result["selected_tasks"] != 1:
        raise AssertionError(f"Expected the real router-cable task to be selected, got {result}")

    selected = args.output / args.dataset.name
    if not selected.is_symlink() or selected.resolve() != args.dataset.resolve():
        raise AssertionError(f"Selected task link is invalid: {selected}")

    summary = json.loads((args.output / "task_filter_summary.json").read_text(encoding="utf-8"))
    if summary["scanned_tasks"] != 1 or summary["selected_tasks"] != 1:
        raise AssertionError(f"Unexpected summary: {summary}")

    print("ACCEPTANCE PASSED: selected complete task " f"{summary['selected_task_names'][0]}")


if __name__ == "__main__":
    main()
