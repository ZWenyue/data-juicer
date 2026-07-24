# -*- coding: utf-8 -*-
"""Select complete LeRobot task directories by task-level keywords."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence

from data_juicer.utils.constant import Fields

from ..ops.filter.robot_task_filter import RobotTaskFilter
from ..utils.embodiment_prompt import load_jsonl


def is_lerobot_task(path: Path) -> bool:
    return (path / "data").is_dir() and (path / "meta" / "info.json").is_file()


def discover_tasks(root: Path) -> List[Path]:
    """Return a LeRobot root itself or its immediate task children."""
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    if is_lerobot_task(root):
        return [root]

    tasks = sorted(path for path in root.iterdir() if path.is_dir() and is_lerobot_task(path))
    if not tasks:
        raise FileNotFoundError(f"No LeRobot task directories under {root} " "(expected data/ and meta/info.json)")
    return tasks


def load_task_labels(task_dir: Path) -> List[str]:
    labels = []
    for row in load_jsonl(task_dir / "meta" / "tasks.jsonl"):
        label = str(row.get("task", "") or "").strip()
        if label:
            labels.append(label)
    return labels


def evaluate_task(task_dir: Path, task_filter: RobotTaskFilter) -> dict:
    sample = {
        "task_name": task_dir.name,
        "task_labels": load_task_labels(task_dir),
        Fields.stats: {},
        Fields.meta: {},
    }
    task_filter.compute_stats_single(sample)
    report = json.loads(sample[Fields.meta][task_filter.report_field])
    return {
        "task_name": task_dir.name,
        "source": str(task_dir),
        "task_labels": sample["task_labels"],
        **report,
    }


def materialize_task(
    task_dir: Path,
    output_dir: Path,
    mode: str,
    on_existing: str,
) -> None:
    destination = output_dir / task_dir.name
    if destination.exists() or destination.is_symlink():
        if on_existing == "skip":
            return
        raise FileExistsError(
            f"Destination already exists: {destination}; " "use --on-existing skip or choose an empty output directory"
        )

    if mode == "symlink":
        destination.symlink_to(task_dir.resolve(), target_is_directory=True)
    elif mode == "copy":
        shutil.copytree(task_dir, destination, symlinks=True)
    elif mode != "list":
        raise ValueError(f"Unsupported output mode: {mode}")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def filter_task_root(
    dataset_root: Path,
    output_dir: Path,
    *,
    skill: str = "",
    include_keywords: Sequence[str] | None = None,
    exclude_keywords: Sequence[str] | None = None,
    mode: str = "symlink",
    case_sensitive: bool = False,
    include_task_labels: bool = True,
    on_existing: str = "error",
    dry_run: bool = False,
) -> dict:
    task_filter = RobotTaskFilter(
        skill=skill,
        include_keywords=include_keywords,
        exclude_keywords=exclude_keywords,
        case_sensitive=case_sensitive,
        text_fields=(("task_name", "task_labels") if include_task_labels else ("task_name",)),
    )
    task_dirs = discover_tasks(dataset_root)
    results = [evaluate_task(task_dir, task_filter) for task_dir in task_dirs]
    selected = [row for row in results if row["keep"]]

    if not dry_run:
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        for row in selected:
            materialize_task(Path(row["source"]), output_dir, mode=mode, on_existing=on_existing)
        write_jsonl(output_dir / "task_filter_results.jsonl", results)
        (output_dir / "task_filter_summary.json").write_text(
            json.dumps(
                {
                    "dataset_root": str(dataset_root.expanduser().resolve()),
                    "output_dir": str(output_dir),
                    "skill": skill,
                    "include_keywords": task_filter.include_keywords,
                    "exclude_keywords": task_filter.exclude_keywords,
                    "include_task_labels": include_task_labels,
                    "mode": mode,
                    "scanned_tasks": len(results),
                    "selected_tasks": len(selected),
                    "selected_task_names": [row["task_name"] for row in selected],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    return {
        "results": results,
        "selected": selected,
        "scanned_tasks": len(results),
        "selected_tasks": len(selected),
    }


def _split_keywords(values: Sequence[str] | None) -> List[str]:
    result = []
    for value in values or ():
        result.extend(word.strip() for word in value.split(",") if word.strip())
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter complete LeRobot task directories by task-level text.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skill",
        default="",
        help="Built-in skill vocabulary. Currently supported: open_door.",
    )
    parser.add_argument(
        "--keyword",
        action="append",
        default=[],
        help="Include keyword; repeat the option or pass a comma-separated list.",
    )
    parser.add_argument(
        "--exclude-keyword",
        action="append",
        default=[],
        help="Reject a task if this keyword occurs; repeat or comma-separate.",
    )
    parser.add_argument("--mode", choices=("symlink", "copy", "list"), default="symlink")
    parser.add_argument("--on-existing", choices=("error", "skip"), default="error")
    parser.add_argument("--case-sensitive", action="store_true")
    parser.add_argument(
        "--include-task-labels",
        dest="include_task_labels",
        action="store_true",
        default=True,
        help="Also match every meta/tasks.jsonl label (default).",
    )
    parser.add_argument(
        "--task-name-only",
        dest="include_task_labels",
        action="store_false",
        help="Only match the task directory name.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = filter_task_root(
        args.dataset_root,
        args.output,
        skill=args.skill,
        include_keywords=_split_keywords(args.keyword),
        exclude_keywords=_split_keywords(args.exclude_keyword),
        mode=args.mode,
        case_sensitive=args.case_sensitive,
        include_task_labels=args.include_task_labels,
        on_existing=args.on_existing,
        dry_run=args.dry_run,
    )

    for row in result["results"]:
        status = "KEEP" if row["keep"] else "SKIP"
        matches = ", ".join(row["include_matches"]) or "-"
        print(f"[{status}] {row['task_name']} (matches: {matches})")
    print(f"Scanned {result['scanned_tasks']} tasks; " f"selected {result['selected_tasks']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
