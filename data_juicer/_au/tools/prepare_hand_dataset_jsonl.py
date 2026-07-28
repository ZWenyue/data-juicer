# -*- coding: utf-8 -*-
"""Build a Data-Juicer jsonl from egocentric videos or an EgoDex LeRobot root.

Examples:
  python -m data_juicer._au.tools.prepare_hand_dataset_jsonl \\
    --input /mnt/r/DATA/EgoDex/test_lerobot \\
    --output /tmp/egodex.jsonl --max-videos 4

  python -m data_juicer._au.tools.prepare_hand_dataset_jsonl \\
    --input /path/to/videos_or_jsonl --output /tmp/out.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _is_egodex_root(path: Path) -> bool:
    return (path / "meta" / "info.json").is_file() and (path / "videos").is_dir()


def _iter_videos_under(root: Path) -> List[Path]:
    found: List[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            found.append(p.resolve())
    return found


def collect_video_paths(input_path: Path) -> List[Path]:
    """Resolve input to a list of absolute video paths."""
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if input_path.is_file():
        if input_path.suffix.lower() == ".jsonl":
            videos: List[Path] = []
            for line in input_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                for v in row.get("videos") or []:
                    videos.append(Path(v).expanduser().resolve())
            return videos
        if input_path.suffix.lower() in VIDEO_EXTS:
            return [input_path]
        raise ValueError(f"Unsupported file type: {input_path}")

    # Directory: EgoDex LeRobot or a plain video folder
    if _is_egodex_root(input_path):
        ego = input_path / "videos" / "observation.images.ego"
        if ego.is_dir():
            return _iter_videos_under(ego)
        return _iter_videos_under(input_path / "videos")

    return _iter_videos_under(input_path)


def write_jsonl(videos: Iterable[Path], output: Path, text: str = "") -> int:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with output.open("w", encoding="utf-8") as f:
        for v in videos:
            if not v.is_file():
                raise FileNotFoundError(f"video missing: {v}")
            f.write(
                json.dumps(
                    {"videos": [str(v)], "text": text, "__dj__meta__": {}},
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
    return n


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="EgoDex LeRobot root, video dir, single mp4, or existing jsonl",
    )
    parser.add_argument("--output", "-o", required=True, type=Path, help="Output jsonl path")
    parser.add_argument("--max-videos", type=int, default=0, help="0 = all")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N videos")
    parser.add_argument("--text", default="", help="Optional default caption text")
    args = parser.parse_args(argv)

    videos = collect_video_paths(args.input)
    if args.offset:
        videos = videos[args.offset :]
    if args.max_videos and args.max_videos > 0:
        videos = videos[: args.max_videos]
    if not videos:
        raise SystemExit(f"No videos found under {args.input}")

    n = write_jsonl(videos, args.output, text=args.text)
    print(f"wrote {n} samples → {args.output.resolve()}")
    for v in videos[:5]:
        print(f"  - {v}")
    if n > 5:
        print(f"  ... +{n - 5} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
