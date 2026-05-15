#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("Arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip())
    return slug or "cluster"


def _make_sheet(paths: list[tuple[str, Path]], output_path: Path, *, cols: int, thumb_width: int) -> None:
    font = _font(18)
    label_h = 28
    pad = 12
    thumbs: list[tuple[str, Image.Image]] = []
    for label, path in paths:
        if not path.exists():
            continue
        with Image.open(path) as image:
            image = image.convert("RGB")
            scale = thumb_width / image.width
            thumb = image.resize((thumb_width, max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
            thumbs.append((label, thumb))

    if not thumbs:
        raise FileNotFoundError(f"No input images found for {output_path}")

    cell_w = thumb_width + 2 * pad
    cell_h = max(image.height for _, image in thumbs) + label_h + 2 * pad
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)

    for idx, (label, image) in enumerate(thumbs):
        row = idx // cols
        col = idx % cols
        x = col * cell_w + pad
        y = row * cell_h + pad
        draw.text((x, y), label[:42], fill="black", font=font)
        sheet.paste(image, (x, y + label_h))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build contact sheets for selected Chronos plot PNGs.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", default="parsec")
    parser.add_argument("--selected", type=Path, default=None)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--thumb-width", type=int, default=560)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    selected_path = args.selected or run_dir / "analysis" / "selected_clusters_for_visual_review.txt"
    names = [line.strip() for line in selected_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    posterior_paths = [
        (name, run_dir / args.model / "posterior_plots" / f"{_slugify(name)}_posterior.png")
        for name in names
    ]
    isochrone_paths = [
        (name, run_dir / args.model / "isochrone_plots" / f"{_slugify(name)}_isochrone.png")
        for name in names
    ]

    analysis_dir = run_dir / "analysis"
    _make_sheet(
        posterior_paths,
        analysis_dir / "selected_posterior_contact_sheet.png",
        cols=args.cols,
        thumb_width=args.thumb_width,
    )
    _make_sheet(
        isochrone_paths,
        analysis_dir / "selected_isochrone_contact_sheet.png",
        cols=args.cols,
        thumb_width=args.thumb_width,
    )
    print(analysis_dir / "selected_posterior_contact_sheet.png")
    print(analysis_dir / "selected_isochrone_contact_sheet.png")


if __name__ == "__main__":
    main()
