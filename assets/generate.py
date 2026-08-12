"""Generate a block-based SVG asset from a blocks.txt source."""

import argparse
from math import radians, tan
from pathlib import Path


def parse_blocks(source: str) -> list[str]:
    return source.strip().splitlines()


def bounds(rows: list[str], block: int, grid: int, skew: int) -> tuple[float, float]:
    shear = tan(radians(skew))
    blocks = [
        (column * grid + shear * row * grid, column * grid + block + shear * row * grid)
        for row, line in enumerate(rows)
        for column, value in enumerate(line)
        if value == "#"
    ]
    return min(left for left, _ in blocks), max(right for _, right in blocks)


def render(
    input_path: Path,
    width: int,
    height: int,
    block: int,
    grid: int,
    skew: int,
    baseline: int,
    title: str,
    description: str,
) -> str:
    rows = parse_blocks(input_path.read_text())
    left, right = bounds(rows, block, grid, skew)
    x = (width - (right - left)) / 2 - left
    y = baseline - ((len(rows) - 1) * grid + block)
    blocks = "\n      ".join(
        f'<use href="#block" x="{column * grid}" y="{row * grid}"/>'
        for row, line in enumerate(rows)
        for column, value in enumerate(line)
        if value == "#"
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">{title}</title>
  <desc id="desc">{description}</desc>
  <rect width="{width}" height="{height}" fill="#000"/>
  <defs>
    <g id="block">
      <rect width="{block}" height="{block}" rx="3" fill="#fff" stroke="#fff" stroke-width="1"/>
    </g>
  </defs>
  <g aria-label="{title}">
    <g transform="translate({x:.1f} {y}) skewX({skew})">
      {blocks}
    </g>
  </g>
</svg>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--block", type=int, default=27)
    parser.add_argument("--grid", type=int, default=24)
    parser.add_argument("--skew", type=int, default=-16)
    parser.add_argument("--baseline", type=int, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--description", required=True)
    args = parser.parse_args()
    args.output.write_text(
        render(
            args.input,
            args.width,
            args.height,
            args.block,
            args.grid,
            args.skew,
            args.baseline,
            args.title,
            args.description,
        )
    )


if __name__ == "__main__":
    main()
