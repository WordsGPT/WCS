#!/usr/bin/env python
"""Build an HTML page visualizing the target words in samples.jsonl."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path


def load_words(samples_path: Path) -> list[dict]:
    words: dict[str, dict] = {}
    with samples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            word = row["word"]
            entry = words.setdefault(
                word,
                {
                    "word": word,
                    "rank": row.get("rank"),
                    "count": row.get("count"),
                    "contexts": 0,
                    "metadata": row.get("metadata", {}),
                },
            )
            entry["contexts"] += 1
    return sorted(words.values(), key=lambda item: (int(item["rank"]), item["word"]))


def load_frequency_curve(frequency_path: Path, max_rank: int = 120_000) -> list[dict]:
    points: list[dict] = []
    if not frequency_path.exists():
        return points
    with frequency_path.open("r", encoding="utf-8") as handle:
        for rank, line in enumerate(handle, start=1):
            if rank > max_rank:
                break
            fields = line.strip().split()
            if len(fields) < 2 or not fields[-1].isdigit():
                continue
            points.append({"rank": rank, "word": fields[0], "count": int(fields[-1])})
    return points


def infer_band(words: list[dict]) -> tuple[int, int]:
    for item in words:
        metadata = item.get("metadata") or {}
        rank_min = metadata.get("rank_min")
        rank_max = metadata.get("rank_max")
        if rank_min and rank_max:
            return int(rank_min), int(rank_max)
    ranks = [int(item["rank"]) for item in words if item.get("rank") is not None]
    return (min(ranks), max(ranks)) if ranks else (0, 0)


def make_frequency_path(
    points: list[dict],
    width: int,
    height: int,
    pad: dict[str, int],
    *,
    rank_min: int | None = None,
    rank_max: int | None = None,
) -> tuple[str, dict[str, float]]:
    domain_points = [
        point
        for point in points
        if (rank_min is None or point["rank"] >= rank_min)
        and (rank_max is None or point["rank"] <= rank_max)
    ]
    if not domain_points:
        return "", {}
    x_min = math.log10(max(1, rank_min or min(point["rank"] for point in domain_points)))
    x_max = math.log10(rank_max or max(point["rank"] for point in domain_points))
    y_min = math.log10(max(1, min(point["count"] for point in domain_points)))
    y_max = math.log10(max(point["count"] for point in domain_points))

    def sx(rank: int) -> float:
        return pad["left"] + ((math.log10(rank) - x_min) / (x_max - x_min)) * (width - pad["left"] - pad["right"])

    def sy(count: int) -> float:
        return height - pad["bottom"] - ((math.log10(count) - y_min) / (y_max - y_min)) * (height - pad["top"] - pad["bottom"])

    target_points = 900
    stride = max(1, math.ceil(len(domain_points) / target_points))
    sampled = domain_points[::stride]
    if domain_points[-1] is not sampled[-1]:
        sampled.append(domain_points[-1])
    path = " ".join(
        ("M" if index == 0 else "L") + f" {sx(point['rank']):.2f} {sy(point['count']):.2f}"
        for index, point in enumerate(sampled)
    )
    return path, {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}


def render_chart(words: list[dict], frequency_points: list[dict], rank_min: int, rank_max: int) -> str:
    width = 980
    height = 560
    main_pad = {"top": 152, "right": 34, "bottom": 62, "left": 124}
    overview_pad = {"top": 36, "right": 34, "bottom": 446, "left": 124}
    zoom_min = max(1, int(rank_min * 0.72))
    zoom_max = int(rank_max * 1.18)
    overview_path, overview_scale = make_frequency_path(frequency_points, width, height, overview_pad)
    path, scale = make_frequency_path(
        frequency_points,
        width,
        height,
        main_pad,
        rank_min=zoom_min,
        rank_max=zoom_max,
    )
    if not path or not scale or not overview_path or not overview_scale:
        return "<p class=\"chart-fallback\">Frequency curve unavailable; selected-word ranks are listed below.</p>"

    def scaler(active_scale: dict[str, float], active_pad: dict[str, int]):
        def sx(rank: int) -> float:
            return active_pad["left"] + ((math.log10(rank) - active_scale["x_min"]) / (active_scale["x_max"] - active_scale["x_min"])) * (width - active_pad["left"] - active_pad["right"])

        def sy(count: int) -> float:
            return height - active_pad["bottom"] - ((math.log10(count) - active_scale["y_min"]) / (active_scale["y_max"] - active_scale["y_min"])) * (height - active_pad["top"] - active_pad["bottom"])

        return sx, sy

    sx, sy = scaler(scale, main_pad)
    osx, osy = scaler(overview_scale, overview_pad)

    x_ticks = [8_000, 10_000, 15_000, 20_000, 30_000, 40_000, 50_000]
    y_ticks = [500_000, 750_000, 1_000_000, 2_000_000, 3_000_000, 5_000_000]
    grid = []
    labels = []
    for tick in x_ticks:
        if scale["x_min"] <= math.log10(tick) <= scale["x_max"]:
            x = sx(tick)
            grid.append(f'<line x1="{x:.2f}" y1="{main_pad["top"]}" x2="{x:.2f}" y2="{height - main_pad["bottom"]}" />')
            labels.append(f'<text x="{x:.2f}" y="{height - 24}" text-anchor="middle">{tick:,}</text>')
    for tick in y_ticks:
        if scale["y_min"] <= math.log10(tick) <= scale["y_max"]:
            y = sy(tick)
            grid.append(f'<line x1="{main_pad["left"]}" y1="{y:.2f}" x2="{width - main_pad["right"]}" y2="{y:.2f}" />')
            labels.append(f'<text x="{main_pad["left"] - 12}" y="{y + 4:.2f}" text-anchor="end">{tick:,}</text>')

    band_x = sx(rank_min)
    band_width = sx(rank_max) - band_x
    overview_band_x = osx(rank_min)
    overview_band_width = osx(rank_max) - overview_band_x
    dots = []
    for item in words:
        rank = int(item["rank"])
        count = int(item["count"])
        dots.append(
            f'<circle cx="{sx(rank):.2f}" cy="{sy(count):.2f}" r="3.8">'
            f'<title>{html.escape(str(item["word"]))}: rank {rank:,}, count {count:,}</title>'
            "</circle>"
        )

    label_words = [words[0], words[len(words) // 2], words[-1]] if len(words) >= 3 else words
    callouts = []
    for item in label_words:
        rank = int(item["rank"])
        count = int(item["count"])
        x = sx(rank)
        y = sy(count)
        label_y = y - 16 if y > 70 else y + 22
        callouts.append(f'<line class="callout-line" x1="{x:.2f}" y1="{y:.2f}" x2="{x:.2f}" y2="{label_y:.2f}" />')
        callouts.append(f'<text class="callout" x="{x:.2f}" y="{label_y - 5:.2f}" text-anchor="middle">{html.escape(str(item["word"]))}</text>')

    return f"""
    <section class="chart-panel" aria-labelledby="freqChartTitle">
      <div class="chart-heading">
        <div>
          <h2 id="freqChartTitle">Frequency Selection Band</h2>
          <p>Norvig word-frequency curve with the sampled middle-long-tail rank segment highlighted.</p>
        </div>
        <div class="legend" aria-label="Chart legend">
          <span><i class="legend-line"></i>Frequency list</span>
          <span><i class="legend-band"></i>Selected rank band</span>
          <span><i class="legend-dot"></i>Target words</span>
        </div>
      </div>
      <svg class="freq-chart" viewBox="0 0 {width} {height}" role="img" aria-label="Word frequency by rank, highlighting ranks {rank_min:,} to {rank_max:,}">
        <rect class="overview-bg" x="{overview_pad["left"]}" y="{overview_pad["top"]}" width="{width - overview_pad["left"] - overview_pad["right"]}" height="{height - overview_pad["top"] - overview_pad["bottom"]}" />
        <path class="overview-curve" d="{overview_path}" />
        <rect class="overview-band" x="{overview_band_x:.2f}" y="{overview_pad["top"]}" width="{overview_band_width:.2f}" height="{height - overview_pad["top"] - overview_pad["bottom"]}" />
        <text class="overview-label" x="{overview_band_x + overview_band_width / 2:.2f}" y="{overview_pad["top"] - 10}" text-anchor="middle">selected band in the full list</text>
        <rect class="plot-bg" x="{main_pad["left"]}" y="{main_pad["top"]}" width="{width - main_pad["left"] - main_pad["right"]}" height="{height - main_pad["top"] - main_pad["bottom"]}" />
        <g class="grid">{"".join(grid)}</g>
        <rect class="band" x="{band_x:.2f}" y="{main_pad["top"]}" width="{band_width:.2f}" height="{height - main_pad["top"] - main_pad["bottom"]}" />
        <path class="curve" d="{path}" />
        <g class="selected-dots">{"".join(dots)}</g>
        <g class="callouts">{"".join(callouts)}</g>
        <line class="axis" x1="{main_pad["left"]}" y1="{height - main_pad["bottom"]}" x2="{width - main_pad["right"]}" y2="{height - main_pad["bottom"]}" />
        <line class="axis" x1="{main_pad["left"]}" y1="{main_pad["top"]}" x2="{main_pad["left"]}" y2="{height - main_pad["bottom"]}" />
        <g class="axis-labels">{"".join(labels)}</g>
        <text class="axis-title" x="{width / 2:.2f}" y="{height - 6}" text-anchor="middle">Frequency rank, zoomed log scale</text>
        <text class="axis-title" x="-{(main_pad["top"] + height - main_pad["bottom"]) / 2:.2f}" y="20" text-anchor="middle" transform="rotate(-90)">Word count, log scale</text>
        <text class="band-label" x="{band_x + band_width / 2:.2f}" y="{main_pad["top"] + 28}" text-anchor="middle">Sampled ranks {rank_min:,}-{rank_max:,}</text>
      </svg>
    </section>
"""


def render_html(
    words: list[dict],
    frequency_points: list[dict],
    frequency_path: Path,
    samples_path: Path,
) -> str:
    rank_min, rank_max = infer_band(words)
    chart = render_chart(words, frequency_points, rank_min, rank_max)
    selected_min = min(int(item["rank"]) for item in words)
    selected_max = max(int(item["rank"]) for item in words)
    metadata = words[0].get("metadata") or {}
    dataset = str(metadata.get("dataset") or samples_path.name)
    config = metadata.get("config")
    dataset_label = f"{dataset} ({config})" if config else dataset
    is_fineweb = "fineweb" in dataset.lower()
    page_title = "FineWeb WCS Target Words" if is_fineweb else "WCS Target Words"
    rows = "\n".join(
        f'<tr data-word="{html.escape(str(item["word"]).lower(), quote=True)}" '
        f'data-rank="{int(item["rank"])}" data-count="{int(item["count"])}">'
        f"<td>{index}</td>"
        f"<td>{html.escape(str(item['word']))}</td>"
        f"<td>{int(item['rank']):,}</td>"
        f"<td>{int(item['count']):,}</td>"
        f"<td>{int(item['contexts']):,}</td>"
        "</tr>"
        for index, item in enumerate(words, start=1)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(page_title)}</title>
  <style>
    :root {{
      --bg: #f4f6f8;
      --panel: #fff;
      --ink: #172033;
      --muted: #5e6b7c;
      --line: #dce2ea;
      --accent: #2563eb;
      --accent-soft: #eff6ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    main {{
      max-width: 1160px;
      margin: 0 auto;
      padding: 28px 28px 48px;
    }}
    .site-head {{
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 24px;
      margin-bottom: 8px;
    }}
    .eyebrow {{
      margin: 0 0 6px;
      color: var(--accent);
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.11em;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: clamp(28px, 4vw, 40px);
      line-height: 1.08;
      letter-spacing: -0.025em;
    }}
    p {{
      margin: 0 0 20px;
      color: var(--muted);
      line-height: 1.55;
    }}
    .lede {{ max-width: 760px; margin-bottom: 0; }}
    nav {{
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 8px 16px;
      padding-bottom: 4px;
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
      font-size: 14px;
      font-weight: 700;
      white-space: nowrap;
    }}
    a:hover {{ text-decoration: underline; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 24px 0;
    }}
    .stat {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 16px 18px;
      box-shadow: 0 7px 24px rgba(15, 23, 42, 0.035);
    }}
    .stat strong {{
      display: block;
      font-size: 20px;
      color: #102a43;
    }}
    .stat span {{
      color: #62748a;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .chart-panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      margin: 0 0 24px;
      padding: 18px 18px 10px;
      box-shadow: 0 8px 28px rgba(15, 23, 42, 0.04);
    }}
    .chart-heading {{
      display: flex;
      gap: 16px;
      justify-content: space-between;
      align-items: start;
      margin-bottom: 8px;
    }}
    h2 {{
      margin: 0 0 4px;
      font-size: 18px;
    }}
    .chart-heading p {{
      margin: 0;
      font-size: 14px;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 14px;
      justify-content: flex-end;
      color: #52606d;
      font-size: 13px;
    }}
    .legend span {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      white-space: nowrap;
    }}
    .legend i {{
      display: inline-block;
      width: 18px;
      height: 10px;
    }}
    .legend-line {{
      border-top: 3px solid #24527a;
    }}
    .legend-band {{
      background: rgba(245, 158, 11, 0.22);
      border: 1px solid rgba(217, 119, 6, 0.35);
    }}
    .legend-dot {{
      width: 10px !important;
      height: 10px !important;
      border-radius: 999px;
      background: #c2410c;
    }}
    .freq-chart {{
      display: block;
      width: 100%;
      height: auto;
      overflow: visible;
    }}
    .plot-bg {{
      fill: #fbfcfd;
    }}
    .overview-bg {{
      fill: #f8fafc;
      stroke: #d9e2ec;
      stroke-width: 1;
    }}
    .overview-curve {{
      fill: none;
      stroke: #8da2b5;
      stroke-width: 2;
      stroke-linecap: round;
      stroke-linejoin: round;
    }}
    .overview-band {{
      fill: rgba(194, 65, 12, 0.24);
      stroke: #c2410c;
      stroke-width: 1.5;
    }}
    .overview-label {{
      fill: #7c2d12;
      font-size: 12px;
      font-weight: 700;
    }}
    .grid line {{
      stroke: #e4e7eb;
      stroke-width: 1;
    }}
    .band {{
      fill: rgba(245, 158, 11, 0.18);
      stroke: rgba(217, 119, 6, 0.55);
    }}
    .curve {{
      fill: none;
      stroke: #24527a;
      stroke-width: 3.4;
      stroke-linecap: round;
      stroke-linejoin: round;
    }}
    .selected-dots circle {{
      fill: #c2410c;
      stroke: white;
      stroke-width: 1.6;
      filter: drop-shadow(0 1px 1px rgba(16, 42, 67, 0.24));
    }}
    .axis {{
      stroke: #718096;
      stroke-width: 1.4;
    }}
    .axis-labels text, .axis-title {{
      fill: #52606d;
      font-size: 12px;
    }}
    .band-label {{
      fill: #92400e;
      font-size: 13px;
      font-weight: 700;
    }}
    .callout {{
      fill: #7c2d12;
      font-size: 12px;
      font-weight: 700;
    }}
    .callout-line {{
      stroke: #c2410c;
      stroke-width: 1;
      stroke-dasharray: 3 3;
    }}
    .chart-fallback {{
      padding: 16px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
    }}
    .table-panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 8px 28px rgba(15, 23, 42, 0.04);
    }}
    .table-toolbar {{
      display: grid;
      grid-template-columns: minmax(220px, 1fr) 220px auto;
      gap: 10px;
      align-items: center;
      padding: 14px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfe;
    }}
    .search-wrap {{
      position: relative;
    }}
    .search-wrap span {{
      position: absolute;
      left: 12px;
      top: 50%;
      transform: translateY(-50%);
      color: #7b8797;
      pointer-events: none;
    }}
    input, select, button {{
      font: inherit;
    }}
    input, select {{
      width: 100%;
      min-height: 40px;
      border: 1px solid #cfd7e3;
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      padding: 9px 11px;
    }}
    input {{
      padding-left: 36px;
    }}
    input:focus, select:focus {{
      outline: 3px solid rgba(37, 99, 235, 0.15);
      border-color: var(--accent);
    }}
    button {{
      min-height: 40px;
      border: 1px solid #cfd7e3;
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      padding: 8px 14px;
      font-weight: 700;
      cursor: pointer;
    }}
    button:hover {{ background: #f4f7fb; }}
    .table-meta {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 14px;
      color: var(--muted);
      font-size: 13px;
      border-bottom: 1px solid var(--line);
    }}
    .table-wrap {{
      max-height: 680px;
      overflow: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid #e4e7eb;
      text-align: left;
      font-size: 14px;
    }}
    th {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: #f1f4f8;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: #52606d;
    }}
    td:nth-child(1), td:nth-child(3), td:nth-child(4), td:nth-child(5) {{
      font-variant-numeric: tabular-nums;
    }}
    td:nth-child(2) {{
      font-size: 16px;
      font-weight: 650;
    }}
    tr:last-child td {{
      border-bottom: 0;
    }}
    tbody tr:nth-child(even) {{ background: #fbfcfe; }}
    tbody tr:hover {{ background: var(--accent-soft); }}
    .no-results {{
      padding: 26px;
      text-align: center;
      color: var(--muted);
    }}
    .no-results[hidden] {{ display: none; }}
    @media (max-width: 760px) {{
      main {{ padding: 20px 14px 36px; }}
      .site-head {{ display: block; }}
      nav {{ justify-content: flex-start; margin-top: 14px; }}
      .summary {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      .chart-heading {{
        display: block;
      }}
      .legend {{
        justify-content: flex-start;
        margin-top: 12px;
      }}
      .table-toolbar {{ grid-template-columns: 1fr; }}
      .table-wrap {{ max-height: 72vh; }}
      th, td {{ padding: 9px 10px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header class="site-head">
      <div>
        <p class="eyebrow">FineWeb · sample-10BT · target inventory</p>
        <h1>{html.escape(page_title)}</h1>
        <p class="lede">{len(words)} unique words from {html.escape(dataset_label)}, sampled from the middle-long-tail band of {html.escape(str(frequency_path))}.</p>
      </div>
      <nav aria-label="Related WCS views">
        <a href="index.html">Decoder curves</a>
        <a href="temperature.html">Temperature comparison</a>
      </nav>
    </header>
    <section class="summary" aria-label="Selection summary">
      <div class="stat"><strong>{len(words)}</strong><span>target words</span></div>
      <div class="stat"><strong>{rank_min:,}-{rank_max:,}</strong><span>selection band</span></div>
      <div class="stat"><strong>{selected_min:,}-{selected_max:,}</strong><span>observed target ranks</span></div>
      <div class="stat"><strong>{sum(int(item["contexts"]) for item in words):,}</strong><span>contexts</span></div>
    </section>
{chart}
    <section class="table-panel" aria-labelledby="wordTableTitle">
      <div class="table-toolbar">
        <label class="search-wrap">
          <span aria-hidden="true">⌕</span>
          <input id="wordSearch" type="search" placeholder="Filter target words…" autocomplete="off" aria-label="Filter target words">
        </label>
        <label>
          <select id="wordSort" aria-label="Sort target words">
            <option value="rank-asc">Rank: low to high</option>
            <option value="rank-desc">Rank: high to low</option>
            <option value="word-asc">Word: A to Z</option>
            <option value="count-desc">Frequency: high to low</option>
          </select>
        </label>
        <button id="clearFilters" type="button">Reset</button>
      </div>
      <div class="table-meta">
        <strong id="wordTableTitle">Target-word inventory</strong>
        <span id="resultCount" aria-live="polite">{len(words)} words</span>
      </div>
      <div class="table-wrap">
        <table id="wordTable">
          <thead>
            <tr>
              <th>#</th>
              <th>Word</th>
              <th>Rank</th>
              <th>Count</th>
              <th>Contexts</th>
            </tr>
          </thead>
          <tbody>
{rows}
          </tbody>
        </table>
        <div class="no-results" id="noResults" hidden>No target words match that filter.</div>
      </div>
    </section>
  </main>
  <script>
    const searchInput = document.getElementById("wordSearch");
    const sortSelect = document.getElementById("wordSort");
    const tableBody = document.querySelector("#wordTable tbody");
    const allRows = Array.from(tableBody.rows);
    const resultCount = document.getElementById("resultCount");
    const noResults = document.getElementById("noResults");

    function updateTable() {{
      const query = searchInput.value.trim().toLowerCase();
      const [field, direction] = sortSelect.value.split("-");
      const ordered = [...allRows].sort((a, b) => {{
        if (field === "word") return a.dataset.word.localeCompare(b.dataset.word);
        const key = field === "count" ? "count" : "rank";
        return Number(a.dataset[key]) - Number(b.dataset[key]);
      }});
      if (direction === "desc") ordered.reverse();

      let visible = 0;
      ordered.forEach(row => {{
        const matches = !query || row.dataset.word.includes(query);
        row.hidden = !matches;
        if (matches) {{
          visible += 1;
          row.cells[0].textContent = String(visible);
        }}
        tableBody.append(row);
      }});
      resultCount.textContent = `${{visible}} of ${{allRows.length}} words`;
      noResults.hidden = visible !== 0;
    }}

    searchInput.addEventListener("input", updateTable);
    sortSelect.addEventListener("change", updateTable);
    document.getElementById("clearFilters").addEventListener("click", () => {{
      searchInput.value = "";
      sortSelect.value = "rank-asc";
      updateTable();
      searchInput.focus();
    }});
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an HTML list of target words.")
    parser.add_argument("--samples", type=Path, default=Path("data/processed/samples.jsonl"))
    parser.add_argument("--frequency", type=Path, default=Path("data/raw/norvig_count_1w.txt"))
    parser.add_argument("--output", type=Path, default=Path("word_list.html"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    words = load_words(args.samples)
    frequency_points = load_frequency_curve(args.frequency)
    args.output.write_text(
        render_html(words, frequency_points, args.frequency, args.samples),
        encoding="utf-8",
    )
    print(f"Wrote {len(words)} words to {args.output}")


if __name__ == "__main__":
    main()
