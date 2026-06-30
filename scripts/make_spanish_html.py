#!/usr/bin/env python
"""Generate the spanish_temperature.html report by cloning and modifying the template."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def main() -> None:
    temp_html = ROOT / "temperature.html"
    if not temp_html.exists():
        print("Error: temperature.html template not found")
        return

    content = temp_html.read_text(encoding="utf-8")

    # Customize title and target CSV file
    content = content.replace(
        "<title>WCS Temperature Comparison</title>", 
        "<title>WCS Spanish Temperature Comparison</title>"
    )
    content = content.replace(
        "<h1>WCS Temperature Comparison</h1>", 
        "<h1>WCS Spanish Temperature Comparison</h1>"
    )
    content = content.replace(
        "wcs_word_summary_punct_all_temperatures.csv", 
        "results/spanish_pd_books/wcs_word_summary.csv"
    )

    # Save output
    output_file = ROOT / "spanish_temperature.html"
    output_file.write_text(content, encoding="utf-8")
    print(f"Wrote Spanish temperature report to {output_file}")


if __name__ == "__main__":
    main()
