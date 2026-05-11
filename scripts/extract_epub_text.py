#!/usr/bin/env python
"""Extract readable plain text from an EPUB into one UTF-8 text file."""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path
from urllib.parse import unquote
from xml.etree import ElementTree

from bs4 import BeautifulSoup


CONTAINER = "META-INF/container.xml"
HTML_SUFFIXES = (".html", ".xhtml", ".htm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract EPUB reading-order text.")
    parser.add_argument("epub", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def package_path(epub: zipfile.ZipFile) -> str:
    try:
        root = ElementTree.fromstring(epub.read(CONTAINER))
    except KeyError as exc:
        raise SystemExit(f"EPUB is missing {CONTAINER}") from exc
    rootfile = root.find(".//{*}rootfile")
    if rootfile is None or not rootfile.attrib.get("full-path"):
        raise SystemExit("EPUB container does not point to an OPF package")
    return unquote(rootfile.attrib["full-path"])


def spine_items(epub: zipfile.ZipFile, opf_path: str) -> list[str]:
    root = ElementTree.fromstring(epub.read(opf_path))
    manifest = {
        item.attrib["id"]: item.attrib.get("href", "")
        for item in root.findall(".//{*}manifest/{*}item")
        if item.attrib.get("id")
    }
    base = Path(opf_path).parent
    ordered: list[str] = []
    for itemref in root.findall(".//{*}spine/{*}itemref"):
        href = manifest.get(itemref.attrib.get("idref", ""))
        if href:
            ordered.append(str((base / unquote(href)).as_posix()))
    return ordered


def html_items(epub: zipfile.ZipFile) -> list[str]:
    return sorted(
        name
        for name in epub.namelist()
        if name.lower().endswith(HTML_SUFFIXES) and not Path(name).name.startswith(".")
    )


def extract_html_text(raw_html: bytes) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "nav"]):
        tag.decompose()
    blocks: list[str] = []
    for tag in soup.find_all(["h1", "h2", "h3", "p", "blockquote", "li"]):
        text = " ".join(tag.get_text(" ", strip=True).split())
        if text:
            blocks.append(text)
    if not blocks:
        text = soup.get_text("\n", strip=True)
        blocks = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n\n".join(blocks)


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def main() -> int:
    args = parse_args()
    if not args.epub.exists():
        raise SystemExit(f"Missing EPUB: {args.epub}")

    with zipfile.ZipFile(args.epub) as epub:
        opf_path = package_path(epub)
        items = [item for item in spine_items(epub, opf_path) if item.lower().endswith(HTML_SUFFIXES)]
        if not items:
            items = html_items(epub)
        sections = []
        missing = []
        for item in items:
            try:
                sections.append(extract_html_text(epub.read(item)))
            except KeyError:
                missing.append(item)
        if missing:
            print(f"Skipped {len(missing)} missing spine items", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(clean_text("\n\n".join(sections)), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Sections: {len(sections)}")
    print(f"Characters: {args.output.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
