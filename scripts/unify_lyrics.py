import csv
import json
import os
from pathlib import Path

def unify_eminem(input_path, output_path):
    print(f"Processing Eminem from {input_path}")
    with open(input_path, "r", encoding="iso-8859-1") as f:
        # The file seems to use tabs based on the 'head' output (Album_Name\tSong_Name\t...)
        reader = csv.DictReader(f, delimiter="\t")
        with open(output_path, "w", encoding="utf-8") as out:
            for row in reader:
                lyrics = row.get("Lyrics", "")
                if lyrics:
                    out.write(lyrics + "\n\n")

def unify_bobdylan(input_path, output_path):
    print(f"Processing Bob Dylan from {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        with open(output_path, "w", encoding="utf-8") as out:
            for song in data:
                lyrics_blocks = song.get("lyrics", [])
                for block in lyrics_blocks:
                    out.write("\n".join(block) + "\n\n")

def unify_mfdoom(input_path, output_path):
    print(f"Processing MF DOOM from {input_path}")
    with open(output_path, "w", encoding="utf-8") as out:
        with open(input_path, "r", encoding="utf-8") as f:
            # It seems to be a list of strings in mfdoom.json or lines of JSON
            try:
                data = json.load(f)
                if isinstance(data, dict) and "train" in data:
                    for text in data["train"]:
                        out.write(text + "\n\n")
                elif isinstance(data, list):
                    for text in data:
                        out.write(text + "\n\n")
            except json.JSONDecodeError:
                # Try line by line if it's JSONL
                f.seek(0)
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        if isinstance(item, str):
                            out.write(item + "\n\n")
                        elif isinstance(item, dict) and "text" in item:
                            out.write(item["text"] + "\n\n")

def unify_taylorswift(input_dir, output_path):
    print(f"Processing Taylor Swift from {input_dir}")
    input_path = Path(input_dir)
    with open(output_path, "w", encoding="utf-8") as out:
        for txt_file in input_path.rglob("*.txt"):
            content = txt_file.read_text(encoding="utf-8")
            out.write(content + "\n\n")

def main():
    base_lyrics = Path("lyrics")
    output_dir = Path("data/raw/lyrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    unify_eminem(base_lyrics / "Eminem_Lyrics.csv", output_dir / "eminem.txt")
    unify_bobdylan(base_lyrics / "bobdylan.json", output_dir / "bobdylan.txt")
    unify_mfdoom(base_lyrics / "mfdoom.json", output_dir / "mfdoom.txt")
    unify_taylorswift(base_lyrics / "taylorswift", output_dir / "taylorswift.txt")

if __name__ == "__main__":
    main()
