import re
from collections import Counter
from pathlib import Path

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")

def make_freq(input_path, output_path):
    print(f"Generating frequency list for {input_path}")
    text = Path(input_path).read_text(encoding="utf-8")
    words = WORD_RE.findall(text.lower())
    counts = Counter(words)
    
    # Sort by frequency descending
    ranked = counts.most_common()
    
    with open(output_path, "w", encoding="utf-8") as f:
        for i, (word, count) in enumerate(ranked, start=1):
            f.write(f"{i}\t{word}\t{count}\n")

def main():
    corpus_dir = Path("data/raw/lyrics")
    freq_dir = Path("data/raw/lyrics_freq")
    freq_dir.mkdir(parents=True, exist_ok=True)
    
    for corpus_file in corpus_dir.glob("*.txt"):
        singer = corpus_file.stem
        make_freq(corpus_file, freq_dir / f"{singer}_freq.tsv")

if __name__ == "__main__":
    main()
