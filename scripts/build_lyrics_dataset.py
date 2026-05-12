import json
import random
import re
from pathlib import Path
from dataclasses import dataclass, asdict

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")

@dataclass
class Sample:
    id: str
    word: str
    rank: int
    count: int
    prefix: str
    matched_text: str
    source_path: str
    match_start_char: int
    match_end_char: int
    context_token_count: int
    search_start_char: int
    metadata: dict

def build_lyrics_dataset(singer, corpus_path, freq_path, output_path, rank_min, rank_max, sample_size, contexts_per_word):
    print(f"Building dataset for {singer}")
    
    entries = []
    with open(freq_path, "r", encoding="utf-8") as f:
        for line in f:
            rank, word, count = line.strip().split("\t")
            entries.append({"rank": int(rank), "word": word.lower(), "count": int(count)})
            
    band = [e for e in entries if rank_min <= e["rank"] <= rank_max]
    random.seed(42)
    random.shuffle(band)
    
    text = Path(corpus_path).read_text(encoding="utf-8")
    tokens = list(WORD_RE.finditer(text))
    
    # Pre-index tokens
    token_index = {}
    for i, m in enumerate(tokens):
        w = m.group(0).lower()
        if w not in token_index:
            token_index[w] = []
        token_index[w].append(i)
        
    samples = []
    words_processed = 0
    
    for entry in band:
        if words_processed >= sample_size:
            break
            
        word = entry["word"]
        indices = token_index.get(word, [])
        valid_indices = [i for i in indices if i >= 200]
        
        if len(valid_indices) >= contexts_per_word:
            selected_indices = random.sample(valid_indices, contexts_per_word)
            for i in selected_indices:
                m = tokens[i]
                start_token = tokens[i - 200]
                prefix = text[start_token.start() : m.start()].strip()
                
                sample = Sample(
                    id=f"sample-{len(samples)+1:06d}",
                    word=word,
                    rank=entry["rank"],
                    count=entry["count"],
                    prefix=prefix,
                    matched_text=m.group(0),
                    source_path=str(corpus_path),
                    match_start_char=m.start(),
                    match_end_char=m.end(),
                    context_token_count=200,
                    search_start_char=0,
                    metadata={"singer": singer, "dataset": "lyrics"}
                )
                samples.append(sample)
            words_processed += 1
            print(".", end="", flush=True)
            
    print(f"\nWrote {len(samples)} samples to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")

def main():
    output_dir = Path("data/processed/lyrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    singers = ["eminem", "bobdylan", "mfdoom", "taylorswift"]
    for singer in singers:
        build_lyrics_dataset(
            singer,
            f"data/raw/lyrics/{singer}.txt",
            f"data/raw/lyrics_freq/{singer}_freq.tsv",
            output_dir / f"{singer}.jsonl",
            100, 2000, 100, 5 # Adjust rank_min to 100
        )

if __name__ == "__main__":
    main()
