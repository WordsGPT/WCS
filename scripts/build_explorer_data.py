import json
import sys
import glob

def main():
    if len(sys.argv) < 3:
        print("Usage: python build_explorer_data.py <samples.jsonl> <audit1.jsonl> <audit2.jsonl> ...")
        sys.exit(1)

    samples_file = sys.argv[1]
    
    # Handle wildcard expansion for Windows if necessary
    audit_files = []
    for arg in sys.argv[2:]:
        audit_files.extend(glob.glob(arg))

    # Data structure for the UI
    data = {
        "models": [],
        "words": {}
    }

    contexts_map = {}
    
    print(f"Loading samples from {samples_file}...")
    with open(samples_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            sid = d.get('id', d.get('sample_id'))
            word = d.get('word', d.get('matched_text', ''))
            prefix = d.get('prefix', '')
            
            if word not in data["words"]:
                data["words"][word] = {
                    "word": word,
                    "contexts": []
                }
            
            ctx = {
                "id": sid,
                "prefix": prefix,
                "target": word,
                "results": {}
            }
            data["words"][word]["contexts"].append(ctx)
            contexts_map[sid] = ctx

    models_seen = set()
    print(f"Processing {len(audit_files)} audit logs...")
    for audit_file in audit_files:
        print(f"  -> {audit_file}")
        with open(audit_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                sid = d.get('sample_id')
                if sid not in contexts_map: continue
                
                model = d.get('model')
                if model not in models_seen:
                    models_seen.add(model)
                    data["models"].append(model)
                
                # We only care about T=1.0 for the explorer, or we can just grab whatever is there
                temp = d.get('temperature', 1.0)
                if temp != 1.0 and temp != 0.0:
                    continue # Skip high temperatures to avoid UI clutter
                
                word_idx = d.get('word_token_index', 0)
                rank = d.get('rank', 9999)
                prob = d.get('probability', 0.0)
                
                existing = contexts_map[sid]["results"].get(model)
                if existing:
                    existing["prob"] *= prob
                    existing["rank"] = max(existing["rank"], rank)
                else:
                    top_5_tokens = d.get('top_5_tokens', [])
                    top_5_probs = d.get('top_5_probs', [])
                    contexts_map[sid]["results"][model] = {
                        "rank": rank,
                        "prob": prob,
                        "top5": [{"t": t, "p": p} for t, p in zip(top_5_tokens, top_5_probs)]
                    }

    data["models"].sort()
    
    out_file = "explorer_data.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, separators=(',', ':'))
        
    print(f"Successfully wrote {out_file} ({(len(json.dumps(data))/1024/1024):.2f} MB)")

if __name__ == '__main__':
    main()
