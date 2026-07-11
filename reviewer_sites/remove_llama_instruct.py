#!/usr/bin/env python3
import json
from pathlib import Path

DATA_DIR = Path("reviewer_sites/data")
EXCLUDE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
EXCLUDE_SLUG = "llama31-8b-instruct"
EXCLUDE_LABEL = "Llama 3.1 8B Instruct"

def holm_correction(raw_pvals):
    n = len(raw_pvals)
    indexed = sorted(enumerate(raw_pvals), key=lambda x: x[1])
    corrected = [0.0] * n
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted = p * (n - rank)
        cummax = max(cummax, adjusted)
        corrected[orig_idx] = min(cummax, 1.0)
    return corrected

def filter_downstream():
    path = DATA_DIR / "downstream.json"
    with open(path) as f:
        data = json.load(f)

    # Filter rows
    data["rows"] = [r for r in data["rows"] if r.get("model") != EXCLUDE_MODEL and
                    r.get("model_slug") != EXCLUDE_SLUG and
                    r.get("model_label") != EXCLUDE_LABEL]

    # Filter primary_correlations and recalculate Holm
    corrs = [r for r in data["primary_correlations"] if r["model"] != EXCLUDE_MODEL]
    n_tests = len(corrs)
    raw_pvals = [float(r["spearman_p_raw"]) for r in corrs]
    holm_pvals = holm_correction(raw_pvals)

    pearson_raw = [float(r["pearson_p_raw"]) for r in corrs]
    pearson_holm = holm_correction(pearson_raw)

    # Clean old holm keys and set new ones
    for i, r in enumerate(corrs):
        for k in list(r.keys()):
            if "holm" in k and k not in [f"spearman_p_holm_{n_tests}", "spearman_significant_holm_05", f"pearson_p_holm_{n_tests}", "pearson_significant_holm_05"]:
                del r[k]
        
        r[f"spearman_p_holm_{n_tests}"] = str(holm_pvals[i])
        r[f"spearman_significant_holm_05"] = str(holm_pvals[i] < 0.05)
        r[f"pearson_p_holm_{n_tests}"] = str(pearson_holm[i])
        r[f"pearson_significant_holm_05"] = str(pearson_holm[i] < 0.05)
        r["family_definition"] = f"{n_tests} model×temperature×metric primary tests"

    data["primary_correlations"] = corrs
    data["metadata"]["primary_family"] = f"{n_tests} aggregate Spearman tests; Holm FWER correction"

    # Filter sampler_correlations
    if "sampler_correlations" in data:
        data["sampler_correlations"] = [r for r in data["sampler_correlations"] if r.get("model") != EXCLUDE_MODEL]

    # Filter paired_effects and recalculate Holm
    if "paired_effects" in data:
        paired = [r for r in data["paired_effects"] if r.get("model_slug") != EXCLUDE_SLUG]
        n_sec = len(paired)
        raw_sec = [float(r["wilcoxon_p_raw"]) for r in paired]
        holm_sec = holm_correction(raw_sec)
        
        for i, r in enumerate(paired):
            for k in list(r.keys()):
                if "holm" in k and k not in [f"wilcoxon_p_holm_{n_sec}", "significant_holm_05"]:
                    del r[k]
            r[f"wilcoxon_p_holm_{n_sec}"] = str(holm_sec[i])
            r["significant_holm_05"] = str(holm_sec[i] < 0.05)
            r["family_definition"] = f"{n_sec} secondary endpoint tests"
        
        data["paired_effects"] = paired
        data["metadata"]["secondary_family"] = f"{n_sec} paired endpoint Wilcoxon tests; Holm FWER correction"

    # Filter completion
    if "completion" in data:
        data["completion"] = [r for r in data["completion"] if r.get("model_slug") != EXCLUDE_SLUG]

    with open(path, "w") as f:
        json.dump(data, f)
    
    # Calculate stats
    sig_count = sum(1 for r in corrs if r["spearman_significant_holm_05"] == "True")
    
    instruct_models = {"meta-llama/Llama-3.1-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"}
    base_models = {r["model"] for r in corrs} - instruct_models
    
    base_entries = [r for r in corrs if r["model"] in base_models]
    instruct_entries = [r for r in corrs if r["model"] in instruct_models]
    
    base_sig = sum(1 for r in base_entries if r["spearman_significant_holm_05"] == "True")
    
    instruct_t1 = [r for r in instruct_entries if r["temperature"] == "1.0"]
    instruct_t1_sig = sum(1 for r in instruct_t1 if r["spearman_significant_holm_05"] == "True")
    
    print(f"Updated stats:")
    print(f"  {sig_count}/{n_tests} primary tests survive Holm")
    print(f"  {base_sig}/{len(base_entries)} Base relationships survive")
    print(f"  {instruct_t1_sig}/{len(instruct_t1)} Instruct T=1 survive")
    print(f"  n_tests={n_tests}")

if __name__ == "__main__":
    filter_downstream()
