# Corrected downstream-validation findings

## Primary analysis and multiplicity correction

We designate the 52 aggregate Spearman tests as the primary family: 13 models × 2 temperatures × two mean diversity metrics. Holm correction controls the family-wise error rate across these 52 tests. Pearson correlations are reported as a separately corrected sensitivity analysis.

- 25/52 primary Spearman relationships remain significant after Holm correction.
- 22/24 Base-model relationships remain significant.
- 3/14 temperature-1.0 Instruct relationships remain significant.

## Secondary paired endpoint analysis

Endpoint contrasts compare the same contexts under restrictive and permissive settings using paired Wilcoxon tests. Holm correction is applied across all 156 model × temperature × sampler × metric tests.

- 24/156 endpoint effects remain significant after correction.

## Completion and interpretation

The lowest configuration-level completion rate is 78% (mistral7b-v03-base, T=1). Completion rates must be reported beside diversity because metrics are evaluated on generations reaching the fixed 100-word window.

The corrected evidence supports the following claim:

> Across matched prompts and decoding configurations, greater forced-path word reachability is generally associated with greater realized lexical diversity. The association is strongest and most consistent for Base models, while effects for Instruct models depend on model and temperature.

It does not establish that exact-word exclusion necessarily causes semantic or qualitative impoverishment.

## Corrected primary results

| Model | T | Metric | Spearman ρ | Raw p | Holm p |
|---|---:|---|---:|---:|---:|
| Qwen2.5-14B | 0.7 | MTLD | 0.81 | .003 | .092 |
| Qwen2.5-14B | 0.7 | TTR | 0.96 | <.001 | .009 |
| Qwen2.5-14B | 1 | MTLD | 0.91 | <.001 | .009 |
| Qwen2.5-14B | 1 | TTR | 0.97 | <.001 | .005 |
| Qwen2.5-14B-Instruct | 0.7 | MTLD | 0.44 | .163 | 1.000 |
| Qwen2.5-14B-Instruct | 0.7 | TTR | 0.26 | .422 | 1.000 |
| Qwen2.5-14B-Instruct | 1 | MTLD | 0.55 | .083 | 1.000 |
| Qwen2.5-14B-Instruct | 1 | TTR | 0.61 | .052 | .929 |
| Qwen3.5-9B | 0.7 | MTLD | 0.02 | .969 | 1.000 |
| Qwen3.5-9B | 0.7 | TTR | 0.18 | .587 | 1.000 |
| Qwen3.5-9B | 1 | MTLD | 0.20 | .560 | 1.000 |
| Qwen3.5-9B | 1 | TTR | 0.15 | .657 | 1.000 |
| Qwen3.5-9B-Base | 0.7 | MTLD | 0.79 | .007 | .178 |
| Qwen3.5-9B-Base | 0.7 | TTR | 0.90 | <.001 | .015 |
| Qwen3.5-9B-Base | 1 | MTLD | 0.96 | <.001 | .005 |
| Qwen3.5-9B-Base | 1 | TTR | 0.99 | <.001 | .005 |
| DeepSeek-R1-Distill-Qwen-14B | 0.7 | MTLD | 0.51 | .110 | 1.000 |
| DeepSeek-R1-Distill-Qwen-14B | 0.7 | TTR | -0.28 | .412 | 1.000 |
| DeepSeek-R1-Distill-Qwen-14B | 1 | MTLD | 0.93 | <.001 | .005 |
| DeepSeek-R1-Distill-Qwen-14B | 1 | TTR | 0.67 | .027 | .552 |
| gemma-3-12b-it | 0.7 | MTLD | 0.46 | .150 | 1.000 |
| gemma-3-12b-it | 0.7 | TTR | 0.16 | .641 | 1.000 |
| gemma-3-12b-it | 1 | MTLD | 0.87 | <.001 | .026 |
| gemma-3-12b-it | 1 | TTR | 0.77 | .007 | .165 |
| gemma-3-12b-pt | 0.7 | MTLD | 0.91 | <.001 | .009 |
| gemma-3-12b-pt | 0.7 | TTR | 0.96 | <.001 | .005 |
| gemma-3-12b-pt | 1 | MTLD | 0.92 | <.001 | .015 |
| gemma-3-12b-pt | 1 | TTR | 0.92 | <.001 | .015 |
| gemma-4-E4B | 0.7 | MTLD | 0.92 | <.001 | .015 |
| gemma-4-E4B | 0.7 | TTR | 0.86 | .002 | .045 |
| gemma-4-E4B | 1 | MTLD | 0.96 | <.001 | .005 |
| gemma-4-E4B | 1 | TTR | 0.96 | <.001 | .005 |
| gemma-4-E4B-it | 0.7 | MTLD | -0.11 | .750 | 1.000 |
| gemma-4-E4B-it | 0.7 | TTR | 0.59 | .060 | 1.000 |
| gemma-4-E4B-it | 1 | MTLD | 0.74 | .012 | .271 |
| gemma-4-E4B-it | 1 | TTR | 0.85 | .001 | .038 |
| Llama-3.1-8B | 0.7 | MTLD | 0.88 | <.001 | .026 |
| Llama-3.1-8B | 0.7 | TTR | 0.93 | <.001 | .009 |
| Llama-3.1-8B | 1 | MTLD | 0.87 | .001 | .033 |
| Llama-3.1-8B | 1 | TTR | 0.90 | <.001 | .020 |
| Llama-3.1-8B-Instruct | 0.7 | MTLD | -0.06 | .865 | 1.000 |
| Llama-3.1-8B-Instruct | 0.7 | TTR | 0.13 | .706 | 1.000 |
| Llama-3.1-8B-Instruct | 1 | MTLD | 0.49 | .142 | 1.000 |
| Llama-3.1-8B-Instruct | 1 | TTR | 0.60 | .060 | 1.000 |
| Mistral-7B-Instruct-v0.3 | 0.7 | MTLD | 0.68 | .026 | .552 |
| Mistral-7B-Instruct-v0.3 | 0.7 | TTR | 0.79 | .005 | .133 |
| Mistral-7B-Instruct-v0.3 | 1 | MTLD | 0.68 | .024 | .526 |
| Mistral-7B-Instruct-v0.3 | 1 | TTR | 0.65 | .039 | .739 |
| Mistral-7B-v0.3 | 0.7 | MTLD | 0.92 | <.001 | .009 |
| Mistral-7B-v0.3 | 0.7 | TTR | 0.90 | <.001 | .020 |
| Mistral-7B-v0.3 | 1 | MTLD | 0.94 | <.001 | .005 |
| Mistral-7B-v0.3 | 1 | TTR | 0.97 | <.001 | .005 |
