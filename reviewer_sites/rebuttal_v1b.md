# Rebuttal v1b — URL placeholders

Evidence links to replace before submission:

- **URL1:** `[URL1_WCS_DIVERSITY_CORRELATION]` — WCS correlation with TTR and MTLD across top-k, top-p, and min-p.
- **URL2:** `[URL2_TOPK_TO_80]` — focused top-k sweep through k=80.
- **URL3:** `[URL3_ALL_SAMPLERS_WITH_CI]` — top-k, top-p, and min-p WCS with 95% confidence intervals.
- **URL4:** `[URL4_TOKEN_COUNT_STRATIFICATION]` — WCS stratified by target-word token count.

## Response to qqwR

We agree that tail truncation itself is well known. We have revised the paper to state this explicitly and to define our contribution more narrowly: WCS is a context-sensitive, word-level diagnostic that identifies which human-attested lexical paths remain reachable under a particular model and inference configuration. The complete sampler sweeps are available at [URL3_ALL_SAMPLERS_WITH_CI].

We retain binary WCS because it answers a precise question: whether every token in an exact word path survives the decoding filter. We now distinguish this reachability construct from probability degradation, semantic quality, and generated-text diversity, and discuss probability-weighted coverage as a complementary extension rather than treating binary survival as a complete account of lexical behavior.

We have removed normative terms such as “censorship” and “lexical erasure” and replaced them with decoding-induced exclusion, lexical truncation, and reachability constraints. We likewise frame possible long-term language effects as hypotheses outside the causal scope of this study.

To evaluate downstream relevance, we ran 7,200 matched generations: six models × 50 shared FineWeb contexts × two temperatures × twelve decoding conditions. Base models receive raw continuation prompts; Instruct models use their native chat templates and an explicit continuation instruction. Generation stops normally at EOS, and TTR/MTLD are evaluated over a fixed 100-word window with completion reported separately. Across the 24 primary model × temperature × metric Spearman tests, 15 remain significant after Holm family-wise-error correction, including all 12 Base-model relationships. Effects for Instruct models depend on model and temperature. Full results and corrected tests are provided at [URL1_WCS_DIVERSITY_CORRELATION].

These additions support a narrower practical conclusion: greater forced-path reachability is generally associated with greater realized lexical diversity, most consistently in Base models. They do not establish that exact-word exclusion necessarily causes semantic or qualitative impoverishment.

## Response to oHfT

We agree that k≤20 did not cover the range needed for a practical diagnostic. We extended top-k from 1 through 20 and then in increments of five through 80. Increasing k from 20 to 80 raises WCS by approximately 14–15 percentage points in every evaluated model, showing that k=20 was not a saturated endpoint. The focused result is available at [URL2_TOPK_TO_80], while [URL3_ALL_SAMPLERS_WITH_CI] places it alongside top-p and min-p.

We do not describe k=80 as a universal recommended default. Instead, it is an upper evaluation boundary that brackets encountered configurations such as k=20, k=50, and k=64.

We have also corrected our model terminology. We now use Base, Instruct, and post-trained according to each released checkpoint and describe SFT/preference optimization only where documented, rather than treating “instruction-tuned” and “aligned” as interchangeable.

Finally, we have shifted the paper’s framing from claims about language evolution to a general-purpose diagnostic for debugging decoding hyperparameters. Potential cognitive or societal consequences are presented as open questions requiring longitudinal and psycholinguistic evidence.

## Response to bYax

We agree that the original PG-19 evaluation could conflate decoding behavior with a register mismatch. We replaced it with 200 target words × 50 contexts sampled from HuggingFaceFW/FineWeb (10,000 contexts per model). All six models are audited on identical sample IDs. Substantial exclusion persists in these contemporary web contexts: at top-k 50, WCS ranges from 0.415 to 0.760. Thus, PG-19 mismatch was a valid concern but does not explain away the observed reachability constraint. Full curves and uncertainty intervals are available at [URL3_ALL_SAMPLERS_WITH_CI].

We report 95% percentile confidence intervals from 2,000 target-word cluster-bootstrap draws, preserving the dependence among the 50 contexts belonging to each word. We also add paired endpoint tests for the downstream experiment and apply Holm correction separately to the 24 primary correlation tests and the 72 secondary paired tests.

To address the tokenizer confound, we stratify WCS by the number of subword tokens in each target. Because segmentation differs sharply across tokenizers, the analysis uses separate model panels and token-count lines rather than pooling strata across models. The results confirm that tokenization is a genuine cross-model confound, but they do not support a simple universal relationship in which more tokens always imply lower WCS. The complete stratified analysis is available at [URL4_TOKEN_COUNT_STRATIFICATION].

Finally, we agree that exact-word unreachability does not itself prove semantic impoverishment: a model may use synonyms, paraphrases, or alternative constructions. We now state this construct boundary directly. The matched generation experiment at [URL1_WCS_DIVERSITY_CORRELATION] establishes an association with output-side lexical diversity while retaining this limitation.

## Revised central claim

Tail truncation by decoding filters is well known. WCS contributes a context-sensitive, word-level audit of its consequences by measuring whether human-attested lexical paths remain reachable under a specified model and inference configuration. On contemporary FineWeb contexts, decoding filters exclude substantial exact-word paths across commonly encountered parameter ranges. Greater reachability is generally associated with greater output-side lexical diversity, with the strongest and most consistent evidence in Base models. These findings establish a decoding-layer reachability constraint; they do not by themselves establish intentional suppression, semantic impoverishment, cognitive harm, or long-term language change.
