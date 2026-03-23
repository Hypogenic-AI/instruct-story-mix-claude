# Resources Catalog

## Summary

This document catalogs all resources gathered for the Instruct-StoryMix research project on using LLMs to decompose stories into components, generate synthetic data, and improve controllability in story generation.

---

## Papers

Total papers downloaded: **25**

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | Plan-And-Write | Yao et al. | 2019 | `1811.05701_yao2019_plan_and_write.pdf` | Foundational plan-then-write; RAKE storylines |
| 2 | DOC: Detailed Outline Control | Yang et al. | 2023 | `2212.10077_yang2023_DOC_outline.pdf` | Hierarchical outlines + FUDGE controller |
| 3 | DSR: Decomposed Screenwriting | Alibaba/PKU | 2025 | `2510.23163_Beyond_Direct_...pdf` | Hybrid synthesis; narrative directives |
| 4 | Weaver | Xie et al. | 2024 | `2401.17268_xie2024_weaver.pdf` | Instruction backtranslation; Constitutional DPO |
| 5 | Agents' Room | Huot et al. | 2024 | `2410.02603_hu2024_agents_room.pdf` | Multi-agent decomposition (ICLR 2025) |
| 6 | Strategies for Structuring | Fan et al. | 2019 | `1902.01109_fan2019_strategies_structuring.pdf` | SRL-based decomposition |
| 7 | Outline to Story (O2S) | Fang et al. | 2021 | `2101.00822_fang2021_outline_to_story.pdf` | FIST: simple outline-conditioned generation |
| 8 | PlotMachines | Rashkin et al. | 2020 | `2004.14967_rashkin2020_plotmachines.pdf` | Dynamic plot state tracking |
| 9 | Aristotelian Rescoring | Goldfarb-Tarrant et al. | 2020 | `2009.09870_goldfarb2020_aristotelian.pdf` | SRL plots + learned structural constraints |
| 10 | Controllable Plot via RL | Tambwekar et al. | 2018 | `1809.10736_Controllable_Neural_...pdf` | RL reward shaping for plots |
| 11 | Collective Critics | Bae, Kim | 2024 | `2410.02428_wang2024_collective_critics.pdf` | Multi-critic iterative refinement |
| 12 | Re3 | Yang et al. | 2022 | `2210.06774_yang2022_re3.pdf` | Recursive reprompting + revision |
| 13 | Story Realization | Fan et al. | 2019 | `1909.03480_fan2019_story_realization.pdf` | Plot events → sentences |
| 14 | MEGATRON-CNTRL | Xu et al. | 2020 | `2010.00840_xu2020_megatron_cntrl.pdf` | External knowledge + large LM |
| 15 | IBSEN | Wang et al. | 2024 | `2407.01093_wang2024_ibsen.pdf` | Director-actor drama generation |
| 16 | StoryWriter | Zhang et al. | 2025 | `2506.16445_zhang2025_storywriter.pdf` | Multi-agent long stories |
| 17 | Character-Centric Story | Zhao et al. | 2024 | `2409.16667_zhao2024_character_centric.pdf` | Character-driven generation |
| 18 | SARD | — | 2024 | `2403.01575_sard2024.pdf` | Human-AI collaborative stories |
| 19 | Art or Artifice? | Chakrabarty et al. | 2023 | `2309.14556_chakrabarty2023_art_artifice.pdf` | LLM creative writing limitations |
| 20 | Story Eval Benchmark | Chhun et al. | 2022 | `2208.11646_chhun2022_benchmark_story.pdf` | Evaluation metric comparison |
| 21 | LitBench | — | 2025 | `2507.00769_litbench2025.pdf` | Creative writing benchmark |
| 22 | CS4 | — | 2024 | `2410.04197_cs4_creativity.pdf` | Constraint-based creativity measurement |
| 23 | Modifying LLM for Diverse Writing | — | 2025 | `2503.17126_modifying_llm_creative.pdf` | Post-training diversity methods |
| 24 | Wordcraft | Yuan et al. | 2022 | `2107.07430_yuan2022_wordcraft.pdf` | Human-AI writing tool |
| 25 | ASP-guided Story Generation | — | 2024 | `2406.00554_Guiding_...pdf` | Logic programming for diversity |

See `papers/README.md` for detailed descriptions.

---

## Datasets

Total datasets downloaded: **4**

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| WritingPrompts | HuggingFace `euclaise/writingprompts` | 303K examples, 602 MB | Prompt→story generation | `datasets/writingprompts/` | Most widely used benchmark |
| ROCStories | HuggingFace `mintujupally/ROCStories` | 98K examples, 15 MB | Short commonsense stories | `datasets/rocstories/` | 5-sentence structured stories |
| WikiPlots | HuggingFace `vishnupriyavr/wiki-movie-plots-with-summaries` | 35K examples, 59 MB | Plot summaries + metadata | `datasets/wikiplots/` | Genre/metadata available |
| Tell Me a Story | HuggingFace `TAUR-dev/tell_me_a_story_*` | 230 examples, 1.4 MB | High-quality story generation | `datasets/tell_me_a_story/` | Professional writer quality |

See `datasets/README.md` for download instructions and detailed descriptions.

---

## Code Repositories

Total repositories cloned: **5**

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| DOC Story Generation | github.com/yangkevin2/doc-story-generation | Outline-controlled long story (ACL 2023) | `code/doc-story-generation/` | Requires legacy OpenAI API |
| DOC V2 | github.com/facebookresearch/doc-storygen-v2 | Modern DOC with LLaMA2/ChatGPT | `code/doc-storygen-v2/` | Best for practical use |
| Tell Me a Story | github.com/google-deepmind/tell_me_a_story | Dataset + eval (ICLR 2025) | `code/tell_me_a_story/` | Dataset-only, CC-BY 4.0 |
| PlotMachines | github.com/hrashkin/plotmachines | Outline-conditioned generation (EMNLP 2020) | `code/plotmachines/` | Old transformers==2.0.0 |
| Awesome-Story-Generation | github.com/yingpengma/Awesome-Story-Generation | Survey/paper list | `code/Awesome-Story-Generation/` | Reference only |

See `code/README.md` for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy
- Used paper-finder service with 7 different query combinations covering: controllable story generation, story decomposition, synthetic narrative data, LLM creative writing evaluation, plan-and-write approaches, story mixing/recombination
- Identified 216 papers with relevance score ≥ 2; scored and ranked by topic relevance
- Downloaded top 25 papers spanning 2018-2025

### Selection Criteria
- Direct relevance to story decomposition, controllable generation, or synthetic data for narratives
- Priority on papers with code/data availability
- Coverage of both methods (decomposition, control) and evaluation (benchmarks, metrics)
- Temporal range from foundational (2018) to state-of-the-art (2025)

### Challenges Encountered
- ArXiv IDs not always available from Semantic Scholar API; required direct arXiv title search
- Plan-And-Write code repository (Bitbucket) is down (404)
- One arXiv ID collision (1808.08933 maps to wrong paper)
- DOC original requires legacy OpenAI API and OPT-175B access

### Gaps and Workarounds
- No "Instruct-StoryMix" or "story mixing" specific papers found → research appears novel
- "Towards Controllable Story Generation" (Peng et al., 2018) PDF not downloadable (wrong arXiv mapping)
- No single unified framework for story decomposition + synthetic data exists → key opportunity

---

## Recommendations for Experiment Design

### 1. Primary Dataset(s)
- **WritingPrompts** for training/development (largest, most established)
- **ROCStories** for controlled experiments (short, structured)
- **Tell Me a Story** for gold-standard evaluation

### 2. Baseline Methods
- End-to-end LLM generation (zero-shot and few-shot)
- DOC outline-controlled generation (using doc-storygen-v2 codebase)
- Simple RAKE outline → GPT generation (following Plan-And-Write)

### 3. Evaluation Metrics
- **Coherence**: Human pairwise preference + LLM-as-judge
- **Controllability**: Plan/outline adherence rate
- **Diversity**: Inter/intra-story trigram repetition, unique word ratio
- **Quality**: Professional evaluator scoring (DSR-style) or human preference

### 4. Code to Adapt/Reuse
- **doc-storygen-v2**: Modern, clean implementation of outline-controlled generation; best starting point for building a controllable pipeline
- **plotmachines/src/preprocessing/extract_outlines.py**: RAKE outline extraction from stories
- **tell_me_a_story**: High-quality evaluation dataset and framework
