# Literature Review: Instruct-StoryMix

## Research Area Overview

This review covers the literature on **controllable story generation**, **story decomposition**, and **synthetic data generation for narrative tasks** — the three pillars of the Instruct-StoryMix hypothesis: that LLMs can decompose stories into components, generate synthetic data from these components, and improve controllability in story generation.

The field has evolved from early neural story generation (event-based, 2017-2019) through plan-and-write paradigms (2018-2020) to modern LLM-based multi-agent and decomposition frameworks (2023-2025). A consistent finding is that **decomposing generation into planning + writing stages significantly outperforms end-to-end generation** across coherence, relevance, and controllability metrics.

---

## Key Papers

### 1. Plan-And-Write: Towards Better Automatic Storytelling
- **Authors:** Yao, Peng, Weischedel, Knight, Zhao & Yan
- **Year:** 2019 (AAAI) | arXiv: 1811.05701
- **Key Contribution:** Foundational two-stage decomposition: Title → Storyline (keyword sequence via RAKE) → Story sentences. Establishes that automatically extracted story components can train controllable generation models.
- **Methodology:** Static schema (full plan before writing) and Dynamic schema (interleaved). RAKE keyword extraction creates storylines without human annotation. BiGRU/BiLSTM encoder-decoders.
- **Datasets:** ROCStories (98K five-sentence stories)
- **Results:** Static schema preferred by humans on coherence (49.5% vs 28.3%) and overall quality (50.1% vs 30.1%). Planning reduces inter-story repetition significantly.
- **Code:** https://bitbucket.org/VioletPeng/language-model (currently 404)
- **Relevance:** Direct template for automatic story decomposition into components. RAKE-based extraction is a prototype for synthetic data pipelines.

### 2. DOC: Improving Long Story Coherence With Detailed Outline Control
- **Authors:** Yang, Klein, Peng, Tian
- **Year:** 2023 (ACL) | arXiv: 2212.10077
- **Key Contribution:** Most complete story component taxonomy: premise, settings, characters (with temporal state), hierarchical plot events (3 levels), summaries. Uses FUDGE controller for outline adherence.
- **Methodology:** Breadth-first outline expansion, event candidate filtering/reranking, character detection + coreference, temporal character development. All via LLM prompting (InstructGPT3).
- **Datasets:** WritingPrompts (for controller training), synthetic stories for ordering model
- **Results:** +22.5pp coherence, +28.2pp relevance over Re3 baseline. Interactive: 80% user preference on intent/control/quality.
- **Code:** https://github.com/yangkevin2/doc-story-generation
- **Relevance:** Blueprint for what story components need to be extracted and tracked. Shows LLM prompting can perform all decomposition steps.

### 3. Beyond Direct Generation: A Decomposed Approach to Screenwriting (DSR)
- **Authors:** Alibaba Group + Peking University
- **Year:** 2025 | arXiv: 2510.23163
- **Key Contribution:** Proves empirically that decomposing creative generation into functionally distinct sub-tasks (narrative creation vs. format conversion) produces significantly better results than end-to-end. Introduces "Narrative Directives" (Chain-of-Thought over pacing, exposition, character arcs).
- **Methodology:** Stage 1: Outline → CoT analysis → Novel prose (SFT). Stage 2: Novel → Screenplay (inference-only with GPT-4). Hybrid data synthesis: reverse-compress inputs + extract directives + forward-distill outputs.
- **Datasets:** 50K samples from 200+ TV series; 32-scene expert evaluation set
- **Results:** 82.7% of human-level quality, 75% win rate vs. Claude-Sonnet-4 and Gemini-2.5-Pro. Decomposition gives +2.4 points over end-to-end.
- **Relevance:** Direct blueprint for synthetic data creation via reverse synthesis + forward distillation. Shows reverse synthesis alone is insufficient (input-output misalignment). Narrative Directives operationalize authorial strategy as trainable components.

### 4. Weaver: Foundation Models for Creative Writing
- **Authors:** Xie et al. (AIWaves Inc.)
- **Year:** 2024 | arXiv: 2401.17268
- **Key Contribution:** Domain-specific LLM for writing via continual pre-training + instruction backtranslation + Constitutional DPO. 500K synthetic instruction-response pairs from high-quality human-written text.
- **Methodology:** Backtranslation: collect human text → GPT-4 generates instruction with CoT → filter by quality. Constitutional DPO: 200+ expert-written principles generate principled negative examples.
- **Datasets:** Curated writing corpus (fiction:non-fiction 1:1); 400K filtered SFT pairs; 25K preference pairs
- **Results:** Weaver Ultra (34B) outperforms GPT-4 on Style and Creativity. 47% productivity improvement over GPT-4 in user study.
- **Relevance:** Instruction backtranslation is directly applicable to creating story instruction data. Constitutional DPO provides a principled approach to preference data for creative writing.

### 5. Agents' Room: Narrative Generation through Multi-step Collaboration
- **Authors:** Huot et al. (Google DeepMind)
- **Year:** 2024 (ICLR 2025) | arXiv: 2410.02603
- **Key Contribution:** Most structured decomposition into specialized agents: 4 planning agents (Conflict, Character, Setting, Plot) + 5 writing agents (Exposition, Rising Action, Climax, Falling Action, Resolution). Shared scratchpad communication.
- **Methodology:** Centralized orchestration with deterministic agent ordering following Freytag's dramatic structure. Backtranslation from stories using Gemini Ultra as teacher. LoRA fine-tuning of Gemini 1.5 Flash agents.
- **Datasets:** Tell Me a Story (230 high-quality workshop stories, avg 1498 tokens)
- **Results:** AR plan+write (FT) most preferred by humans. Fine-tuned agents outperform zero-shot. Decomposition via separate agents > decomposition within single model.
- **Code:** https://github.com/google-deepmind/tell_me_a_story
- **Relevance:** Definitive evidence that multi-agent decomposition outperforms single-model approaches. Backtranslation from stories to structured components is the exact pipeline needed for synthetic data generation.

### 6. Strategies for Structuring Story Generation
- **Authors:** Fan, Lewis, Dauphin
- **Year:** 2019 | arXiv: 1902.01109
- **Key Contribution:** Coarse-to-fine three-stage pipeline: SRL action sequences → entity-anonymized narrative → surface realization with name generation.
- **Datasets:** WritingPrompts (300K stories)
- **Relevance:** Demonstrates NLP tool-based story decomposition (SRL, NER, coreference) as foundation for structured synthetic data.

### 7. Outline to Story (O2S / FIST)
- **Authors:** Fang et al.
- **Year:** 2021 | arXiv: 2101.00822
- **Key Contribution:** Simple but effective outline-conditioned generation via interleaved special tokens in GPT-2 fine-tuning. No architecture changes needed.
- **Datasets:** WritingPrompts (303K), WikiPlots (113K)
- **Relevance:** Shows RAKE keyword outlines can be automatically extracted and used as control signals. Supports partial/incomplete outlines for flexible control.

### 8. PlotMachines: Outline-Conditioned Generation with Dynamic Plot State Tracking
- **Authors:** Rashkin, Celikyilmaz, Choi, Gao
- **Year:** 2020 (EMNLP) | arXiv: 2004.14967
- **Key Contribution:** Dynamic memory tracking which outline elements have been used, with discourse-position labels for narrative structure.
- **Datasets:** WikiPlots, WritingPrompts, NYTimes
- **Code:** https://github.com/hrashkin/plotmachines
- **Relevance:** Formalizes outline-conditioned generation with explicit state tracking—key for ensuring controllability.

### 9. Content Planning with Aristotelian Rescoring
- **Authors:** Goldfarb-Tarrant, Chakrabarty, Weischedel, Peng
- **Year:** 2020 | arXiv: 2009.09870
- **Key Contribution:** SRL-based plot generation with Aristotelian rescoring classifiers (event arrangement, character tracking, relevance, diction) for principled plot quality control.
- **Datasets:** WritingPrompts
- **Relevance:** Principled structural constraints on plot quality via learned classifiers.

### 10. Collective Critics for Creative Story Generation (CRITICS)
- **Authors:** Bae, Kim
- **Year:** 2024 (EMNLP) | arXiv: 2410.02428
- **Key Contribution:** Multi-critic iterative refinement at both plan and text levels, with persona-assigned LLM critics evaluating creativity dimensions.
- **Relevance:** Control mechanism for steering creativity via iterative critique—complementary to decomposition approaches.

### 11. Re3: Generating Longer Stories with Recursive Reprompting and Revision
- **Authors:** Yang et al.
- **Year:** 2022 (EMNLP) | arXiv: 2210.06774
- **Key Contribution:** Recursive reprompting with plan, draft, and revision stages for long story generation.
- **Relevance:** Predecessor to DOC; establishes the plan-draft-revise paradigm for long-form generation.

---

## Common Methodologies

### Story Decomposition Approaches
1. **Keyword/Outline extraction** (RAKE): Used in Plan-And-Write, O2S, PlotMachines. Automatic, no annotation needed.
2. **SRL-based action sequences**: Used in Fan et al. 2019, Goldfarb-Tarrant 2020. More structured but heavier.
3. **Hierarchical outline trees**: DOC generates multi-depth outline trees. Most detailed but complex.
4. **Narrative component agents**: Agents' Room assigns separate agents per story dimension.
5. **Reverse synthesis from finished text**: DSR, Weaver, Agents' Room all reverse-engineer structured components from complete stories.

### Synthetic Data Generation Methods
1. **Instruction backtranslation**: Weaver (GPT-4 generates instruction from human text)
2. **Reverse compression + forward distillation**: DSR (reverse inputs, extract directives, forward synthesize)
3. **Teacher-student backtranslation**: Agents' Room (Gemini Ultra → Gemini Flash)
4. **Automatic extraction**: Plan-And-Write, O2S (RAKE keywords from stories)

### Control Mechanisms
1. **Plan-conditioned generation**: All plan-and-write approaches
2. **FUDGE token-level control**: DOC (auxiliary classifier steers generation)
3. **Aristotelian rescoring**: Goldfarb-Tarrant (learned classifiers rerank candidates)
4. **Dynamic plot state tracking**: PlotMachines (memory over outline coverage)
5. **Multi-agent specialization**: Agents' Room (each agent handles one aspect)

---

## Standard Baselines
- **End-to-end LLM generation** (no planning): GPT-3/4, LLaMA, etc.
- **Rolling context window**: Generate text with only recent context (ROLLING-OPT in DOC)
- **Re3**: Recursive reprompting with plan + revision (DOC's predecessor)
- **Conditional language model**: Standard fine-tuned LM with prompt conditioning

---

## Evaluation Metrics

### Automatic Metrics
- **Inter/Intra-story repetition**: Trigram repetition rates (Plan-And-Write)
- **BLEU scores**: Standard but noted as inappropriate for open-ended creative generation
- **ROUGE-L, BERTScore**: Reference-based; ROUGE rewards similarity to gold stories
- **Unique word ratio**: Vocabulary diversity
- **Prompt overlap**: How much story copies from prompt

### Human Evaluation (Consensus Dimensions)
- **Coherence/Plot**: Logical consistency, connected events
- **Relevance/Fidelity**: Adherence to prompt/outline/plan
- **Creativity/Interestingness**: Engagement, originality, avoiding clichés
- **Character Development**: Believable, well-developed characters
- **Language Use**: Varied, rich language; literary devices
- **Overall preference**: Holistic quality judgment

### LLM-as-Judge
- GPT-4 judge (Weaver, Agents' Room): Moderate correlation with human preferences (ρ = 0.41-0.62)
- Professional evaluator panels: DSR uses 20+ professional screenwriters (12-point scale)

---

## Datasets in the Literature

| Dataset | Size | Domain | Used By | Notes |
|---------|------|--------|---------|-------|
| **WritingPrompts** | 303K | Reddit stories | Fan 2019, DOC, O2S, PlotMachines, Aristotelian | Most widely used; prompt→story pairs |
| **ROCStories** | 98K | 5-sentence commonsense | Plan-And-Write | Short, structured stories with titles |
| **WikiPlots** | 113K-135K | Wikipedia movie plots | O2S, PlotMachines | Plot summaries; structured metadata |
| **Tell Me a Story** | 230 | Workshop fiction | Agents' Room | Highest quality; detailed prompts |
| **NYTimes** | 240K | News articles | PlotMachines | Requires LDC license |

---

## Gaps and Opportunities

1. **No unified decomposition-to-synthetic-data pipeline**: Each paper creates its own ad-hoc extraction method. A systematic framework for decomposing stories into reusable components and generating synthetic training data is missing.

2. **Limited component vocabulary**: Most work uses only keywords/outlines. Richer components (character arcs, emotional trajectories, setting transitions, narrative pacing) are identified but rarely used as explicit training signals.

3. **Reverse synthesis quality**: DSR shows reverse-only synthesis creates input-output misalignment. The hybrid approach (reverse + forward) is better but not widely adopted.

4. **Evaluation gap**: Human evaluation is expensive and inconsistent. LLM-as-judge shows moderate but imperfect correlation with humans, especially for creativity.

5. **Scale vs. quality tradeoff**: Large datasets (WritingPrompts) have variable quality; high-quality datasets (Tell Me a Story) are tiny. Synthetic data generation could bridge this gap.

6. **LLM story writing limitations**: Multiple papers (Art or Artifice, Echoes in AI) document that LLMs produce less diverse, more formulaic stories than humans. Controllability may help by enabling explicit specification of diverse story elements.

---

## Recommendations for Our Experiment

### Recommended Datasets
1. **WritingPrompts** (primary): Largest story corpus with prompts; widely used as benchmark
2. **ROCStories** (secondary): Short, structured stories ideal for component extraction experiments
3. **WikiPlots** (supplementary): Movie plots with rich metadata; good for genre-conditioned experiments
4. **Tell Me a Story** (evaluation): High-quality gold standard for human evaluation

### Recommended Baselines
1. End-to-end LLM generation (GPT-4/Claude/LLaMA)
2. DOC outline-controlled generation
3. Simple plan-and-write (RAKE keywords → generation)
4. Agents' Room multi-agent approach

### Recommended Metrics
1. **Human evaluation**: Coherence, creativity, character development, overall preference (pairwise)
2. **Controllability**: Outline/plan adherence rate (detailed-relevance metric from DOC)
3. **Diversity**: Inter/intra-story repetition, unique word ratio
4. **LLM-as-judge**: GPT-4 scoring on style, relevance, creativity (as validation)

### Methodological Considerations
- **Hybrid synthesis** (DSR approach) is strongly recommended over pure reverse or pure forward synthesis for creating training data
- **Instruction backtranslation** (Weaver approach) is the most scalable method for creating instruction-response pairs from existing stories
- **Component granularity matters**: Start with coarse components (outline, characters, setting) before attempting fine-grained decomposition
- **Multi-agent decomposition** (Agents' Room) empirically outperforms single-model decomposition, even with identical information
- **Constitutional principles** (Weaver) provide a principled way to create preference data for creative writing tasks
