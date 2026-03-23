# Instruct-StoryMix: Can Story Decomposition Enable Better Controllable Generation?

## 1. Executive Summary

We tested whether the Instruct-SkillMix paradigm—decomposing tasks into skills, creating combinatorial synthetic data, and using it to improve model capability—transfers to creative story generation. Using GPT-4.1, we extracted structured components (setting, character, conflict type, theme, emotional arc, tone) from 100 ROCStories, then generated 50 new stories from novel component combinations and compared them against two baselines (title-only and prompted generation).

**Key finding**: Component-controlled generation achieves near-perfect specification adherence (4.96/5.0) but suffers a statistically significant quality penalty: overall quality drops by 0.40 points (d=-0.80, p=0.002) compared to title-only baselines, with the largest penalties in language quality (d=-0.83) and creativity (d=-0.46). Diversity metrics show modest improvements (larger vocabulary, lower inter-story overlap), but the "arbitrage" hypothesis—that explicit control would yield better stories despite LLMs' writing limitations—is **not supported**. The constraint-quality tradeoff appears fundamental: the model allocates its generation "bandwidth" to following specifications at the expense of expressive, creative writing.

**Practical implication**: The SkillMix approach works beautifully for *controllability* but not for *quality*. To realize the full potential, a two-stage approach is needed: component specification for structure, followed by unconstrained refinement for quality—echoing the decomposition findings from DSR and Agents' Room.

## 2. Goal

### Research Question
Can LLMs decompose stories into discrete components, generate stories from novel component combinations, and improve controllability and quality compared to end-to-end generation?

### Hypotheses
- **H1**: LLMs can reliably extract structured story components from existing stories.
- **H2**: Novel component combinations produce more diverse stories than end-to-end generation.
- **H3**: Component-specified generation achieves higher plan adherence (controllability).
- **H4**: Component-controlled generation produces higher-quality stories than end-to-end generation (the "arbitrage" hypothesis).

### Why This Matters
LLMs are notoriously formulaic storytellers (Chakrabarty et al. 2023). If decomposition enables explicit control over narrative dimensions, it could unlock more diverse, author-directed creative writing. The Instruct-SkillMix framework (Allenzhu et al. 2024) showed this works for general instruction-following; we test whether it transfers to the harder domain of creative writing.

### Gap in Existing Work
While decomposition-based story generation (DOC, Agents' Room, DSR) consistently outperforms end-to-end approaches, no work has applied the SkillMix paradigm of systematic component enumeration → combinatorial synthesis → evaluation. Each existing system uses ad-hoc decompositions; none treat story components as a first-class composable vocabulary.

## 3. Data Construction

### Dataset Description
- **Source**: ROCStories (Mostafazadeh et al., 2016), via HuggingFace `mintujupally/ROCStories`
- **Size**: 78,528 training stories; we sampled 100 for component extraction
- **Characteristics**: 5-sentence commonsense stories, ~43 words average, simple narrative structures
- **Known limitations**: Stories are short and structurally simple; may not generalize to longer fiction

### Example Samples

**Source Story (ROCStories)**:
> "The boy went to a video arcade. He played his favorite machine. His games didn't go very well. He told the owner about his experience. The owner explained that he had made the game settings harder."

**Extracted Components**:
| Component | Value |
|-----------|-------|
| Setting | A video arcade, during a typical day |
| Protagonist | A boy, a regular arcade-goer |
| Conflict | person_vs_technology |
| Theme | unexpected adversity |
| Emotional arc | neutral |
| Tone | lighthearted |

**Component-Controlled Generated Story** (using mixed components):
> "John sat upright in his high school classroom, meticulously reviewing his notes while his classmates whispered and scrolled through their phones... Despite his fear of drawing attention, John raised his hand and politely pointed out the discrepancy..."

### Data Quality
- 100/100 stories successfully decomposed (100% extraction rate)
- All JSON outputs valid and well-formed
- No missing values in any component field

### Preprocessing
1. Random sample of 100 stories (seed=42) from ROCStories train split
2. Component extraction via GPT-4.1 (temperature=0.2, structured JSON output)
3. Novel combinations created by sampling components from 5 different source stories per combination

## 4. Experiment Description

### Methodology

#### High-Level Approach
Inspired by Instruct-SkillMix, we treat story components as composable "skills." We: (1) extract a taxonomy of components from existing stories, (2) create novel specifications by mixing components from different sources, (3) generate stories conditioned on these specifications, and (4) compare quality, controllability, and diversity against unconditioned baselines.

#### Why This Method?
The SkillMix paradigm was chosen because it provides a systematic framework for testing compositional generalization in creative tasks. Alternatives (e.g., Agents' Room multi-agent, DOC hierarchical outlines) are more complex and would confound the core question of whether component decomposition itself enables the "arbitrage."

### Implementation Details

#### Tools and Libraries
- Python 3.12
- OpenAI API v2.29.0 (GPT-4.1)
- NumPy 2.4.3, SciPy 1.17.1
- Matplotlib 3.10+, Seaborn 0.13.2
- HuggingFace Datasets

#### Model
- **GPT-4.1** for all tasks (extraction, generation, evaluation)
- Temperature: 0.2 (extraction), 0.8 (generation), 0.1 (evaluation)

#### Experimental Conditions
| Condition | Input | N |
|-----------|-------|---|
| Component-controlled | Full 7-component specification | 50 |
| Baseline (title-only) | Theme only | 50 |
| Baseline (prompted) | Theme + setting | 50 |
| Supplementary: Coherent specs | All components from same source story | 25 |
| Supplementary: Mixed specs | Components from different source stories | 25 |

#### Hyperparameters
| Parameter | Value | Selection Method |
|-----------|-------|------------------|
| N stories extracted | 100 | Budget constraint |
| N stories generated | 50 per condition | Statistical power |
| Generation temperature | 0.8 | Standard for creative tasks |
| Eval temperature | 0.1 | Minimize judge variance |
| Random seed | 42 | Convention |

### Evaluation Metrics

**Quality (GPT-4.1-as-judge, 1-5 scale)**:
- Coherence, Creativity, Character Quality, Language Quality, Overall Quality

**Controllability (GPT-4.1-as-judge, 1-5 scale)**:
- Per-component adherence (setting, character, conflict, theme, arc, tone) + overall

**Diversity (automatic)**:
- Unique trigram ratio, vocabulary size, inter-story trigram overlap

**Statistical tests**: Wilcoxon signed-rank (paired, non-parametric), Cohen's d effect sizes, Bonferroni correction

### Raw Results

#### Main Experiment: Quality Scores

| Dimension | Component-Controlled | Baseline (Title) | Baseline (Prompted) |
|-----------|---------------------|-------------------|---------------------|
| Coherence | 5.00 ± 0.00 | 5.00 ± 0.00 | 5.00 ± 0.00 |
| Creativity | 3.72 ± 0.45 | **3.94 ± 0.51** | 3.92 ± 0.52 |
| Character Quality | 4.30 ± 0.54 | 4.20 ± 0.45 | 4.34 ± 0.47 |
| Language Quality | 4.38 ± 0.49 | **4.76 ± 0.43** | **4.66 ± 0.47** |
| Overall Quality | 4.26 ± 0.52 | **4.66 ± 0.47** | **4.58 ± 0.49** |

Bold = significantly better (p < 0.05, Wilcoxon signed-rank)

#### Controllability Scores (Component-Controlled Only)

| Component | Adherence (1-5) |
|-----------|----------------|
| Setting | 4.88 ± 0.38 |
| Character | 4.90 ± 0.30 |
| Conflict | 4.82 ± 0.38 |
| Theme | 5.00 ± 0.00 |
| Arc | 4.98 ± 0.14 |
| Tone | 4.76 ± 0.47 |
| **Overall** | **4.96 ± 0.20** |

#### Diversity Metrics

| Metric | Component-Controlled | Baseline (Title) | Baseline (Prompted) |
|--------|---------------------|-------------------|---------------------|
| Unique trigram ratio | **0.988** | 0.980 | 0.988 |
| Vocabulary size | **2,276** | 1,807 | 2,164 |
| Inter-story overlap | **0.0006** | 0.0008 | 0.0006 |
| Mean story length | 110 words | 87 words | 100 words |

#### Statistical Tests (Component vs Title Baseline)

| Dimension | Cohen's d | p-value | Interpretation |
|-----------|----------|---------|----------------|
| Coherence | 0.00 | n/a | No difference (ceiling) |
| Creativity | -0.46 | 0.028* | Small-medium penalty |
| Character Quality | +0.20 | 0.369 | No significant difference |
| Language Quality | **-0.83** | **0.001*** | Large penalty |
| Overall Quality | **-0.80** | **0.002*** | Large penalty |

\* p < 0.05, \*\* p < 0.01, \*\*\* p < 0.001

#### Supplementary: Coherent vs Mixed Component Specifications

| Dimension | Coherent Specs (n=25) | Mixed Specs (n=25) |
|-----------|----------------------|-------------------|
| Coherence | 5.00 ± 0.00 | 4.96 ± 0.20 |
| Creativity | 3.56 ± 0.50 | 3.76 ± 0.43 |
| Character Quality | 4.40 ± 0.49 | 4.48 ± 0.57 |
| Language Quality | 4.60 ± 0.49 | 4.64 ± 0.56 |
| Overall Quality | 4.44 ± 0.50 | 4.52 ± 0.57 |

#### Visualizations

- `results/plots/quality_comparison.png` — Box plots of quality scores across conditions
- `results/plots/controllability.png` — Bar chart of per-component adherence
- `results/plots/diversity_metrics.png` — Diversity comparison across conditions
- `results/plots/effect_sizes.png` — Forest plot of effect sizes with significance

### Component Taxonomy (Extracted from 100 ROCStories)

| Component | Distribution |
|-----------|-------------|
| **Conflict type** | person_vs_self (54%), person_vs_nature (17%), person_vs_person (13%), person_vs_society (9%), person_vs_fate (4%), person_vs_technology (2%) |
| **Emotional arc** | positive_resolution (53%), negative_resolution (22%), bittersweet (17%), neutral (6%), comedic (1%) |
| **Tone** | heartwarming (35%), serious (27%), lighthearted (21%), melancholic (11%), dramatic (3%), suspenseful (2%) |
| **Narrative technique** | linear (100%) — limitation of short ROCStories |

## 5. Result Analysis

### Key Findings

1. **H1 SUPPORTED — Reliable component extraction**: GPT-4.1 extracted structured components from 100% of stories with valid JSON output. The taxonomy reveals meaningful distributions (e.g., person_vs_self is the dominant conflict at 54%, positive_resolution the dominant arc at 53%).

2. **H2 PARTIALLY SUPPORTED — Modest diversity improvement**: Component-controlled stories show a larger vocabulary (2,276 vs 1,807 words) and slightly higher unique trigram ratio (0.988 vs 0.980). However, the differences are small and may be partially explained by the longer average length of component-controlled stories (110 vs 87 words).

3. **H3 STRONGLY SUPPORTED — Excellent controllability**: Near-perfect specification adherence (4.96/5.0 overall). Every component dimension scored above 4.76/5.0. The model reliably follows detailed multi-dimensional specifications.

4. **H4 REFUTED — No quality arbitrage**: Component-controlled stories scored significantly *lower* on language quality (d=-0.83, p=0.001), creativity (d=-0.46, p=0.028), and overall quality (d=-0.80, p=0.002) compared to title-only baselines. The "arbitrage" hypothesis is not supported.

### The Constraint-Quality Tradeoff

The central finding is a **constraint-quality tradeoff**: as more structural constraints are imposed, the model follows them faithfully but produces less creative, less linguistically polished output.

Qualitative examination reveals why:
- **Baseline stories** use vivid imagery, sensory language, and unexpected turns (e.g., "She inhaled, leapt, and for a glorious moment soared above the world, wind screaming victory in her ears")
- **Component-controlled stories** read more like competent plot summaries, hitting all specification points but with flatter prose (e.g., "Against the odds, the team rallied... Bob's heartfelt words echoed through the stadium")

The model appears to allocate its generation "bandwidth" between two competing objectives: (1) satisfying constraints and (2) producing expressive writing. More constraints leave less room for creative expression.

### Supplementary Finding: Mixing Doesn't Hurt

The supplementary experiment comparing coherent specs (all components from one source) vs mixed specs (components from different sources) found **no quality difference** (overall: 4.44 vs 4.52). Mixed specs actually scored slightly higher on creativity (3.76 vs 3.56), suggesting that unusual component combinations may stimulate more original generation. The quality penalty comes from the *number* of constraints, not from component incoherence.

### Surprises and Insights

1. **Coherence is trivially easy**: All conditions achieved perfect or near-perfect coherence (5.0/5.0). For 5-sentence stories, GPT-4.1 never produces incoherent narratives regardless of constraint level.

2. **Narrative technique bottleneck**: 100% of extracted stories were classified as "linear" narrative technique. ROCStories are too short and simple to exhibit flashbacks, frame narratives, or in medias res. This limits one dimension of the taxonomy.

3. **Length effect**: Component-controlled stories are ~26% longer (110 vs 87 words), likely because the model needs more words to satisfy all specified elements. This may partially inflate the diversity metrics.

4. **The judge ceiling**: GPT-4.1-as-judge gave very high scores across all conditions (rarely below 3/5), suggesting it may be too generous. A more discriminating evaluation (human judges, or a custom rubric) might reveal larger or different patterns.

### Limitations

1. **Same model for generation and evaluation**: GPT-4.1 generates and judges the stories, creating potential self-preference bias. However, this bias would favor all conditions equally.

2. **Short stories only**: ROCStories (5 sentences) may not reveal benefits that emerge at longer scales where planning and control become more important (DOC, Agents' Room show larger gains on longer stories).

3. **No fine-tuning**: We tested zero-shot component-controlled generation. The SkillMix paradigm's full potential requires *training* on synthetic component data, which we did not test due to scope constraints.

4. **Single model**: All experiments use GPT-4.1. Results may differ with other models, especially smaller models where controllability is harder.

5. **Automatic evaluation only**: No human evaluation was conducted. GPT-4.1-as-judge has moderate correlation with human judgments (ρ ≈ 0.4-0.6 per literature).

6. **Limited taxonomy**: The narrative_technique dimension collapsed to a single value, reducing the effective component space.

## 6. Conclusions

### Summary

The Instruct-SkillMix approach transfers to story generation for **controllability** but not for **quality**. LLMs can reliably decompose stories into structured components and generate new stories that faithfully follow novel component combinations. However, component-controlled generation produces significantly lower-quality stories than unconstrained generation—a constraint-quality tradeoff where specification adherence competes with creative expression.

### Implications

**For the "arbitrage" hypothesis**: The arbitrage does not exist at the single-generation level. Models that are bad at writing stories become *worse* when given more structural constraints to satisfy simultaneously. However, the arbitrage may exist in a *two-stage* pipeline: use component specifications for structure, then apply unconstrained refinement for quality (as in DSR's hybrid synthesis approach).

**For synthetic data generation**: The component taxonomy and combinatorial synthesis pipeline is viable for creating diverse training data. The extracted components form a meaningful vocabulary with natural distributions. This could be valuable for training smaller models via distillation, even if the teacher model's constrained outputs need refinement.

**For creative writing tools**: Component specification is highly effective for *steering* generation (4.96/5.0 adherence). This has immediate practical value for interactive writing tools where authors want to control narrative elements while the model handles prose.

### Confidence in Findings
- **High confidence** in the controllability finding (near-ceiling scores, clear mechanism)
- **Medium confidence** in the quality penalty finding (consistent across comparisons, but single judge model)
- **Low confidence** in the diversity finding (small differences, confounded by length)

## 7. Next Steps

### Immediate Follow-ups
1. **Two-stage pipeline**: Generate with component specs, then refine with "make this more creative and expressive" post-processing. Test whether this recovers quality while maintaining controllability.
2. **Longer stories**: Repeat on WritingPrompts (300K stories, longer narratives) where controllability benefits should be larger.
3. **Fine-tuning experiment**: Use the component-controlled synthetic data to fine-tune a smaller model (e.g., LLaMA-8B) and measure whether it develops better controllability than baseline instruction tuning.

### Alternative Approaches
- **Fewer, more impactful components**: Test whether 2-3 components (e.g., theme + conflict only) achieve a better quality-controllability balance.
- **Hierarchical generation**: Use components for outline, then unconstrained prose generation.
- **Constitutional DPO**: Use Weaver's approach to create preference data from component-controlled vs baseline pairs.

### Open Questions
1. Does the constraint-quality tradeoff diminish at larger model scales?
2. Would training on component-controlled data transfer controllability to unconstrained generation?
3. Is there a "sweet spot" number of components that maximizes the quality-controllability product?
4. Do human evaluators agree with the GPT-4.1 judge's relative rankings?

## References

1. Allen-Zhu, Z. et al. (2024). "Instruct-SkillMix: Skill-Mixture Data for General Instruction Following." arXiv:2408.14774
2. Yao, L. et al. (2019). "Plan-And-Write: Towards Better Automatic Storytelling." AAAI.
3. Yang, K. et al. (2023). "DOC: Improving Long Story Coherence With Detailed Outline Control." ACL.
4. Xie, X. et al. (2024). "Weaver: Foundation Models for Creative Writing." arXiv:2401.17268
5. Huot, F. et al. (2024). "Agents' Room: Narrative Generation through Multi-step Collaboration." ICLR 2025.
6. DSR (2025). "Beyond Direct Generation: A Decomposed Approach to Screenwriting." arXiv:2510.23163
7. Chakrabarty, T. et al. (2023). "Art or Artifice? Large Language Models and the False Promise of Creativity." arXiv:2309.14556
8. Mostafazadeh, N. et al. (2016). "A Corpus and Cloze Evaluation for Deeper Understanding of Commonsense Stories." NAACL.

---

*Experiment conducted 2026-03-23. Model: GPT-4.1. Total API calls: ~550. Seed: 42.*
