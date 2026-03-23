# Instruct-StoryMix: Research Plan

## Motivation & Novelty Assessment

### Why This Research Matters
LLMs are notoriously formulaic storytellers—they default to predictable arcs, flat characters, and repetitive language (Chakrabarty et al. 2023). If we can decompose stories into discrete, controllable components (like Instruct-SkillMix does for instruction-following skills), we could generate synthetic training data that teaches models to explicitly control each narrative dimension, potentially breaking through the "formulaic ceiling."

### Gap in Existing Work
The literature review reveals that while decomposition-based story generation (DOC, Agents' Room, DSR) consistently outperforms end-to-end approaches, **no work applies the SkillMix paradigm**—systematically enumerating story components as "skills," creating combinatorial synthetic data from novel component mixtures, and measuring whether this improves controllability. Each existing system uses ad-hoc decompositions; none treat components as a first-class composable vocabulary.

### Our Novel Contribution
We test whether the Instruct-SkillMix "skill decomposition → combinatorial synthesis → evaluation" pipeline transfers to creative writing. Specifically: (1) Can LLMs reliably extract a rich taxonomy of story components? (2) Does generating stories from novel component combinations produce more diverse/controlled output than end-to-end generation? (3) Is there "arbitrage"—can a model that's bad at holistic story writing still produce better stories when given explicit component specifications?

### Experiment Justification
- **Experiment 1 (Component Extraction)**: Tests whether LLMs can reliably decompose stories into structured components—prerequisite for the entire pipeline.
- **Experiment 2 (Taxonomy Construction)**: Validates that extracted components form a coherent, reusable vocabulary—analogous to SkillMix's skill inventory.
- **Experiment 3 (Combinatorial Generation)**: Core test—generates stories from novel component combinations and compares to baselines on quality, diversity, and controllability.
- **Experiment 4 (Controllability Assessment)**: Directly measures whether component-specified generation produces stories that actually follow the specifications.

## Research Question
Can LLMs decompose stories into discrete components, and does generating stories from novel combinations of these components improve controllability and diversity compared to end-to-end generation?

## Hypothesis Decomposition
- **H1**: LLMs can extract structured story components (characters, plot structure, setting, theme, narrative technique, conflict type, emotional arc) from existing stories with high inter-rater agreement.
- **H2**: Novel combinations of extracted components produce more diverse stories than end-to-end generation.
- **H3**: Component-specified generation achieves higher plan adherence (controllability) than unconstrained generation.
- **H4**: Despite LLMs being "bad" at stories, component-controlled generation produces higher-quality stories than end-to-end generation (the "arbitrage" hypothesis).

## Proposed Methodology

### Approach
Use ROCStories (short, 5-sentence stories) as the base corpus. Extract components via GPT-4.1, build a taxonomy, then generate new stories by mixing components from different source stories. Compare against end-to-end baselines using GPT-4.1-as-judge evaluation.

### Experimental Steps
1. **Sample 100 ROCStories** for component extraction
2. **Extract components** from each story using GPT-4.1 with structured output (JSON)
3. **Build taxonomy**: Cluster and categorize extracted components
4. **Create 50 novel component specifications** by mixing components from different stories
5. **Generate stories** from specifications (component-controlled condition)
6. **Generate baseline stories** from titles only (end-to-end condition)
7. **Evaluate** all stories using GPT-4.1-as-judge on 5 dimensions
8. **Statistical analysis** comparing conditions

### Baselines
1. **End-to-end (title only)**: Generate story from just the title
2. **End-to-end (title + prompt)**: Generate story from title + brief writing prompt
3. **Component-controlled**: Generate story from full component specification

### Evaluation Metrics
1. **Coherence** (1-5): Logical consistency, connected events
2. **Creativity** (1-5): Originality, avoiding clichés
3. **Controllability** (1-5): How well story follows the component specification
4. **Character quality** (1-5): Believable, developed characters
5. **Overall quality** (1-5): Holistic judgment
6. **Diversity**: Unique trigram ratio, vocabulary diversity across generated stories

### Statistical Analysis Plan
- Paired t-tests / Wilcoxon signed-rank for quality comparisons
- Cohen's d for effect sizes
- Bootstrap confidence intervals (1000 resamples)
- α = 0.05 with Bonferroni correction for multiple comparisons

## Expected Outcomes
- H1 supported: Components extracted consistently (inter-extraction agreement > 0.7)
- H2 supported: Component-mixed stories show higher lexical diversity than baselines
- H3 supported: Component-controlled stories achieve higher plan adherence scores
- H4 partially supported: Quality improvement exists but may be modest; the "arbitrage" may be real but limited

## Timeline
- Phase 1-2: Setup + Data prep (15 min)
- Phase 3: Component extraction + taxonomy (30 min, ~100 API calls)
- Phase 4: Story generation (30 min, ~200 API calls)
- Phase 5: Evaluation (30 min, ~250 API calls)
- Phase 6: Analysis + Documentation (30 min)

## Potential Challenges
- API rate limits → use exponential backoff
- Component extraction inconsistency → validate with repeated extraction
- GPT-4.1-as-judge bias → compare with diversity metrics as independent signal
- Short stories (ROCStories) may limit component richness → document as limitation

## Success Criteria
- Clear evidence for or against each hypothesis
- Statistically significant results with appropriate effect sizes
- Reproducible pipeline with documented prompts and code
- Honest reporting regardless of whether results support the hypothesis
