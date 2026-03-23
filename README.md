# Instruct-StoryMix

**Can the Instruct-SkillMix paradigm improve controllable story generation?**

## Key Findings

- **Controllability works**: Component-specified generation achieves 4.96/5.0 adherence to specifications across 6 narrative dimensions (setting, character, conflict, theme, emotional arc, tone)
- **Quality penalty exists**: Component-controlled stories score significantly lower on language quality (d=-0.83, p=0.001) and overall quality (d=-0.80, p=0.002) vs unconstrained baselines
- **Mixing components is fine**: Novel combinations from different source stories produce similar quality to coherent same-story specifications—and slightly higher creativity
- **No quality "arbitrage"**: More constraints make stories more controllable but less creative. The model trades expressive writing for specification adherence.
- **Diversity modestly improves**: Component-controlled stories show larger vocabulary (2,276 vs 1,807 unique words) and lower inter-story overlap

## Project Structure

```
├── REPORT.md                    # Full research report with all results
├── planning.md                  # Research plan and motivation
├── src/
│   ├── experiment.py            # Main experiment pipeline
│   └── supplementary_experiment.py  # Coherent vs mixed specs test
├── results/
│   ├── components.json          # Extracted story components (n=100)
│   ├── taxonomy.json            # Component taxonomy
│   ├── combinations.json        # Novel component combinations
│   ├── evaluated_*.json         # Generated stories with quality scores
│   ├── statistical_results.json # Full statistical analysis
│   ├── diversity_metrics.json   # Lexical diversity metrics
│   ├── supplementary_results.json
│   └── plots/
│       ├── quality_comparison.png
│       ├── controllability.png
│       ├── diversity_metrics.png
│       └── effect_sizes.png
├── datasets/                    # ROCStories, WritingPrompts, etc.
├── papers/                      # 25 downloaded research papers
├── code/                        # Baseline implementations
└── literature_review.md         # Comprehensive literature review
```

## Reproduction

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install openai numpy scipy matplotlib seaborn pandas datasets pyarrow

# Run (requires OPENAI_API_KEY)
export OPENAI_API_KEY=your_key
python src/experiment.py
python src/supplementary_experiment.py
```

- **Model**: GPT-4.1
- **API calls**: ~550 total
- **Runtime**: ~20 minutes
- **Seed**: 42

## See Also

- [REPORT.md](REPORT.md) for full analysis and discussion
- [literature_review.md](literature_review.md) for background on story generation research
