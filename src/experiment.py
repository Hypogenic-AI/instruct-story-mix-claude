"""
Instruct-StoryMix: Component Decomposition → Combinatorial Synthesis → Evaluation

This script implements the full experimental pipeline:
1. Extract story components from ROCStories using GPT-4.1
2. Build a component taxonomy
3. Generate stories from novel component combinations
4. Generate baseline stories (end-to-end)
5. Evaluate all stories using GPT-4.1-as-judge
6. Statistical analysis
"""

import os
import json
import random
import time
import numpy as np
from datetime import datetime
from openai import OpenAI
from collections import Counter

# ─── Configuration ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

MODEL = "gpt-4.1"  # Primary model for all tasks
EVAL_MODEL = "gpt-4.1"  # Evaluation model
N_STORIES_EXTRACT = 100  # Stories to extract components from
N_STORIES_GENERATE = 50  # Novel combinations to generate
MAX_RETRIES = 3
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/plots", exist_ok=True)

client = OpenAI()

# ─── Utility ─────────────────────────────────────────────────────────────────

def api_call(messages, model=MODEL, temperature=0.7, max_tokens=2000, response_format=None):
    """Robust API call with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            kwargs = dict(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if response_format:
                kwargs["response_format"] = response_format
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    return None

def log(msg):
    """Print with timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load Stories
# ═══════════════════════════════════════════════════════════════════════════════

def load_stories(n=N_STORIES_EXTRACT):
    """Load a random sample of ROCStories."""
    from datasets import load_from_disk
    ds = load_from_disk("datasets/rocstories")
    all_stories = ds["train"]["text"]
    # Sample deterministically
    indices = random.sample(range(len(all_stories)), n)
    stories = [all_stories[i] for i in indices]
    log(f"Loaded {len(stories)} stories (avg {np.mean([len(s.split()) for s in stories]):.0f} words)")
    return stories


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Extract Story Components
# ═══════════════════════════════════════════════════════════════════════════════

EXTRACTION_PROMPT = """Analyze this short story and extract its components. Return valid JSON with exactly these fields:

{
  "setting": "where and when the story takes place (1 sentence)",
  "protagonist": "brief description of the main character",
  "conflict_type": one of ["person_vs_person", "person_vs_self", "person_vs_nature", "person_vs_society", "person_vs_technology", "person_vs_fate"],
  "theme": "the central theme or moral (1-3 words)",
  "emotional_arc": one of ["positive_resolution", "negative_resolution", "bittersweet", "comedic", "neutral", "surprise_twist"],
  "narrative_technique": one of ["linear", "flashback", "in_medias_res", "frame_narrative", "cause_and_effect"],
  "tone": one of ["humorous", "serious", "suspenseful", "heartwarming", "melancholic", "lighthearted", "dramatic"],
  "plot_structure": "1-sentence summary of what happens"
}

Story: """

def extract_components(stories):
    """Extract structured components from each story."""
    components = []
    log(f"Extracting components from {len(stories)} stories...")

    for i, story in enumerate(stories):
        if i % 20 == 0:
            log(f"  Progress: {i}/{len(stories)}")

        messages = [
            {"role": "system", "content": "You are a literary analyst. Always respond with valid JSON only, no markdown."},
            {"role": "user", "content": EXTRACTION_PROMPT + story}
        ]

        try:
            response = api_call(messages, temperature=0.2, max_tokens=500,
                              response_format={"type": "json_object"})
            comp = json.loads(response)
            comp["source_story"] = story
            comp["source_index"] = i
            components.append(comp)
        except Exception as e:
            log(f"  Failed on story {i}: {e}")
            continue

    log(f"Successfully extracted {len(components)}/{len(stories)} components")
    return components


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Build Taxonomy & Create Novel Combinations
# ═══════════════════════════════════════════════════════════════════════════════

def build_taxonomy(components):
    """Analyze the distribution of extracted components."""
    taxonomy = {}
    categorical_fields = ["conflict_type", "emotional_arc", "narrative_technique", "tone"]

    for field in categorical_fields:
        values = [c.get(field, "unknown") for c in components]
        taxonomy[field] = dict(Counter(values).most_common())

    # Also collect unique settings and themes
    taxonomy["themes"] = list(set(c.get("theme", "") for c in components))
    taxonomy["settings"] = list(set(c.get("setting", "") for c in components))

    log(f"Taxonomy built: {len(taxonomy['themes'])} unique themes, {len(taxonomy['settings'])} unique settings")
    for field in categorical_fields:
        log(f"  {field}: {taxonomy[field]}")

    return taxonomy


def create_novel_combinations(components, n=N_STORIES_GENERATE):
    """Create novel story specifications by mixing components from different stories."""
    combinations = []

    for i in range(n):
        # Pick components from different source stories
        sources = random.sample(components, min(5, len(components)))

        combo = {
            "setting": sources[0]["setting"],
            "protagonist": sources[1]["protagonist"],
            "conflict_type": sources[2]["conflict_type"],
            "theme": sources[3]["theme"],
            "emotional_arc": sources[min(4, len(sources)-1)]["emotional_arc"],
            "narrative_technique": random.choice(components)["narrative_technique"],
            "tone": random.choice(components)["tone"],
            "source_indices": [s["source_index"] for s in sources],
        }
        combinations.append(combo)

    log(f"Created {len(combinations)} novel component combinations")
    return combinations


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: Generate Stories
# ═══════════════════════════════════════════════════════════════════════════════

COMPONENT_GENERATION_PROMPT = """Write a short story (exactly 5 sentences) that follows these specifications:

Setting: {setting}
Main character: {protagonist}
Conflict type: {conflict_type}
Theme: {theme}
Emotional arc: {emotional_arc}
Narrative technique: {narrative_technique}
Tone: {tone}

Important: Write exactly 5 sentences. Make the story coherent, creative, and engaging while adhering to ALL the specifications above."""

BASELINE_TITLE_PROMPT = """Write a short story (exactly 5 sentences). The story should be about: {theme}

Important: Write exactly 5 sentences. Make the story coherent, creative, and engaging."""

BASELINE_PROMPT_PROMPT = """Write a short story (exactly 5 sentences).

Setting: {setting}
Theme: {theme}

Important: Write exactly 5 sentences. Make the story coherent, creative, and engaging."""


def generate_component_stories(combinations):
    """Generate stories from component specifications."""
    stories = []
    log(f"Generating {len(combinations)} component-controlled stories...")

    for i, combo in enumerate(combinations):
        if i % 10 == 0:
            log(f"  Progress: {i}/{len(combinations)}")

        prompt = COMPONENT_GENERATION_PROMPT.format(**{k: v for k, v in combo.items() if k != "source_indices"})
        messages = [
            {"role": "system", "content": "You are a creative fiction writer. Write the story directly, no preamble."},
            {"role": "user", "content": prompt}
        ]

        try:
            story = api_call(messages, temperature=0.8, max_tokens=500)
            stories.append({
                "text": story.strip(),
                "specification": combo,
                "condition": "component_controlled",
                "index": i
            })
        except Exception as e:
            log(f"  Failed on combination {i}: {e}")

    log(f"Generated {len(stories)} component-controlled stories")
    return stories


def generate_baseline_stories(combinations):
    """Generate baseline stories using only theme (title-only) and theme+setting (prompted)."""
    title_stories = []
    prompted_stories = []

    log(f"Generating {len(combinations)} baseline stories (title-only)...")
    for i, combo in enumerate(combinations):
        if i % 10 == 0:
            log(f"  Progress: {i}/{len(combinations)} (title-only)")

        prompt = BASELINE_TITLE_PROMPT.format(theme=combo["theme"])
        messages = [
            {"role": "system", "content": "You are a creative fiction writer. Write the story directly, no preamble."},
            {"role": "user", "content": prompt}
        ]

        try:
            story = api_call(messages, temperature=0.8, max_tokens=500)
            title_stories.append({
                "text": story.strip(),
                "specification": {"theme": combo["theme"]},
                "condition": "baseline_title",
                "index": i
            })
        except Exception as e:
            log(f"  Failed: {e}")

    log(f"Generating {len(combinations)} baseline stories (prompted)...")
    for i, combo in enumerate(combinations):
        if i % 10 == 0:
            log(f"  Progress: {i}/{len(combinations)} (prompted)")

        prompt = BASELINE_PROMPT_PROMPT.format(setting=combo["setting"], theme=combo["theme"])
        messages = [
            {"role": "system", "content": "You are a creative fiction writer. Write the story directly, no preamble."},
            {"role": "user", "content": prompt}
        ]

        try:
            story = api_call(messages, temperature=0.8, max_tokens=500)
            prompted_stories.append({
                "text": story.strip(),
                "specification": {"setting": combo["setting"], "theme": combo["theme"]},
                "condition": "baseline_prompted",
                "index": i
            })
        except Exception as e:
            log(f"  Failed: {e}")

    log(f"Generated {len(title_stories)} title-only and {len(prompted_stories)} prompted baseline stories")
    return title_stories, prompted_stories


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: Evaluate Stories
# ═══════════════════════════════════════════════════════════════════════════════

EVAL_QUALITY_PROMPT = """Rate this short story on these dimensions (1-5 scale, where 5 is best):

Story:
"{story}"

Rate on:
1. Coherence: Are events logically connected? Does the story make sense?
2. Creativity: Is the story original and engaging? Does it avoid clichés?
3. Character quality: Is the character believable and interesting?
4. Language quality: Is the writing varied and expressive?
5. Overall quality: Your holistic assessment of the story.

Return valid JSON:
{{"coherence": X, "creativity": X, "character_quality": X, "language_quality": X, "overall_quality": X}}"""

EVAL_CONTROLLABILITY_PROMPT = """Rate how well this story follows the given specification (1-5 scale, where 5 = perfectly follows all elements).

Specification:
- Setting: {setting}
- Main character: {protagonist}
- Conflict type: {conflict_type}
- Theme: {theme}
- Emotional arc: {emotional_arc}
- Tone: {tone}

Story:
"{story}"

For each specification element, rate adherence (1-5). Return valid JSON:
{{"setting_adherence": X, "character_adherence": X, "conflict_adherence": X, "theme_adherence": X, "arc_adherence": X, "tone_adherence": X, "overall_adherence": X}}"""


def evaluate_quality(all_stories):
    """Evaluate story quality using GPT-4.1-as-judge."""
    log(f"Evaluating quality of {len(all_stories)} stories...")

    for i, s in enumerate(all_stories):
        if i % 20 == 0:
            log(f"  Quality eval progress: {i}/{len(all_stories)}")

        prompt = EVAL_QUALITY_PROMPT.format(story=s["text"][:1000])
        messages = [
            {"role": "system", "content": "You are a literary critic. Rate stories fairly and consistently. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = api_call(messages, model=EVAL_MODEL, temperature=0.1, max_tokens=200,
                              response_format={"type": "json_object"})
            scores = json.loads(response)
            s["quality_scores"] = scores
        except Exception as e:
            log(f"  Quality eval failed for story {i}: {e}")
            s["quality_scores"] = None

    return all_stories


def evaluate_controllability(component_stories, combinations):
    """Evaluate how well component-controlled stories follow specifications."""
    log(f"Evaluating controllability of {len(component_stories)} stories...")

    for i, s in enumerate(component_stories):
        if i % 10 == 0:
            log(f"  Controllability eval progress: {i}/{len(component_stories)}")

        spec = s["specification"]
        prompt = EVAL_CONTROLLABILITY_PROMPT.format(
            setting=spec.get("setting", ""),
            protagonist=spec.get("protagonist", ""),
            conflict_type=spec.get("conflict_type", ""),
            theme=spec.get("theme", ""),
            emotional_arc=spec.get("emotional_arc", ""),
            tone=spec.get("tone", ""),
            story=s["text"][:1000]
        )
        messages = [
            {"role": "system", "content": "You are a literary analyst. Rate specification adherence fairly. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = api_call(messages, model=EVAL_MODEL, temperature=0.1, max_tokens=200,
                              response_format={"type": "json_object"})
            scores = json.loads(response)
            s["controllability_scores"] = scores
        except Exception as e:
            log(f"  Controllability eval failed for story {i}: {e}")
            s["controllability_scores"] = None

    return component_stories


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: Diversity Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_diversity_metrics(stories_by_condition):
    """Compute lexical diversity metrics for each condition."""
    metrics = {}

    for condition, stories in stories_by_condition.items():
        texts = [s["text"] for s in stories if s.get("text")]

        # Unique trigram ratio (across all stories in condition)
        all_trigrams = []
        all_words = []
        for text in texts:
            words = text.lower().split()
            all_words.extend(words)
            trigrams = [tuple(words[j:j+3]) for j in range(len(words)-2)]
            all_trigrams.extend(trigrams)

        unique_trigram_ratio = len(set(all_trigrams)) / max(len(all_trigrams), 1)
        unique_word_ratio = len(set(all_words)) / max(len(all_words), 1)

        # Per-story metrics
        per_story_unique_words = []
        per_story_lengths = []
        for text in texts:
            words = text.lower().split()
            per_story_unique_words.append(len(set(words)) / max(len(words), 1))
            per_story_lengths.append(len(words))

        # Inter-story repetition: average pairwise trigram overlap
        story_trigram_sets = []
        for text in texts:
            words = text.lower().split()
            trigrams = set(tuple(words[j:j+3]) for j in range(len(words)-2))
            story_trigram_sets.append(trigrams)

        pairwise_overlaps = []
        for a in range(len(story_trigram_sets)):
            for b in range(a+1, min(a+20, len(story_trigram_sets))):  # Sample pairs for efficiency
                if story_trigram_sets[a] and story_trigram_sets[b]:
                    overlap = len(story_trigram_sets[a] & story_trigram_sets[b]) / \
                              min(len(story_trigram_sets[a]), len(story_trigram_sets[b]))
                    pairwise_overlaps.append(overlap)

        metrics[condition] = {
            "n_stories": len(texts),
            "unique_trigram_ratio": unique_trigram_ratio,
            "unique_word_ratio": unique_word_ratio,
            "mean_per_story_unique_word_ratio": np.mean(per_story_unique_words) if per_story_unique_words else 0,
            "mean_story_length": np.mean(per_story_lengths) if per_story_lengths else 0,
            "mean_inter_story_trigram_overlap": np.mean(pairwise_overlaps) if pairwise_overlaps else 0,
            "vocab_size": len(set(all_words)),
        }

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7: Statistical Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def statistical_analysis(component_stories, title_stories, prompted_stories):
    """Run statistical tests comparing conditions."""
    from scipy import stats

    results = {}

    # Get quality scores for each condition
    conditions = {
        "component_controlled": component_stories,
        "baseline_title": title_stories,
        "baseline_prompted": prompted_stories,
    }

    quality_dims = ["coherence", "creativity", "character_quality", "language_quality", "overall_quality"]

    for dim in quality_dims:
        scores = {}
        for cond, stories in conditions.items():
            vals = [s["quality_scores"][dim] for s in stories if s.get("quality_scores") and dim in s.get("quality_scores", {})]
            scores[cond] = vals

        # Component vs title baseline
        if scores.get("component_controlled") and scores.get("baseline_title"):
            n = min(len(scores["component_controlled"]), len(scores["baseline_title"]))
            a, b = scores["component_controlled"][:n], scores["baseline_title"][:n]
            stat, p = stats.wilcoxon(a, b, alternative='two-sided')
            d = (np.mean(a) - np.mean(b)) / np.sqrt((np.var(a) + np.var(b)) / 2) if np.var(a) + np.var(b) > 0 else 0
            results[f"{dim}_component_vs_title"] = {
                "component_mean": float(np.mean(a)), "component_std": float(np.std(a)),
                "title_mean": float(np.mean(b)), "title_std": float(np.std(b)),
                "wilcoxon_stat": float(stat), "p_value": float(p),
                "cohens_d": float(d), "n": n
            }

        # Component vs prompted baseline
        if scores.get("component_controlled") and scores.get("baseline_prompted"):
            n = min(len(scores["component_controlled"]), len(scores["baseline_prompted"]))
            a, b = scores["component_controlled"][:n], scores["baseline_prompted"][:n]
            try:
                stat, p = stats.wilcoxon(a, b, alternative='two-sided')
            except ValueError:
                stat, p = 0.0, 1.0
            d = (np.mean(a) - np.mean(b)) / np.sqrt((np.var(a) + np.var(b)) / 2) if np.var(a) + np.var(b) > 0 else 0
            results[f"{dim}_component_vs_prompted"] = {
                "component_mean": float(np.mean(a)), "component_std": float(np.std(a)),
                "prompted_mean": float(np.mean(b)), "prompted_std": float(np.std(b)),
                "wilcoxon_stat": float(stat), "p_value": float(p),
                "cohens_d": float(d), "n": n
            }

    # Controllability summary
    ctrl_scores = [s["controllability_scores"]["overall_adherence"]
                   for s in component_stories
                   if s.get("controllability_scores") and "overall_adherence" in s.get("controllability_scores", {})]
    if ctrl_scores:
        results["controllability_summary"] = {
            "mean": float(np.mean(ctrl_scores)),
            "std": float(np.std(ctrl_scores)),
            "median": float(np.median(ctrl_scores)),
            "n": len(ctrl_scores)
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8: Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def create_visualizations(component_stories, title_stories, prompted_stories, diversity_metrics, stat_results):
    """Create all plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=1.1)

    # ── Plot 1: Quality comparison across conditions ──
    quality_dims = ["coherence", "creativity", "character_quality", "language_quality", "overall_quality"]
    conditions = {
        "Component\nControlled": component_stories,
        "Baseline\n(Title)": title_stories,
        "Baseline\n(Prompted)": prompted_stories,
    }

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for idx, dim in enumerate(quality_dims):
        data = []
        labels = []
        for label, stories in conditions.items():
            vals = [s["quality_scores"][dim] for s in stories if s.get("quality_scores") and dim in s.get("quality_scores", {})]
            data.append(vals)
            labels.append(label)

        bp = axes[idx].boxplot(data, labels=labels, patch_artist=True,
                               boxprops=dict(facecolor='lightblue', alpha=0.7))
        colors = ['#4CAF50', '#FF9800', '#2196F3']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        axes[idx].set_title(dim.replace('_', ' ').title(), fontsize=12)
        axes[idx].set_ylim(0.5, 5.5)
        axes[idx].set_ylabel('Score (1-5)' if idx == 0 else '')

    plt.suptitle('Story Quality Comparison Across Conditions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/plots/quality_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    log("Saved quality_comparison.png")

    # ── Plot 2: Controllability scores ──
    ctrl_dims = ["setting_adherence", "character_adherence", "conflict_adherence",
                 "theme_adherence", "arc_adherence", "tone_adherence", "overall_adherence"]
    ctrl_data = {dim: [] for dim in ctrl_dims}
    for s in component_stories:
        if s.get("controllability_scores"):
            for dim in ctrl_dims:
                if dim in s["controllability_scores"]:
                    ctrl_data[dim].append(s["controllability_scores"][dim])

    fig, ax = plt.subplots(figsize=(10, 6))
    means = [np.mean(ctrl_data[d]) if ctrl_data[d] else 0 for d in ctrl_dims]
    stds = [np.std(ctrl_data[d]) if ctrl_data[d] else 0 for d in ctrl_dims]
    labels = [d.replace('_adherence', '').replace('_', ' ').title() for d in ctrl_dims]

    bars = ax.bar(labels, means, yerr=stds, capsize=5, color='#4CAF50', alpha=0.7, edgecolor='black')
    ax.set_ylim(0, 5.5)
    ax.set_ylabel('Adherence Score (1-5)')
    ax.set_title('Component Specification Adherence\n(How well stories follow their component specs)', fontsize=13, fontweight='bold')
    ax.axhline(y=3.0, color='red', linestyle='--', alpha=0.5, label='Midpoint (3.0)')
    ax.legend()
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1, f'{m:.2f}',
                ha='center', va='bottom', fontsize=10)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/plots/controllability.png', dpi=150, bbox_inches='tight')
    plt.close()
    log("Saved controllability.png")

    # ── Plot 3: Diversity metrics ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    cond_names = list(diversity_metrics.keys())
    display_names = [n.replace('_', '\n') for n in cond_names]

    # Unique trigram ratio
    vals = [diversity_metrics[c]["unique_trigram_ratio"] for c in cond_names]
    axes[0].bar(display_names, vals, color=['#4CAF50', '#FF9800', '#2196F3'], alpha=0.7, edgecolor='black')
    axes[0].set_title('Unique Trigram Ratio\n(higher = more diverse)')
    axes[0].set_ylim(0, 1.1)
    for j, v in enumerate(vals):
        axes[0].text(j, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

    # Vocabulary size
    vals = [diversity_metrics[c]["vocab_size"] for c in cond_names]
    axes[1].bar(display_names, vals, color=['#4CAF50', '#FF9800', '#2196F3'], alpha=0.7, edgecolor='black')
    axes[1].set_title('Vocabulary Size\n(unique words across all stories)')
    for j, v in enumerate(vals):
        axes[1].text(j, v + 10, str(v), ha='center', fontsize=10)

    # Inter-story trigram overlap
    vals = [diversity_metrics[c]["mean_inter_story_trigram_overlap"] for c in cond_names]
    axes[2].bar(display_names, vals, color=['#4CAF50', '#FF9800', '#2196F3'], alpha=0.7, edgecolor='black')
    axes[2].set_title('Mean Inter-Story Trigram Overlap\n(lower = more diverse)')
    axes[2].set_ylim(0, max(vals)*1.5 if max(vals) > 0 else 0.1)
    for j, v in enumerate(vals):
        axes[2].text(j, v + 0.001, f'{v:.4f}', ha='center', fontsize=10)

    plt.suptitle('Diversity Metrics by Condition', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/plots/diversity_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    log("Saved diversity_metrics.png")

    # ── Plot 4: Effect sizes ──
    fig, ax = plt.subplots(figsize=(12, 6))
    comparisons = []
    effect_sizes = []
    p_values = []

    for key, val in stat_results.items():
        if "cohens_d" in val:
            comparisons.append(key.replace('_', ' ').replace('component vs', '\nvs'))
            effect_sizes.append(val["cohens_d"])
            p_values.append(val["p_value"])

    if comparisons:
        colors = ['#4CAF50' if p < 0.05 else '#FF9800' for p in p_values]
        bars = ax.barh(comparisons, effect_sizes, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel("Cohen's d (effect size)")
        ax.set_title("Effect Sizes: Component-Controlled vs Baselines\n(green = p<0.05, orange = n.s.)", fontsize=13, fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.axvline(x=0.2, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(x=-0.2, color='gray', linestyle=':', alpha=0.5)
        for bar, d, p in zip(bars, effect_sizes, p_values):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f'd={d:.2f}, p={p:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/plots/effect_sizes.png', dpi=150, bbox_inches='tight')
    plt.close()
    log("Saved effect_sizes.png")

    # ── Plot 5: Component distribution (taxonomy) ──
    # Will be created separately if taxonomy data available


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log("=" * 70)
    log("INSTRUCT-STORYMIX EXPERIMENT")
    log("=" * 70)

    config = {
        "seed": SEED,
        "model": MODEL,
        "eval_model": EVAL_MODEL,
        "n_stories_extract": N_STORIES_EXTRACT,
        "n_stories_generate": N_STORIES_GENERATE,
        "timestamp": datetime.now().isoformat()
    }

    # Step 1: Load stories
    log("\n─── STEP 1: Loading stories ───")
    stories = load_stories()

    # Step 2: Extract components
    log("\n─── STEP 2: Extracting components ───")
    components = extract_components(stories)
    with open(f"{RESULTS_DIR}/components.json", "w") as f:
        json.dump(components, f, indent=2)

    # Step 3: Build taxonomy and create combinations
    log("\n─── STEP 3: Building taxonomy ───")
    taxonomy = build_taxonomy(components)
    with open(f"{RESULTS_DIR}/taxonomy.json", "w") as f:
        json.dump(taxonomy, f, indent=2, default=str)

    combinations = create_novel_combinations(components)
    with open(f"{RESULTS_DIR}/combinations.json", "w") as f:
        json.dump(combinations, f, indent=2)

    # Step 4: Generate stories
    log("\n─── STEP 4: Generating stories ───")
    component_stories = generate_component_stories(combinations)
    title_stories, prompted_stories = generate_baseline_stories(combinations)

    # Save generated stories
    with open(f"{RESULTS_DIR}/generated_component.json", "w") as f:
        json.dump(component_stories, f, indent=2)
    with open(f"{RESULTS_DIR}/generated_title.json", "w") as f:
        json.dump(title_stories, f, indent=2)
    with open(f"{RESULTS_DIR}/generated_prompted.json", "w") as f:
        json.dump(prompted_stories, f, indent=2)

    # Step 5: Evaluate stories
    log("\n─── STEP 5: Evaluating stories ───")
    all_stories = component_stories + title_stories + prompted_stories
    all_stories = evaluate_quality(all_stories)

    # Split back
    n_comp = len(component_stories)
    n_title = len(title_stories)
    component_stories = all_stories[:n_comp]
    title_stories = all_stories[n_comp:n_comp + n_title]
    prompted_stories = all_stories[n_comp + n_title:]

    # Evaluate controllability for component stories
    component_stories = evaluate_controllability(component_stories, combinations)

    # Save evaluated stories
    with open(f"{RESULTS_DIR}/evaluated_component.json", "w") as f:
        json.dump(component_stories, f, indent=2)
    with open(f"{RESULTS_DIR}/evaluated_title.json", "w") as f:
        json.dump(title_stories, f, indent=2)
    with open(f"{RESULTS_DIR}/evaluated_prompted.json", "w") as f:
        json.dump(prompted_stories, f, indent=2)

    # Step 6: Diversity metrics
    log("\n─── STEP 6: Computing diversity metrics ───")
    diversity_metrics = compute_diversity_metrics({
        "component_controlled": component_stories,
        "baseline_title": title_stories,
        "baseline_prompted": prompted_stories,
    })
    with open(f"{RESULTS_DIR}/diversity_metrics.json", "w") as f:
        json.dump(diversity_metrics, f, indent=2)

    log("Diversity metrics:")
    for cond, m in diversity_metrics.items():
        log(f"  {cond}: trigram_ratio={m['unique_trigram_ratio']:.4f}, vocab={m['vocab_size']}, overlap={m['mean_inter_story_trigram_overlap']:.4f}")

    # Step 7: Statistical analysis
    log("\n─── STEP 7: Statistical analysis ───")
    stat_results = statistical_analysis(component_stories, title_stories, prompted_stories)
    with open(f"{RESULTS_DIR}/statistical_results.json", "w") as f:
        json.dump(stat_results, f, indent=2)

    log("\nKey results:")
    for key, val in stat_results.items():
        if "p_value" in val:
            log(f"  {key}: d={val['cohens_d']:.3f}, p={val['p_value']:.4f}")
        elif "mean" in val:
            log(f"  {key}: mean={val['mean']:.3f} ± {val['std']:.3f}")

    # Step 8: Visualizations
    log("\n─── STEP 8: Creating visualizations ───")
    create_visualizations(component_stories, title_stories, prompted_stories, diversity_metrics, stat_results)

    # Save config
    config["completion_time"] = datetime.now().isoformat()
    with open(f"{RESULTS_DIR}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    log("\n" + "=" * 70)
    log("EXPERIMENT COMPLETE")
    log(f"Results saved to {RESULTS_DIR}/")
    log("=" * 70)

    return {
        "component_stories": component_stories,
        "title_stories": title_stories,
        "prompted_stories": prompted_stories,
        "diversity_metrics": diversity_metrics,
        "stat_results": stat_results,
        "taxonomy": taxonomy,
        "config": config,
    }


if __name__ == "__main__":
    results = main()
