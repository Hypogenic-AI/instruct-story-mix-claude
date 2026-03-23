"""
Supplementary Experiment: Testing the "Constraint Bandwidth" Hypothesis

Does the quality penalty come from:
(a) Having more constraints (constraint overload), or
(b) Mismatched/incoherent component combinations (mixing artifacts)?

Test: Generate stories with COHERENT specs (from same source story) vs
MIXED specs (from different stories) vs BASELINES.
"""

import os, json, random, time, numpy as np
from datetime import datetime
from openai import OpenAI

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
MODEL = "gpt-4.1"
client = OpenAI()

def api_call(messages, temperature=0.7, max_tokens=500):
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
                response_format={"type": "json_object"} if "JSON" in messages[-1]["content"] else None
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"  Retry {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    return None

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

COMPONENT_PROMPT = """Write a short story (exactly 5 sentences) that follows these specifications:

Setting: {setting}
Main character: {protagonist}
Conflict type: {conflict_type}
Theme: {theme}
Emotional arc: {emotional_arc}
Tone: {tone}

Important: Write exactly 5 sentences. Make the story coherent, creative, and engaging while adhering to ALL the specifications above."""

EVAL_PROMPT = """Rate this short story on these dimensions (1-5 scale, where 5 is best):

Story:
"{story}"

Rate on:
1. Coherence: Are events logically connected? Does the story make sense?
2. Creativity: Is the story original and engaging? Does it avoid clichés?
3. Character quality: Is the character believable and interesting?
4. Language quality: Is the writing varied and expressive?
5. Overall quality: Your holistic assessment of the story.

Return valid JSON only:
{{"coherence": X, "creativity": X, "character_quality": X, "language_quality": X, "overall_quality": X}}"""

def main():
    # Load previously extracted components
    with open("results/components.json") as f:
        components = json.load(f)

    log(f"Loaded {len(components)} component sets")

    # Create COHERENT specs (all components from same story)
    coherent_specs = []
    for c in components[:25]:
        coherent_specs.append({
            "setting": c["setting"],
            "protagonist": c["protagonist"],
            "conflict_type": c["conflict_type"],
            "theme": c["theme"],
            "emotional_arc": c["emotional_arc"],
            "tone": c["tone"],
            "type": "coherent"
        })

    # Create MIXED specs (components from different stories - same as main experiment)
    mixed_specs = []
    for i in range(25):
        sources = random.sample(components, 5)
        mixed_specs.append({
            "setting": sources[0]["setting"],
            "protagonist": sources[1]["protagonist"],
            "conflict_type": sources[2]["conflict_type"],
            "theme": sources[3]["theme"],
            "emotional_arc": sources[4]["emotional_arc"],
            "tone": random.choice(components)["tone"],
            "type": "mixed"
        })

    # Generate stories for both conditions
    all_stories = []
    for spec_type, specs in [("coherent", coherent_specs), ("mixed", mixed_specs)]:
        log(f"Generating {len(specs)} {spec_type} stories...")
        for i, spec in enumerate(specs):
            prompt = COMPONENT_PROMPT.format(**{k:v for k,v in spec.items() if k != "type"})
            messages = [
                {"role": "system", "content": "You are a creative fiction writer. Write the story directly, no preamble."},
                {"role": "user", "content": prompt}
            ]
            try:
                story = api_call(messages, temperature=0.8)
                all_stories.append({"text": story.strip(), "condition": spec_type, "spec": spec, "index": i})
            except:
                pass

    # Evaluate
    log(f"Evaluating {len(all_stories)} stories...")
    for i, s in enumerate(all_stories):
        if i % 10 == 0:
            log(f"  Eval progress: {i}/{len(all_stories)}")
        prompt = EVAL_PROMPT.format(story=s["text"][:1000])
        messages = [
            {"role": "system", "content": "You are a literary critic. Rate stories fairly. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ]
        try:
            response = api_call(messages, temperature=0.1, max_tokens=200)
            s["quality"] = json.loads(response)
        except:
            s["quality"] = None

    # Analyze
    coherent_scores = [s["quality"] for s in all_stories if s["condition"] == "coherent" and s.get("quality")]
    mixed_scores = [s["quality"] for s in all_stories if s["condition"] == "mixed" and s.get("quality")]

    log("\n=== SUPPLEMENTARY RESULTS ===")
    for dim in ["coherence", "creativity", "character_quality", "language_quality", "overall_quality"]:
        c_vals = [s[dim] for s in coherent_scores if dim in s]
        m_vals = [s[dim] for s in mixed_scores if dim in s]
        log(f"  {dim}: coherent={np.mean(c_vals):.2f}±{np.std(c_vals):.2f}, mixed={np.mean(m_vals):.2f}±{np.std(m_vals):.2f}")

    results = {
        "coherent": {dim: {"mean": float(np.mean([s[dim] for s in coherent_scores if dim in s])),
                           "std": float(np.std([s[dim] for s in coherent_scores if dim in s]))}
                     for dim in ["coherence", "creativity", "character_quality", "language_quality", "overall_quality"]},
        "mixed": {dim: {"mean": float(np.mean([s[dim] for s in mixed_scores if dim in s])),
                        "std": float(np.std([s[dim] for s in mixed_scores if dim in s]))}
                  for dim in ["coherence", "creativity", "character_quality", "language_quality", "overall_quality"]},
        "n_coherent": len(coherent_scores),
        "n_mixed": len(mixed_scores),
    }

    with open("results/supplementary_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("results/supplementary_stories.json", "w") as f:
        json.dump(all_stories, f, indent=2)

    log("Supplementary experiment complete.")
    return results

if __name__ == "__main__":
    main()
