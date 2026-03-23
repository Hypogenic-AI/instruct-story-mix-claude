# Cloned Repositories

## 1. DOC Story Generation (Original)
- **URL**: https://github.com/yangkevin2/doc-story-generation
- **Purpose**: Outline-controlled long story generation (ACL 2023)
- **Location**: `code/doc-story-generation/`
- **Key files**: `scripts/main.py`, `story_generation/` library
- **Dependencies**: Python 3.8.15, PyTorch, OpenAI API (legacy v0.16), OPT-175B via Alpa
- **Notes**: Requires legacy OpenAI API. Has downloadable data: `doc_data.zip`, `doc_outputs.zip` from S3.

## 2. DOC Story Generation V2 (Facebook Research)
- **URL**: https://github.com/facebookresearch/doc-storygen-v2
- **Purpose**: Modernized DOC for LLaMA-2/ChatGPT (simplified codebase)
- **Location**: `code/doc-storygen-v2/`
- **Key files**: `scripts/{premise,plan,story}/generate.py`, `prompts.json`
- **Dependencies**: Python 3.9+, VLLM, transformers
- **Notes**: Best choice for running DOC pipeline with modern open-source LLMs. Human eval data available.

## 3. Tell Me a Story / Agents' Room (Google DeepMind)
- **URL**: https://github.com/google-deepmind/tell_me_a_story
- **Purpose**: Dataset + evaluation for multi-agent story generation (ICLR 2025)
- **Location**: `code/tell_me_a_story/`
- **Key files**: `README.md` (decryption instructions), `keys.zip`
- **Dependencies**: `cryptography` package only
- **Notes**: Dataset-only repo (no model code). Encrypted JSONL files downloadable from GCS. CC-BY 4.0.

## 4. PlotMachines (EMNLP 2020)
- **URL**: https://github.com/hrashkin/plotmachines
- **Purpose**: Outline-conditioned story generation with dynamic plot state tracking
- **Location**: `code/plotmachines/`
- **Key files**: `src/model/{train,generate_stories,model}.py`, `src/preprocessing/extract_outlines.py`
- **Dependencies**: `transformers==2.0.0` (very old), PyTorch, spacy, nltk
- **Notes**: Uses old transformers version. WikiPlots via separate scraper. NYTimes requires LDC license.

## 5. Awesome-Story-Generation (Survey)
- **URL**: https://github.com/yingpengma/Awesome-Story-Generation
- **Purpose**: Curated paper list covering story generation in the LLM era
- **Location**: `code/Awesome-Story-Generation/`
- **Notes**: Reference/survey only, no runnable code. ~596 stars. Useful as literature map.

## Not Cloned
- **Plan-And-Write** (https://bitbucket.org/VioletPeng/language-model): **404 — repository is down**
