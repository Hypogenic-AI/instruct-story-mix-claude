# Downloaded Datasets

Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: WritingPrompts

### Overview
- **Source**: HuggingFace `euclaise/writingprompts` (originally Fan et al., 2018)
- **Size**: 303,358 examples (~602 MB)
- **Format**: HuggingFace Dataset
- **Task**: Prompt-conditioned story generation
- **Splits**: train (272,600), validation (15,620), test (15,138)
- **Columns**: `prompt`, `story`
- **License**: Research use

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("euclaise/writingprompts")
dataset.save_to_disk("datasets/writingprompts")
```

### Loading the Dataset

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/writingprompts")
print(dataset['train'][0])  # {'prompt': '...', 'story': '...'}
```

### Notes
- Most widely used story generation benchmark
- Prompts are from Reddit r/WritingPrompts; stories are user-written responses
- Variable quality; some stories are very short or low-effort
- Used by: DOC, Re3, PlotMachines, Goldfarb-Tarrant et al., Fan et al.

---

## Dataset 2: ROCStories

### Overview
- **Source**: HuggingFace `mintujupally/ROCStories`
- **Size**: 98,161 examples (~15 MB)
- **Format**: HuggingFace Dataset
- **Task**: Short commonsense story generation
- **Splits**: train (88,344), test (9,817)
- **Columns**: `text` (full 5-sentence story concatenated)

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("mintujupally/ROCStories")
dataset.save_to_disk("datasets/rocstories")
```

### Loading the Dataset

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/rocstories")
```

### Notes
- Five-sentence commonsense stories capturing causal/temporal relations
- This version has sentences concatenated into single `text` field
- Used by Plan-And-Write (Yao et al., 2019) as primary dataset
- Good for component extraction experiments due to consistent structure

---

## Dataset 3: WikiPlots (Movie Plots with Summaries)

### Overview
- **Source**: HuggingFace `vishnupriyavr/wiki-movie-plots-with-summaries`
- **Size**: 34,886 examples (~59 MB)
- **Format**: HuggingFace Dataset
- **Task**: Plot-conditioned story generation
- **Splits**: train only
- **Columns**: `Title`, `Plot`, `PlotSummary`, `Genre`, `Director`, `Cast`, `Release Year`, `Origin/Ethnicity`, `Wiki Page`

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("vishnupriyavr/wiki-movie-plots-with-summaries")
dataset.save_to_disk("datasets/wikiplots")
```

### Loading the Dataset

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/wikiplots")
```

### Notes
- Wikipedia movie plot summaries with rich metadata
- Both full `Plot` and shorter `PlotSummary` fields available
- Genre information enables genre-conditioned experiments
- Used by PlotMachines and O2S (Fang et al.)

---

## Dataset 4: Tell Me a Story

### Overview
- **Source**: HuggingFace `TAUR-dev/tell_me_a_story_{train,test,validation}`
- **Size**: 230 total examples (~1.4 MB)
- **Format**: HuggingFace Dataset
- **Task**: High-quality story generation from detailed prompts
- **Splits**: train (123), validation (52), test (55)
- **Columns**: `example_id`, `inputs` (detailed prompt), `targets` (story)
- **License**: CC-BY 4.0

### Download Instructions

```python
from datasets import load_dataset
# Load each split separately
train = load_dataset("TAUR-dev/tell_me_a_story_train")
val = load_dataset("TAUR-dev/tell_me_a_story_validation")
test = load_dataset("TAUR-dev/tell_me_a_story_test")
```

### Loading the Dataset

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/tell_me_a_story")
```

### Notes
- Highest quality dataset; written by professional/skilled writers in workshops
- Average prompt length: 113 tokens; average story length: 1,498 tokens
- Predominantly sci-fi and fantasy genres
- Used by Agents' Room (Huot et al., ICLR 2025)
- Small size limits use as training data; best for evaluation

---

## Sample Data

Small sample files (10 examples each) are included for reference:
- `writingprompts_sample.json`
- `rocstories_sample.json`
- `wikiplots_sample.json`
- `tell_me_a_story_sample.json`
