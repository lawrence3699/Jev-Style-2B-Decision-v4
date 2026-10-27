---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B
base_model_relation: finetune
library_name: jev-style
pipeline_tag: text-classification
tags:
  - decision-model
  - decision-making
  - transformers
  - jev-style
  - system-one
  - calibration
  - classification
  - long-context
  - vision
  - multimodal
  - qwen3.5
  - llm-routing
  - guardrails
---

# Jev-Style-2B-Decision-v4

**Website:** [jevstyle.com](https://jevstyle.com) · **GitHub:** [jev-style](https://github.com/lawrence3699/jev-style)

**Jev-Style decision series:** [v1 · 2B](https://huggingface.co/chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-GGUF) → [v2 · 2B](https://huggingface.co/chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-v2) → [v3 · 0.8B](https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3) → [v3 · 2B](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v3) → **v4 · 2B (this model)**

**Other formats (text only):** [GGUF for llama.cpp](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4-GGUF) · [MLX for Apple silicon](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4-MLX)

**Jev-style decisions about text and images.** Give it a state (text, JSON, and now images) and typed questions; it
returns a calibrated probability for every option in one pass. 2.21B parameters (Qwen3.5-2B with its vision
tower), open weights, Apache-2.0.

| | **Jev-Style 2B v4** (this model) | Jev-Style 2B v3 |
|---|---:|---:|
| Decision Index 0.3 public index (self-run, complete) | **44.37** | 34.05 |
| Decision Index Vision public index (self-run, complete) | **57.55** | no image input |
| JevBench v1.4.1, public items (231) | **74.9%** | 73.6% |
| Inputs | **text, JSON and images** | text, JSON |
| Context per call | **32,768 tokens** | 25,600 tokens |

<sub>Protocols under [Results](#results). One request at a time on an A100 (bfloat16, CUDA graphs), the median latency
is 16 ms.</sub>

## What it does

A state and a typed question go in; a probability for every option comes out. The model never generates text and
cannot answer outside the options it is given.

- **choice**: pick one of N named options;
- **noul** (yes/no): the probability that a statement about the state is true;
- **score**: a distribution over 2 to 10 ordered levels.

New in v4: **images in the state** (photos, screenshots, receipts, charts, scanned forms), **32,768 tokens** per
call (v3: 25,600), and **one calibrated temperature per question type**.

```python
from jev_style_decision import JevStyleDecision

m = JevStyleDecision(".")            # float32 on the best available device: CUDA > MPS > CPU

m.decide({"messages": ["Refund still missing after 3 weeks"]},
         "Which team should handle this ticket?",
         options={"billing": "payments, refunds", "tech": "bugs, crashes", "sales": "pricing, plans"})
# answer "billing"   probabilities {"billing": 0.994, "tech": 0.004, "sales": 0.002}

# several questions about one state: one call, the state is read once
state = {"order": {"id": "4471", "status": "packed, not shipped yet"},
         "policy": "Orders can be cancelled free of charge until they ship.",
         "message": "Please cancel my order, I bought the wrong size. I need the money back before Friday!"}
m.decide_many(state, [
    "Can this order still be cancelled free of charge?",                    # true / false
    {"t": "choice", "ins": "What should the agent do next?",
     "crit": {"cancel": "cancel the order and refund it", "exchange": "offer a different size instead",
              "escalate": "hand over to a human"}},
    {"t": "score", "ins": "How urgent is the customer's request?",
     "crit": ["not urgent", "somewhat urgent", "very urgent"]},
])
# "true" 0.912 (false 0.088) · "cancel" 0.965 (exchange 0.027, escalate 0.008) · level "2" 0.830 (1: 0.163, 0: 0.007)

# images go in the state
receipt = {"receipt": "<|image_1|>", "policy": "Meals over 50 EUR need a manager approval."}
m.decide(receipt, "Does this receipt need a manager approval?", images=["assets/receipt.png"])
# answer "true"   probabilities {"false": 0.116, "true": 0.884}   (440 image tokens)
m.decide(receipt, "Which expense category is this receipt?", images=["assets/receipt.png"],
         options=["meals", "travel", "accommodation", "office_supplies"])
# answer "meals"  probabilities {"meals": 0.996, "travel": 0.002, "accommodation": 0.001, "office_supplies": 0.001}
```

<sub>Outputs of this repository's runtime on the released weights (float32, one A100), rounded; every example was
run once as written. `assets/receipt.png` is a made-up receipt (total 66.90 EUR) drawn for this card.</sub>

## Quick start

```bash
pip install -U huggingface_hub
hf download chaoliangUNSW/Jev-Style-2B-Decision-v4 --local-dir jev-v4-2b && cd jev-v4-2b
pip install -r requirements.txt     # torch 2.14.1, torchvision 0.29.1, transformers 5.17.0, tokenizers 0.23.3, pillow

python jev_style_decision.py --model-dir . \
  --state "The user asked to cancel the order" \
  --question "What should happen?" \
  --options '{"cancel": "cancel the order", "ship": "ship it"}'

python jev_style_decision.py --model-dir . \
  --state-json '{"receipt": "<|image_1|>", "policy": "Meals over 50 EUR need a manager approval."}' \
  --image assets/receipt.png --question "Does this receipt need a manager approval?"
```

`decide` returns `answer`, `probabilities`, the raw `scores`, the `temperature` used, `top_probability`,
`entropy_concentration`, token counts (`input_tokens`, `state_tokens`, `head_tokens`, `image_tokens`), `blocks`,
`catalogue_overflow`, `model` and `backend`. Batch mode reads JSON lines (`--jsonl file|-`); consecutive rows with
the same state and images share one state computation. `--verify` first checks the weights, tokenizer, configs,
runtime and reference code against the sha256 manifest. A true/false question returns `{"false": p, "true": p}`;
a score question returns the level indices `"0"`, `"1"`, ... as option names.

**Images.** Put `<|image_1|>` … `<|image_n|>` in the state where each image belongs (each exactly once) and pass the
images in that order (`images=[...]`: paths, bytes, data URLs, base64 or PIL images; CLI `--image`, repeatable). A
state without placeholders gets them added the way the training data writes them (`{"image": "<|image_1|>", ...}`).
Each image is read at its native resolution up to 1.6 MP (`max_pixels`; larger images are downscaled, aspect kept).
Image tokens count toward the 32,768-token budget (one token per 32 × 32 px: about 1,600 for a 1.6 MP image).

Devices: CUDA (bfloat16 or float32), Apple MPS and CPU (float32). On CUDA two speed paths are on by default; both
compute the reference scorer's numbers up to floating-point order:
- **CUDA graphs** (bfloat16; `cuda_graphs=None`, CLI `--cuda-graphs auto|on|off`): when the model loads, one graph
  per padded input length (64 to 8,192 tokens) is recorded (about a minute the first time, while Triton compiles
  its kernels; seconds after that), and each text question then runs as one replay instead of a few thousand
  kernel launches. Calls with several questions use them while that is cheaper than the fast path.
- **Fast path** (`fast=True`, CLI `--fast auto|on|off`): the other text calls with two or more questions compute
  the state once and score every question on its cache.

Image calls and single questions over 8,192 tokens use the reference scorer. In float32 the forward is
compute-bound and the graphs gain little, so `auto` records them only in bfloat16. Parity on a 2,000-request
stratified sample of the Decision Index 0.3 public suite (16,537 questions, one A100): bfloat16 picks the same top
option as float32 on 99.6% of the questions (median |Δp| 0.0014); with CUDA graphs and the fast path, bfloat16
agrees with the float32 reference on 99.7% and with the bfloat16 reference scorer on 99.6%.

## Other formats (text only)

| Build | Size | Runtime |
|---|---:|---|
| **Transformers safetensors (bf16) · this repository** | 4.43 GB (3 shards) | PyTorch on CUDA, Apple MPS or CPU (`jev_style_decision.py`); text, JSON and images |
| [GGUF F16 / Q8_0 / Q4_K_M](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4-GGUF) | 3.78 / 2.01 / 1.27 GB | llama.cpp (libllama) + the bundled `jev-score-v2` scorer; text and JSON |
| [MLX bf16 / 8-bit](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4-MLX) (one repository) | 3.76 / 2.00 GB | Apple silicon, mlx-lm 0.31.3 (`--model-dir bf16\|8bit`); text and JSON |

Every build was checked against this repository's reference scorer in float32, with gates declared before any format
was built or scored. All five pass.

| Format | Same top-1 as float32 (1,000 questions) | Max abs Δp | Accuracy (float32: 84.4%) | Long fixture (97 questions, up to 32,768 tokens) |
|---|---:|---:|---:|---:|
| GGUF F16 | 99.7% | 0.029 | 84.5% | 97 / 97 |
| GGUF Q8_0 | 99.6% | 0.062 | 84.3% | 97 / 97 |
| GGUF Q4_K_M | 97.4% | 0.321 | 83.6% | 95 / 97 |
| MLX bf16 | 99.7% | 0.036 | 84.4% | 97 / 97 |
| MLX 8-bit (affine, group 64) | 99.9% | 0.048 | 84.5% | 97 / 97 |

- **Long documents:** GGUF F16 or Q8_0, or either MLX folder; Q4_K_M is noticeably noisier on long inputs.
- **Images** need this repository's PyTorch runtime: the GGUF and MLX files hold the language model only.

<sub>Gate fixture: 1,000 questions from our own held-out development rows (≤ 4,096 tokens), accuracy = top-1 in the
gold set. Long fixture: 70 requests / 97 questions, development rows of 4,097 to 32,768 tokens plus edge cases (255
options, catalogue questions, an input of exactly 32,768 tokens); the column counts questions with the same top-1 as
the reference. For 4-bit the pre-declared gate is the accuracy drop (≤ 1.0 point). GGUF scored on an NVIDIA L4
(CUDA), MLX on an Apple M1 Max. Details, speed and validation records on each build's card.</sub>

## Results

All evaluations were self-run once with protocols fixed before the run. The temperatures were fitted on our own
calibration rows, never on benchmark items.

### Decision Index 0.3 public index: 44.37 (v3: 34.05)

| | Public index | Raw index | Knowledge & Reasoning | Language | Retrieval & Classification | Tools & Automation | Arts & Taste |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Jev-Style 2B v4** (this model, self-run) | **44.37** | **57.68** | **23.6** | **42.5** | **65.4** | **61.8** | **29.0** |
| Jev-Style 2B v3 (board) | 34.05 | 49.95 | 17.2 | 29.9 | 51.6 | 52.8 | 19.0 |

<sub>Decision Index public suite 0.3 (kit `25c6ec7`, the kit's own runner and scorer), one complete run: 140,178
scoreable requests, all answered, 0 unsupported, 0 errors (`scores.json` says `"complete": true`). bfloat16 on one
A100 80GB, fast path on. Area columns are the chance-corrected area skills × 100. v3 row: the public numbers on the
Decision Index 0.4.1 board (2026-10-26). The board's headline Full score comes only from private tests (60% the
same skills, 40% new domains) run by the maintainers; this public index is not part of it, and this is not an
official board entry until the maintainers review it. Several benchmarks' train splits are in the training data
(see [Training data and licences](#training-data-and-licences)), so those results are not zero-shot.</sub>

Every area is up. Among the benchmarks that count towards the index, the largest gains over v3 (raw metric) are
RAGTruth† 0.18 → 0.70, When2Call† 0.50 → 0.87, iSarcasmEval† 0.18 → 0.47, HoVer† 0.60 → 0.86, ContractNLI† 0.59 →
0.84, Amazon ESCI† 0.35 → 0.56, WinoGrande† 0.61 → 0.77, BPoMP‡ 0.64 → 0.77, New Yorker† 0.53 → 0.65 and CLadder‡
0.62 → 0.72. Intent and tool routing went up as well: BANKING77† 0.80 → 0.88, CLINC150† 0.92 → 0.94, BFCL‡ 0.88 → 0.92.
<sub>† official train split in the training data; ‡ a code generator imitating the benchmark's format is in the
training data (own vocabulary and templates, no benchmark rows).</sub>

### Speed: 16 ms median per request

| 750 Decision Index 0.3 public requests, one at a time, A100 80GB, bfloat16 | Median | Mean | p80 | p95 |
|---|---:|---:|---:|---:|
| CUDA graphs + fast path (the default on CUDA in bfloat16) | **16.4 ms** | **58.8 ms** | **60.9 ms** | **211 ms** |
| fast path only (`cuda_graphs=False`) | 71.5 ms | 104.0 ms | 80.5 ms | 214 ms |

<sub>Requests sampled uniformly from the public suite files (seed 20261027), run with this repository's own code
only (`src/` on the path, this folder as the model). 663 of them are also in the full run (the others are suite rows
that the 0.3 edition leaves out of the run): with the fast path only their answers match the full run exactly
(1,585 questions); with CUDA graphs 1,579 of the 1,585 have the same top option (median |Δp| 0.000002, max 0.036).
In-process request wall time, model loading and graph recording excluded, nothing else on the GPU. Not the
maintainers' RTX PRO 6000 measurement.</sub>

### JevBench v1.4.1: 74.9% on the public items

| System | Public accuracy (231) | Easy (48) | Standard (72) | Hard (111) |
|---|---:|---:|---:|---:|
| **Jev-Style 2B v4** (this model, self-run) | **74.9% (173)** | 100% | 94.4% | **51.4%** |
| Jev-Style 2B v3 (self-run) | 73.6% (170) | 100% | 95.8% | 47.7% |
| decider-2b (Mapika, Qwen3.5-2B-Base) | 71.0% | | | |
| Open-Jev 2B (Zefan Cai, Qwen3.5-2B + LoRA) | 64.5% | | | |

<sub>JevBench v1.4.1 (github.com/fstandhartinger/jevbench, tag v1.4.1), public items only, self-run once with the
official harness through our `/v1/systemone` server over the same scorer (float32, Apple MPS); 95% CI 68.9–80.0% (Wilson); not
an official leaderboard entry. Other rows: public accuracy as published in the board's v1.4.1 results file. Brier
score 0.299 (v3 0.315); paraphrase agreement 34 / 36.</sub>

### Consistency and multilingual checks (same harness for v4 and v3)

| Check | v4 | v3 | n |
|---|---:|---:|---:|
| Same answer when the options are listed in reverse order | **94.0%** | 88.0% | 705 / 643 pairs from our held-out rows (v3: text only) |
| CREAK true/false claims, accuracy | **72.2%** | 62.6% | 1,371 (development split) |
| MASSIVE 1.1 intent (59 intents), macro accuracy over 51 locales | **74.3%** | 74.0% | 5,100 (100 test utterances per locale) |

<sub>float32 on CUDA (A100) for the option-order and MASSIVE checks, Apple MPS for CREAK. None of these rows is in
the training data: the option-order pairs come from our held-out development and calibration rows; CREAK is
evaluation-only; MASSIVE uses its test split (its train split is in the training data).</sub>

### Decision Index Vision public suite: 57.55

The first Jev-Style model that reads images.

| Benchmark | Rows | Accuracy | Skill |
|---|---:|---:|---:|
| CV-Bench | 2,038 | 80.4% | 67.6 |
| BLINK | 961 | 62.7% | 42.3 |
| RealWorldQA | 589 | 72.0% | 54.2 |
| CharXiv | 532 | 75.8% | 67.7 |
| InfographicVQA | 244 | 88.5% | 84.7 |
| Mind2Web | 973 | 67.3% | 56.4 |
| Winoground | 800 | 84.3% | 68.5 |
| KIE (CORD + FUNSD)† | 2,465 | 91.9% | 86.7 |
| Moderation (Hateful Memes) | 1,213 | 60.3% | 20.7 |
| R-Bench-M (weight 0.5) | 665 | 30.4% | 16.4 |
| MMMU-Pro vision (weight 0.25) | 1,730 | 26.9% | 16.2 |
| **Public vision index** | **12,210** | | **57.55** |

<sub>Decision Index Vision public suite (kit `25c6ec7`, the kit's own vision runner and scorer), one complete run:
12,210 rows, all answered (`vision-scores.json` says `"complete": true`). bfloat16 on one A100 80GB; every image at
its native resolution up to 1.6 MP (the run's effective budget: 1,638,400 pixels / 1,600 image tokens per image).
Skill = (accuracy − chance) / (1 − chance) × 100; the index weighs nine benchmarks 1, R-Bench-M 0.5 and MMMU-Pro
vision 0.25. † CORD's train split is in the training data, so KIE is not zero-shot. The Vision board's Full score is
50% this public index and 50% private tests run by the maintainers; this is not an official board entry until the
maintainers review it.</sub>

## How it differs from v3

| | [Jev-Style 2B v3](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v3) | **Jev-Style 2B v4 (this model)** |
|---|---|---|
| Base | Qwen/Qwen3.5-2B, text only | **Qwen/Qwen3.5-2B with its vision tower** (frozen) |
| Parameters | 1,881,825,088 | **2,213,241,664** (language 1,881,825,088 + vision 331,416,576) |
| Inputs | text / JSON | **text / JSON + images** (native resolution up to 1.6 MP) |
| Context per call | 25,600 tokens | **32,768 tokens** (image tokens included) |
| Training tokens | 60M (one run) | **about 294M per run, two runs averaged** |
| Calibration | one global temperature (T = 0.828) | **one temperature per question type** (choice 0.770 / noul 0.825 / score 0.584) |
| Labels | public datasets and generators | the same **plus soft labels distilled from TypeSafe Jev 1.14** via its public API |

Same input protocol as the 2B v3 (render v2 with long-option catalogues, block attention in the full-attention
layers, the same verdict readout), so the v3 notes on layout and readout still apply; the v3 runtime cannot load
this checkpoint (it is a `Qwen3_5ForConditionalGeneration` with a vision tower).

### Input format and readout

<details>
<summary><strong>Prompt layout, block attention, images and the score</strong></summary>

Each segment is tokenised on its own and the pieces are concatenated, so slot positions are exact. Text inside the
state, question and options is tokenised with special tokens disabled: `<|im_end|>` in user text stays plain text.

```text
State:
<state: plain text, or any JSON value; each <|image_k|> becomes <|vision_start|> + image tokens + <|vision_end|>>

Question [<choice|noul|score>]: <question>
Options:
- <option 1> ->
- <option 2> ->
```

When the question, options and slots together exceed 2,048 tokens, the options are written once as a numbered
catalogue (`Option 1: <option 1>` ...), followed by `Judge each numbered option in the complete catalogue above:` and
one `Option k ->` slot per option.

- **Block attention.** The input is cut into blocks of at most 2,048 tokens (state blocks, then the question block,
  or catalogue and rubric blocks). In the 6 full-attention layers each block attends to everything before it and to
  itself, with no causal mask inside the block. The Gated DeltaNet layers are ordinary recurrent layers. This is
  why the shipped runtime is required.
- **Images.** The checkpoint's own Qwen3.5 vision tower and image processor (`preprocessor_config.json`); images are
  sized to multiples of 32 px (patch 16 × merge 2), one image token per 32 × 32 px.
- **Score.** Option k's score is `logit(" yes") − logit(" no")` at its ` ->` slot, computed in float32 from the
  final normalised hidden state and the tied embedding rows. No parameters are added.
- **Probabilities.** `softmax(scores / T)` with the temperature of the question's type (choice 0.7698, noul 0.8247,
  score 0.5836; `readout_config.json`), fitted on 3,325 held-out calibration rows (NLL 0.2618 → 0.2457, ECE
  0.049 → 0.021). `temperature=1.0` gives the raw scores.

</details>

## Scope and limits

- **Runtime required.** Decisions come from the runtime in this repository (`jev_style_decision.py`, which calls the
  reference scoring code shipped under `src/macjev/`). Stock transformers pipelines, llama.cpp, Ollama or LM Studio
  cannot produce the decision scores and would run causal attention. The text-only [GGUF](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4-GGUF) and
  [MLX](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4-MLX) builds ship their own runtimes.
- **32,768 tokens** is the limit for the whole input (state including image tokens + question + options + readout);
  larger inputs raise `InputBudgetError`, nothing is truncated.
- **Images in the state only.** Questions and options are text. Box (region) questions are not supported.
- **Decision Index.** The training data was de-duplicated against the public Decision Index 0.3 suite (row hashes,
  8-gram and MinHash gates) and uses no suite rows or suite build scripts; it does include the official train splits
  of several benchmarks that the index tests on (listed under [Training data and licences](#training-data-and-licences)),
  so results on those benchmarks are not zero-shot.
- **Decisions only.** The model scores the options you give it. It does not generate text and takes no actions.

## Training

- **Base:** [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) (revision `15852e8c`), Apache-2.0. The vision
  tower was kept frozen; the multi-token-prediction head was removed.
- **Runs:** two full fine-tunes with different seeds on one A100 80GB each: 2,245 steps / 294.1M training tokens
  and 2,244 steps / 294.0M tokens. The released weights are the uniform average of the two runs' final weights,
  chosen by a pre-declared rule (lowest component-macro NLL on a fixed development sample of 2,400 rows; the average
  beat each single run).
- **Calibration:** one temperature per question type, fitted on 3,325 independent calibration rows.

<!-- BEGIN DATA_LICENCES -->
## Training data and licences

- **Base model:** Qwen/Qwen3.5-2B (Qwen team, Alibaba Cloud), Apache-2.0; see `LICENSE` and `NOTICE`.
- **Mixture:** a pool of 916,923 training rows from 88 sources (589M tokens); each run drew about 294M tokens from it
  by per-component token quotas (at most 1.5 passes over a component). English is 91% of the rows and Chinese 7%;
  MASSIVE adds more languages. 370,318 of the rows are reused from the v3 training pool.

| Component | Rows in the pool | Share of training tokens | Sources |
|---|---:|---:|---|
| Public reading tasks | 271,053 | 19.2% | 13 sets: WANLI, TabFact, TempReason, DROP, ShARC, TimeQA, AQuA-RAT, LexGLUE, FinQA, ROPES, MAUD, QASPER, CondaQA |
| Tools and automation | 97,773 | 17.6% | code-generated tool and API catalogues, a home-device simulator, model selection and tool retrieval; When2Call |
| Retrieval and classification | 61,594 | 9.6% | BANKING77, Amazon ESCI, HoVer, RAGTruth, SGD; code-generated link-safety and relevance items; model-written intent messages |
| Judging and scoring | 75,608 | 9.6% | HelpSteer2 and HelpSteer3 (with their human rating distributions), AQuA-RAT and FinQA scoring views, Habermas Machine |
| New-domain decisions | 31,874 | 8.6% | documents, questions and answers written by Claude and GPT models (answers kept where two models agreed) |
| Knowledge and reasoning | 87,079 | 7.0% | GSM8K; code-generated word problems, causal graphs, select-all, algorithmic and code-execution (from MBPP) items; Lichess puzzles; ARC, QASC, MedMCQA, CommonsenseQA; ScienceQA; model-written deduction stories |
| Long documents | 5,141 | 6.4% | code-generated long-context items (facts planted in long documents) and long tables |
| Arts and taste | 46,991 | 5.8% | POP909 chords, New Yorker caption contest, code-generated colour-palette preferences, model-written limericks |
| Images | 73,304 | 5.4% | A-OKVQA, VQAv2, TextVQA, CORD, ScienceQA |
| Language understanding | 83,778 | 3.8% | WinoGrande, HellaSwag, ContractNLI, iSarcasmEval, SNLI; model-written reviews, stance / sarcasm and multi-entity sentiment texts; code-generated clinical-trial reports |
| Typed decisions (v3) | 27,300 | 2.6% | model-written business workflows |
| Intents and yes/no QA | 21,427 | 1.6% | MASSIVE, CLINC150, BoolQ |
| Safety themes | 25,536 | 1.0% | Civil Comments, SQuAD v2, five jailbreak and prompt-injection sets, model-written prompts |
| Hard cases | 1,519 | 0.9% | five generated families (policy, multi-hop, numeric, abstention, judging) |
| Mac agent checks | 5,159 | 0.8% | the project's own simulators |
| Option-format views | 1,787 | 0.2% | CLINC150 with all 151 options |

- **Benchmark train splits** (official train splits only, no test or validation split; results on these benchmarks
  are not zero-shot): BANKING77 (CC BY 4.0 upstream at PolyAI/banking77; rows from the MTEB mirror, whose card says
  MIT), CLINC150 (CC BY 3.0), SGD (CC BY-SA 4.0), ContractNLI (CC BY 4.0), ARC (CC BY-SA 4.0), WinoGrande
  (CC BY 2.0), HellaSwag (MIT; see below), GSM8K (MIT), POP909 (MIT), Amazon ESCI (Apache-2.0), iSarcasmEval (MIT),
  Habermas Machine (Apache-2.0), CommonsenseQA (MIT), RAGTruth (MIT), HoVer (CC BY-SA 4.0), When2Call (CC BY 4.0),
  New Yorker caption contest (CC BY 4.0), and for the Vision suite CORD (CC BY 4.0, part of its KIE benchmark).
  Code generators that imitate the format of other benchmarks (tool catalogues, a home-device simulator, model
  routing, limericks, colour palettes, chess, select-all, causal graphs, code execution, deduction stories, aspect
  sentiment, stance, clinical-trial reports, reasoning relevance) are also in the mixture, with their own
  vocabularies and templates.
- **Full per-source list:** [validation/data_sources.json](validation/data_sources.json) (all 88 sources with the
  licence recorded for each, the mixture rows they feed and a link where available).
- **Sources with terms of their own** (kept in the pool; check each source's terms before commercial use), for
  example:
  - Jailbreak prompts from the jailbreak_llms collection, which states it is for research purposes only:
    In-the-Wild Jailbreak Prompts (TrustAIRLab, MIT; 1,077 rows) and the jailbreak prompts in
    jackhhao/jailbreak-classification (Apache-2.0; 931 rows in total).
  - HellaSwag: its card gives MIT only in the text (no licence field), and the original GitHub repository is blocked
    after a DMCA notice from wikiHow. Only the ActivityNet-caption items were used.
  - neuralchemy Prompt-injection-dataset (1,403 rows): the upstream rows its card marks research-only were removed.
  - Images: A-OKVQA and VQAv2 use COCO photos (Flickr, under their own licences and the COCO terms of use);
    TextVQA uses Open Images photos (CC BY 2.0). No images are included in this repository.
  - Share-alike (CC BY-SA): SQuAD v2, BoolQ, DROP, SNLI, ARC, SGD, HoVer, ShARC, TempReason, ScienceQA (mirror).
- **Outputs of other models:**
  - **Soft labels distilled from TypeSafe Jev 1.14 via its public API:** 334,739 training rows use Jev 1.14's
    probabilities as their target. Where Jev's top choice was not a correct answer, the gold label was kept instead
    (58,586 rows). No Jev weights or code are included.
  - Anthropic Claude and OpenAI GPT models (through the Claude Code and Codex command-line tools) wrote the
    new-domain documents and questions, the reviews, the stance / sarcasm, multi-entity sentiment and intent texts,
    the limericks, the deduction stories and adversarial SNLI hypotheses. Their labels come from code, from the
    source data, or from two models that answered independently; rows were kept only where a blind second model
    (or both models) agreed.
  - From the v3 pool: OpenAI GPT and Anthropic Claude models designed the typed-decision workflows (27,300 rows);
    Claude models labelled them. A Claude model wrote and labelled 837 jailbreak and toxicity prompt rows. OpenAI
    GPT models wrote two hard-case families and paraphrased part of the Mac goal wording.
  - Third-party data with model-written text: WANLI (GPT-3, revised by crowdworkers), HelpSteer2 and HelpSteer3
    (responses written by language models, rated by humans), When2Call (synthetic conversations made with language
    models by NVIDIA), RAGTruth (model responses with human-annotated hallucination spans) and part of the jackhhao
    benign prompts (from GPTeacher, generated by GPT-4).
  - The providers' terms of use may restrict how models trained on such outputs may be used, so check them for
    your use case.
- **Decision Index hygiene:** every training row was checked against the public Decision Index 0.3 suite (155,192
  rows): 3,136 rows with an exact text match and 16,251 rows caught by 8-gram or MinHash gates were dropped. No
  suite rows and no suite build scripts were used.
- **Evaluation-only data** (0 training rows): the JevBench items, the Decision Index public suites (text and
  vision), the MASSIVE 1.1 test split and the CREAK development split used for the checks below, and our own
  development and calibration rows (held out from training).
<!-- END DATA_LICENCES -->

## Disclaimers

> **Independent project.** Jev-Style is not affiliated with, endorsed by or connected to TypeSafe AI or Jev. Soft
> labels for part of the training rows were distilled from TypeSafe Jev 1.14 via its public API; no Jev weights or
> code are included. It is also not affiliated with the Laya authors or the Qwen team.

**AI disclosure:** code written with AI coding assistants (Claude Code) under my direction; I designed the project,
trained the models and verified the results.

## Licence

Apache-2.0. Built on Qwen/Qwen3.5-2B (Apache-2.0); the Apache License 2.0 text is in `LICENSE`, and `NOTICE` lists
the modifications.

## Citation

```bibtex
@misc{jevstyle2026v4_2b,
  title        = {Jev-Style-2B-Decision-v4: a 2B decision model for text and images with calibrated probabilities for every option},
  author       = {chaoliangUNSW},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4}},
  note         = {Fine-tuned from Qwen/Qwen3.5-2B}
}
```

## Contact

I welcome internship, employment, and research collaboration opportunities. Please contact me at [**yanchaoliang369@gmail.com**](mailto:yanchaoliang369@gmail.com).

欢迎提供实习、工作及科研合作机会，请邮件联系：[yanchaoliang369@gmail.com](mailto:yanchaoliang369@gmail.com)。
