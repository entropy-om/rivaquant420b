# rivaquant420b

BitNet b1.58 ternary decoder-only LLM, scaling the proven
[PeetPedro/rivaquant](https://huggingface.co/PeetPedro/rivaquant) (162.2M
params) recipe toward 420B. The house "train in the open" run for
[attestal](https://attestal.ai)'s free tier — the whole loop (submit →
train → verify → publish) proven on this run before any external user
touches a slot. Status lives at `attestal.ai/api/slots` → `house_run`
(source: `functions/api/slots.js` in `entropy-om/witness-ai`).

**No vapor: this README, and that API endpoint, only ever say what the
artifact backs.** Until a checkpoint exists at a given stage, that stage is
`provisioning` or `training`, never `done`.

## What this actually is right now

A parametrized version of the exact architecture that already works —
`model/bitlinear.py` and `model/transformer.py` are unchanged from
`PeetPedro/rivaquant`, not reinvented. `configs.py` names five stages on
the 162m->420b ladder (`162m`, `1b`, `8b`, `70b`, `420b`), plus one
sibling one-off (`quantal`) — all run through the same `train/train.py`;
only the shape (`n_layer`/`n_embd`/`n_head`) changes.

| stage | params (actual) | status |
|---|---|---|
| 162m | 162,213,888 | **trained** — val ppl 5.455, TinyStories, 17,500 steps |
| 1b | 1,414,258,688 (~1.41B) | untrained shape, not started |
| 8b | 6,855,344,128 (~6.86B) | untrained shape, not started |
| 70b | 65,253,834,752 (~65.3B) | untrained shape, not started |
| 420b | 407,539,843,072 (~407.5B) | untrained shape, not started |
| quantal | 998,714,496 | **training** — OVH compute-optimized (CPU, intentional, no GPU); closest sane-architecture exact integer to a 1,000,000,000 target |

(The `params` column was corrected 2026-07-26: an earlier version of the
formula in `configs.py` omitted RMSNorm weight vectors and undercounted
every stage — found by comparing this file's claimed count against two
live runs' own `model.num_params()`. The 162m and quantal rows above are
confirmed against real, running models; 1b/8b/70b/420b are the same
corrected formula applied to untrained shapes.)

The `1b`/`8b`/`70b`/`420b` labels are roadmap stage names, not exact
parameter promises — each is a real, precedented depth/width shape (see
`configs.py` for exactly which known architectures each is modeled on),
and the table above states what it actually comes out to.

## Staging (why 420B isn't "in progress" yet)

1. **Pipeline at small scale (current)** — reproduce the 162M run
   end-to-end through *this* pipeline (data → tokenizer → config → train
   → eval → publish), not the original ad-hoc scripts. Green here = the
   loop is real, not that 420B is close.
2. **Mid checkpoints** — 1B → 8B → ~70B on real corpus, proving the
   ternary recipe holds as width/depth grow (loss curves, val ppl,
   throughput, memory headroom per step) before spending anything at the
   next scale up.
3. **420B target** — only once stage 2 is stable and the compute for it
   is actually booked.

## Compute

RunPod (community cloud) for training bursts, autoscale-on-demand — pods
start for a run and stop when it's done or a cost cap is hit
(`remote/cost_watcher.py`), no idle burn. Stage 1 runs on a single RTX
4090 (~$0.34/hr), the same tier the original 162M run used — there is no
reason to reach for an H200 to re-run a 162M model. Larger stages will
need bigger cards and real parallelism (FSDP/TP/PP), sized to whatever's
actually booked at that stage, not provisioned speculatively now.

A persistent Hetzner box (`rivaquant420b-pipeline`) holds data staging and
orchestration between training bursts.

## Data

Stage 1 uses TinyStories (`train/data.py`) — the same corpus the proven
162M run used, on purpose: it isolates the one real unknown (does the
ternary recipe hold at each new shape) from a second unknown (a new
corpus). Real, licensed, publishable corpus work (dedup, shard, provenance
manifest) is Stage 2 scope, not built yet.

## Tokenizer

GPT-2 BPE (`tiktoken`, 50257 vocab) reused as-is through at least Stage 2.
Revisit before Stage 3 (70B) if vocab size becomes a real bottleneck.

## Running Stage 1

```sh
export RUNPOD_API_KEY=...   # entheai/.env
python3 remote/provision.py                     # RIVAQUANT_STAGE defaults to 162m
bash remote/launch.sh
python3 remote/cost_watcher.py &                # hard cost cap, stops the pod
```

## Publishing

Checkpoints + an honest eval card go to `PeetPedro/rivaquant420b` on
HuggingFace (models → HF, not git) once a stage actually finishes. The
card states exactly what was trained, on what, verified how — nothing
more.
