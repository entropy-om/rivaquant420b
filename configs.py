"""Named RivaQuant420b stage configs — one file spanning 162M -> 420B.

Every stage uses the exact same architecture (model/transformer.py,
model/bitlinear.py — unchanged from the proven PeetPedro/rivaquant recipe).
Only shape changes: n_layer / n_embd / n_head. head_dim is pinned to 128
(standard practice) for every stage except 162M, which keeps its own
already-proven head_dim=64 rather than being retrofit to match.

Param counts below are the ACTUAL result of each shape — matches
model.num_params() exactly (embedding + head + attention/MLP projections
+ every RMSNorm weight vector), not a number force-fit to the stage's own
label. "8b" is a roadmap stage name, not a promise of exactly 8.00e9
parameters — the real count for its shape is 6.86B, stated honestly here
so nothing downstream has to guess or round.

162M is the only stage that has actually been trained (PeetPedro/rivaquant,
val ppl 5.455 on TinyStories, 17,500 steps). Every other stage is an
untrained shape, staged per the handoff doc's own gate: mid checkpoints
(1B -> 8B -> ~70B) prove the ternary recipe holds as width/depth grow
before 420B is ever attempted, and each only starts once the stage before
it is stable and the compute for it is actually booked.
"""
from dataclasses import dataclass

from model import RivaQuantConfig

VOCAB_SIZE = 50257  # GPT-2 BPE, reused through at least Stage 2 (1B) — see README


@dataclass
class Stage:
    name: str
    config: RivaQuantConfig
    trained: bool
    note: str


def _params(n_layer: int, n_embd: int, vocab: int = VOCAB_SIZE) -> int:
    # Matches model.num_params() exactly (verified against two live runs:
    # 162m reported 162,213,888, quantal reported 998,714,496 — both match
    # this formula bit for bit). An earlier version of this function
    # omitted the RMSNorm weight vectors (one per BitLinear call, plus
    # Block's ln1/ln2, plus the model's own ln_f) and undercounted every
    # stage below by n_embd*(9*n_layer+2) — found by comparing this file's
    # claimed count against the real, running model, not by inspection.
    big = 2 * vocab * n_embd + n_layer * 12 * n_embd**2
    norms = n_embd * (9 * n_layer + 2)
    return big + norms


STAGES = {
    "162m": Stage(
        name="162m",
        config=RivaQuantConfig(vocab_size=VOCAB_SIZE, n_layer=12, n_head=12, n_embd=768, block_size=256),
        trained=True,
        note=f"PROVEN. {_params(12, 768):,} params (matches the live run's own "
             "model.num_params() exactly). val ppl 5.455, TinyStories, 17,500 steps. "
             "This is the pipeline's own regression test, not a placeholder.",
    ),
    "1b": Stage(
        name="1b",
        config=RivaQuantConfig(vocab_size=VOCAB_SIZE, n_layer=24, n_head=16, n_embd=2048, block_size=1024),
        trained=False,
        note=f"UNTRAINED shape. {_params(24, 2048):,} params (~1.41B) — GPT-2-XL-ish width, deeper.",
    ),
    "8b": Stage(
        name="8b",
        config=RivaQuantConfig(vocab_size=VOCAB_SIZE, n_layer=32, n_head=32, n_embd=4096, block_size=2048),
        trained=False,
        note=f"UNTRAINED shape. {_params(32, 4096):,} params (~6.86B) — Llama-3-8B-shaped (depth/width), "
             "not param-count-matched to it (different vocab, no GQA, no tied embeddings here).",
    ),
    "70b": Stage(
        name="70b",
        config=RivaQuantConfig(vocab_size=VOCAB_SIZE, n_layer=80, n_head=64, n_embd=8192, block_size=2048),
        trained=False,
        note=f"UNTRAINED shape. {_params(80, 8192):,} params (~65.3B) — Llama-2-70B-shaped.",
    ),
    "420b": Stage(
        name="420b",
        config=RivaQuantConfig(vocab_size=VOCAB_SIZE, n_layer=126, n_head=128, n_embd=16384, block_size=2048),
        trained=False,
        note=f"UNTRAINED shape. {_params(126, 16384):,} params (~407.5B) — extrapolated dense shape at this "
             "scale. Not started; gated on the 70B stage being stable and H200 capacity actually booked, "
             "per the handoff doc. Calling this stage \"420b\" is a roadmap label, not a claim this shape "
             "or anything trained under it exists yet.",
    ),
    # "quantal" — a sibling one-off, not a rung on the 162m->420b ladder.
    # Target: exactly 1,000,000,000 params. No integer (n_layer, n_embd)
    # shape with a standard head_dim hits that exactly — checked
    # numerically (including the RMSNorm fix above), not assumed. This is
    # the closest real match with a sane depth/width ratio and
    # head_dim=64: 998,714,496, off by 1,285,504 (99.871% of the way
    # there). Forcing an exact hit meant an absurd shape (hundreds of
    # layers at a sliver of width) — architecturally broken just to
    # satisfy a digit. The honest number, not a fudged one, same rule as
    # every other stage in this file. Confirmed live: the actual running
    # model reported exactly this count via model.num_params().
    "quantal": Stage(
        name="quantal",
        config=RivaQuantConfig(vocab_size=VOCAB_SIZE, n_layer=10, n_head=39, n_embd=2496, block_size=512),
        trained=False,
        note=f"UNTRAINED shape. {_params(10, 2496):,} params — closest sane-architecture exact integer "
             "to a 1,000,000,000 target (see comment above). Deliberately CPU-trained (OVH compute-"
             "optimized, no GPU) — a real, explicit choice, not a fallback.",
    ),
}


def get_stage(name: str) -> Stage:
    try:
        return STAGES[name.lower()]
    except KeyError:
        raise ValueError(f"unknown stage {name!r}, choose one of {sorted(STAGES)}") from None


if __name__ == "__main__":
    for stage in STAGES.values():
        tag = "TRAINED" if stage.trained else "untrained"
        print(f"{stage.name:>5}  [{tag:>9}]  {stage.note}")
