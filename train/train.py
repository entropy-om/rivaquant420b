"""RivaQuant420b pretraining loop. Same training mechanics as the proven
PeetPedro/rivaquant recipe (AdamW, cosine LR with warmup, gradient
clipping) — unchanged on purpose, so a bad run can only be the ternary-
weights-at-this-shape bet, not a training-loop bug. What's new here vs.
the 162M-only original: stage selection by name (configs.py spans
162m -> 420b through one file) and checkpoint/resume, since a multi-day
run at 1B+ can't afford to restart from step 0 after a pod interruption.
"""
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import RivaQuant, RivaQuantConfig
from configs import get_stage

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_DIR = os.environ.get("RIVAQUANT_OUT", "/workspace/rivaquant420b-out")
STATUS_PATH = os.path.join(OUT_DIR, "status.json")
LOG_PATH = os.path.join(OUT_DIR, "train.log")
CKPT_PATH = os.path.join(OUT_DIR, "checkpoint.pt")

STAGE = get_stage(os.environ.get("RIVAQUANT_STAGE", "162m"))

BLOCK_SIZE = int(os.environ.get("RIVAQUANT_BLOCK_SIZE", str(STAGE.config.block_size)))
# BitLinear's STE (activation_quant(x) kept alongside x for the backward
# graph, per BitLinear call, x4 per block x n_layer) uses far more peak
# memory per sample than a plain nn.Linear block at the same size — a batch
# of 32 at block_size=512 OOM'd a 24GB card at 162M. Small micro-batch +
# gradient accumulation keeps the same effective batch size at a fraction
# of the peak memory; larger stages will need this tuned down further.
BATCH_SIZE = int(os.environ.get("RIVAQUANT_BATCH_SIZE", "8"))
GRAD_ACCUM_STEPS = int(os.environ.get("RIVAQUANT_GRAD_ACCUM_STEPS", "4"))
MAX_STEPS = int(os.environ.get("RIVAQUANT_MAX_STEPS", "20000"))
# Was hardcoded to 50 — fine on GPU (a step takes seconds), but on a slow
# CPU run a single step can take many minutes, so a 50-step log/status
# cadence can leave the public status.json (and anything reading it, like
# a monitoring dashboard) showing stale data for hours with no way to
# tell "still running" from "actually stuck." Configurable so a slow run
# can log every step instead.
LOG_INTERVAL = int(os.environ.get("RIVAQUANT_LOG_INTERVAL", "50"))
EVAL_INTERVAL = int(os.environ.get("RIVAQUANT_EVAL_INTERVAL", "250"))
CKPT_INTERVAL = int(os.environ.get("RIVAQUANT_CKPT_INTERVAL", "250"))
LR = float(os.environ.get("RIVAQUANT_LR", "3e-4"))
WARMUP_STEPS = int(os.environ.get("RIVAQUANT_WARMUP_STEPS", "500"))
DEVICE = os.environ.get("RIVAQUANT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
SEED = int(os.environ.get("RIVAQUANT_SEED", "1337"))


def write_status(**fields) -> None:
    import json
    os.makedirs(OUT_DIR, exist_ok=True)
    fields["stage_name"] = STAGE.name
    fields["updated_at"] = time.time()
    with open(STATUS_PATH, "w") as f:
        json.dump(fields, f, indent=2)


def log(msg: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def get_batch(split: str, cfg: RivaQuantConfig):
    path = os.path.join(DATA_DIR, f"{split}.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - cfg.block_size - 1, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy(data[i:i + cfg.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + cfg.block_size].astype(np.int64)) for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


def lr_at(step: int) -> float:
    if step < WARMUP_STEPS:
        return LR * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    return 0.5 * LR * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_val_loss(model: RivaQuant, cfg: RivaQuantConfig, iters: int = 20) -> float:
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch("val", cfg)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    torch.manual_seed(SEED)
    cfg = STAGE.config
    model = RivaQuant(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)

    start_step = 0
    best_val = float("inf")
    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        best_val = ckpt.get("best_val_loss", float("inf"))
        log(f"resumed from {CKPT_PATH} at step {start_step}")

    log(f"stage: {STAGE.name}  model params: {model.num_params():,}  device: {DEVICE}  "
        f"start_step: {start_step}/{MAX_STEPS}")
    write_status(stage="training", step=start_step, max_steps=MAX_STEPS, params=model.num_params())

    for step in range(start_step, MAX_STEPS):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        opt.zero_grad(set_to_none=True)
        train_loss = 0.0
        for _ in range(GRAD_ACCUM_STEPS):
            x, y = get_batch("train", cfg)
            _, loss = model(x, y)
            (loss / GRAD_ACCUM_STEPS).backward()
            train_loss += loss.item() / GRAD_ACCUM_STEPS
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % LOG_INTERVAL == 0:
            log(f"step {step}/{MAX_STEPS}  train_loss {train_loss:.4f}  lr {lr_at(step):.2e}")
            write_status(stage="training", step=step, max_steps=MAX_STEPS,
                          train_loss=train_loss, best_val_loss=best_val)

        if step > 0 and step % EVAL_INTERVAL == 0:
            val_loss = estimate_val_loss(model, cfg)
            log(f"step {step}  val_loss {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model": model.state_dict(), "cfg": cfg, "step": step},
                           os.path.join(OUT_DIR, "best.pt"))
            write_status(stage="training", step=step, max_steps=MAX_STEPS,
                          train_loss=train_loss, val_loss=val_loss, best_val_loss=best_val)

        if step > 0 and step % CKPT_INTERVAL == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "cfg": cfg,
                "step": step,
                "best_val_loss": best_val,
            }, CKPT_PATH)

    torch.save({"model": model.state_dict(), "cfg": cfg, "step": MAX_STEPS},
               os.path.join(OUT_DIR, "final.pt"))
    write_status(stage="done", step=MAX_STEPS, best_val_loss=best_val)
    log(f"done. best_val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()
