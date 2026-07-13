"""onnx_export.py — Export Exp C3 (TwoTowerGRUEnriched) to ONNX for serving.

Run (from the repo root): python -m serving.onnx_export

Steps:
  1. Load `models/exp_c3_gru_enriched.pt` (state dict + hyperparams/scaling).
  2. Build `TwoTowerGRUEnrichedServing` (pack-free, ONNX-exportable forward)
     and load the SAME weights.
  3. Parity check #1 — serving-model (torch, eager) vs. reference model
     (torch, pack_padded_sequence, the ORIGINAL training forward) on random
     synthetic batches with varied sequence lengths. Must match to float
     tolerance — this proves the pack-free rewrite changed nothing.
  4. Export the serving model to ONNX (dynamic batch axis).
  5. Parity check #2 — ONNX Runtime output vs. the PyTorch serving-model
     output on the SAME synthetic batches. Must match to float tolerance.
  6. Copy `exp_c3_isotonic.pkl` to `serving/models/` (byte-identical copy —
     the calibrator itself is not touched, only relocated for serving).
  7. Write `serving/models/exp_c3_scaling.json` — the bundled train-only
     scaling constants + metadata, loaded verbatim at serving time.

This is the "ONNX output matches the PyTorch output to float tolerance on a
few sample pairs" parity verification for the serving export.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from .feature_pipeline import MAX_LEN, N_FEATS, N_HEAD
from .model import (
    build_reference_model,
    build_serving_model,
    dump_scaling_json,
    load_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / 'models'
SERVING_MODELS_DIR = Path(__file__).resolve().parent / 'models'

PT_PATH = MODELS_DIR / 'exp_c3_gru_enriched.pt'
ISOTONIC_SRC = MODELS_DIR / 'exp_c3_isotonic.pkl'
ONNX_OUT = SERVING_MODELS_DIR / 'exp_c3.onnx'
ISOTONIC_DST = SERVING_MODELS_DIR / 'exp_c3_isotonic.pkl'
SCALING_JSON_OUT = SERVING_MODELS_DIR / 'exp_c3_scaling.json'

RANDOM_STATE = 42
ATOL = 1e-5
RTOL = 1e-4


def make_synthetic_batch(rng: np.random.RandomState, batch_size: int = 16):
    """Random-but-plausible synthetic inputs covering the length range
    [1, MAX_LEN] on both sides (including debut-match length=1 and
    full-length=MAX_LEN cases), for the parity check only — never real data."""
    player_seq = rng.randn(batch_size, MAX_LEN, N_FEATS).astype(np.float32)
    opp_seq = rng.randn(batch_size, MAX_LEN, N_FEATS).astype(np.float32)
    player_len = rng.randint(1, MAX_LEN + 1, size=batch_size).astype(np.int64)
    opp_len = rng.randint(1, MAX_LEN + 1, size=batch_size).astype(np.int64)
    # Zero out padding beyond each sample's own length, exactly like real data
    # (build_sequence_at only ever fills [:length], leaving the rest at 0).
    for i in range(batch_size):
        player_seq[i, player_len[i]:] = 0.0
        opp_seq[i, opp_len[i]:] = 0.0
    head_feats = rng.randn(batch_size, N_HEAD).astype(np.float32)
    return player_seq, player_len, opp_seq, opp_len, head_feats


def main() -> None:
    SERVING_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f'Loading checkpoint: {PT_PATH}')
    ckpt = load_checkpoint(PT_PATH)

    ref_model = build_reference_model(ckpt)
    serving_model = build_serving_model(ckpt)

    # ── Parity check #1 — serving (pack-free) vs. reference (packed), PyTorch only ──
    rng = np.random.RandomState(RANDOM_STATE)
    max_diff_1 = 0.0
    N_BATCHES = 5
    for b in range(N_BATCHES):
        p_seq, p_len, o_seq, o_len, hf = make_synthetic_batch(rng, batch_size=32)
        with torch.no_grad():
            ref_logit = ref_model(
                torch.from_numpy(p_seq), torch.from_numpy(p_len),
                torch.from_numpy(o_seq), torch.from_numpy(o_len), torch.from_numpy(hf),
            ).numpy()
            serv_logit = serving_model(
                torch.from_numpy(p_seq), torch.from_numpy(p_len),
                torch.from_numpy(o_seq), torch.from_numpy(o_len), torch.from_numpy(hf),
            ).numpy()
        diff = np.max(np.abs(ref_logit - serv_logit))
        max_diff_1 = max(max_diff_1, diff)
    print(f'Parity #1 (serving-torch vs reference-torch, pack-free rewrite): max|diff| = {max_diff_1:.3e}')
    assert max_diff_1 < ATOL, 'Pack-free serving model does NOT match the original packed reference model!'
    print('OK — pack-free rewrite is numerically equivalent to the original training forward.')

    # ── Export to ONNX (dynamic batch axis) ──────────────────────────────────
    dummy_p_seq, dummy_p_len, dummy_o_seq, dummy_o_len, dummy_hf = make_synthetic_batch(rng, batch_size=4)
    dummy_inputs = (
        torch.from_numpy(dummy_p_seq), torch.from_numpy(dummy_p_len),
        torch.from_numpy(dummy_o_seq), torch.from_numpy(dummy_o_len), torch.from_numpy(dummy_hf),
    )
    input_names = ['player_seq', 'player_len', 'opp_seq', 'opp_len', 'head_feats']
    output_names = ['logit']
    dynamic_axes = {name: {0: 'batch'} for name in input_names + output_names}

    print(f'Exporting ONNX model to: {ONNX_OUT}')
    torch.onnx.export(
        serving_model,
        dummy_inputs,
        str(ONNX_OUT),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        dynamo=False,  # legacy TorchScript-based exporter — avoids the optional
                        # `onnxscript` dependency the new dynamo exporter needs
                        # (torch>=2.13 default), not installed in the serving env
    )
    onnx_model = onnx.load(str(ONNX_OUT))
    onnx.checker.check_model(onnx_model)
    print('ONNX model structurally valid (onnx.checker passed).')

    # ── Parity check #2 — ONNX Runtime vs. PyTorch serving model ─────────────
    sess = ort.InferenceSession(str(ONNX_OUT), providers=['CPUExecutionProvider'])
    max_diff_2 = 0.0
    for b in range(N_BATCHES):
        p_seq, p_len, o_seq, o_len, hf = make_synthetic_batch(rng, batch_size=32)
        with torch.no_grad():
            torch_logit = serving_model(
                torch.from_numpy(p_seq), torch.from_numpy(p_len),
                torch.from_numpy(o_seq), torch.from_numpy(o_len), torch.from_numpy(hf),
            ).numpy()
        onnx_logit = sess.run(
            output_names,
            {
                'player_seq': p_seq, 'player_len': p_len,
                'opp_seq': o_seq, 'opp_len': o_len, 'head_feats': hf,
            },
        )[0]
        diff = np.max(np.abs(torch_logit - onnx_logit))
        max_diff_2 = max(max_diff_2, diff)
    print(f'Parity #2 (ONNX Runtime vs PyTorch serving model): max|diff| = {max_diff_2:.3e}')
    assert max_diff_2 < ATOL, 'ONNX Runtime output does NOT match PyTorch output!'
    print('OK — ONNX Runtime output matches PyTorch output within float tolerance.')

    # ── Copy isotonic calibrator (byte-identical) + write scaling JSON ───────
    shutil.copyfile(ISOTONIC_SRC, ISOTONIC_DST)
    print(f'Copied isotonic calibrator: {ISOTONIC_SRC} -> {ISOTONIC_DST}')

    dump_scaling_json(ckpt, SCALING_JSON_OUT)
    print(f'Wrote scaling/metadata bundle: {SCALING_JSON_OUT}')

    onnx_size_kb = ONNX_OUT.stat().st_size / 1024
    print(f'\nDone. {ONNX_OUT.name}: {onnx_size_kb:.1f} KB')
    print(f'Max parity diff (pack-free rewrite):  {max_diff_1:.3e}  (tolerance {ATOL:.0e})')
    print(f'Max parity diff (ONNX vs PyTorch):     {max_diff_2:.3e}  (tolerance {ATOL:.0e})')


if __name__ == '__main__':
    main()
