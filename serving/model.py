"""model.py — Exp C3 (TwoTowerGRUEnriched) model definitions + checkpoint loading.

Two forward-pass implementations sharing the IDENTICAL parameter names/shapes
(so both can `load_state_dict()` the same `exp_c3_gru_enriched.pt` checkpoint):

  - `TwoTowerGRUEnriched`         — the ORIGINAL training-notebook architecture
                                    (`pack_padded_sequence`), kept here only as
                                    the parity-check reference implementation.
  - `TwoTowerGRUEnrichedServing`  — an ONNX-exportable equivalent: runs the GRU
                                    over the full MAX_LEN-padded sequence and
                                    gathers the hidden state at each sample's
                                    own `length - 1` timestep, instead of
                                    `pack_padded_sequence` (which does not
                                    export reliably to ONNX — dynamic per-batch
                                    control flow). Mathematically identical:
                                    a GRU's hidden state at step t never
                                    depends on padding steps AFTER t, so
                                    gathering at `length-1` from the full-length
                                    run equals the packed run's final hidden
                                    state. Verified numerically in
                                    `onnx_export.py`'s parity check.

Both classes' weights come from `models/exp_c3_gru_enriched.pt`
(hidden_size=24, dropout=0.3, N_FEATS=9, N_HEAD=15) — see
`notebooks/10_enriched_gru.ipynb` Step 9 for the training-time definition
this file mirrors.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from .feature_pipeline import HEAD_COLS, N_FEATS, N_HEAD

HIDDEN_SIZE = 24
DROPOUT = 0.3
MLP_HIDDEN = 16


class TwoTowerGRUEnriched(nn.Module):
    """Original training architecture (pack_padded_sequence) — reference only,
    used for the ONNX parity check. Not used at serving time."""

    def __init__(self, input_size=N_FEATS, hidden_size=HIDDEN_SIZE, dropout=DROPOUT,
                 mlp_hidden=MLP_HIDDEN, n_head=N_HEAD):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.drop_embed = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 3 + n_head, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def encode(self, seq, lengths):
        packed = pack_padded_sequence(
            seq, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False,
        )
        _, h_n = self.gru(packed)
        return self.drop_embed(h_n[-1])

    def forward(self, player_seq, player_len, opp_seq, opp_len, head_feats):
        h_p = self.encode(player_seq, player_len)
        h_o = self.encode(opp_seq, opp_len)
        combined = torch.cat([h_p, h_o, h_p - h_o, head_feats], dim=1)
        return self.head(combined).squeeze(-1)


class TwoTowerGRUEnrichedServing(nn.Module):
    """ONNX-exportable equivalent — no pack_padded_sequence. Same parameter
    names/shapes as `TwoTowerGRUEnriched` (`gru.*`, `head.0.*`, `head.3.*`),
    so `load_state_dict()` from the SAME checkpoint works unmodified.

    Dropout layers are present (for state_dict compatibility) but the module
    is always used in `eval()` mode at serving/export time, so they are no-ops.
    """

    def __init__(self, input_size=N_FEATS, hidden_size=HIDDEN_SIZE, dropout=DROPOUT,
                 mlp_hidden=MLP_HIDDEN, n_head=N_HEAD):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.drop_embed = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 3 + n_head, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def encode(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # seq: [B, MAX_LEN, N_FEATS] (zero-padded AFTER the real steps),
        # lengths: [B] int64, each in [1, MAX_LEN].
        output, _ = self.gru(seq)  # output: [B, MAX_LEN, hidden]
        idx = (lengths - 1).clamp(min=0)  # [B]
        batch_idx = torch.arange(seq.shape[0], device=seq.device)
        h = output[batch_idx, idx]  # [B, hidden] -- gather, ONNX-exportable
        return self.drop_embed(h)

    def forward(self, player_seq, player_len, opp_seq, opp_len, head_feats):
        h_p = self.encode(player_seq, player_len)
        h_o = self.encode(opp_seq, opp_len)
        combined = torch.cat([h_p, h_o, h_p - h_o, head_feats], dim=1)
        logit = self.head(combined).squeeze(-1)
        return logit


def load_checkpoint(pt_path: Path | str) -> dict:
    """Loads `exp_c3_gru_enriched.pt` — returns the full checkpoint dict
    (model_state_dict, hyperparams, scaling, calibration, feature names, ...)."""
    return torch.load(pt_path, map_location='cpu', weights_only=False)


def build_serving_model(checkpoint: dict) -> TwoTowerGRUEnrichedServing:
    hp = checkpoint['hyperparams']
    model = TwoTowerGRUEnrichedServing(
        input_size=hp['n_feats'], hidden_size=hp['hidden_size'],
        dropout=hp['dropout'], n_head=hp['n_head'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def build_reference_model(checkpoint: dict) -> TwoTowerGRUEnriched:
    hp = checkpoint['hyperparams']
    model = TwoTowerGRUEnriched(
        input_size=hp['n_feats'], hidden_size=hp['hidden_size'],
        dropout=hp['dropout'], n_head=hp['n_head'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def dump_scaling_json(checkpoint: dict, out_path: Path | str) -> None:
    """Bundles the train-only scaling constants + metadata for serving —
    loaded VERBATIM at serving time, never recomputed. Mirrors
    `models/exp_c3_gru_enriched.pt`'s own `scaling`/`hyperparams`/
    `calibration` dicts exactly (this JSON is just a lightweight, torch-free
    way for `precompute_gru_matrix.py` to read the same constants without
    needing `torch` if it ever runs in a torch-less environment)."""
    payload = {
        'model_version': 'exp_c3_v1',
        'hyperparams': checkpoint['hyperparams'],
        'scaling': checkpoint['scaling'],
        'step_feature_names': checkpoint['step_feature_names'],
        'head_feature_names': checkpoint['head_feature_names'],
        'calibration': checkpoint['calibration'],
        'test_metrics': checkpoint['test_metrics'],
    }
    assert checkpoint['head_feature_names'] == HEAD_COLS, (
        'Checkpoint head_feature_names order does not match feature_pipeline.HEAD_COLS!'
    )
    Path(out_path).write_text(json.dumps(payload, indent=2), encoding='utf-8')
