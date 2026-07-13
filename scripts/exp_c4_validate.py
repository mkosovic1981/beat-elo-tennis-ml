"""
Exp C4 validation script -- confirms the text-skill join (vectorized) + ONE GRU train
finishes in reasonable time BEFORE assembling the notebook (runtime-safety instruction,
same discipline as `exp_c3_validate.py`).

This is Exp C3's exact pipeline with ONE change: the neutral text_p/text_o head slot is
replaced with 6 real columns (text_skill_p/o, text_conf_p/o, has_text_p/o) built from
`trenutci_scores.csv`.

Run from scripts/:  python exp_c4_validate.py
"""
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
torch.use_deterministic_algorithms(False)

DATA_DIR = Path('../data')
device = torch.device('cpu')

t_start = time.time()
print(f'PyTorch {torch.__version__}, device={device}')

# ---------------------------------------------------------------------------
# Step 1 -- load raw + sym (identical to 05/08/10)
# ---------------------------------------------------------------------------
raw = pd.read_csv(DATA_DIR / 'matches_export.csv', parse_dates=['Datum'])
raw['Datum'] = pd.to_datetime(raw['Datum']).dt.tz_localize(None)
raw = raw.sort_values(['Datum', 'SusretId']).reset_index(drop=True)

sym = pd.read_parquet(DATA_DIR / 'features.parquet')
with open(DATA_DIR / 'feature_tags.json', encoding='utf-8') as f:
    TAGS = json.load(f)
print(f'raw: {raw.shape}, sym: {sym.shape}')

# ---------------------------------------------------------------------------
# Step 2 -- long-format player-match log, extended with OppEloBefore (identical to 10)
# ---------------------------------------------------------------------------
common_cols = ['SusretId', 'Datum', 'SezonaId', 'LigaId', 'LigaTip', 'LigaKoeficjent', 'Lokacija']

winner_rows = raw[common_cols + [
    'WinnerId', 'LoserId', 'WinnerEloBefore', 'LoserEloBefore',
    'W_Set1', 'W_Set2', 'W_Set3', 'L_Set1', 'L_Set2', 'L_Set3',
]].rename(columns={
    'WinnerId': 'IgracId', 'LoserId': 'OpponentId',
    'WinnerEloBefore': 'MyEloBefore', 'LoserEloBefore': 'OppEloBefore',
    'W_Set1': 'MySet1', 'W_Set2': 'MySet2', 'W_Set3': 'MySet3',
    'L_Set1': 'OppSet1', 'L_Set2': 'OppSet2', 'L_Set3': 'OppSet3',
}).assign(Won=1)

loser_rows = raw[common_cols + [
    'LoserId', 'WinnerId', 'LoserEloBefore', 'WinnerEloBefore',
    'L_Set1', 'L_Set2', 'L_Set3', 'W_Set1', 'W_Set2', 'W_Set3',
]].rename(columns={
    'LoserId': 'IgracId', 'WinnerId': 'OpponentId',
    'LoserEloBefore': 'MyEloBefore', 'WinnerEloBefore': 'OppEloBefore',
    'L_Set1': 'MySet1', 'L_Set2': 'MySet2', 'L_Set3': 'MySet3',
    'W_Set1': 'OppSet1', 'W_Set2': 'OppSet2', 'W_Set3': 'OppSet3',
}).assign(Won=0)

player_log = pd.concat([winner_rows, loser_rows], ignore_index=True)
player_log['LokacijaNorm'] = player_log['Lokacija'].str.strip().str.lower()
player_log = player_log.sort_values(['IgracId', 'Datum', 'SusretId'], kind='mergesort').reset_index(drop=True)
print(f'player_log: {player_log.shape}')

# ---------------------------------------------------------------------------
# Step 3 -- per-step point-in-time features (identical to Exp C/C2/C3)
# ---------------------------------------------------------------------------
g = player_log.groupby('IgracId', sort=False)
player_log['MatchCountBefore'] = g.cumcount()
player_log['HomeCountBefore'] = player_log.groupby(['IgracId', 'LokacijaNorm'], sort=False).cumcount()
player_log['HomeShareBefore'] = np.where(
    player_log['MatchCountBefore'] > 0,
    player_log['HomeCountBefore'] / player_log['MatchCountBefore'],
    np.nan,
)
player_log['DaysSinceLastMatch'] = player_log.groupby('IgracId')['Datum'].diff().dt.days
player_log['CareerWinRateBefore'] = g['Won'].transform(lambda s: s.shift().expanding().mean())

for c in ['MySet1', 'MySet2', 'MySet3', 'OppSet1', 'OppSet2', 'OppSet3']:
    player_log[c + '_f'] = player_log[c].fillna(0)
player_log['SetMargin'] = (
    (player_log['MySet1_f'] + player_log['MySet2_f'] + player_log['MySet3_f'])
    - (player_log['OppSet1_f'] + player_log['OppSet2_f'] + player_log['OppSet3_f'])
)
player_log = player_log.drop(columns=[c + '_f' for c in ['MySet1', 'MySet2', 'MySet3', 'OppSet1', 'OppSet2', 'OppSet3']])

check_w = raw[['SusretId', 'WinnerId', 'WinnerMatchCountBefore']].rename(columns={'WinnerId': 'IgracId', 'WinnerMatchCountBefore': 'Exp'})
check_l = raw[['SusretId', 'LoserId', 'LoserMatchCountBefore']].rename(columns={'LoserId': 'IgracId', 'LoserMatchCountBefore': 'Exp'})
chk = pd.concat([check_w, check_l])
mm = player_log.merge(chk, on=['SusretId', 'IgracId'], how='left')
assert (mm['MatchCountBefore'] != mm['Exp']).sum() == 0, 'pandas point-in-time ordering mismatch!'
print('MatchCountBefore cross-check OK.')

opp_cwr = player_log[['SusretId', 'IgracId', 'CareerWinRateBefore']].rename(
    columns={'IgracId': 'OpponentId', 'CareerWinRateBefore': 'OpponentCareerWinRateBefore'})
player_log = player_log.merge(opp_cwr, on=['SusretId', 'OpponentId'], how='left', validate='many_to_one')

# ---------------------------------------------------------------------------
# Step 3b -- H2H per-step features (identical to 10) -- vectorized, no per-row loops.
# ---------------------------------------------------------------------------
h2h_g = player_log.groupby(['IgracId', 'OpponentId'], sort=False)
player_log['H2HMeetingsBefore'] = h2h_g.cumcount()
player_log['WasH2H'] = (player_log['H2HMeetingsBefore'] > 0).astype(np.float32)
player_log['H2HSetsMarginBefore'] = h2h_g['SetMargin'].transform(lambda s: s.cumsum().shift(fill_value=0.0))

sample_pair = player_log.loc[player_log['H2HMeetingsBefore'] >= 2].iloc[0]
pair_hist = player_log[
    (player_log['IgracId'] == sample_pair['IgracId'])
    & (player_log['OpponentId'] == sample_pair['OpponentId'])
    & (player_log['Datum'] < sample_pair['Datum'])
]
assert len(pair_hist) == sample_pair['H2HMeetingsBefore'], 'WasH2H/H2HMeetingsBefore leakage!'
expected_margin = pair_hist['SetMargin'].sum()
assert abs(expected_margin - sample_pair['H2HSetsMarginBefore']) < 1e-6, 'H2HSetsMarginBefore leakage/mismatch!'
print('H2H leakage spot-check OK.')

# ---------------------------------------------------------------------------
# Step 4 -- build 9-dim per-step feature array + history precompute (identical to 10)
# ---------------------------------------------------------------------------
MAX_LEN = 24
N_FEATS = 9

SETMARGIN_SCALE = 12.0
DAYSREST_CLIP = 90.0

_split_date = pd.Timestamp(TAGS['split_date'])
_train_mask_pl = player_log['Datum'] < _split_date
LIGAKOEF_SCALE = float(player_log.loc[_train_mask_pl, 'LigaKoeficjent'].max())

train_elo_mask = sym['Split'] == 'train'
ELO_MEAN = float(pd.concat([sym.loc[train_elo_mask, 'player_elo'], sym.loc[train_elo_mask, 'opponent_elo']]).mean())
ELO_STD = float(pd.concat([sym.loc[train_elo_mask, 'player_elo'], sym.loc[train_elo_mask, 'opponent_elo']]).std())
print(f'LIGAKOEF_SCALE={LIGAKOEF_SCALE}  ELO_MEAN={ELO_MEAN:.2f}  ELO_STD={ELO_STD:.2f}')

step_feats = np.zeros((len(player_log), N_FEATS), dtype=np.float32)
step_feats[:, 0] = player_log['Won'].values
step_feats[:, 1] = np.clip(player_log['SetMargin'].values / SETMARGIN_SCALE, -3, 3)
days_rest = player_log['DaysSinceLastMatch'].fillna(DAYSREST_CLIP).clip(upper=DAYSREST_CLIP).values
step_feats[:, 2] = days_rest / DAYSREST_CLIP
step_feats[:, 3] = player_log['LigaKoeficjent'].values / LIGAKOEF_SCALE
step_feats[:, 4] = player_log['HomeShareBefore'].fillna(0.0).values
step_feats[:, 5] = player_log['OpponentCareerWinRateBefore'].fillna(0.5).values
step_feats[:, 6] = np.clip((player_log['OppEloBefore'].values - ELO_MEAN) / ELO_STD, -4, 4)
step_feats[:, 7] = player_log['WasH2H'].values
step_feats[:, 8] = 0.0

PLACEHOLDER_VEC = np.zeros(N_FEATS, dtype=np.float32)
PLACEHOLDER_VEC[5] = 0.5
PLACEHOLDER_VEC[6] = 0.0
PLACEHOLDER_VEC[7] = 0.0
PLACEHOLDER_VEC[8] = 1.0

player_log['_gpos'] = np.arange(len(player_log))
first_idx = player_log.groupby('IgracId')['_gpos'].min().to_dict()
idx_lookup = dict(zip(zip(player_log['IgracId'], player_log['SusretId']), player_log['_gpos']))

def build_history(igrac_id, susret_id):
    gpos = idx_lookup[(igrac_id, susret_id)]
    start = first_idx[igrac_id]
    p = gpos - start
    seq = np.zeros((MAX_LEN, N_FEATS), dtype=np.float32)
    if p == 0:
        seq[0] = PLACEHOLDER_VEC
        length = 1
    else:
        lo = max(start, gpos - MAX_LEN)
        chunk = step_feats[lo:gpos]
        length = chunk.shape[0]
        seq[:length] = chunk
    return seq, length

n_rows = len(player_log)
HIST_ARRAY = np.zeros((n_rows, MAX_LEN, N_FEATS), dtype=np.float32)
LEN_ARRAY = np.zeros(n_rows, dtype=np.int64)
igrac_arr, susret_arr = player_log['IgracId'].values, player_log['SusretId'].values
t_hist = time.time()
for i in range(n_rows):
    HIST_ARRAY[i], LEN_ARRAY[i] = build_history(igrac_arr[i], susret_arr[i])
print(f'History precompute: {n_rows:,} rows in {time.time()-t_hist:.1f}s')

p_check = (player_log['_gpos'] - player_log['IgracId'].map(first_idx))
assert (p_check == player_log['MatchCountBefore']).all(), 'group contiguity assumption violated!'
print('OK -- group contiguity holds.')

# ---------------------------------------------------------------------------
# Step 5 -- resolve sym rows onto history positions + chronological carve (identical to 10)
# ---------------------------------------------------------------------------
sym = sym.sort_values(['Datum', 'SusretId', 'Perspective'], kind='mergesort').reset_index(drop=True)
player_pos = np.array([idx_lookup[(pid, sid)] for pid, sid in zip(sym['PlayerId'], sym['SusretId'])])
opp_pos = np.array([idx_lookup[(oid, sid)] for oid, sid in zip(sym['OpponentId'], sym['SusretId'])])

train_matches = (
    sym.loc[sym['Split'] == 'train', ['SusretId', 'Datum']]
    .drop_duplicates().sort_values(['Datum', 'SusretId'], kind='mergesort').reset_index(drop=True)
)
n_train_matches = len(train_matches)
CALIB_FRAC, EARLY_STOP_FRAC = 0.15, 0.10
n_calib = int(round(n_train_matches * CALIB_FRAC))
fit_order = train_matches.iloc[: n_train_matches - n_calib].reset_index(drop=True)
calib_ids = set(train_matches.iloc[n_train_matches - n_calib:]['SusretId'])
n_fit = len(fit_order)
n_eval = int(round(n_fit * EARLY_STOP_FRAC))
core_ids = set(fit_order.iloc[: n_fit - n_eval]['SusretId'])
eval_ids = set(fit_order.iloc[n_fit - n_eval:]['SusretId'])
sym['Bucket'] = np.where(sym['Split'] == 'test', 'test',
                  np.where(sym['SusretId'].isin(calib_ids), 'calib',
                  np.where(sym['SusretId'].isin(eval_ids), 'eval', 'core')))
assert sym.loc[sym['Bucket'] == 'test'].shape[0] == TAGS['n_matches_test'] * 2
print(sym['Bucket'].value_counts())

# ---------------------------------------------------------------------------
# Step 6 -- head-level scalar blocks (identical to 10, EXCEPT the text slot -- Step 6b)
# ---------------------------------------------------------------------------
sym['elo_p_scaled'] = (sym['player_elo'] - ELO_MEAN) / ELO_STD
sym['elo_o_scaled'] = (sym['opponent_elo'] - ELO_MEAN) / ELO_STD

sym['cwr_p'] = sym['player_career_winrate_before'].fillna(0.5)
sym['cwr_o'] = sym['opponent_career_winrate_before'].fillna(0.5)
sym['forma_p'] = sym['player_forma_last10'].fillna(0.5)
sym['forma_o'] = sym['opponent_forma_last10'].fillna(0.5)

MATCHCOUNT_SCALE = 100.0
sym['mc_p'] = np.clip(sym['player_match_count_before'] / MATCHCOUNT_SCALE, 0, 4)
sym['mc_o'] = np.clip(sym['opponent_match_count_before'] / MATCHCOUNT_SCALE, 0, 4)

train_age_mask = sym['Split'] == 'train'
AGE_MEAN = float(pd.concat([sym.loc[train_age_mask, 'player_age'], sym.loc[train_age_mask, 'opponent_age']]).mean())
AGE_STD = float(pd.concat([sym.loc[train_age_mask, 'player_age'], sym.loc[train_age_mask, 'opponent_age']]).std())
sym['age_p'] = ((sym['player_age'] - AGE_MEAN) / AGE_STD).fillna(0.0)
sym['age_o'] = ((sym['opponent_age'] - AGE_MEAN) / AGE_STD).fillna(0.0)

# ---------------------------------------------------------------------------
# Step 6b -- NEW: real TextSkillScore join (THE change vs Exp C3)
# ---------------------------------------------------------------------------
scores = pd.read_csv(DATA_DIR / 'trenutci_scores.csv')[['IgracId', 'skill_1_10', 'confidence']]
print(f'trenutci_scores.csv: {scores.shape[0]} scored players')

# TRAIN-only mean/std for skill_1_10 -- pooled over player+opponent perspectives of TRAIN rows
# that actually HAVE a text score (same discipline as ELO_MEAN/AGE_MEAN: never touch TEST).
train_rows = sym.loc[sym['Split'] == 'train', ['PlayerId', 'OpponentId']]
train_player_ids = pd.concat([train_rows['PlayerId'], train_rows['OpponentId']])
train_scored = scores[scores['IgracId'].isin(set(train_player_ids))]
TEXT_SKILL_MEAN = float(train_scored['skill_1_10'].mean())
TEXT_SKILL_STD = float(train_scored['skill_1_10'].std())
print(f'TEXT_SKILL_MEAN (train-only, scored players) = {TEXT_SKILL_MEAN:.3f}, '
      f'TEXT_SKILL_STD = {TEXT_SKILL_STD:.3f}  (n={len(train_scored)})')

skill_lookup = scores.set_index('IgracId')['skill_1_10'].to_dict()
conf_lookup = scores.set_index('IgracId')['confidence'].to_dict()

for who, id_col in [('p', 'PlayerId'), ('o', 'OpponentId')]:
    raw_skill = sym[id_col].map(skill_lookup)
    has_text = raw_skill.notna().astype(np.float32)
    skill_scaled = ((raw_skill - TEXT_SKILL_MEAN) / TEXT_SKILL_STD).fillna(0.0)  # neutral (=train mean) when missing
    conf_val = sym[id_col].map(conf_lookup).fillna(0.0)  # 0 confidence when no text at all
    sym[f'text_skill_{who}'] = skill_scaled
    sym[f'text_conf_{who}'] = conf_val
    sym[f'has_text_{who}'] = has_text

coverage_p = sym['has_text_p'].mean()
print(f'has_text_p coverage (fraction of ALL sym rows with a scored PlayerId): {coverage_p:.2%}')
assert sym[['text_skill_p', 'text_skill_o', 'text_conf_p', 'text_conf_o', 'has_text_p', 'has_text_o']].isna().sum().sum() == 0

H2HTOTAL_SCALE = 10.0
sym['h2h_wr'] = sym['player_h2h_win_rate_before']
sym['h2h_total_scaled'] = np.clip(sym['h2h_total_before'] / H2HTOTAL_SCALE, 0, 3)
H2H_SETSMARGIN_SCALE = 20.0
sym['h2h_sets_margin_scaled'] = np.clip(player_log['H2HSetsMarginBefore'].values[player_pos] / H2H_SETSMARGIN_SCALE, -3, 3)

HEAD_COLS = ['elo_p_scaled', 'elo_o_scaled', 'cwr_p', 'cwr_o', 'forma_p', 'forma_o',
             'mc_p', 'mc_o', 'age_p', 'age_o',
             'text_skill_p', 'text_skill_o', 'text_conf_p', 'text_conf_o', 'has_text_p', 'has_text_o',
             'h2h_wr', 'h2h_total_scaled', 'h2h_sets_margin_scaled']
N_HEAD = len(HEAD_COLS)
print(f'N_HEAD scalar features: {N_HEAD} (Exp C3 had 15 -- +4 net: 6 new text cols replace 2 neutral ones)')
assert sym[HEAD_COLS].isna().sum().sum() == 0, 'NaN leaked into head scalar block!'

# ---------------------------------------------------------------------------
# Step 7 -- model + training (ONE run, reused hyperparams from Exp C/C2/C3)
# ---------------------------------------------------------------------------
HIST_T = torch.from_numpy(HIST_ARRAY)
LEN_T = torch.from_numpy(LEN_ARRAY)


class TwoTowerGRUEnriched(nn.Module):
    def __init__(self, input_size=N_FEATS, hidden_size=24, dropout=0.3, mlp_hidden=16, n_head=N_HEAD):
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
        packed = pack_padded_sequence(seq, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        return self.drop_embed(h_n[-1])

    def forward(self, player_seq, player_len, opp_seq, opp_len, head_feats):
        h_p = self.encode(player_seq, player_len)
        h_o = self.encode(opp_seq, opp_len)
        combined = torch.cat([h_p, h_o, h_p - h_o, head_feats], dim=1)
        return self.head(combined).squeeze(-1)


def bucket_tensors(bucket_name):
    mask = (sym['Bucket'] == bucket_name).values
    pp = torch.as_tensor(player_pos[mask], dtype=torch.long)
    op = torch.as_tensor(opp_pos[mask], dtype=torch.long)
    y = torch.as_tensor(sym.loc[mask, 'Label'].values, dtype=torch.float32)
    hf = torch.as_tensor(sym.loc[mask, HEAD_COLS].values, dtype=torch.float32)
    return HIST_T[pp], LEN_T[pp], HIST_T[op], LEN_T[op], y, hf

core_pseq, core_plen, core_oseq, core_olen, core_y, core_hf = bucket_tensors('core')
eval_pseq, eval_plen, eval_oseq, eval_olen, eval_y, eval_hf = bucket_tensors('eval')
calib_pseq, calib_plen, calib_oseq, calib_olen, calib_y, calib_hf = bucket_tensors('calib')
test_pseq, test_plen, test_oseq, test_olen, test_y, test_hf = bucket_tensors('test')
print(f'core={core_y.shape[0]:,}  eval={eval_y.shape[0]:,}  calib={calib_y.shape[0]:,}  test={test_y.shape[0]:,}')

HIDDEN_SIZE, DROPOUT, WEIGHT_DECAY, LR = 24, 0.3, 1e-4, 2e-3
BATCH_SIZE, MAX_EPOCHS, PATIENCE = 512, 60, 8

torch.manual_seed(RANDOM_STATE)
model = TwoTowerGRUEnriched(hidden_size=HIDDEN_SIZE, dropout=DROPOUT).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.BCEWithLogitsLoss()
n_core = core_y.shape[0]

def eval_bucket(p_seq, p_len, o_seq, o_len, hf, y):
    model.eval()
    with torch.no_grad():
        logit = model(p_seq, p_len, o_seq, o_len, hf)
    return brier_score_loss(y.numpy(), torch.sigmoid(logit).numpy())

best_brier, best_state, epochs_no_improve = np.inf, None, 0
rng = np.random.RandomState(RANDOM_STATE)
t0 = time.time()
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    perm = rng.permutation(n_core)
    total_loss, n_batches = 0.0, 0
    for start in range(0, n_core, BATCH_SIZE):
        idx = perm[start:start + BATCH_SIZE]
        optimizer.zero_grad()
        logit = model(core_pseq[idx], core_plen[idx], core_oseq[idx], core_olen[idx], core_hf[idx])
        loss = criterion(logit, core_y[idx])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    val_brier = eval_bucket(eval_pseq, eval_plen, eval_oseq, eval_olen, eval_hf, eval_y)
    improved = val_brier < best_brier - 1e-5
    if improved:
        best_brier, best_state, epochs_no_improve = val_brier, {k: v.clone() for k, v in model.state_dict().items()}, 0
    else:
        epochs_no_improve += 1
    if epoch % 5 == 0 or improved or epoch == 1:
        print(f'epoch {epoch:3d}  train_loss={total_loss/n_batches:.4f}  val_brier={val_brier:.5f}'
              f"{'  <- best' if improved else ''}")
    if epochs_no_improve >= PATIENCE:
        print(f'Early stopping at epoch {epoch}.')
        break

model.load_state_dict(best_state)
print(f'Training done in {time.time()-t0:.1f}s. Best eval-bucket Brier = {best_brier:.5f}')

# calibration -- ISOTONIC only, by design (C3 did a temp-vs-isotonic bake-off)
model.eval()
with torch.no_grad():
    calib_logit = model(calib_pseq, calib_plen, calib_oseq, calib_olen, calib_hf)
    test_logit = model(test_pseq, test_plen, test_oseq, test_olen, test_hf)
calib_logit_np, calib_y_np = calib_logit.numpy(), calib_y.numpy()
test_logit_np, test_y_np = test_logit.numpy(), test_y.numpy()

p_calib_raw = 1 / (1 + np.exp(-calib_logit_np))
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(p_calib_raw, calib_y_np)
brier_iso = brier_score_loss(calib_y_np, iso.predict(p_calib_raw))
print(f'Calib Brier -- isotonic: {brier_iso:.5f}')

p_test_raw = 1 / (1 + np.exp(-test_logit_np))
p_test_calibrated = iso.predict(p_test_raw)

def compute_metrics(y_true, p, label):
    p_c = np.clip(p, 1e-6, 1 - 1e-6)
    return {'experiment': label, 'accuracy': accuracy_score(y_true, (p >= 0.5).astype(int)),
            'brier': brier_score_loss(y_true, p), 'log_loss': log_loss(y_true, p_c),
            'auc': roc_auc_score(y_true, p)}

preds4 = pd.read_parquet(DATA_DIR / 'test_predictions_4way.parquet')
elo_brier = brier_score_loss(preds4['Label'], preds4['p_exp0_elo'])
c4_metrics = compute_metrics(test_y_np, p_test_calibrated, 'Exp C4 - enriched GRU + text skill (calibrated)')
c4_metrics['brier_delta_vs_elo_pct'] = (c4_metrics['brier'] - elo_brier) / elo_brier * 100
print(c4_metrics)

preds_c3 = pd.read_parquet(DATA_DIR / 'test_predictions_c3.parquet')
c3_brier = brier_score_loss(preds_c3['Label'], preds_c3['p_expC3_calibrated'])
print(f"\nExp C3 (no text) test Brier for reference: {c3_brier:.5f}")
print(f"Exp C4 (this run) test Brier:                {c4_metrics['brier']:.5f}")
print(f"Delta (C4-C3):                                {c4_metrics['brier']-c3_brier:+.5f}")


# ---------------------------------------------------------------------------
# Step 8 (validation-only) -- cold-start bucket sanity check + permutation importance,
# proving out the analysis logic BEFORE it gets assembled into notebook cells.
# ---------------------------------------------------------------------------
test_keys = sym.loc[sym['Bucket'] == 'test', ['SusretId', 'PlayerId', 'OpponentId', 'player_match_count_before']].reset_index(drop=True)
test_keys['p_expC4_calibrated'] = p_test_calibrated
merged8 = p8 = pd.read_parquet(DATA_DIR / 'test_predictions_8way.parquet')
cold = test_keys.merge(p8[['SusretId', 'PlayerId', 'OpponentId', 'Label', 'p_exp0_elo', 'p_expC2_calibrated', 'p_expC3_calibrated']],
                        on=['SusretId', 'PlayerId', 'OpponentId'], how='inner')
assert len(cold) == len(test_keys), 'row-set mismatch merging onto 8way file!'

cold['bucket'] = pd.cut(cold['player_match_count_before'], bins=[-1, 9, 30, 1e9],
                         labels=['very_low(<10)', 'low(10-30)', 'established(>30)'])
print('\nCold-start bucket sizes:')
print(cold['bucket'].value_counts())

print('\nBrier per bucket (ELO / C2 / C3 / C4):')
for b in ['very_low(<10)', 'low(10-30)', 'established(>30)']:
    sub = cold[cold['bucket'] == b]
    row = {
        'ELO': brier_score_loss(sub['Label'], sub['p_exp0_elo']),
        'C2': brier_score_loss(sub['Label'], sub['p_expC2_calibrated']),
        'C3': brier_score_loss(sub['Label'], sub['p_expC3_calibrated']),
        'C4': brier_score_loss(sub['Label'], sub['p_expC4_calibrated']),
    }
    print(f'  {b} (n={len(sub)}): ' + '  '.join(f'{k}={v:.5f}' for k, v in row.items()) + f"  (C4-C3={row['C4']-row['C3']:+.5f})")

# permutation importance for the 6 text head columns (cheap, no retrain)
baseline_brier = brier_score_loss(test_y_np, p_test_calibrated)
def perm_importance_head(col_idx, n_repeats=3, seed=42):
    rng_local = np.random.RandomState(seed)
    degradations = []
    for _ in range(n_repeats):
        perm_idx = rng_local.permutation(test_hf.shape[0])
        hf_perm = test_hf.clone()
        hf_perm[:, col_idx] = test_hf[perm_idx, col_idx]
        with torch.no_grad():
            logit_perm = model(test_pseq, test_plen, test_oseq, test_olen, hf_perm)
        p_perm = iso.predict(1 / (1 + np.exp(-logit_perm.numpy())))
        degradations.append(brier_score_loss(test_y_np, p_perm) - baseline_brier)
    return np.mean(degradations)

print('\nPermutation importance, text-related head columns:')
for name in ['text_skill_p', 'text_skill_o', 'text_conf_p', 'text_conf_o', 'has_text_p', 'has_text_o']:
    idx = HEAD_COLS.index(name)
    print(f'  {name}: {perm_importance_head(idx):+.5f}')

print(f'\nTotal script time (incl. validation analysis): {time.time()-t_start:.1f}s')
