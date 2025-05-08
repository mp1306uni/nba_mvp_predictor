# streamlit_app.py

import os
import json
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

# ─── 1. Paths & config ────────────────────────────────────────────────────
BASE_DIR        = os.path.abspath(os.path.dirname(__file__))
FEATURES_CSV    = os.path.join(BASE_DIR, 'data', 'processed', 'features.csv')
LABELS_CSV      = os.path.join(BASE_DIR, 'data', 'processed', 'mvp_labels_full.csv')
CLASS_MAP_FILE  = os.path.join(BASE_DIR, 'models', 'class_map.json')

# Best hyperparameters from cross-validation
BEST_PARAMS = {
    'subsample':     0.6,
    'max_depth':     5,
    'learning_rate': 0.01,
    'n_estimators': 100,
    'objective':    'multi:softprob',
    'eval_metric':  'mlogloss'
}

# ─── 2. Load & merge data ──────────────────────────────────────────────────
df_feat = pd.read_csv(FEATURES_CSV)
df_lbl  = pd.read_csv(LABELS_CSV)

# Load class_map and convert keys to ints
_raw_cm   = json.load(open(CLASS_MAP_FILE))
class_map = {int(k): v for k, v in _raw_cm.items()}
inv_map   = {v: k for k, v in class_map.items()}

# Merge features + labels, then encode
df_all = df_feat.merge(df_lbl, on=['PLAYER_ID','SEASON'], how='inner')
df_all['MVP_enc'] = df_all['MVP_class'].map(class_map).astype(int)

# ─── 3. Sidebar: select season & contenders ────────────────────────────────
st.sidebar.title("Settings")
all_seasons = sorted(df_all['SEASON'].unique())
# skip the first season since there’s no prior data to train on
season = st.sidebar.selectbox(
    "Season to predict",
    options=all_seasons[1:], 
    index=len(all_seasons)-2
)
top_n = st.sidebar.slider("Contenders per season", 3, 10, 5)

st.title(f"NBA MVP Predictor — {season}")
st.write(f"Training on seasons before {season}, scoring top-{top_n} contenders")

# ─── 4. Helper: pick top-N per season ───────────────────────────────────────
def top_n_per_season(df, n):
    return (
        df
        .sort_values(['SEASON','composite_score'], ascending=[True, False])
        .groupby('SEASON', group_keys=False)
        .head(n)
        .reset_index(drop=True)
    )

# ─── 5. Split into train vs test ───────────────────────────────────────────
df_train = df_all[df_all['SEASON'] < season]
df_test  = df_all[df_all['SEASON'] == season]

df_train = top_n_per_season(df_train, top_n)
df_test  = top_n_per_season(df_test,  top_n)

if df_train.empty:
    st.error(f"No training data for seasons before {season}.")
    st.stop()
if df_test.empty:
    st.error(f"No contenders for {season}.")
    st.stop()

st.subheader("Training seasons")
st.write(sorted(df_train['SEASON'].unique()))

st.subheader("Test contenders")
st.table(df_test[['PLAYER_NAME','composite_score']])

# ─── 6. Prepare features & labels ─────────────────────────────────────────
drop_cols    = ['PLAYER_ID','PLAYER_NAME','SEASON','MVP_class','MVP_enc']
feature_cols = [c for c in df_all.columns if c not in drop_cols]

X_train = df_train[feature_cols]
y_train = df_train['MVP_enc']
X_test  = df_test[feature_cols]

# ─── 7. Train fresh XGBoost ───────────────────────────────────────────────
model = XGBClassifier(num_class=len(class_map), **BEST_PARAMS)
model.fit(X_train, y_train)

# ─── 8. Predict & format results ──────────────────────────────────────────
proba   = model.predict_proba(X_test)
results = []
for i, row in df_test.iterrows():
    name   = row['PLAYER_NAME']
    scores = proba[i]
    topk   = scores.argsort()[::-1][:3]
    for rank, code in enumerate(topk, start=1):
        results.append({
            'Player':      name,
            'Prediction #': rank,
            'Class':       inv_map[code],        # 3=MVP, 2=runner-up, 1=3rd
            'Probability': round(scores[code], 3)
        })

res_df = pd.DataFrame(results)

st.subheader("Top-3 model predictions per contender")
st.dataframe(
    res_df.pivot(index='Player', columns='Prediction #', values=['Class','Probability'])
)
