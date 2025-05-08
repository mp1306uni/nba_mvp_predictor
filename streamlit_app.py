import os, json
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier
import numpy as np

# ─── 1. Paths & config ────────────────────────────────────────────────────
BASE_DIR        = os.path.abspath(os.path.dirname(__file__))
FEATURES_CSV    = os.path.join(BASE_DIR, 'data', 'processed', 'features.csv')
LABELS_CSV      = os.path.join(BASE_DIR, 'data', 'processed', 'mvp_labels_full.csv')
VOTES_CSV       = os.path.join(BASE_DIR, 'data', 'external', 'mvp_votes.csv')

BEST_PARAMS = {
    'subsample':     0.6,
    'max_depth':     5,
    'learning_rate': 0.01,
    'n_estimators': 100,
    'objective':    'multi:softprob',
    'eval_metric':  'mlogloss'
}

# ─── 2. Load data & merge ─────────────────────────────────────────────────
df_feat = pd.read_csv(FEATURES_CSV)
df_lbl  = pd.read_csv(LABELS_CSV)

# build label encoder map
_raw_cm   = json.load(open(os.path.join(BASE_DIR,'models','class_map.json')))
class_map = {int(k): v for k,v in _raw_cm.items()}
inv_map   = {v: k for k,v in class_map.items()}

df_all = df_feat.merge(df_lbl, on=['PLAYER_ID','SEASON'], how='inner')
df_all['MVP_enc'] = df_all['MVP_class'].map(class_map).astype(int)

# valid seasons = those with real ballots
votes_df     = pd.read_csv(VOTES_CSV)
valid_seasons = sorted(votes_df['SEASON'].unique())

# ─── 3. Sidebar ───────────────────────────────────────────────────────────
st.sidebar.title("Settings")
st.sidebar.markdown(
    "> **composite_score** = avg of normalized PTS, REB, AST, vote points & first-team votes"
)
season = st.sidebar.selectbox("Season to predict", valid_seasons, index=len(valid_seasons)-1)
top_n  = st.sidebar.slider("Contenders per season", 3, 10, 5)

st.title(f"NBA MVP Predictor — {season}")
st.write(f"Training on seasons before {season}, scoring top-{top_n} contenders")

# ─── 4. Helper: top-N per season ───────────────────────────────────────────
def top_n_per_season(df, n):
    return (
        df
        .sort_values(['SEASON','composite_score'], ascending=[True, False])
        .groupby('SEASON', group_keys=False)
        .head(n)
        .reset_index(drop=True)
    )

# ─── 5. Split train vs test ───────────────────────────────────────────────
df_train = df_all[df_all['SEASON'] < season]
df_test  = df_all[df_all['SEASON'] == season]

df_train = top_n_per_season(df_train, top_n)
df_test  = top_n_per_season(df_test,  top_n)

if df_train.empty or df_test.empty:
    st.error("Not enough data to train/test for that season.")
    st.stop()

st.subheader("Contenders by Composite Score")
st.table(df_test[['PLAYER_NAME','composite_score']])

# ─── 6. Train model ───────────────────────────────────────────────────────
drop_cols    = ['PLAYER_ID','PLAYER_NAME','SEASON','MVP_class','MVP_enc']
feature_cols = [c for c in df_all.columns if c not in drop_cols]

X_train = df_train[feature_cols]; y_train = df_train['MVP_enc']
X_test  = df_test[feature_cols]

model = XGBClassifier(num_class=len(class_map), **BEST_PARAMS)
model.fit(X_train, y_train)

# ─── 7. Predict ───────────────────────────────────────────────────────────
proba   = model.predict_proba(X_test)
players = df_test['PLAYER_NAME'].tolist()
df_proba = pd.DataFrame(proba, index=players, columns=[inv_map[i] for i in range(proba.shape[1])])

# 7a. Bar chart of probabilities
st.subheader("Prediction Probabilities (MVP, Runner-up, 3rd)")
st.bar_chart(df_proba.loc[players].T)

# ─── 8. Historical accuracy ───────────────────────────────────────────────
st.subheader("Season-over-Season Top-1 Accuracy")
acc = []
for s in valid_seasons:
    # train on prior
    past = df_all[df_all.SEASON < s]
    if past['MVP_enc'].nunique() < 2: 
        acc.append(np.nan); continue
    pts = top_n_per_season(past, top_n)
    Xp, yp = pts[feature_cols], pts['MVP_enc']
    mdl = XGBClassifier(num_class=len(class_map), **BEST_PARAMS).fit(Xp, yp)

    test = df_all[df_all.SEASON == s]
    test = top_n_per_season(test, top_n)
    yp_pred = mdl.predict(test[feature_cols])
    acc.append((yp_pred == test['MVP_enc']).mean())

hist = pd.Series(acc, index=valid_seasons)
st.line_chart(hist)

# ─── 9. Detailed pivot table ──────────────────────────────────────────────
st.subheader("Top-3 model picks per contender")
results = []
for i, name in enumerate(players):
    row = proba[i]
    top3 = row.argsort()[::-1][:3]
    for rank, idx in enumerate(top3,1):
        results.append({
            'Player': name,
            'Rank':   rank,
            'Class':  inv_map[idx],
            'Prob':   float(f"{row[idx]:.3f}")
        })
df_res = pd.DataFrame(results)
st.dataframe(df_res.pivot(index='Player', columns='Rank', values=['Class','Prob']))
