# src/streamlit_app.py
import os, json
import pandas as pd
import numpy as np
import streamlit as st

#calls the training model and predict model 
from src.train   import train_model, top_n_per_season
from src.predict import predict_proba

# ─── 1. Paths & load ───────────────────────────────────────────────────────
BASE_DIR       = os.path.abspath(os.path.dirname(__file__))
FEAT_CSV       = os.path.join(BASE_DIR, 'data/processed/features.csv')
LBL_CSV        = os.path.join(BASE_DIR, 'data/processed/mvp_labels_full.csv')
VOTE_CSV       = os.path.join(BASE_DIR, 'data/external/mvp_votes.csv')
CLASS_MAP_JSON = os.path.join(BASE_DIR, 'models/class_map.json')

df_feat = pd.read_csv(FEAT_CSV)
df_lbl  = pd.read_csv(LBL_CSV)
votes   = pd.read_csv(VOTE_CSV)

with open(CLASS_MAP_JSON) as f:
    raw_map   = json.load(f)
# map MVP_class→enc and invert
class_map = {int(k):v for k,v in raw_map.items()}
inv_map   = {v:int(k) for k,v in class_map.items()}

valid_seasons = sorted(votes.SEASON.unique())

# ─── 2. Sidebar ───────────────────────────────────────────────────────────
st.sidebar.title("Settings")
st.sidebar.markdown(
    "> **composite_score** = avg of normalized PTS, REB, AST, vote points & first-team votes"
)
season = st.sidebar.selectbox("Season to predict", valid_seasons, index=len(valid_seasons)-1)
top_n  = st.sidebar.slider("Contenders per season", 3, 10, 5)

st.title(f"NBA MVP Predictor — {season}")
st.write(f"Training on seasons before {season}, scoring top-{top_n} contenders")

# ─── 3. Show composite scores ─────────────────────────────────────────────
df_all = df_feat.merge(df_lbl, on=['PLAYER_ID','SEASON'], how='inner')
df_test = top_n_per_season(df_all[df_all.SEASON == season], top_n)
st.subheader("Contenders by Composite Score")
st.table(df_test[['PLAYER_NAME','composite_score']])

# ─── 4. Train & Predict ───────────────────────────────────────────────────
BEST_PARAMS = {
    'subsample':     0.6,
    'max_depth':     5,
    'learning_rate': 0.01,
    'n_estimators': 100,
    'objective':    'multi:softprob',
    'eval_metric':  'mlogloss'
}
model, feature_cols = train_model(df_feat, df_lbl, class_map, season, top_n, BEST_PARAMS)
df_proba           = predict_proba(model, df_feat, season, top_n, inv_map, feature_cols)

# ─── 5. MVP Pick card ─────────────────────────────────────────────────────
mvp_probs  = df_proba['3'].sort_values(ascending=False)
top_player = mvp_probs.index[0]
top_p      = mvp_probs.iloc[0]
st.metric("🪙 Model’s MVP Pick", top_player, f"{top_p:.1%} chance")

# ─── 6. Sorted MVP probabilities ──────────────────────────────────────────
st.subheader("All Contenders — MVP Probability")
mvp_table = (
    pd.DataFrame({"Player": mvp_probs.index, "P(MVP)": mvp_probs.values})
      .assign(**{"P(MVP)": lambda d: d["P(MVP)"].map("{:.1%}".format)})
      .reset_index(drop=True)
)
st.table(mvp_table)

# ─── 7. Full probability breakdown ─────────────────────────────────────────
st.subheader("Contender Probabilities by Class")
full = df_proba.rename(columns={
    '3': 'P(MVP)',
    '2': 'P(Runner-up)',
    '1': 'P(3rd)',
    '0': 'P(Out)'
})
full = full.loc[mvp_probs.index]
st.dataframe(full.style.format("{:.1%}"))
