# src/predict.py

import os
import json
import joblib
import pandas as pd
from datetime import datetime

# 1. Config
BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MODELS_DIR    = os.path.join(BASE_DIR, 'models')
FEATURES_CSV  = os.path.join(BASE_DIR, 'data', 'processed', 'features.csv')
MODEL_PATH    = os.path.join(MODELS_DIR, 'xgb_mvp_tuned.pkl')
CLASS_MAP     = os.path.join(MODELS_DIR, 'class_map.json')
TOP_N         = 5   # number of contenders
TOP_K         = 3   # top-K predictions

# 2. Season
now    = datetime.now()
year   = now.year if now.month >= 10 else now.year - 1
SEASON = f"{year-1}-{str(year)[-2:]}"
print(f"Predicting MVP for season {SEASON} (top {TOP_N} contenders)")

# 3. Load model & map
model     = joblib.load(MODEL_PATH)
class_map = json.load(open(CLASS_MAP))
inv_map   = {v:int(k) for k,v in class_map.items()}

# 4. Load features
df = pd.read_csv(FEATURES_CSV)
df = df[df['SEASON']==SEASON].copy()
if df.empty:
    raise RuntimeError(f"No data for season {SEASON} in features.csv")

# 5. Top-N contenders by composite_score
df = df.sort_values('composite_score', ascending=False).head(TOP_N).reset_index(drop=True)
print("Top contenders:")
print(df[['PLAYER_NAME','composite_score']])

# 6. Prepare X_pred (keep composite_score!)
drop_cols = ['PLAYER_ID','PLAYER_NAME','SEASON']
X_pred    = df.drop(columns=drop_cols, errors='ignore')

# 7. Align columns exactly to what the model saw
if hasattr(model, 'feature_names_in_'):
    X_pred = X_pred[model.feature_names_in_]

# 8. Predict
proba = model.predict_proba(X_pred)

# 9. Collect top-K for each player
results = []
for i, row in df.iterrows():
    name   = row['PLAYER_NAME']
    scores = proba[i]
    idxs   = scores.argsort()[::-1][:TOP_K]
    for rank, code in enumerate(idxs, start=1):
        results.append({
            'Player':      name,
            'Prediction #': rank,
            'Class':       inv_map[code],
            'Probability': round(scores[code], 3)
        })

out = pd.DataFrame(results)
print("\nPredictions (top-3 for each contender):")
print(out.pivot(index='Player', columns='Prediction #', values=['Class','Probability']))
