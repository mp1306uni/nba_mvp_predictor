# src/train.py

import os
import json
import pandas as pd
import joblib

from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

# 0. Paths & setup
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PROC_DIR    = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
FULL_CSV    = os.path.join(PROC_DIR, 'mvp_labels_full.csv')
FEAT_CSV    = os.path.join(PROC_DIR, 'features.csv')
os.makedirs(MODELS_DIR, exist_ok=True)

# 1. Load features & labels
features = pd.read_csv(FEAT_CSV)
labels   = pd.read_csv(FULL_CSV)
labels['MVP_class'] = labels['MVP_class'].astype(int)

# 2. Merge then filter to top-5 contenders per season
df = features.merge(labels, on=['PLAYER_ID','SEASON'], how='inner')
df = df.sort_values(['SEASON','composite_score'], ascending=[True, False])
df = df.groupby('SEASON').head(5).reset_index(drop=True)

# 3. Build & save class map
raw_classes = sorted(df['MVP_class'].unique())  # e.g. [0,1,2,3]
class_map   = {int(c): i for i, c in enumerate(raw_classes)}
with open(os.path.join(MODELS_DIR, 'class_map.json'), 'w') as f:
    json.dump(class_map, f, indent=2)
df['MVP_enc'] = df['MVP_class'].map(class_map)

# 4. Prepare X, y, groups
drop_cols = ['PLAYER_ID','PLAYER_NAME','SEASON','MVP_class','MVP_enc']
X         = df.drop(columns=drop_cols)
y         = df['MVP_enc']
groups    = df['SEASON']

# 5. Pipeline: oversample + XGBoost
ros = RandomOverSampler(random_state=42)
xgb = XGBClassifier(
    objective='multi:softprob',
    num_class=len(class_map),
    eval_metric='mlogloss',
    n_estimators=100
)
pipeline = Pipeline([('ros', ros), ('xgb', xgb)])

param_dist = {
    'xgb__learning_rate': [0.01, 0.05, 0.1],
    'xgb__max_depth':      [3, 5, 7],
    'xgb__subsample':      [0.6, 0.8, 1.0],
}

# 6. Hyperparameter tuning with season-based CV
gkf = GroupKFold(n_splits=5)
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=20,
    cv=gkf.split(X, y, groups),
    scoring='accuracy',
    verbose=1,
    random_state=42
)
search.fit(X, y)
best = search.best_estimator_
print("Best params:", search.best_params_)

# 7. Cross-val metrics (Top-1 & Top-2 on true contenders 1–3)
top1, top2 = [], []
eval_raw    = [1, 2, 3]                                         # 3rd, runner-up, MVP
eval_labels = [class_map[c] for c in eval_raw if c in class_map]

for tr_idx, te_idx in gkf.split(X, y, groups):
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]
    best.fit(X_tr, y_tr)

    mask = y_te.isin(eval_labels)
    if not mask.any():
        continue

    y_te_filt = y_te[mask]
    proba_all = best.predict_proba(X_te)[mask]
    proba_filt= proba_all[:, eval_labels]
    pred      = best.predict(X_te)[mask]

    top1.append(accuracy_score(y_te_filt, pred))
    top2.append(top_k_accuracy_score(y_te_filt, proba_filt, k=2, labels=eval_labels))

print(f"CV Top-1 accuracy: {sum(top1)/len(top1):.3f}")
print(f"CV Top-2 accuracy: {sum(top2)/len(top2):.3f}")

# 8. Final train & save
best.fit(X, y)
joblib.dump(best, os.path.join(MODELS_DIR, 'xgb_mvp_tuned.pkl'))
print(f"Saved tuned model → {MODELS_DIR}/xgb_mvp_tuned.pkl")
