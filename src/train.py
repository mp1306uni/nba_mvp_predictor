# src/train.py
import pandas as pd
from xgboost import XGBClassifier

# ─── Helpers ────────────────────────────────────────────────────────────────
def top_n_per_season(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    For each SEASON in df, keep the top-n rows by composite_score.
    """
    return (
        df
        .sort_values(['SEASON', 'composite_score'], ascending=[True, False])
        .groupby('SEASON', group_keys=False)
        .head(n)
        .reset_index(drop=True)
    )

def train_model(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    class_map: dict[int,int],
    season: int,
    top_n: int,
    params: dict
) -> tuple[XGBClassifier, list[str]]:
    """
    Train an XGBClassifier on all seasons < `season`.
    Returns the fitted model and the list of feature column names.
    """
    # 1. Merge features + labels, encode target
    df = features_df.merge(labels_df, on=['PLAYER_ID','SEASON'], how='inner')
    df['MVP_enc'] = df['MVP_class'].map(class_map).astype(int)

    # 2. Filter to seasons before `season` and top-n contenders each year
    train_df = df[df.SEASON < season]
    train_df = top_n_per_season(train_df, top_n)

    # 3. Prepare X, y
    drop_cols   = ['PLAYER_ID','PLAYER_NAME','SEASON','MVP_class','MVP_enc']
    feature_cols= [c for c in train_df.columns if c not in drop_cols]
    X_train     = train_df[feature_cols]
    y_train     = train_df['MVP_enc']

    # 4. Fit model
    model = XGBClassifier(num_class=len(class_map), **params)
    model.fit(X_train, y_train)

    return model, feature_cols
