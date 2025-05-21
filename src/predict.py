# src/predict.py
import pandas as pd

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

def predict_proba(
    model,
    features_df: pd.DataFrame,
    season: int,
    top_n: int,
    inv_map: dict[int,int],
    feature_cols: list[str]
) -> pd.DataFrame:
    """
    Returns a DataFrame of class-probabilities for the top_n
    contenders in `season`. Index is PLAYER_NAME, columns are class labels as strings.
    """
    # 1. Select that season’s top-n by composite_score
    df_season = features_df[features_df.SEASON == season]
    df_test   = top_n_per_season(df_season, top_n)

    # 2. Prep X and predict
    X_test    = df_test[feature_cols]
    proba     = model.predict_proba(X_test)

    # 3. Build DataFrame
    df_proba = pd.DataFrame(
        proba,
        index=df_test.PLAYER_NAME,
        columns=[inv_map[i] for i in range(proba.shape[1])]
    )
    # force column names to strings so you can do df_proba['3']
    df_proba.columns = df_proba.columns.map(str)

    return df_proba
