# src/features.py

import os
import pandas as pd

BASE_DIR        = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PROCESSED_DIR   = os.path.join(BASE_DIR, 'data', 'processed')
FEATURES_CSV    = os.path.join(PROCESSED_DIR, 'features.csv')
MVP_VOTES_CSV   = os.path.join(BASE_DIR, 'data', 'external', 'mvp_votes.csv')

def main():
    # 1. Load cleaned player & team stats
    players = pd.read_csv(os.path.join(PROCESSED_DIR, 'player_stats_processed.csv'))
    teams   = pd.read_csv(os.path.join(PROCESSED_DIR, 'team_stats_processed.csv'))

    # 2. Merge on TEAM_ID + SEASON
    df = players.merge(teams, on=['TEAM_ID', 'SEASON'], how='left')

    # 3. Merge MVP voting data
    if os.path.exists(MVP_VOTES_CSV):
        votes = pd.read_csv(MVP_VOTES_CSV)
        df   = df.merge(votes, on=['PLAYER_ID', 'SEASON'], how='left')
        df['VOTE_POINTS'] = df['VOTE_POINTS'].fillna(0)
        df['VOTE_FIRST']  = df['VOTE_FIRST'].fillna(0)
    else:
        print(f"Warning: {MVP_VOTES_CSV} not found—skipping MVP vote features")
        df['VOTE_POINTS'] = 0
        df['VOTE_FIRST']  = 0

    # 4. Compute composite_score for contender filtering
    #    include core box‐score stats plus MVP vote features
    metrics = ['PTS', 'REB', 'AST', 'VOTE_POINTS', 'VOTE_FIRST']
    present = [m for m in metrics if m in df.columns]
    if not present:
        raise RuntimeError("No composite_score metrics found in features.")
    for m in present:
        norm_col = f"{m}_norm"
        df[norm_col] = df.groupby('SEASON')[m].transform(
            lambda x: (x - x.min()) / (x.max() - x.min())
                      if x.max() != x.min() else 0
        )
    df['composite_score'] = df[[f"{m}_norm" for m in present]].mean(axis=1)

    # 5. Save full feature set
    df.to_csv(FEATURES_CSV, index=False)
    print(f"✔ features.csv written to {FEATURES_CSV}")

if __name__ == '__main__':
    main()
