# src/make_auto_labels.py

import os
import pandas as pd

def main():
    # project root (parent of src/)
    ROOT          = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    VOTES_CSV     = os.path.join(ROOT, 'data', 'external', 'mvp_votes.csv')
    AUTO_LABELS_CSV = os.path.join(ROOT, 'data', 'processed', 'mvp_labels.csv')

    if not os.path.exists(VOTES_CSV):
        raise FileNotFoundError(f"Cannot find vote file at {VOTES_CSV}")

    # 1. Load the raw vote points
    df = pd.read_csv(VOTES_CSV)

    # 2. For each season, pick the top-3 by VOTE_POINTS
    labels = []
    for season, grp in df.groupby('SEASON'):
        top3 = grp.sort_values('VOTE_POINTS', ascending=False).head(3).copy()
        # assign classes: 3=MVP (most points), 2=runner-up, 1=3rd
        top3['MVP_class'] = [3, 2, 1][:len(top3)]
        labels.append(top3[['PLAYER_ID','SEASON','MVP_class']])
    
    result = pd.concat(labels, ignore_index=True)
    os.makedirs(os.path.dirname(AUTO_LABELS_CSV), exist_ok=True)
    result.to_csv(AUTO_LABELS_CSV, index=False)
    print(f"Wrote auto labels → {AUTO_LABELS_CSV}")

if __name__ == '__main__':
    main()
