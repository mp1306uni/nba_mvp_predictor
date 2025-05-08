# src/make_full_labels.py

import os
import pandas as pd

BASE     = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PROC     = os.path.join(BASE, 'data', 'processed')
PLAYERS  = os.path.join(PROC, 'player_stats_processed.csv')
TOP3     = os.path.join(PROC, 'mvp_labels.csv')           # manual or auto top-3
FULL_CSV = os.path.join(PROC, 'mvp_labels_full.csv')

# 1. Load every player-season and your top-3 labels
players = pd.read_csv(PLAYERS)[['PLAYER_ID','SEASON']]
top3    = pd.read_csv(TOP3)

# 2. Merge and fill non-top3 with 0
full    = players.merge(top3, on=['PLAYER_ID','SEASON'], how='left')
full['MVP_class'] = full['MVP_class'].fillna(0).astype(int)

# 3. Save complete label set
full.to_csv(FULL_CSV, index=False)
print(f"Wrote full labels → {FULL_CSV}")
