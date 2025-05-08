# src/fetch_mvp_votes.py

import os
import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
from nba_api.stats.static import players

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
OUTPUT_CSV = os.path.join(BASE_DIR, 'data', 'external', 'mvp_votes.csv')
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# map player names → IDs
all_players = players.get_players()
name_to_id  = {p['full_name']: p['id'] for p in all_players}

records = []

for year in range(1997, 2025):  # seasons 1996-97 up through 2024-25
    season = f"{year-1}-{str(year)[-2:]}"
    url    = f"https://www.basketball-reference.com/awards/awards_{year}.html"
    print(f"Fetching MVP votes for {season}…", end=" ")

    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        html = res.text
    except Exception as e:
        print(f"❌ network error ({e}), skipping")
        continue

    # First try pandas.read_html
    df_mvp = None
    try:
        tables = pd.read_html(html, header=[0,1], flavor='bs4')
        for tbl in tables:
            # flatten multiindex
            if isinstance(tbl.columns, pd.MultiIndex):
                tbl.columns = [
                    " ".join(filter(lambda x: str(x)!='nan', col)).strip()
                    for col in tbl.columns.values
                ]
            if any("Pts Won" in c for c in tbl.columns):
                df_mvp = tbl
                break
    except Exception:
        df_mvp = None

    # If that failed, fall back to comment‐scraping
    if df_mvp is None:
        soup = BeautifulSoup(html, 'html.parser')
        comment = None
        for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
            if 'table id="mvp"' in c:
                comment = c
                break
        if comment:
            tbl_soup = BeautifulSoup(comment, 'html.parser')
            df_mvp = pd.read_html(str(tbl_soup), header=[0,1], flavor='bs4')[0]
            # flatten columns
            df_mvp.columns = [
                " ".join(filter(lambda x: str(x)!='nan', col)).strip()
                for col in df_mvp.columns.values
            ]

    if df_mvp is None:
        print("⚠️ no MVP table found, skipping")
        continue

    # identify columns
    cols       = df_mvp.columns.tolist()
    col_player = next((c for c in cols if "Player"   in c), None)
    col_pts    = next((c for c in cols if "Pts Won"  in c), None)
    col_first  = next((c for c in cols if "First"    in c), None)

    if not col_player or not col_pts:
        print("⚠️ table missing key columns, skipping")
        continue

    # extract rows
    for _, row in df_mvp.iterrows():
        name     = row[col_player]
        try:
            pts_won  = int(row[col_pts])
        except:
            pts_won  = 0
        try:
            first_tm = int(row[col_first]) if col_first else 0
        except:
            first_tm = 0

        pid = name_to_id.get(name)
        if pid is None:
            continue

        records.append({
            'PLAYER_ID':   pid,
            'SEASON':      season,
            'VOTE_POINTS': pts_won,
            'VOTE_FIRST':  first_tm
        })

    print("✔")

# write CSV
out = pd.DataFrame(records)
out.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote MVP votes → {OUTPUT_CSV}")