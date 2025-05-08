# src/data_ingestion.py

import os
import time
from datetime import datetime

from nba_api.stats.endpoints import LeagueDashPlayerStats, LeagueDashTeamStats

def fetch_player_stats(season, out_dir):
    """
    Fetch per-game player stats (basic + advanced) for a given season.
    Saves CSV to out_dir/<season>.csv.
    """
    df = LeagueDashPlayerStats(
        season=season,
        season_type_all_star='Regular Season',
        per_mode_detailed='PerGame'
    ).get_data_frames()[0]
    path = os.path.join(out_dir, f"{season}.csv")
    df.to_csv(path, index=False)
    print(f"[players] wrote {path}")

def fetch_team_stats(season, out_dir):
    """
    Fetch per-game team stats for a given season.
    Saves CSV to out_dir/<season>.csv.
    """
    df = LeagueDashTeamStats(
        season=season,
        season_type_all_star='Regular Season',
        per_mode_detailed='PerGame'
    ).get_data_frames()[0]
    path = os.path.join(out_dir, f"{season}.csv")
    df.to_csv(path, index=False)
    print(f"[teams]   wrote {path}")

def main():
    base_dir    = os.path.abspath(os.path.dirname(__file__))
    raw_dir     = os.path.join(base_dir, "..", "data", "raw")
    players_dir = os.path.join(raw_dir, "players")
    teams_dir   = os.path.join(raw_dir, "teams")
    os.makedirs(players_dir, exist_ok=True)
    os.makedirs(teams_dir,   exist_ok=True)

    # build season list from 1996-97 up through current
    current_year = datetime.now().year
    seasons = [f"{y-1}-{str(y)[-2:]}" for y in range(1997, current_year+1)]

    for season in seasons:
        print(f"▶ {season}")
        try:
            fetch_player_stats(season, players_dir)
        except Exception as e:
            print(f"[players] ERROR {season}: {e}")
        time.sleep(1)  # throttle

        try:
            fetch_team_stats(season, teams_dir)
        except Exception as e:
            print(f"[teams]   ERROR {season}: {e}")
        time.sleep(1)

if __name__ == "__main__":
    main()
