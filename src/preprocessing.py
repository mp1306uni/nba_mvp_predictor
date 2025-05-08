import os, glob, warnings, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

BASE_DIR        = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RAW_PLAYERS_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'players')
RAW_TEAMS_DIR   = os.path.join(BASE_DIR, 'data', 'raw', 'teams')
PROCESSED_DIR   = os.path.join(BASE_DIR, 'data', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_and_tag(csv_paths):
    dfs = []
    for fp in csv_paths:
        df = pd.read_csv(fp)
        if df.empty:
            warnings.warn(f"Skipping empty file {fp}")
            continue
        season = os.path.splitext(os.path.basename(fp))[0]
        df['SEASON'] = season
        dfs.append(df)
    if not dfs:
        raise RuntimeError("No valid CSVs to concatenate.")
    return pd.concat(dfs, ignore_index=True)

def clean_player_data():
    # only non-empty season CSVs
    paths = [fp for fp in glob.glob(f"{RAW_PLAYERS_DIR}/*.csv") if os.path.getsize(fp) > 0]
    if not paths:
        raise RuntimeError("No player files found.")
    df = load_and_tag(paths)

    # always keep these cols
    non_num = ['PLAYER_ID','PLAYER_NAME','TEAM_ID','SEASON']
    # numeric stats you want if present
    desired_num = {
      'PTS','REB','AST','STL','BLK',
      'FG_PCT','FG3_PCT','FT_PCT','MIN','GP',
      'PER','TS_PCT','USG_PCT','WS','WS_PER_48','BPM','VORP'
    }
    present_num = [c for c in desired_num if c in df.columns]
    missing = desired_num - set(present_num)
    if missing:
        warnings.warn(f"Missing player columns, skipping: {missing}")

    keep = non_num + present_num
    df = df[keep]

    # impute & scale
    imputer = SimpleImputer(strategy='median')
    df[present_num] = imputer.fit_transform(df[present_num])
    scaler = StandardScaler()
    df[present_num] = scaler.fit_transform(df[present_num])

    return df

def clean_team_data():
    paths = [fp for fp in glob.glob(f"{RAW_TEAMS_DIR}/*.csv") if os.path.getsize(fp) > 0]
    if not paths:
        raise RuntimeError("No team files found.")
    df = load_and_tag(paths)

    # always keep TEAM_ID & SEASON
    non_num = ['TEAM_ID','SEASON']
    desired_num = {'W','W_PCT','NET_RATING','OFF_RATING','DEF_RATING'}
    present_num = [c for c in desired_num if c in df.columns]
    missing = desired_num - set(present_num)
    if missing:
        warnings.warn(f"Missing team cols, skipping: {missing}")

    keep = non_num + present_num
    return df[keep]

def main():
    players_df = clean_player_data()
    players_df.to_csv(os.path.join(PROCESSED_DIR, 'player_stats_processed.csv'), index=False)
    teams_df   = clean_team_data()
    teams_df.to_csv(os.path.join(PROCESSED_DIR, 'team_stats_processed.csv'),   index=False)
    print("✔ Cleaned data written to data/processed/")

if __name__ == '__main__':
    main()
