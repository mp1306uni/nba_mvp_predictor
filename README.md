# NBA MVP Predictor

This Streamlit app predicts the NBA MVP, runner-up, and third place for any season from 1996–97 onward, using XGBoost trained on box-score stats and real MVP ballot data.

## Features

- **Data pipeline**: ingests NBA stats & MVP voting ballots, cleans and merges them  
- **Model**: XGBoost multi-class classifier with cross-validation and tuning  
- **Interactive dashboard**:  
  - Pick any season (1996–97 → latest)  
  - Slider to choose number of contenders  
  - Composite-score ranking (normalized PTS, REB, AST, votes)  
  - Top-3 predictions per contender  
  - Bar chart of prediction probabilities  
  - Historical accuracy chart  

## Local Setup

```bash
git clone https://github.com/mp1306uni/nba_mvp_predictor.git
cd nba_mvp_predictor
python -m venv .venv
.venv/Scripts/Activate      # Windows PowerShell
pip install -r requirements.txt

# Build data & model
python src/data_ingestion.py
python src/preprocessing.py
python src/features.py
python src/make_auto_labels.py
python src/make_full_labels.py
python src/train.py

# Run the dashboard
streamlit run streamlit_app.py
