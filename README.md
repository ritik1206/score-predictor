# Cricket Analytics Pro (BA Group Project)

An educational IPL analytics dashboard with team/player analysis, match exploration and a Random Forest match-outcome estimation module.

## Included files
- `app.py` — Streamlit application
- `notebooks/Cricket_Analytics_Colab.ipynb` — Colab-ready analysis notebook
- `data/sample_matches.csv` — synthetic demo match data
- `data/sample_deliveries.csv` — synthetic demo ball-by-ball data
- `requirements.txt` — dependencies
- `presentation/Cricket_Analytics_Pro_Presentation.pptx` — editable presentation

## Important data note
The bundled CSVs are **synthetic demonstration data**, not official IPL records. They allow the app to run immediately and are clearly labelled as demo data. For academic analysis and credible results, replace them with a verified dataset whose licence permits your use. Do not present demo-model accuracy as real-world predictive performance.

## Run in Google Colab
1. Upload the notebook to Google Colab.
2. Run cells from top to bottom.
3. Upload your match-level and ball-by-ball CSV files when prompted, or use the bundled demo CSVs.
4. The notebook performs data checks, EDA, player/team summaries, model training and evaluation.

## Run the Streamlit app locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
To run from Colab, install requirements and use a trusted Streamlit tunnel method approved by your instructor. Never publish Colab secrets or private tokens.

## GitHub setup
1. Create a public or private repository named `cricket-analytics-pro`.
2. Upload the contents of this folder (not the ZIP file alone).
3. Commit the notebook, app, data, README and requirements.
4. Add all four group members as collaborators if needed.
5. In your presentation, open the repository and demonstrate the running app.

## Data schemas
### Match CSV (required)
`team1, team2, winner, venue`
Optional: `match_id, date, season, toss_winner, toss_decision, win_by_runs, win_by_wickets, result`

### Ball-by-ball CSV (for player analysis)
`batter, bowler, batsman_runs, is_wicket`
Optional: `total_runs, match_id, inning, batting_team, bowling_team, over, ball`

## Modelling notes
The baseline model uses pre-match fields: team1, team2, venue, toss_winner and toss_decision. It uses a train/test split and reports hold-out accuracy plus a classification report. With a small or synthetic dataset, the score is not a meaningful estimate of real match-prediction ability. Improve the project by using multiple real seasons, chronological validation, class-wise metrics and careful feature availability checks.

## Suggested academic workflow
Problem definition → dataset understanding → cleaning → EDA → descriptive analytics → feature engineering → model training → evaluation → dashboard → limitations and conclusion.
