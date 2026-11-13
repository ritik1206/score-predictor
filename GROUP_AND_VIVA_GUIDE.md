# Group division, code explanation and viva guide

## Suggested four-member responsibility split

| Member | Ownership | What to demonstrate |
|---|---|---|
| Member 1 — Data lead | Dataset selection, schema, loading, missing values, duplicates and cleaning | Explain `read_csv`, `info`, `isna`, `dropna`, `drop_duplicates` |
| Member 2 — Analytics lead | Team KPIs, player batting/bowling metrics and charts | Explain `groupby`, `agg`, strike rate, economy and Plotly charts |
| Member 3 — ML lead | Feature selection, encoding, train/test split, Random Forest and evaluation | Explain target vs features, leakage, OneHotEncoder, accuracy, precision, recall and F1 |
| Member 4 — App/GitHub lead | Streamlit navigation, file upload, dashboard integration, README and GitHub | Explain `st.sidebar`, `file_uploader`, `st.metric`, `st.plotly_chart`, repository and deployment |

**Everyone must understand the full workflow.** Sir may direct a question to any member, so each person should be able to explain the project objective, dataset, one chart, model inputs/target and limitations.

## Code walkthrough: important code blocks

### 1. Importing libraries
- `pandas`: tabular data loading and transformation.
- `numpy`: numerical operations.
- `plotly.express`: interactive charts.
- `sklearn`: preprocessing, model training and evaluation.
- `streamlit`: web app interface.

### 2. Loading data
`pd.read_csv()` reads the CSV into a DataFrame. The app first loads bundled sample CSVs; if a user uploads a file, the uploaded file replaces the demo file for that session.

### 3. Cleaning and validation
- Standardise column names to lowercase with underscores.
- Check that the match file has `team1`, `team2`, `winner`, `venue`.
- Remove rows with missing team or winner labels.
- In a real project, also check duplicate match IDs, invalid team names, innings values, extras and date formats.

### 4. Team KPIs
For a selected team:
- Matches played = count of rows where team is Team 1 or Team 2.
- Wins = count of rows where winner equals the selected team.
- Losses = matches played − wins.
- Win rate (%) = wins ÷ matches played × 100.

### 5. Player KPIs
- Batting runs = sum of `batsman_runs` grouped by batter.
- Strike rate = runs ÷ balls faced × 100.
- Wickets = sum of `is_wicket` grouped by bowler.
- Bowling economy = runs conceded ÷ overs bowled.

**Real-data caution:** Correct balls faced and legal deliveries require careful treatment of wides, no-balls, byes, leg-byes and run-outs. The starter notebook uses a simplified row-count demonstration and explains this limitation.

### 6. Prediction model
- Features (`X`): Team 1, Team 2, venue, toss winner and toss decision.
- Target (`y`): winner.
- `OneHotEncoder` converts categorical labels into numeric indicator columns.
- `RandomForestClassifier` trains multiple decision trees and combines their votes.
- `train_test_split` holds out a test subset to evaluate performance.
- `accuracy_score` calculates the proportion of correct predictions.
- `classification_report` reports precision, recall and F1-score by class.

**Avoid data leakage:** Do not include the actual winner, final score, player-of-the-match or post-match statistics among prediction features. For a stronger evaluation, split chronologically by season rather than randomly.

## 12 likely viva questions with short answers

1. **What is Business Analytics?**  
   Business Analytics uses data, statistical methods and models to generate insights and support decisions.

2. **What is the objective of your application?**  
   To analyse cricket team/player performance and provide an educational estimate of match outcomes from historical data.

3. **What is the difference between match-level and ball-by-ball data?**  
   Match-level data has one record per match; ball-by-ball data has a record for each delivery.

4. **Why did you use Pandas?**  
   Pandas makes it easier to load, clean, group, aggregate and analyse structured CSV data.

5. **What is a KPI? Give an example.**  
   A Key Performance Indicator measures performance. Team win rate and batting strike rate are examples.

6. **How is batting strike rate calculated?**  
   Runs scored divided by balls faced, multiplied by 100.

7. **How is bowling economy calculated?**  
   Runs conceded divided by overs bowled. In real cricket data, legal balls must be counted correctly.

8. **Why Random Forest?**  
   It combines multiple decision trees, can model non-linear relationships and works well as a baseline for tabular classification.

9. **What is the target variable?**  
   The target is the match winner. The other selected pre-match fields are model features.

10. **Why do we split data into training and testing sets?**  
    Training data fits the model; unseen test data gives an estimate of how well it generalises.

11. **What is overfitting?**  
    Overfitting occurs when a model learns training-data noise and performs poorly on unseen data.

12. **What is data leakage?**  
    Data leakage occurs when information unavailable at prediction time, or information derived from the target, enters model training and makes evaluation unrealistically optimistic.

13. **Can the model guarantee the winner?**  
    No. It provides a probability-based estimate. Cricket outcomes are uncertain and depend on many factors not represented in the dataset.

14. **Why is accuracy not enough?**  
    With imbalanced classes, accuracy can hide poor performance for less frequent teams. Precision, recall, F1-score and a confusion matrix add detail.

15. **What are the limitations of this version?**  
    The bundled data is synthetic; player metrics are simplified; the baseline model has limited features and random hold-out validation. Real, multi-season data and chronological validation are needed.

## Suggested 5–7 minute presentation flow

1. Member 1 (60–75 sec): Problem statement, dataset and cleaning.
2. Member 2 (75–90 sec): Team/player KPIs and visualisations.
3. Member 3 (90–120 sec): ML features, algorithm and evaluation.
4. Member 4 (90–120 sec): Streamlit live demo, GitHub and conclusion.
5. All members: Prepare to answer questions outside their assigned section.

## GitHub checklist

- Create repository `cricket-analytics-pro`.
- Upload the contents of this project folder.
- Confirm `README.md`, `requirements.txt`, `app.py`, `notebooks/` and `data/` appear in the repository.
- Add all group members as collaborators if the repository is private.
- Commit changes with clear messages such as `Add initial notebook`, `Add Streamlit dashboard`, `Add demo datasets`.
- Do not upload passwords, API keys, private tokens or personal data.
- For Streamlit Community Cloud, connect the GitHub repository, choose `app.py` as the main file and deploy. Check current platform requirements before deployment.
