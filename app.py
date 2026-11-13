import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Cricket Analytics Pro", page_icon="🏏", layout="wide")
st.title("🏏 Cricket Analytics Pro")
st.caption("Business Analytics dashboard | Team insights, player statistics and match outcome estimation")

DATA_DIR = Path(__file__).parent / "data"

@st.cache_data
def read_default_data():
    match_path = DATA_DIR / "sample_matches.csv"
    ball_path = DATA_DIR / "sample_deliveries.csv"
    if not match_path.exists() or not ball_path.exists():
        raise FileNotFoundError("The data folder must contain sample_matches.csv and sample_deliveries.csv.")
    return pd.read_csv(match_path), pd.read_csv(ball_path)

def normalise_columns(frame):
    frame = frame.copy()
    frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame.columns]
    return frame

try:
    default_matches, default_balls = read_default_data()
except Exception as exc:
    st.error(f"Could not load bundled data: {exc}")
    st.stop()

st.sidebar.header("Data")
uploaded_matches = st.sidebar.file_uploader("Optional match CSV", type=["csv"], key="matches")
uploaded_balls = st.sidebar.file_uploader("Optional ball-by-ball CSV", type=["csv"], key="balls")

try:
    matches = normalise_columns(pd.read_csv(uploaded_matches)) if uploaded_matches else default_matches.copy()
    balls = normalise_columns(pd.read_csv(uploaded_balls)) if uploaded_balls else default_balls.copy()
except Exception as exc:
    st.error(f"Could not read uploaded CSV: {exc}")
    st.stop()

required_match_cols = {"team1", "team2", "winner", "venue"}
missing = required_match_cols - set(matches.columns)
if missing:
    st.error("Match CSV is missing required columns: " + ", ".join(sorted(missing)))
    st.stop()

for col in ["team1", "team2", "winner", "venue"]:
    matches[col] = matches[col].astype("string")
matches = matches.dropna(subset=["team1", "team2", "winner", "venue"]).copy()
if matches.empty:
    st.error("No valid match rows remain after removing missing values.")
    st.stop()

for col, default in [("toss_winner", "Unknown"), ("toss_decision", "Unknown"), ("season", "Unknown")]:
    if col not in matches.columns:
        matches[col] = default
    matches[col] = matches[col].fillna(default).astype(str)

team_names = sorted(set(matches["team1"].astype(str)) | set(matches["team2"].astype(str)))
page = st.sidebar.radio("Choose a module", [
    "Overview", "Team Analytics", "Player Analytics",
    "Match Explorer", "Match Prediction", "About"
])

if page == "Overview":
    st.subheader("Tournament overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches", f"{len(matches):,}")
    c2.metric("Teams", len(team_names))
    c3.metric("Venues", matches["venue"].nunique())
    c4.metric("Most wins", str(matches["winner"].value_counts().index[0]))
    win_counts = matches["winner"].value_counts().rename_axis("Team").reset_index(name="Wins")
    venue_counts = matches["venue"].value_counts().rename_axis("Venue").reset_index(name="Matches")
    left, right = st.columns(2)
    left.plotly_chart(px.bar(win_counts, x="Team", y="Wins", title="Wins by team"), use_container_width=True)
    right.plotly_chart(px.pie(venue_counts, names="Venue", values="Matches", title="Matches by venue"), use_container_width=True)
    st.caption("Bundled data is synthetic demonstration data.")

elif page == "Team Analytics":
    st.subheader("Team performance")
    team = st.selectbox("Select team", team_names)
    played = matches[(matches["team1"] == team) | (matches["team2"] == team)]
    wins = int((played["winner"] == team).sum())
    total = len(played)
    a, b, c = st.columns(3)
    a.metric("Matches played", total)
    b.metric("Wins", wins)
    c.metric("Win rate", f"{wins / total * 100:.1f}%" if total else "N/A")
    venue_wins = (played[played["winner"] == team]
                  .groupby("venue").size().reset_index(name="Wins"))
    if not venue_wins.empty:
        st.plotly_chart(px.bar(venue_wins, x="venue", y="Wins", title=f"{team}: wins by venue"), use_container_width=True)
    else:
        st.info("No wins found for this team in the selected data.")

elif page == "Player Analytics":
    st.subheader("Player statistics")
    required_ball_cols = {"batter", "bowler", "batsman_runs", "is_wicket", "total_runs"}
    missing_ball = required_ball_cols - set(balls.columns)
    if missing_ball:
        st.warning("Ball-by-ball CSV is missing: " + ", ".join(sorted(missing_ball)))
    else:
        for col in ["batsman_runs", "is_wicket", "total_runs"]:
            balls[col] = pd.to_numeric(balls[col], errors="coerce").fillna(0)
        batting = (balls.groupby("batter")
                   .agg(Runs=("batsman_runs", "sum"), Balls=("batsman_runs", "count"))
                   .reset_index())
        batting["Strike Rate"] = (batting["Runs"] / batting["Balls"].replace(0, np.nan) * 100).round(2)
        bowling = (balls.groupby("bowler")
                   .agg(Wickets=("is_wicket", "sum"), Runs=("total_runs", "sum"),
                        Balls=("total_runs", "count")).reset_index())
        bowling["Economy (approx.)"] = (bowling["Runs"] / (bowling["Balls"] / 6).replace(0, np.nan)).round(2)
        x, y = st.columns(2)
        x.plotly_chart(px.bar(batting.nlargest(10, "Runs"), x="batter", y="Runs", title="Top 10 run scorers"), use_container_width=True)
        y.plotly_chart(px.bar(bowling.nlargest(10, "Wickets"), x="bowler", y="Wickets", title="Top 10 wicket takers"), use_container_width=True)
        st.write("Batting leaderboard")
        st.dataframe(batting.sort_values("Runs", ascending=False), use_container_width=True, hide_index=True)
        st.write("Bowling leaderboard")
        st.dataframe(bowling.sort_values("Wickets", ascending=False), use_container_width=True, hide_index=True)
        st.caption("Economy is approximate in this demo. Real data needs correct legal-ball and extras handling.")

elif page == "Match Explorer":
    st.subheader("Historical match explorer")
    selected_team = st.selectbox("Filter by team", ["All teams"] + team_names)
    view = matches if selected_team == "All teams" else matches[
        (matches["team1"] == selected_team) | (matches["team2"] == selected_team)
    ]
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.write(f"Records displayed: {len(view)}")

elif page == "Match Prediction":
    st.subheader("Match outcome estimation")
    st.info("Educational estimate only. Not a guaranteed result. Model quality depends on the quality and size of historical data.")
    feature_cols = ["team1", "team2", "venue", "toss_winner", "toss_decision"]
    model_data = matches.dropna(subset=["winner"]).copy()
    model_data = model_data[model_data["winner"].isin(team_names)]
    if len(model_data) < 20 or model_data["winner"].nunique() < 2:
        st.error("Prediction requires at least 20 labelled matches and at least two winner classes.")
    else:
        X = model_data[feature_cols].fillna("Unknown").astype(str)
        y = model_data["winner"].astype(str)
        stratify_y = y if y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=stratify_y
        )
        prep = ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore"), feature_cols)
        ])
        model = Pipeline([
            ("preprocessor", prep),
            ("classifier", RandomForestClassifier(
                n_estimators=250, max_depth=8, min_samples_leaf=2,
                class_weight="balanced", random_state=42
            ))
        ])
        model.fit(X_train, y_train)
        test_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, test_pred)
        st.metric("Hold-out accuracy", f"{accuracy * 100:.1f}%")
        st.caption("Random split score on the currently loaded data; for a serious evaluation, use chronological season-based testing.")
        col1, col2 = st.columns(2)
        team1 = col1.selectbox("Team 1", team_names, key="pred_team1")
        team2_options = [t for t in team_names if t != team1]
        team2 = col2.selectbox("Team 2", team2_options, key="pred_team2")
        venue_options = sorted(matches["venue"].dropna().astype(str).unique())
        venue = col1.selectbox("Venue", venue_options)
        toss_winner = col2.selectbox("Toss winner", [team1, team2])
        toss_decision = st.radio("Toss decision", ["bat", "field"], horizontal=True)
        if st.button("Predict match outcome", type="primary"):
            input_row = pd.DataFrame([{
                "team1": team1, "team2": team2, "venue": venue,
                "toss_winner": toss_winner, "toss_decision": toss_decision
            }])
            predicted = model.predict(input_row)[0]
            probabilities = model.predict_proba(input_row)[0]
            classes = model.named_steps["classifier"].classes_
            st.success(f"Model's estimated winner: {predicted}")
            probability_df = pd.DataFrame({
                "Team": classes, "Estimated probability": probabilities
            }).sort_values("Estimated probability", ascending=False)
            st.plotly_chart(px.bar(probability_df, x="Team", y="Estimated probability",
                                   title="Estimated class probabilities"), use_container_width=True)
        with st.expander("View model evaluation report"):
            st.text(classification_report(y_test, test_pred, zero_division=0))

else:
    st.subheader("About this project")
    st.markdown("""
    **Project:** Cricket Analytics Pro — Business Analytics group application.

    **Workflow:** Data loading → validation → descriptive analytics → model training → evaluation → prediction.

    **Demo data:** The bundled CSV files are synthetic and are included only so the project runs immediately. Replace them with verified, appropriately licensed historical IPL data before making factual claims about real teams or players.

    **Required match CSV columns:** `team1`, `team2`, `winner`, `venue`.

    **Required ball-by-ball CSV columns:** `batter`, `bowler`, `batsman_runs`, `is_wicket`, `total_runs`.

    **Model:** Random Forest classifier using team, venue and toss fields. The actual match winner is the target and must never be included among input features.
    """)
