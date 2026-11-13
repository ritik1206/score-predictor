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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.figure_factory as ff

st.set_page_config(page_title="Cricket Analytics Pro", page_icon="🏏", layout="wide")
st.title("🏏 Cricket Analytics Pro")
st.caption("A Business Analytics application for IPL performance analysis and match-outcome estimation.")

DATA_DIR = Path(__file__).parent / "data"
@st.cache_data
def load_default():
    m = pd.read_csv(DATA_DIR / "sample_matches.csv")
    d = pd.read_csv(DATA_DIR / "sample_deliveries.csv")
    return m, d

def clean_columns(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

default_m, default_d = load_default()
st.sidebar.header("Data input")
match_file = st.sidebar.file_uploader("Upload match-level CSV (optional)", type=["csv"])
ball_file = st.sidebar.file_uploader("Upload ball-by-ball CSV (optional)", type=["csv"])
matches = clean_columns(pd.read_csv(match_file)) if match_file else default_m.copy()
balls = clean_columns(pd.read_csv(ball_file)) if ball_file else default_d.copy()

required_match = {"team1","team2","winner","venue"}
if not required_match.issubset(matches.columns):
    st.error("Match CSV must contain: team1, team2, winner, venue. See README for the expected schema.")
    st.stop()
for c in ["team1","team2","winner","venue"]:
    matches[c] = matches[c].astype(str)
matches = matches.dropna(subset=["team1","team2","winner"])
if "season" not in matches: matches["season"] = "All seasons"
if "toss_winner" not in matches: matches["toss_winner"] = "Unknown"
if "toss_decision" not in matches: matches["toss_decision"] = "Unknown"

pages = ["Overview","Team Analytics","Player Analytics","Match Explorer","Match Prediction","About & Data Guide"]
page = st.sidebar.radio("Navigate", pages)

if page == "Overview":
    st.subheader("Tournament snapshot")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Matches", f"{len(matches):,}")
    c2.metric("Teams", matches[["team1","team2"]].stack().nunique())
    c3.metric("Venues", matches["venue"].nunique())
    c4.metric("Most wins", matches["winner"].value_counts().index[0] if len(matches) else "—")
    left,right = st.columns(2)
    wins = matches["winner"].value_counts().rename_axis("Team").reset_index(name="Wins")
    left.plotly_chart(px.bar(wins,x="Team",y="Wins",title="Wins by team", color="Wins"),use_container_width=True)
    venue = matches["venue"].value_counts().rename_axis("Venue").reset_index(name="Matches")
    right.plotly_chart(px.pie(venue,names="Venue",values="Matches",title="Matches by venue"),use_container_width=True)
    if "season" in matches:
        season = matches.groupby("season").size().reset_index(name="Matches")
        st.plotly_chart(px.line(season,x="season",y="Matches",markers=True,title="Matches by season"),use_container_width=True)

elif page == "Team Analytics":
    st.subheader("Team performance")
    team_list = sorted(set(matches.team1).union(matches.team2))
    selected = st.selectbox("Select a team",team_list)
    played = matches[(matches.team1==selected)|(matches.team2==selected)]
    wins = int((played.winner==selected).sum())
    losses = int(len(played)-wins)
    a,b,c = st.columns(3)
    a.metric("Matches played",len(played)); b.metric("Wins",wins); c.metric("Win rate",f"{wins/len(played)*100:.1f}%" if len(played) else "—")
    venue_wins = played[played.winner==selected].groupby("venue").size().reset_index(name="Wins")
    st.plotly_chart(px.bar(venue_wins,x="venue",y="Wins",title=f"{selected}: wins by venue"),use_container_width=True)
    summary = pd.DataFrame({"Metric":["Matches","Wins","Losses","Win rate (%)"],"Value":[len(played),wins,losses,round(wins/len(played)*100,1) if len(played) else 0]})
    st.dataframe(summary,use_container_width=True,hide_index=True)

elif page == "Player Analytics":
    st.subheader("Player performance")
    needed = {"batter","bowler","batsman_runs","is_wicket"}
    if not needed.issubset(balls.columns):
        st.warning("Player analytics needs ball-by-ball columns: batter, bowler, batsman_runs, is_wicket. Upload a compatible file or use the bundled demo.")
    else:
        bat = balls.groupby("batter").agg(Runs=("batsman_runs","sum"), Balls=("batsman_runs","count")).reset_index()
        bat["Strike Rate"] = (bat.Runs / bat.Balls * 100).round(2)
        bat = bat.sort_values("Runs",ascending=False).head(15)
        bowl = balls.groupby("bowler").agg(Wickets=("is_wicket","sum"), Runs_Conceded=("total_runs","sum"), Balls=("total_runs","count")).reset_index()
        bowl["Economy"] = (bowl.Runs_Conceded / (bowl.Balls/6)).round(2)
        bowl = bowl.sort_values("Wickets",ascending=False).head(15)
        x,y = st.columns(2)
        x.plotly_chart(px.bar(bat,x="batter",y="Runs",title="Top run scorers"),use_container_width=True)
        y.plotly_chart(px.bar(bowl,x="bowler",y="Wickets",title="Top wicket takers"),use_container_width=True)
        st.markdown("**Batting leaderboard**"); st.dataframe(bat,use_container_width=True,hide_index=True)
        st.markdown("**Bowling leaderboard**"); st.dataframe(bowl,use_container_width=True,hide_index=True)

elif page == "Match Explorer":
    st.subheader("Explore historical matches")
    teams_all = sorted(set(matches.team1).union(matches.team2))
    filt = st.selectbox("Filter by team",["All teams"]+teams_all)
    view = matches if filt=="All teams" else matches[(matches.team1==filt)|(matches.team2==filt)]
    st.dataframe(view.sort_values("date",ascending=False) if "date" in view else view,use_container_width=True,hide_index=True)
    if len(view):
        st.info(f"Displayed {len(view)} match records.")

elif page == "Match Prediction":
    st.subheader("Predict a match outcome")
    st.write("This model learns from historical match records. It is an educational estimate, not a betting tool or guarantee.")
    feature_cols = ["team1","team2","venue","toss_winner","toss_decision"]
    train = matches.dropna(subset=["winner"]).copy()
    train = train[train["winner"].isin(set(train.team1).union(train.team2))]
    if len(train)<20 or train["winner"].nunique()<2:
        st.error("At least 20 labelled historical matches and two winner classes are required.")
    else:
        X=train[feature_cols].fillna("Unknown").astype(str)
        y=train["winner"].astype(str)
        strat = y if y.value_counts().min() >= 2 else None
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=strat)
        prep=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),feature_cols)])
        model=Pipeline([("prep",prep),("rf",RandomForestClassifier(n_estimators=250,max_depth=8,min_samples_leaf=2,class_weight="balanced",random_state=42))])
        model.fit(Xtr,ytr)
        pred=model.predict(Xte)
        acc=accuracy_score(yte,pred)
        st.metric("Hold-out accuracy",f"{acc*100:.1f}%")
        st.caption("Accuracy is measured on a held-out portion of the available dataset. It will vary with dataset size, seasons and data quality.")
        c1,c2=st.columns(2)
        teams_sorted=sorted(set(train.team1).union(train.team2))
        team1=c1.selectbox("Team 1",teams_sorted,index=0)
        team2=c2.selectbox("Team 2",teams_sorted,index=min(1,len(teams_sorted)-1))
        venue=c1.selectbox("Venue",sorted(train.venue.unique()))
        toss_winner=c2.selectbox("Toss winner",[team1,team2])
        toss_decision=st.radio("Toss decision",["bat","field"],horizontal=True)
        if st.button("Estimate match winner",type="primary"):
            if team1==team2:
                st.error("Choose two different teams.")
            else:
                row=pd.DataFrame([{"team1":team1,"team2":team2,"venue":venue,"toss_winner":toss_winner,"toss_decision":toss_decision}])
                proba=model.predict_proba(row)[0]
                classes=model.named_steps["rf"].classes_
                order=np.argsort(proba)[::-1]
                top=classes[order[0]]
                st.success(f"Model estimate: **{top}**")
                probs=pd.DataFrame({"Team":classes,"Estimated probability":proba})
                st.plotly_chart(px.bar(probs.sort_values("Estimated probability",ascending=False),x="Team",y="Estimated probability",title="Model class probabilities"),use_container_width=True)
        with st.expander("Model evaluation details"):
            st.write("Accuracy:",round(acc,3))
            st.text(classification_report(yte,pred,zero_division=0))
            labels=sorted(y.unique())
            cm=confusion_matrix(yte,pred,labels=labels)
            st.dataframe(pd.DataFrame(cm,index=labels,columns=labels),use_container_width=True)

else:
    st.subheader("About this application")
    st.markdown("""
    **Purpose:** Demonstrate how Business Analytics can turn cricket records into actionable summaries.

    **Workflow:** Data upload → cleaning → exploratory analysis → feature preparation → model training → evaluation → prediction.

    **Important:** Bundled sample data is synthetic and created only to demonstrate the application. Replace it with a verified, licensed historical IPL dataset before making real-world claims.

    **Expected match CSV columns:** `team1, team2, winner, venue`; optional: `season, toss_winner, toss_decision, date`.

    **Expected ball-by-ball CSV columns:** `batter, bowler, batsman_runs, is_wicket`; optional: `total_runs, match_id, inning, over, ball`.
    """)
