from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.metrics import brier_score_loss, mean_absolute_error
st.set_page_config(page_title='NFL Free Research Lab',layout='wide')
st.title('NFL Free Research Lab')
st.caption('Out-of-sample model research • no betting execution • no verified ROI')
path=Path(__file__).resolve().parent/'data'/'walkforward.csv'
if not path.exists():
    st.info('Run: python engine.py all — then reload this dashboard.')
    st.stop()
df=pd.read_csv(path)
seasons=sorted(df.season.unique().tolist())
selected=st.multiselect('Test seasons',seasons,default=seasons)
df=df[df.season.isin(selected)].copy()
if df.empty: st.stop()
a,b,c=st.columns(3)
a.metric('Out-of-sample games',len(df))
b.metric('Margin MAE',f'{mean_absolute_error(df.margin,df.pred_margin):.2f} points')
nt=df[~df.is_tie.astype(bool)]
c.metric('Win probability Brier',f'{brier_score_loss(nt.home_win_actual,nt.home_win_prob):.3f}' if len(nt) else 'N/A')
st.subheader('Margin forecast vs result')
st.plotly_chart(px.scatter(df,x='pred_margin',y='margin',color='season',hover_data=['home_team','away_team','week']),use_container_width=True)
st.subheader('Win probability calibration')
if len(nt):
    bins=pd.cut(nt.home_win_prob,bins=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1],include_lowest=True)
    calibration=nt.groupby(bins,observed=True).agg(predicted=('home_win_prob','mean'),observed=('home_win_actual','mean'),games=('home_win_actual','size')).reset_index(drop=True)
    st.plotly_chart(px.line(calibration,x='predicted',y='observed',markers=True,hover_data=['games']),use_container_width=True)
st.subheader('Historical forecasts')
st.dataframe(df[['season','week','home_team','away_team','home_win_prob','pred_margin','margin','pred_total','total','margin_sd','total_sd']],use_container_width=True)
st.warning('The source schedule lines are not timestamped executable odds. Paper ROI and closing-line value cannot be validated until timestamped sportsbook snapshots are collected. Player props are not implemented in this starter.')
