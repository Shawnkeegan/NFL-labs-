"""NFL free-data research pipeline. No wagers are placed."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
DATA.mkdir(exist_ok=True)
FEATURES = ['home_margin_avg','away_margin_avg','home_points_avg','away_points_avg',
            'home_allowed_avg','away_allowed_avg','home_rest','away_rest','neutral_site']

def fetch():
    import nflreadpy as nfl
    raw = nfl.load_schedules(list(range(2015, 2027))).to_pandas()
    raw.to_csv(DATA / 'schedules.csv', index=False)
    print(f'Saved {len(raw)} schedule rows')

def prepare():
    df = pd.read_csv(DATA / 'schedules.csv')
    required = ['season','week','gameday','home_team','away_team','home_score','away_score']
    missing = set(required) - set(df.columns)
    if missing: raise ValueError(f'Missing columns: {missing}')
    df['gameday'] = pd.to_datetime(df['gameday'])
    df = df.sort_values(['gameday','week','home_team']).reset_index(drop=True)
    # Date-batched updates prevent same-day results leaking into features.
    history = {}
    rows = []
    for date, games in df.groupby('gameday', sort=True):
        updates = []
        for _, g in games.iterrows():
            home, away = str(g.home_team), str(g.away_team)
            def past(team):
                records = history.get(team, [])
                recent = records[-8:]
                if not recent: return (0., 21., 21., 7.)
                margins = [r[1] for r in recent]
                scored = [r[2] for r in recent]
                conceded = [r[3] for r in recent]
                rest = max(0, min(21, (date - records[-1][0]).days))
                return (float(np.mean(margins)), float(np.mean(scored)), float(np.mean(conceded)), float(rest))
            hm,hp,ha,hr = past(home)
            am,ap,aa,ar = past(away)
            hs,as_ = pd.to_numeric(g.home_score, errors='coerce'),pd.to_numeric(g.away_score, errors='coerce')
            row = dict(season=int(g.season),week=int(g.week),gameday=date,
                       home_team=home,away_team=away,home_margin_avg=hm,away_margin_avg=am,
                       home_points_avg=hp,away_points_avg=ap,home_allowed_avg=ha,
                       away_allowed_avg=aa,home_rest=hr,away_rest=ar,
                       neutral_site=int(str(g.get('location','')).lower()=='neutral'),
                       home_score=hs,away_score=as_)
            for c in ['spread_line','total_line']:
                row[c] = pd.to_numeric(g.get(c,np.nan),errors='coerce')
            rows.append(row)
            if pd.notna(hs) and pd.notna(as_):
                updates.extend([(home, (date,float(hs-as_),float(hs),float(as_))),
                                (away, (date,float(as_-hs),float(as_),float(hs)))])
        for team,record in updates: history.setdefault(team,[]).append(record)
    out = pd.DataFrame(rows)
    out.to_csv(DATA / 'features.csv',index=False)
    print(f'Saved {len(out)} rows with strictly pre-game rolling features')

def run():
    df = pd.read_csv(DATA / 'features.csv',parse_dates=['gameday'])
    finished = df.dropna(subset=['home_score','away_score']).copy()
    finished['margin'] = finished.home_score-finished.away_score
    finished['total'] = finished.home_score+finished.away_score
    seasons = sorted(finished.season.unique())
    outputs=[]
    for season in seasons:
        train = finished[finished.season < season]
        test = finished[finished.season == season].copy()
        if len(train)<300 or len(test)<20: continue
        margin_model=Ridge(alpha=100.)
        total_model=Ridge(alpha=100.)
        margin_model.fit(train[FEATURES],train.margin)
        total_model.fit(train[FEATURES],train.total)
        train_margin=margin_model.predict(train[FEATURES])
        # Fixed training-only residual scale; no test outcome leakage.
        margin_sd=max(3.,float(np.std(train.margin-train_margin,ddof=1)))
        total_sd=max(3.,float(np.std(train.total-total_model.predict(train[FEATURES]),ddof=1)))
        test['pred_margin']=margin_model.predict(test[FEATURES])
        test['pred_total']=total_model.predict(test[FEATURES])
        test['margin_sd']=margin_sd
        test['total_sd']=total_sd
        # Approximate win probability via logistic calibration on prior seasons.
        # Tie games excluded from binary calibration.
        non_ties=train[train.margin!=0]
        if non_ties.margin.gt(0).nunique()<2: continue
        train_scores=margin_model.predict(non_ties[FEATURES]).reshape(-1,1)
        cal=LogisticRegression(C=1.)
        cal.fit(train_scores,(non_ties.margin>0).astype(int))
        test['home_win_prob']=cal.predict_proba(test[['pred_margin']].to_numpy())[:,1]
        test['home_win_actual']=(test.margin>0).astype(int)
        test['is_tie']=(test.margin==0)
        outputs.append(test)
        non_tie_test=test[~test.is_tie]
        print(f'{season}: {len(test)} games, margin MAE={mean_absolute_error(test.margin,test.pred_margin):.2f}, '
              f'Brier={brier_score_loss(non_tie_test.home_win_actual,non_tie_test.home_win_prob):.3f}' if len(non_tie_test) else f'{season}: no non-tie games')
    if not outputs: raise ValueError('Insufficient completed historical seasons for walk-forward testing')
    out=pd.concat(outputs,ignore_index=True)
    out.to_csv(DATA/'walkforward.csv',index=False)
    print(f'Saved {len(out)} out-of-sample forecasts to data/walkforward.csv')
    print('NOTE: schedule spread_line/total_line may be closing lines. Not executable pregame odds; no ROI computed.')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['fetch','prepare','walkforward','all'])
    args=p.parse_args()
    if args.command in ('fetch','all'): fetch()
    if args.command in ('prepare','all'): prepare()
    if args.command in ('walkforward','all'): run()
