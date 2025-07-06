#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble     import IsolationForest
from sklearn.cluster      import DBSCAN, KMeans
from sklearn.neighbors    import NearestNeighbors
from scipy.stats          import entropy as kl_div
import statsmodels.formula.api as smf
from hmmlearn import hmm

def load_and_preprocess(path):
    df = pd.read_csv(path)
    df = df.sort_values('index').reset_index(drop=True)
    df['riichi_turn'] = df['riichi_turn'].replace(-1, np.nan)
    for c in ['tsumo_counts','naki_counts','total_draw_counts',
              'tsumogiri_count','points_gained']:
        df[c] = df[c].fillna(0)
    df['riichi_turn_by_tsumo'] = df.apply(
        lambda r: (r.riichi_turn * r.tsumo_counts / r.total_draw_counts)
                  if (r.riichi_turn>0 and r.total_draw_counts>0) else r.riichi_turn,
        axis=1
    )
    return df

def build_scaled_fairness_matrix(df):
    feats = ['p_runs','markov_p','p_empirical',
             'entropy_bits','cramers_v',
             'riichi_turn_by_tsumo','tsumo_counts','naki_counts']
    # Fill NaN values with 0 before scaling and PCA
    X = df[feats].fillna(0).to_numpy()
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def run_pca_isoforest(df, X):
    Z = PCA(n_components=2).fit_transform(X)
    df['PC1'], df['PC2'] = Z[:,0], Z[:,1]
    df['anomaly_flag'] = IsolationForest(contamination=0.05, random_state=0)\
                         .fit_predict(X)
    plt.figure(figsize=(6,6))
    sns.scatterplot(data=df, x='PC1', y='PC2',
                    hue='anomaly_flag', palette={1:'gray',-1:'red'})
    plt.title('PCA + IsolationForest (Fairness Only)')
    plt.savefig('pca_isoforest.png')
    plt.close()
    return df, Z

def run_dbscan(df, Z):
    # eps 探し用に5-NN距離プロット
    dists,_ = NearestNeighbors(n_neighbors=5).fit(Z).kneighbors(Z)
    d = np.sort(dists[:,4])
    plt.plot(d); plt.title('5-NN distance'); plt.savefig('dbscan_eps_selection.png'); plt.close()
    # eps=0.7、min_samples=4 は例
    df['dbscan'] = DBSCAN(eps=0.7, min_samples=4).fit(Z).labels_
    return df

def simulate_baseline(df):
    obs = (df['p_runs']<0.05).mean()
    sims = np.random.binomial(len(df), 0.05, size=5000)/len(df)
    kl = kl_div([obs,1-obs],[sims.mean(),1-sims.mean()])
    with open('baseline_comparison.txt','w') as f:
        f.write(f"Observed rate = {obs:.3f}\nSimulated rate = {sims.mean():.3f}\nKL = {kl:.3f}\n\n")
        f.write("Regression on points_gained:\n")
        reg = smf.ols('points_gained ~ p_runs + cramers_v + riichi_turn_by_tsumo', data=df).fit()
        f.write(reg.summary().as_text())

def save_report(df):
    df.to_csv('final_report.csv', index=False)
    print("▶ final_report.csv written.")

def main(path):
    df = load_and_preprocess(path)
    X  = build_scaled_fairness_matrix(df)
    df, Z = run_pca_isoforest(df, X)
    df['CUSUM'] = (df['anomaly_flag']==-1).cumsum()
    df = run_dbscan(df, Z)
    # HMM on p_runs
    vals = df['p_runs'].fillna(1).to_numpy().reshape(-1,1)
    model = hmm.GaussianHMM(n_components=2, covariance_type='diag', n_iter=50)
    model.fit(vals)
    df['hmm_state'] = model.predict(vals)
    simulate_baseline(df)
    save_report(df)

if __name__=='__main__':
    main("fairness_comprehensive_report.csv")