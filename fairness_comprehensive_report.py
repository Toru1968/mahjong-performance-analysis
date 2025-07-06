#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler                 # ← CHANGED
from sklearn.decomposition import PCA
from sklearn.ensemble     import IsolationForest
from sklearn.cluster      import DBSCAN
from sklearn.cluster      import KMeans
from sklearn.neighbors    import NearestNeighbors                # ← CHANGED
from scipy.stats          import entropy as kl_div
import statsmodels.api     as sm
import statsmodels.formula.api as smf
from hmmlearn import hmm

def load_and_preprocess(path):
    df = pd.read_csv(path)
    if 'game_index' in df.columns:
        df = df.sort_values('game_index').reset_index(drop=True)
    df['riichi_turn'] = df['riichi_turn'].replace(-1, np.nan)
    # ツモ／鳴き／得失点は後段分析用なので型変換のみ
    for c in ['tsumo_counts','naki_counts','total_draw_counts',
              'tsumogiri_count','points_gained']:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    # 自摸補正リーチ巡目
    df['riichi_turn_by_tsumo'] = df.apply(
        lambda r: (r.riichi_turn * (r.tsumo_counts / r.total_draw_counts))
                  if (r.riichi_turn>0 and r.total_draw_counts>0) else r.riichi_turn,
        axis=1
    )
    return df

def build_scaled_fairness_matrix(df):                         # ← CHANGED
    # 牌山偏り指標のみ抽出
    feats = [
        'p_runs', 'markov_p', 'p_empirical',
        'entropy_bits', 'cramers_v',
        'riichi_turn_by_tsumo',
        'tsumo_counts','naki_counts'
    ]
    X = df[feats].fillna(0).to_numpy()
    scaler = StandardScaler()                                  # ← CHANGED
    X_scaled = scaler.fit_transform(X)                         # ← CHANGED
    return X_scaled

def run_pca_isoforest(df, X_scaled):                           # ← CHANGED
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X_scaled)                            # ← CHANGED
    df['PC1'], df['PC2'] = Z[:,0], Z[:,1]
    iso = IsolationForest(contamination=0.05, random_state=42)
    df['anomaly_flag'] = iso.fit_predict(X_scaled)             # ← CHANGED
    # 可視化
    plt.figure(figsize=(6,6))
    sns.scatterplot(x='PC1', y='PC2', hue='anomaly_flag',
                    data=df, palette={1:'blue', -1:'red'})
    plt.title('PCA + IsolationForest (Fairness Only)')
    plt.savefig('pca_isoforest.png')
    plt.close()
    return df, Z                                                # ← CHANGED

def run_dbscan(df, Z):                                         # ← CHANGED
    # まず5-NN距離をプロットして eps 値を決めます（省略可）
    nbrs = NearestNeighbors(n_neighbors=5).fit(Z)
    dists, _ = nbrs.kneighbors(Z)
    dists = np.sort(dists[:,4])
    plt.figure()
    plt.plot(dists)
    plt.title('5-NN Distance Plot for eps selection')
    plt.savefig('dbscan_eps_selection.png')
    plt.close()

    # eps=0.7, min_samples=4 は一例。5-NNから適宜調整してください
    db = DBSCAN(eps=0.7, min_samples=4).fit(Z)
    df['dbscan'] = db.labels_
    return df

def simulate_baseline(df, path='baseline_comparison.txt'):
    obs = np.mean(df['p_runs'] < 0.05)
    sims = np.random.binomial(len(df), 0.05, size=5000) / len(df)
    kl = kl_div([obs,1-obs],[sims.mean(),1-sims.mean()])
    with open(path,'w') as f:
        f.write(f"Observed p_runs<0.05 rate = {obs:.4f}\n")
        f.write(f"Simulated mean rate     = {sims.mean():.4f}\n")
        f.write(f"KL divergence            = {kl:.4f}\n\n")
        # 得失点回帰
        f.write("=== Points Gained Regression ===\n")
        df_r = df.dropna(subset=['points_gained','p_runs'])
        model = smf.ols('points_gained ~ p_runs + cramers_v + riichi_turn_by_tsumo',
                        data=df_r).fit()
        f.write(model.summary().as_text())
    return sims

def save_report(df, path='fairness_comprehensive_report.csv'):
    cols = [
        'index','name','seat','game_index','riichi_turn',
        'tsumo_counts','naki_counts','total_draw_counts',
        'riichi_turn_by_tsumo','p_runs','markov_p','p_empirical',
        'entropy_bits','cramers_v',
        'tsumogiri_count','points_gained',
        'PC1','PC2','anomaly_flag','CUSUM','kmeans','dbscan','hmm_state'
    ]
    df.to_csv(path, columns=[c for c in cols if c in df.columns],
              index=False)
    print(f"▶ Saved {path}")

def main():
    df = load_and_preprocess('analyzed_report.csv')
    Xs = build_scaled_fairness_matrix(df)                     # ← CHANGED

    # PCA + IsoForest
    df, Z = run_pca_isoforest(df, Xs)                          # ← CHANGED

    # CUSUM（省略:同じまま、anomaly_flag を累積）
    df['CUSUM'] = (df['anomaly_flag']==-1).astype(int).cumsum()

    # k-means (必要なら)
    km = KMeans(n_clusters=3, random_state=0).fit(Z)
    df['kmeans'] = km.labels_

    # DBSCAN
    df = run_dbscan(df, Z)                                     # ← CHANGED

    # HMM (同じまま)
    vals = df['p_runs'].fillna(1).to_numpy().reshape(-1,1)
    model = hmm.GaussianHMM(n_components=2, covariance_type='diag', n_iter=100)
    model.fit(vals)
    df['hmm_state'] = model.predict(vals)

    # シミュレーション＋回帰
    simulate_baseline(df)

    # 最後にレポート出力
    save_report(df)

if __name__ == '__main__':
    main()