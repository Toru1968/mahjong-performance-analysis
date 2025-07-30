import pandas as pd
import numpy as np

try:
    # 1. データ読み込みとクリーニング
    # Use the latest uploaded file
    df = pd.read_csv('player_stats (6).csv')

    shanten_cols = [
        '対局数',
        '平均配牌シャンテン数',
        '親番平均配牌シャンテン数',
        '子番平均配牌シャンテン数'
    ]
    
    # Use .copy() to avoid SettingWithCopyWarning
    df_cleaned = df.dropna(subset=shanten_cols).copy()
    
    for col in shanten_cols:
        # Using .loc to ensure the changes are made to the original DataFrame slice
        df_cleaned.loc[:, col] = pd.to_numeric(df_cleaned[col], errors='coerce')
    
    df_cleaned.dropna(subset=shanten_cols, inplace=True)

    # 2. 安定したプレイヤー群の抽出
    game_threshold = 200
    df_stable = df_cleaned[df_cleaned['対局数'] >= game_threshold]

    print("--- 配牌シャンテン数の分布範囲から外れているプレイヤー数の分析 ---")
    print(f"\n分析対象: 対局数が{game_threshold}局以上のプレイヤー ({len(df_stable)}名)")
    print("基準: 各シャンテン数の平均値から標準偏差の2倍以上離れているプレイヤーを「外れ値」としてカウントします。")
    print("----------------------------------------------------------------------")

    # 3. 各シャンテン数で外れ値を計算
    for col_name in ['平均配牌シャンテン数', '親番平均配牌シャンテン数', '子番平均配牌シャンテン数']:
        
        if not df_stable.empty:
            # 平均と標準偏差を計算
            mean_val = df_stable[col_name].mean()
            std_val = df_stable[col_name].std()
            
            # 安定範囲を定義 (平均 ± 2σ)
            lower_bound = mean_val - 2 * std_val
            upper_bound = mean_val + 2 * std_val
            
            # 範囲から外れているプレイヤーをカウント
            outliers = df_stable[
                (df_stable[col_name] < lower_bound) | (df_stable[col_name] > upper_bound)
            ]
            num_outliers = len(outliers)
            
            print(f"\nカテゴリ: 「{col_name}」")
            print(f"  安定範囲: {lower_bound:.4f}  ～ {upper_bound:.4f}")
            print(f"  この範囲から外れているプレイヤー数: {num_outliers} 名")
        else:
            print(f"\nカテゴリ: 「{col_name}」")
            print(f"  対局数が{game_threshold}以上のプレイヤーがいないため、計算できません。")


except FileNotFoundError:
    print("エラー: 'player_stats (6).csv' が見つかりませんでした。ファイルをアップロードし直してください。")
except Exception as e:
    print(f"エラーが発生しました: {e}")