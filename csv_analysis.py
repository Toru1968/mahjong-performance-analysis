import pandas as pd
from itertools import combinations
import copy
import random # random モジュールを追加
import math
import numpy as np
from collections import Counter, defaultdict
from scipy.stats import chisquare
from scipy import stats
from typing import Any, Dict, List

# ==============================================================================
# 1. 牌の定義と変換ユーティリティ
# ==============================================================================

# 牌の文字列表現とソート用の内部表現の対応
# z: 1=東, 2=南, 3=西, 4=北, 5=白, 6=発, 7=中
# 内部表現はソート可能かつ一意になるように定義
# 通常の数牌: 11-19 (1m-9m), 21-29 (1p-9p), 31-39 (1s-9s)
# 字牌: 41-47 (1z-7z)
# 赤ドラ: 51 (0m), 52 (0p), 53 (0s) - 通常の5とは異なる値とする
TILES_MAP = {
    **{f'{i}m': 10+i for i in range(1, 10)},
    **{f'{i}p': 20+i for i in range(1, 10)},
    **{f'{i}s': 30+i for i in range(1, 10)},
    **{f'{i}z': 40+i for i in range(1, 8)}, # 1z-7z (東南西北白発中)
    '0m': 51, # 赤ドラ5m
    '0p': 52, # 赤ドラ5p
    '0s': 53  # 赤ドラ5s
}
INV_TILES_MAP = {v: k for k, v in TILES_MAP.items()}

def to_str(tile_int):
    """内部表現(int)を文字列に変換"""
    return INV_TILES_MAP.get(tile_int, '?')

def to_int(tile_str):
    """文字列を内部表現(int)に変換"""
    return TILES_MAP.get(tile_str, 0)

def format_hand(hand_ints, sort=True):
    """手牌リストを整形して文字列で表示"""
    if sort:
        hand_ints.sort()
    return ' '.join([to_str(t) for t in hand_ints])

def parse_tile_list_string(tile_list_str: str) -> List[str]:
    if not isinstance(tile_list_str, str) or not tile_list_str: return []
    tiles, i = [], 0
    while i < len(tile_list_str):
        if i + 1 < len(tile_list_str):
            num_char, suit_char = tile_list_str[i], tile_list_str[i+1]
            if (num_char.isdigit() or num_char == '0') and suit_char in ['m', 'p', 's', 'z']:
                tiles.append(num_char + suit_char); i += 2
            else: i += 1
        else: i += 1
    return tiles

# ==============================================================================
# 牌山生成ロジック
# ==============================================================================

def generate_mahjong_wall():
    """
    オンライン麻雀ゲームで一般的に採用されている、シード値を使わない
    標準的な牌山生成ロジック (全136枚) を生成します。
    """
    # 萬子 (マンズ): 1m - 9m 各4枚 (通常の5mも含む)
    manzu = [10 + i for i in range(1, 10)] * 4
    # 筒子 (ピンズ): 1p - 9p 各4枚 (通常の5pも含む)
    pinzu = [20 + i for i in range(1, 10)] * 4
    # 索子 (ソーズ): 1s - 9s 各4枚 (通常の5sも含む)
    souzu = [30 + i for i in range(1, 10)] * 4
    # 字牌 (ツーパイ): 東西南北白発中 各4枚
    zihai = [40 + i for i in range(1, 8)] * 4 # 7種類

    all_tiles = manzu + pinzu + souzu + zihai

    red_dora_ints = [51, 52, 53] # 0m, 0p, 0s の内部表現
    normal_five_ints = [15, 25, 35] # 5m, 5p, 5s の内部表現

    for red_dora_int, normal_five_int in zip(red_dora_ints, normal_five_ints):
        try:
            idx = all_tiles.index(normal_five_int)
            all_tiles[idx] = red_dora_int
        except ValueError:
            print(f"警告: 通常の牌 {to_str(normal_five_int)} が牌山に見つかりませんでした。")
            return None # 異常終了

    if len(all_tiles) != 136:
         print(f"エラー: 牌山生成数が不正です ({len(all_tiles)}枚)")
         return None

    random.shuffle(all_tiles)

    return all_tiles

# ==============================================================================
# 分析ロジック
# ==============================================================================

def verify_sequence_randomness(row_index,tile_sequence: list[int],analyze_str: str = "",player: str = "", alpha: float = 0.0025):
    """
    牌の時系列リストを受け取り、そのシーケンスのランダム性を検証する。

    カイ二乗検定（分布の偏り）とランズ・テスト（並びの偏り）を用いて多角的に評価し、
    結果をまとめた辞書を返す。

    Args:
        tile_sequence (list[int]): 検証対象の牌のシーケンス。
        alpha (float, optional): 統計的検定の有意水準。デフォルトは 0.05。

    Returns:
        dict: 各検定の結果を格納した辞書。
    """
    return_value = 0
    n_samples = len(tile_sequence)

    # --- 1. ランズ・テスト (並びの偏り) ---
    # 中央値より上か下かで系列を2値化 (1 or 0)
    median_val = np.median(tile_sequence)
    binary_sequence = [1 if x > median_val else 0 for x in tile_sequence]
    
    # ランの数を数える
    runs = 1
    for i in range(1, len(binary_sequence)):
        if binary_sequence[i] != binary_sequence[i-1]:
            runs += 1
            
    n1 = sum(binary_sequence)
    n2 = n_samples - n1
    
    # Z検定でp値を計算
    if n1 == 0 or n2 == 0:
        z_score, p_runs = (0, 1.0) # 全て同じ値の場合はランダムでないが検定不能
    else:
        expected_runs = 2 * n1 * n2 / (n1 + n2) + 1
        variance_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2)**2 * (n1 + n2 - 1))
        std_dev_runs = np.sqrt(variance_runs)
        
        z_score = (runs - expected_runs) / std_dev_runs
        # 両側検定のp値
        p_runs = stats.norm.sf(abs(z_score)) * 2

    if p_runs < alpha:
        if analyze_str == "csv":
            print(f"局index:{row_index} プレイヤー:{player} 検定結果:{p_runs} 対象リスト:{[to_str(t) for t in(tile_sequence)]}")

        return_value = 1

    return return_value

# ==============================================================================
# 配牌とツモリストの生成
# ==============================================================================

def analyze_hand_drawn_tile(row_index,wall_ints,analyze_str):
    """
    シャッフルされた内部表現の牌山 (intのリスト) から、
    各プレイヤーの初期手牌とツモ牌リストを生成し、
    分析関数が期待する「文字列リスト」形式で返す。
    """
    analyze_true_count = 0
    # --- 配牌とツモリストの生成 ---
    # players辞書は内部で使用するが、最終的に分析用の全牌リストを返す
    players = {i: {'name': name} for i, name in enumerate(['東家', '南家', '西家', '北家'])}

    current_wall = list(wall_ints) # 入力が整数のリスト wall_ints であることを想定

    # 各プレイヤーの初期手牌とツモ牌を結合したリスト (分析用)
    all_player_tiles_int = []

    # 配牌
    for i in range(4):
        players[i]['hand'] = []
        players[i]['hand_draws'] = []

    # 4枚ずつ3回
    for _ in range(3):
        for i in range(4):
            if not current_wall:
                 print("警告: 配牌中に牌山が不足しました。")
                 break # 牌山がなくなったら終了
            drawn_tiles = current_wall[:4]
            players[i]['hand'].extend(drawn_tiles)
            players[i]['hand_draws'].extend(drawn_tiles)
            current_wall = current_wall[4:]
        if not current_wall and _ < 2: # 3回目の4枚配り切る前に牌山が尽きた場合
             break

    # 1枚ずつ
    if current_wall: # 牌山が残っている場合のみ
        for i in range(4):
             if not current_wall:
                  break # 牌山がなくなったら終了
             drawn_tile = current_wall.pop(0)
             players[i]['hand'].append(drawn_tile)
             players[i]['hand_draws'].append(drawn_tile)

    for _ in range(18): # 18巡分
        for i in range(4):
            if not current_wall: break
            players[i]['hand_draws'].append(current_wall.pop(0))
    ii = 0
    for i in range(4):
         player = players[i]
         all_player_tiles_int = player['hand_draws']
         ii = ii + verify_sequence_randomness(row_index,all_player_tiles_int, analyze_str, player['name'])

    if ii > 0:
        analyze_true_count = 1

    return analyze_true_count

# ==============================================================================
# 実行
# ==============================================================================
def read_csv():
    # --- CSVファイルから牌山を読み込み、なければ生成 ---
    tile_list_str = ""
    try:
        df = pd.read_csv('game_records.csv')
        # CSVのtile_list列をPythonのリストに変換
        tile_list_str = df['tile_List'].iloc[0]
        if tile_list_str == "":
          return ""
    except FileNotFoundError:
        tile_list_str = ""
    
    return tile_list_str

# ==============================================================================# 実行
# ==============================================================================
if __name__ == '__main__':
     num_iterations = 100000 # 例として1000回に設定
     analyze_str = ""

     total_ng_counts = defaultdict(int)
     total_iterations = 0 # 成功した分析回数を記録
     row_index = 0
     wall_list = read_csv()

     if wall_list == "":
        analyze_str = "random"
        for i in range(num_iterations):
           row_index = row_index + 1
           wall_ints = generate_mahjong_wall()

           if wall_ints is None:
               print(f"Iteration {i + 1}: エラー: 牌山の生成に失敗しました。スキップします。")
               continue # 次の繰り返しに進む
           total_iterations = total_iterations + analyze_hand_drawn_tile(row_index,wall_ints,analyze_str)


     else:
         analyze_str = "csv"
         df = pd.read_csv('game_records.csv')
         num_iterations = 0
         for index, row in df.iterrows():
             row_index = row_index + 1
             tile_list_str = row['tile_List']          
             wall_str = parse_tile_list_string(tile_list_str)
             #print(f"{wall_str}")
             wall = [to_int(t) for t in wall_str if t]
             total_iterations = total_iterations + analyze_hand_drawn_tile(row_index,wall,analyze_str)
             num_iterations = num_iterations + 1

# ループ完了後の処理（必要に応じて）
print(f"\nAnalysis completed for {total_iterations} / {num_iterations} iterations.")
print(f"{analyze_str}")