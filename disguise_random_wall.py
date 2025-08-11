import pandas as pd
import sys
import numpy as np
from tqdm import tqdm
from collections import Counter
from typing import List
import random

# ==============================================================================
# --- 関数 ---
# ==============================================================================
initial_players = ['A', 'B', 'C', 'D']
PLAYERS = ['A','B','C','D']
parent_idx = 0
prev_kyoku = None

TILES_MAP = {
    **{f'{i}m': 10+i for i in range(1, 10)},
    **{f'{i}p': 20+i for i in range(1, 10)},
    **{f'{i}s': 30+i for i in range(1, 10)},
    **{f'{i}z': 40+i for i in range(1, 8)}, # 1z-7z
    '0m': 51, # 赤ドラ5m
    '0p': 52, # 赤ドラ5p
    '0s': 53  # 赤ドラ5s
}
INV_TILES_MAP = {v: k for k, v in TILES_MAP.items()}

def to_str(tile_int):
    return INV_TILES_MAP.get(tile_int, '?')

def to_int(tile_str):
    return TILES_MAP.get(tile_str, 0)

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

def convert_tile_counts_34(tiles):
    red_tile_map = {'0m': '5m', '0p': '5p', '0s': '5s'}
    normalized_tiles = [red_tile_map.get(tile, tile) for tile in tiles]
    tile_order = [
        '1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m',
        '1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p',
        '1s', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s',
        '1z', '2z', '3z', '4z', '5z', '6z', '7z'
    ]
    count_map = Counter(normalized_tiles)
    tile_counts_34 = [count_map.get(tile, 0) for tile in tile_order]
    return tile_counts_34

import csv

def convert_to_string_representation(tile_list):
    """
    数値の牌リストを、"1m"や"5z"のような文字列形式に変換する。
    例: [11, 45, 50] -> "1m5z0m"
    
    ルール:
    - 10の位が種類 (1:m, 2:p, 3:s, 4:z), 1の位が数字 (1-9)
    - 特殊牌: 50は赤5m("0m"), 51は赤5p("0p"), 52は赤5s("0s")として扱う。
    """
    suit_map = {1: 'm', 2: 'p', 3: 's', 4: 'z'}
    result_string = ""
    if not tile_list:
        return ""
        
    for tile_code in tile_list:
        # 赤ドラの特殊ケース
        if tile_code == 51:
            result_string += "0m"
        elif tile_code == 52:
            result_string += "0p"
        elif tile_code == 53:
            result_string += "0s"
        else:
            suit_digit = tile_code // 10
            num_digit = tile_code % 10
            if suit_digit in suit_map and 1 <= num_digit <= 9:
                suit_char = suit_map[suit_digit]
                result_string += f"{num_digit}{suit_char}"
            else:
                print(f"Warning: Invalid tile code: {tile_code}")
                result_string += f"?" # 不明なコードのフォールバック
    return result_string

# ==============================================================================
# --- 牌山生成ロジック ---
# ==============================================================================
def reconstruct_initial_walls(final_players_data, final_dead_wall):
    """
    ゲーム完了後のデータから、ゲーム開始直後の各家のシャッフル済み牌山を復元する。
    """
    # --- ステップ1: ツモの逆再生 (ツモ山部分の復元) ---
    reconstructed_draw_wall = []
    num_draws_per_player = [len(p['draws']) for p in final_players_data]
    max_draws = max(num_draws_per_player) if num_draws_per_player else 0

    for i in range(max_draws):
        for p_idx in range(4):
            if i < num_draws_per_player[p_idx]:
                reconstructed_draw_wall.append(final_players_data[p_idx]['draws'][i])
    
    # --- ステップ2: 配牌の逆再生 (配牌部分の復元) ---
    reconstructed_deal_wall = []
    
    # 4枚ずつの部分を復元
    for chunk in range(3): # 0, 1, 2
        start_idx = chunk * 4
        end_idx = start_idx + 4
        for p_idx in range(4):
            reconstructed_deal_wall.extend(final_players_data[p_idx]['hand'][start_idx:end_idx])

    # 1枚ずつの部分を復元
    for p_idx in range(4):
        reconstructed_deal_wall.append(final_players_data[p_idx]['hand'][12])
        
    # --- ステップ3: live_wall (122枚の牌山) の完全復元 ---
    reconstructed_live_wall = reconstructed_deal_wall + reconstructed_draw_wall + final_dead_wall
    
    return reconstructed_live_wall

# ==============================================================================
# --- 配牌とツモリストの生成 ---
# ==============================================================================
def random_hand_tile(wall_ints):
    """
    シャッフルされた内部表現の牌山 (intのリスト) から、
    各プレイヤーの初期手牌とツモ牌リストを生成し、
    分析関数が期待する「文字列リスト」形式で返す。
    """
    hand_results = []
    # --- 配牌とツモリストの生成 ---
    # players辞書は内部で使用するが、最終的に分析用の全牌リストを返す
    players = {i: {'name': name, 'seat': f'Seat {i+1}'} for i, name in enumerate(['東', '南', '西', '北'])}

    current_wall = list(wall_ints) # 入力が整数のリスト wall_ints であることを想定

    # 各プレイヤーの初期手牌とツモ牌を結合したリスト (分析用)
    player_tiles_int = []
    player_draw_int = []

    # 配牌
    for i in range(4):
        players[i]['hand'] = []
        players[i]['hand_draws'] = []
        players[i]['draws'] = []

    # 4枚ずつ3回
    for _ in range(3):
        for i in range(4):
            if len(current_wall) < 4:
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

    # Handle red fives after initial deal
    for i in range(4):
       players[i]['hand_draws'] = [15 if x == 51 else x for x in players[i]['hand_draws']]
       players[i]['hand_draws'] = [25 if x == 52 else x for x in players[i]['hand_draws']]
       players[i]['hand_draws'] = [35 if x == 53 else x for x in players[i]['hand_draws']]
       players[i]['hand'] = [15 if x == 51 else x for x in players[i]['hand']]
       players[i]['hand'] = [25 if x == 52 else x for x in players[i]['hand']]
       players[i]['hand'] = [35 if x == 53 else x for x in players[i]['hand']]


    for draw_index in range(18): # 18巡分
        for i in range(4):
            if not current_wall or (i == 2 and draw_index == 17): break
            drawn_tile = current_wall.pop(0)
            players[i]['hand_draws'].append(drawn_tile)
            players[i]['draws'].append(drawn_tile)
            # Handle red fives after drawing
            players[i]['hand_draws'] = [15 if x == 51 else x for x in players[i]['hand_draws']]
            players[i]['hand_draws'] = [25 if x == 52 else x for x in players[i]['hand_draws']]
            players[i]['hand_draws'] = [35 if x == 53 else x for x in players[i]['hand_draws']]
            players[i]['draws'] = [15 if x == 51 else x for x in players[i]['draws']]
            players[i]['draws'] = [25 if x == 52 else x for x in players[i]['draws']]
            players[i]['draws'] = [35 if x == 53 else x for x in players[i]['draws']]


    for i in range(4):
         player = players[i]
         player_hand_int = player['hand']
         player_draws_int = player['draws']
         random.shuffle(player_hand_int)
         random.shuffle(player_draws_int)

         hand_results.append({
             "seat"      :          player['seat'], # Changed 'name' to 'seat' to match the logic in mahjong_verification_logic
             "hand"      :          player_hand_int,
             "draws"     :          player_draws_int
         })

    random.shuffle(current_wall)

    return current_wall,hand_results

# ==============================================================================
# --- メイン処理 ---
# ==============================================================================
def analyze_report():
    """CSVファイルを読み込み、向聴数の進捗を分析する"""



    string_representation = convert_to_string_representation(all_results)

    return string_representation
    
# ==============================================================================
# --- 実行部 ---
# ==============================================================================
if __name__ == "__main__":
    csv_filename = "mahjong_output.csv"

    try:
        # csvモジュールを使い、全てのフィールドをダブルクォートで囲む設定で書き出す
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)

            all_results = []
            player_hand = []

            df = pd.read_csv('_report.csv',
            header=None,
            names=['value'],
            low_memory=False
            )
            pbar = tqdm(df.iterrows(),
                desc="処理中",           # 前につく説明文字列
                unit="局",               # 単位表示（it, items, bytes…）
                ncols=80,                # バーの幅（文字数）
                ascii=True
                )
            for index, row in pbar:
                index = index + 1
                tile_list_str = row['value']
                wall_str_list = parse_tile_list_string(tile_list_str)
                wall_ints = [to_int(t) for t in wall_str_list]

                wall,player_hand = random_hand_tile(wall_ints)
                output_for_game = reconstruct_initial_walls(player_hand,wall)
                string_representation = convert_to_string_representation(output_for_game)
                writer.writerow([string_representation])

        print(f"\n結果が \"{csv_filename}\" に保存されました。")
    except IOError as e:
        print(f"\nエラー: ファイルの書き込みに失敗しました - {e}")
