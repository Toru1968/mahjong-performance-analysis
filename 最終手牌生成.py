import json
import pandas as pd
from collections import Counter
import re
from typing import Dict, List, Tuple
import csv

try:
    from mahjong.shanten import Shanten
    from mahjong.tile import TilesConverter
    from mahjong.hand_calculating.hand import HandCalculator
    from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
    from mahjong.meld import Meld
except Exception as e:
    print(f"インポートエラー: {e}")
    exit()
import re
import numpy as np
from typing import List

# from mahjong.shanten import Shanten
# from mahjong.tile import TilesConverter
# 上記2行は、実際の環境に合わせて有効にしてください。

# ==============================================================================
# 1. ユーティリティ関数
# ==============================================================================

# --- 牌のコードと文字列表現を相互変換するためのマッピング ---
# TILES_MAP: 整数コードから文字列表現（例: 11 -> '1m'）へ変換します。赤ドラは '0' を使って表現します (例: 51 -> '0m')。
# (元のコードは構文エラーがあったため、辞書を結合する形式に修正しました)
TILES_MAP = (
    {11+i: f'{i+1}m' for i in range(9)} |
    {21+i: f'{i+1}p' for i in range(9)} |
    {31+i: f'{i+1}s' for i in range(9)} |
    {41+i: f'{i+1}z' for i in range(7)} |
    {51: '0m', 52: '0p', 53: '0s'}
)
# INV_TILES_MAP: 文字列表現から整数コードへ逆変換します。
INV_TILES_MAP = {v: k for k, v in TILES_MAP.items()}

def parse_kyoku_index(kyoku_index: str) -> tuple[str, int, int]:
    """
    "東1-0" のような局・本場を示す文字列を解析し、風、局、本場に分解して返す。
    (例: "東1-0" -> ("東", 1, 0))
    """
    wind = kyoku_index[0]              # '東','南','西','北'
    rest = kyoku_index[1:]             # "1-0"
    kyoku_str, honba_str = rest.split('-')
    return wind, int(kyoku_str), int(honba_str)


def to_str(tile_int: int) -> str:
    """牌の整数コードを、人間が読める文字列表現に変換する。"""
    if isinstance(tile_int, str):
        return tile_int
    return TILES_MAP.get(tile_int, f'INVALID({tile_int})')

def to_int(tile_str: str) -> int:
    """牌の文字列表現を、ソートや内部処理で使うための整数コードに変換する。"""
    # '0m' のような赤ドラ表記も正しく整数コードに変換する。
    # 不正な入力の場合は、ソートで末尾に来るよう大きな値(999)を返す。
    if not isinstance(tile_str, str) or len(tile_str) < 2:
        return 999
    
    num_char = tile_str[0]
    suit = tile_str[1]
    
    return INV_TILES_MAP.get(tile_str, 999)

def parse_tile_code(action: any) -> int:
    """
    複雑な文字列（ログなど）から正規表現を使って牌の2桁の整数コードを抽出する。
    (例: "Chi(12,13,14)" -> 12)
    """
    text = str(action)
    digit_match = re.search(r'\d+', text)
    if digit_match:
        # 抽出した数字部分から、さらに数字のみを取り出す
        tile_code_str = ''.join(c for c in digit_match.group(0) if c.isdigit())
        if len(tile_code_str) >= 2:
            return int(tile_code_str[:2])
    return action

def parse_tile_list_string(tile_list_str: str) -> List[str]:
    """
    "1m2m3p0s" のように連結された牌の文字列を、['1m', '2m', '3p', '0s'] のようなリストに分割する。
    """
    if not isinstance(tile_list_str, str) or not tile_list_str:
        return []
    
    tiles, i = [], 0
    while i < len(tile_list_str):
        # 2文字で1つの牌を構成するかチェック
        if i + 1 < len(tile_list_str):
            num_char, suit_char = tile_list_str[i], tile_list_str[i+1]
            if (num_char.isdigit() or num_char == '0') and suit_char in ['m', 'p', 's', 'z']:
                tiles.append(num_char + suit_char)
                i += 2
                continue
        # 牌として解釈できない場合は1文字進む
        i += 1
    return tiles

def get_dora_tiles(dorahyo_list: list[int]) -> list[int]:
    """
    ドラ表示牌のリストを受け取り、それに対応するドラ牌のリストを返す。
    (例: [11] (1m) -> [12] (2m))
    (元のコードはバグがあったため修正しました)
    """
    if not dorahyo_list:
        return []

    dora_list = []
    for tile in dorahyo_list:
        dora_tile = None
        # 数牌 (マンズ:11-19, ピンズ:21-29, ソーズ:31-39)
        if 11 <= tile <= 39 and (tile % 10 != 0):
            # 9の次は1
            if tile % 10 == 9:
                dora_tile = tile - 8
            else:
                dora_tile = tile + 1
        # 風牌 (東南西北: 41-44)
        elif 41 <= tile <= 44:
            # 北の次は東
            if tile == 44:
                dora_tile = 41
            else:
                dora_tile = tile + 1
        # 三元牌 (白発中: 45-47)
        elif 45 <= tile <= 47:
            # 中の次は白
            if tile == 47:
                dora_tile = 45
            else:
                dora_tile = tile + 1
        
        dora_list.append(dora_tile)
            
    return dora_list

def _decode_pai(code: int) -> str:
    """
    整数コードを文字列表現に変換する。to_strの別実装。
    (例: 11 -> '1m', 51 -> '0m')
    """
    if not isinstance(code, int): return str(code)
    # 赤ドラ
    if code == 51: return '0m'
    if code == 52: return '0p'
    if code == 53: return '0s'
    # 通常牌
    suit, num = code // 10, code % 10
    if suit == 1: return f"{num}m"
    if suit == 2: return f"{num}p"
    if suit == 3: return f"{num}s"
    if suit == 4: return f"{num}z"
    return f'{code}'

def hand_strings_to_34_array(hand_strings: list[str]) -> np.ndarray:
    """
    ['1m', '2p'] のような牌の文字列リストを、シャンテン計算などで使われる34種の牌に対応したNumPy配列に変換する。
    """
    man, pin, sou, honors = [], [], [], []
    for tile in hand_strings:
        if not isinstance(tile, str) or len(tile) != 2:
            continue
        num, suit = tile[0], tile[1]
        # 赤ドラ('0')は通常の'5'として扱う
        if num == '0':
            num = '5'
        
        if suit == 'm': man.append(num)
        elif suit == 'p': pin.append(num)
        elif suit == 's': sou.append(num)
        elif suit == 'z': honors.append(num)
        
    # 外部ライブラリの `TilesConverter` を使って変換
    # return TilesConverter.string_to_34_array(man="".join(man), pin="".join(pin), sou="".join(sou), honors="".join(honors))
    # 上記の行が動作しない場合、ダミーの配列を返す
    return np.zeros(34, dtype=int)

def convert_to_tile_notation(num):
    """
    Converts a single number into its mahjong tile string representation.
    e.g., 13 -> '3m', 41 -> '1z', 53 -> '0s'
    """
    # Special handling for red dora
    if num == 51:
        return '0m'
    if num == 52:
        return '0p'
    if num == 53:
        return '0s'

    # Split into suit (tens digit) and number (units digit)
    suit_num = num // 10
    tile_num = num % 10

    # Determine the suit character
    if suit_num == 1:
        suit_char = 'm'  # Manzu
    elif suit_num == 2:
        suit_char = 'p'  # Pinzu
    elif suit_num == 3:
        suit_char = 's'  # Souzu
    elif suit_num == 4:
        suit_char = 'z'  # Jihai (Honors)
    else:
        return '??' # Unknown tile

    return f"{tile_num}{suit_char}"

def convert_hand_to_string_notation(numeric_hand):
    """
    Converts a list of numeric tiles into a list of string tile notations,
    matching the 'initial_hand' format.
    """
    return [convert_to_tile_notation(n) for n in numeric_hand]

from collections import Counter

from collections import Counter

def analyze_wait_tiles_final(
    target_player: str,
    player_hands: dict,
    player_discards: dict,
    player_melds: dict or list, # ★辞書またはリストを受け付ける
    remaining_wall: list,
    player_waits: dict
) -> dict:
    """
    指定プレイヤーの待ち牌がどこにあるかを分析する（鳴き牌リストが空[]でも対応）。
    """
    dead_wall = remaining_wall[-14:]
    live_wall = remaining_wall[:-14]
    
    waits = player_waits
    analysis_result = {}

    opponents = [p for p in player_hands if p != target_player]
    all_discards = [tile for p_discards in player_discards.values() for tile in p_discards]
        
    discards_counter = Counter(all_discards)
    dead_wall_counter = Counter(dead_wall)
    live_wall_counter = Counter(live_wall)
    
    opponent_visible_counters = {}
    for p in opponents:
        opponent_counter = Counter(player_hands.get(p, []))
        
        # --- ★ここからが修正部分 ---
        # player_meldsが辞書の場合のみ、相手の鳴き牌を取得して追加
        if isinstance(player_melds, dict):
            opponent_counter.update(player_melds.get(p, []))
        # --- 修正部分ここまで ---

        opponent_visible_counters[p] = opponent_counter

    for wait_tile in waits:
        in_opponents_visible = sum(counter[wait_tile] for counter in opponent_visible_counters.values())

        analysis_result[wait_tile] = {
            '捨て牌': discards_counter[wait_tile],
            '相手の手（鳴き含む）': in_opponents_visible,
            '王牌': dead_wall_counter[wait_tile],
            '山（ツモ可能）': live_wall_counter[wait_tile]
        }
        
    return analysis_result

def parse_melds_to_list(data_input):
    """
    文字列または文字列のリストから特定の文字を除去し、2桁の数値リストに変換する。
    引数が文字列でもリストでも対応可能。
    """
    
    # --- ここからが改良部分 ---
    input_list = []
    # 引数が文字列の場合、リストに変換してあげる
    if isinstance(data_input, str):
        input_list = [data_input]
    # 引数がリストの場合は、そのまま使う
    elif isinstance(data_input, list):
        input_list = data_input
    # --- 改良部分ここまで ---
    
    melds_list = []
    for s in input_list:
        cleaned_s = re.sub(r'[acpkm]', '', s)
        for i in range(0, len(cleaned_s), 2):
            tile_str = cleaned_s[i:i+2]
            if tile_str:
                melds_list.append(int(tile_str))
                
    return melds_list

def get_shanten_correct(hand_int_list: list[int], melds_str_list: list[str] = None) -> int:
    """
    【最終修正版】数値リスト形式の手牌と、文字列リスト形式の副露から向聴数を正しく計算する。
    """
    try:
        # --- 1. 手牌と副露牌を全て結合し、1つの手牌として扱う ---
        all_tiles_int = list(hand_int_list) # 手牌をコピー

        if melds_str_list:
            for s in melds_str_list:
                cleaned_s = re.sub(r'[cpkam]', '', s)
                for i in range(0, len(cleaned_s), 2):
                    if tile_str := cleaned_s[i:i+2]:
                        all_tiles_int.append(int(tile_str))
        
        # --- 2. 結合した手牌をライブラリが使う34種形式の「枚数リスト」に変換 ---
        tiles_34_counts = [0] * 34
        if not all_tiles_int: return 8
        
        for t in all_tiles_int:
            if 11 <= t <= 19: tiles_34_counts[t - 11] += 1
            elif 21 <= t <= 29: tiles_34_counts[t - 21 + 9] += 1
            elif 31 <= t <= 39: tiles_34_counts[t - 31 + 18] += 1
            elif 41 <= t <= 47: tiles_34_counts[t - 41 + 27] += 1

        # --- 3. 手牌全体の状態で向聴数を計算 ---
        shanten_calculator = Shanten()
        # ★★★手牌全体の枚数リストを渡すだけで、ライブラリが正しく計算してくれる★★★
        result = shanten_calculator.calculate_shanten(tiles_34_counts)
        
        return result
        
    except Exception:
        return 8

def handle_kakan(current_melds,tile_to_add: int) -> bool:
    """
    プレイヤーの鳴きリストに対して加槓の処理を行う。

    Args:
        player_melds (dict): 全プレイヤーの鳴きリストが入った辞書。
        player_name (str): 加槓を行うプレイヤーの名前。
        tile_to_add (int): 加槓する牌の数値。

    Returns:
        bool: 処理が成功した場合は True、失敗した場合は False。
    """
    tile_str = str(tile_to_add)

    # 削除対象となる可能性のあるポンのパターン
    possible_pons = [
        f'p{tile_str}{tile_str}{tile_str}',
        f'{tile_str}p{tile_str}{tile_str}',
        f'{tile_str}{tile_str}p{tile_str}'
    ]
    # 新しく追加するカンの文字列
    kan_to_add = f'k{tile_str}{tile_str}{tile_str}{tile_str}'

    for pon_pattern in possible_pons:
        if pon_pattern in current_melds:
            # ポンを削除し、カンを追加
            current_melds.remove(pon_pattern)
            current_melds.append(kan_to_add)

            return True # 処理成功
 
    return False # 処理失敗

def calculate_effective_tiles(hand_13_tiles: list) -> tuple[int, list]:
    """
    手牌の有効牌の数とその種類を計算します。
    """
    base_shanten = get_shanten_correct(hand_13_tiles)
    if base_shanten <= -1: return 0, []
    effective_tiles, count = [], 0
    all_tile_types = [11, 12, 13, 14, 15, 16, 17,
                      18, 19, 21, 22, 23, 24, 25, 
                      26, 27, 28, 29, 31, 32, 33, 34, 
                      35, 36, 37, 38, 39, 41, 42, 43, 
                      44, 45, 46, 47, 51, 52, 53]
    current_hand_counter = Counter(hand_13_tiles)
    for tile in all_tile_types:
        if current_hand_counter[tile] < 4:
            if get_shanten_correct(hand_13_tiles + [tile]) < base_shanten:
                num_available = 4 - current_hand_counter[tile]
                count += num_available
                effective_tiles.append(tile)

    return count, effective_tiles

# ==============================================================================
# 3. 実行と表示
# ==============================================================================
import re
import json
from collections import Counter
from typing import Dict, List, Any

def simulate_rounds_fixed(players_data: str, wall: List[int], kyoku_id: str):
    """
    元のアルゴリズムを変更せず、値の代入やデータ定義のバグのみを修正したバージョン。
    """
    # (extract_tile_code と parse_kyoku_index は別途定義が必要です)
    def extract_tile_code(s: Any) -> int | None:
        text = str(s)
        digit_groups = re.findall(r'\d+', text)
        for group in digit_groups:
            for i in range(0, len(group), 2):
                chunk = group[i:i+2]
                if len(chunk) == 2: return int(chunk)
        return None

    # --- 1. 初期化フェーズ (変更なし) ---
    data = json.loads(players_data.replace("'", "\""))
    log_data = data.get('log', [[]])[0]

    # --- ★ 変更点1: 局の最終結果を先に取得 ---
    end_game_info = log_data[-1]
    point_changes = end_game_info[1]
    initial_point = log_data[1]
    raw_dora = get_dora_tiles(log_data[2])

    dora = []
    for item in raw_dora:
        dora.append(_decode_pai(item))

    raw_ura_dora = get_dora_tiles(log_data[3])
    ura_dora = []
    for item in raw_ura_dora:
        ura_dora.append(_decode_pai(item))

    if not wall: return {}
    
    player_names = ["", "", "", ""]
    wind,kyoku,honba=  parse_kyoku_index(kyoku_id)
    if kyoku == 1:
        player_names[0] = "Aさん"
        player_names[1] = "Bさん"
        player_names[2] = "Cさん"
        player_names[3] = "Dさん"
    elif kyoku == 2:
        player_names[3] = "Aさん"
        player_names[0] = "Bさん"
        player_names[1] = "Cさん"
        player_names[2] = "Dさん"
    elif kyoku == 3:
        player_names[2] = "Aさん"
        player_names[3] = "Bさん"
        player_names[0] = "Cさん"
        player_names[1] = "Dさん"
    elif kyoku == 4:
        player_names[1] = "Aさん"
        player_names[2] = "Bさん"
        player_names[3] = "Cさん"
        player_names[0] = "Dさん"

    summary = {name: {
        'initial_hand': log_data[4 + i * 3][:],
        'draws': log_data[5 + i * 3][:],
        'discards': log_data[6 + i * 3][:]
    } for i, name in enumerate(player_names)}

    # --- 2. 捨て牌パート (バグ修正) ---
    wall_dummy = wall.copy()
    draws_tiles = {p: [] for p in player_names}
    discards_tiles = {p: [] for p in player_names}
    open_melds = {p: [] for p in player_names}
    open_melds_org = {p: [] for p in player_names}
    turn = 0
    # ループを安定したwhileに変更
    while wall_dummy:
        turn += 1
        if turn > 60: break
        action_taken_this_cycle = False
        
        # プレイヤー検索ループ
        kan_str = []
        ponchikan = 0
        for p in player_names:
            #ツモリスト
            expected_draws = summary[p]['draws']
            #捨て牌リスト
            expected_discards = summary[p]['discards']

            if not expected_draws: continue

            actual_tile = wall_dummy[0]
            draw_action = expected_draws[0]

            if isinstance(draw_action, str):
                if "m" in str(draw_action):
                    open_melds_org[p].append(draw_action)
                    parsed_tiles1 = parse_melds_to_list(draw_action)
                    # ★プレイヤー名 'Aさん' をキーに指定して、そのリストに追加
                    open_melds[p].extend(parsed_tiles1)
                    expected_draws.pop(0)
                    ponchikan = extract_tile_code(draw_action)
                    draws_tiles[p].append(ponchikan)
                    # 槓子を捨てた後の捨て牌リストの先頭を取得
                    discard_action = expected_discards[0]
                    expected_discards.pop(0)
                    # 嶺上ツモをツモリストから取得
                    minkan_dummy = expected_draws.pop(0)
                    # 捨て牌が60又は0の場合の分岐
                    if discard_action == 60:
                        # 60又は0の場合はツモった牌をそのまま正規化ツモリストに追加する。
                        draws_tiles[p].append(minkan_dummy)
                        # 60又は0の場合はツモった牌をそのまま正規化捨て牌リストに追加する。
                        discards_tiles[p].append(minkan_dummy)
                        expected_discards.pop(0)
                        continue
                    elif discard_action == 0:
                        expected_discards.pop(0)
                        continue
                    else:
                        # 60又は0の場合はツモった牌をそのまま正規化ツモリストに追加する。
                        draws_tiles[p].append(minkan_dummy)
                        # 60又は0の場合はツモった牌をそのまま正規化捨て牌リストに追加する。
                        discards_tiles[p].append(discard_action)
                        continue
                else:
                    open_melds_org[p].append(draw_action)
                    parsed_tiles1 = parse_melds_to_list(draw_action)
                    # ★プレイヤー名 'Aさん' をキーに指定して、そのリストに追加
                    open_melds[p].extend(parsed_tiles1)
                    expected_draws.pop(0)
                    ponchikan = extract_tile_code(draw_action)
                    draws_tiles[p].append(ponchikan)
                    action_taken_this_cycle = True

            elif draw_action == actual_tile:
                wall_dummy.pop(0)
                expected_draws.pop(0)
                draws_tiles[p].append(actual_tile)
                if expected_discards:
                    discard_action = expected_discards[0]

                if "a" in str(discard_action) or "k" in str(discard_action):
                    if "a" in str(discard_action):
                        open_melds_org[p].append(discard_action)
                        parsed_tiles1 = parse_melds_to_list(discard_action)
                        # ★プレイヤー名 'Aさん' をキーに指定して、そのリストに追加
                        open_melds[p].extend(parsed_tiles1)
                    elif "k" in str(discard_action):
                        parsed_tiles1 = extract_tile_code(discard_action)
                        kan_discard_action = handle_kakan(open_melds_org[p],parsed_tiles1)
                        kan_str.append(parsed_tiles1)
                        # ★プレイヤー名 'Aさん' をキーに指定して、そのリストに追加
                        open_melds[p].extend(kan_str)

                    # 捨て牌リストから'XXXXXXaXX'を捨てる
                    expected_discards.pop(0) 
                    # 槓子を捨てた後の捨て牌リストの先頭を取得
                    discard_action = expected_discards[0] 
                    # 嶺上ツモをツモリストから取得
                    ankan_dummy = expected_draws.pop(0)
                    # 捨て牌が60又は0の場合の分岐
                    if discard_action == 60:
                        # 60又は0の場合はツモった牌をそのまま正規化ツモリストに追加する。
                        draws_tiles[p].append(ankan_dummy)
                        # 60又は0の場合はツモった牌をそのまま正規化捨て牌リストに追加する。
                        discards_tiles[p].append(ankan_dummy)
                        continue
                    elif discard_action == 0:
                        expected_discards.pop(0)
                        continue
                    else:
                        # 60又は0の場合はツモった牌をそのまま正規化ツモリストに追加する。
                        draws_tiles[p].append(ankan_dummy)
                        # 60又は0の場合はツモった牌をそのまま正規化捨て牌リストに追加する。
                        discards_tiles[p].append(discard_action)
                        continue

                action_taken_this_cycle = True
            else:
                draws_tiles[p].append(99)
                action_taken_this_cycle = True
                continue
            
            if not expected_discards:
                break
            else:
                discards_tile = expected_discards[0]
                if discards_tile == 60: 
                    discards_tiles[p].append(draw_action)
                    expected_discards.pop(0)

                elif ("r" in str(expected_discards[0])): 
                    ponchikan = extract_tile_code(expected_discards[0]) 
                    if ponchikan == 60:
                        discards_tiles[p].append(draw_action)
                        expected_discards.pop(0)
                    else:
                        ponchikan = extract_tile_code(expected_discards[0]) 
                        expected_discards.pop(0)
                        discards_tiles[p].append(ponchikan)

                elif discards_tile == 0: 
                    expected_discards.pop(0)
                else: 
                    discards_tiles[p].append(expected_discards.pop(0))

    # このコードを実行するには、向聴数を計算する関数が必要です。
    # 例: from mahjong_utils import calculate_shanten 

    # --- 3. 最終手牌計算 (Counterを使い正確に) ---
    fuuro_list_for_p = []
    fuuro_list_for_p_org = []
    final_hand = {}
    final_hand_dummy = {}
    waits = []
    waits_analysis_dict = {}
    waits_list_formatted = []
    for p in player_names:
        waits = []
        fuuro_list_for_p_org = open_melds_org.get(p, [])
        fuuro_list_for_p = open_melds.get(p, [])
        fuuro_counter = Counter(fuuro_list_for_p)
        # Counterを使って加算・減算
        hand_counter = Counter(summary[p]['initial_hand'])
        hand_counter_dummy = Counter(summary[p]['initial_hand'])        
        # 99(スキップ)を除いたツモ牌を加える
        valid_draws = [tile for tile in draws_tiles[p] if tile != 99 and tile is not None]
        hand_counter.update(valid_draws)
        hand_counter_dummy.update(valid_draws)
        # 捨て牌を引く
        valid_discards = [tile for tile in discards_tiles[p] if tile is not None]
        hand_counter.subtract(valid_discards)
        hand_counter_dummy.subtract(valid_discards)

        hand_counter.subtract(fuuro_counter)
        # Counterからリストに復元
        hand_list = []
        for tile, count in hand_counter.items():
            hand_list.extend([tile] * count)
        # 赤ドラを通常牌に戻してソート
        replacement_map = {51: 15, 52: 25, 53: 35}
        hand_list = [replacement_map.get(t, t) for t in hand_list]
        hand_list.sort()

        hand_list_dummy = []
        for tile_dummy, count_dummy in hand_counter_dummy.items():
            hand_list_dummy.extend([tile_dummy] * count_dummy)
        # 赤ドラを通常牌に戻してソート
        replacement_map = {51: 15, 52: 25, 53: 35}
        hand_list_dummy = [replacement_map.get(t, t) for t in hand_list_dummy]
        hand_list_dummy.sort()

        final_hand_dummy[p] = (hand_list_dummy)
        shanten = get_shanten_correct(hand_list, fuuro_list_for_p_org)
        if shanten == 0:
            dummy_count,waits = calculate_effective_tiles(hand_list_dummy)
            waits_analysis_dict = analyze_wait_tiles_final(p,final_hand_dummy,discards_tiles,fuuro_list_for_p,wall_dummy,waits)
            waits_str_list = []
            for tile, locations in waits_analysis_dict.items():
                s = locations['捨て牌']
                o = locations['相手の手（鳴き含む）']
                d = locations['王牌']
                w = locations['山（ツモ可能）']
                waits_str_list.append(f"{tile}(s:{s},o:{o},d:{d},w:{w})")
                
            waits_list_formatted = "|".join(waits_str_list)

        # 元の出力形式に合わせる
        final_hand[p] = {'final_hand': convert_hand_to_string_notation(hand_list),
                         'fuuro_list': convert_hand_to_string_notation(fuuro_list_for_p),
                         'final_shanten': shanten,
                         'waits_list': waits_list_formatted
                         }

    if kyoku == 1:
        final_hand["東"] = final_hand["Aさん"]  # 新しいキーに値を移す
        del final_hand["Aさん"]                # 古いキーを削除
        final_hand["南"] = final_hand["Bさん"]  # 新しいキーに値を移す
        del final_hand["Bさん"]                # 古いキーを削除
        final_hand["西"] = final_hand["Cさん"]  # 新しいキーに値を移す
        del final_hand["Cさん"]                # 古いキーを削除
        final_hand["北"] = final_hand["Dさん"]  # 新しいキーに値を移す
        del final_hand["Dさん"]                # 古いキーを削除
        final_hand["Aさん"] = final_hand["東"]  # 新しいキーに値を移す
        del final_hand["東"]                # 古いキーを削除
        final_hand["Bさん"] = final_hand["南"]  # 新しいキーに値を移す
        del final_hand["南"]                # 古いキーを削除
        final_hand["Cさん"] = final_hand["西"]  # 新しいキーに値を移す
        del final_hand["西"]                # 古いキーを削除
        final_hand["Dさん"] = final_hand["北"]  # 新しいキーに値を移す
        del final_hand["北"]                # 古いキーを削除    elif kyoku == 2:
    if kyoku == 2:
        final_hand["北"] = final_hand["Aさん"]  # 新しいキーに値を移す
        del final_hand["Aさん"]               # 古いキーを削除
        final_hand["東"] = final_hand["Bさん"]  # 新しいキーに値を移す
        del final_hand["Bさん"]                # 古いキーを削除
        final_hand["南"] = final_hand["Cさん"]  # 新しいキーに値を移す
        del final_hand["Cさん"]                # 古いキーを削除
        final_hand["西"] = final_hand["Dさん"]  # 新しいキーに値を移す
        del final_hand["Dさん"]                # 古いキーを削除
        final_hand["Aさん"] = final_hand["東"]  # 新しいキーに値を移す
        del final_hand["東"]                # 古いキーを削除
        final_hand["Bさん"] = final_hand["南"]  # 新しいキーに値を移す
        del final_hand["南"]                # 古いキーを削除
        final_hand["Cさん"] = final_hand["西"]  # 新しいキーに値を移す
        del final_hand["西"]                # 古いキーを削除
        final_hand["Dさん"] = final_hand["北"]  # 新しいキーに値を移す
        del final_hand["北"]                # 古いキーを削除
    elif kyoku == 3:
        final_hand["西"] = final_hand["Aさん"]  # 新しいキーに値を移す
        del final_hand["Aさん"]               # 古いキーを削除
        final_hand["北"] = final_hand["Bさん"]  # 新しいキーに値を移す
        del final_hand["Bさん"]                # 古いキーを削除
        final_hand["東"] = final_hand["Cさん"]  # 新しいキーに値を移す
        del final_hand["Cさん"]                # 古いキーを削除
        final_hand["南"] = final_hand["Dさん"]  # 新しいキーに値を移す
        del final_hand["Dさん"]                # 古いキーを削除
        final_hand["Aさん"] = final_hand["東"]  # 新しいキーに値を移す
        del final_hand["東"]                # 古いキーを削除
        final_hand["Bさん"] = final_hand["南"]  # 新しいキーに値を移す
        del final_hand["南"]                # 古いキーを削除
        final_hand["Cさん"] = final_hand["西"]  # 新しいキーに値を移す
        del final_hand["西"]                # 古いキーを削除
        final_hand["Dさん"] = final_hand["北"]  # 新しいキーに値を移す
        del final_hand["北"]                # 古いキーを削除
    elif kyoku == 4:
        final_hand["北"] = final_hand["Aさん"]  # 新しいキーに値を移す
        del final_hand["Aさん"]               # 古いキーを削除
        final_hand["東"] = final_hand["Bさん"]  # 新しいキーに値を移す
        del final_hand["Bさん"]                # 古いキーを削除
        final_hand["南"] = final_hand["Cさん"]  # 新しいキーに値を移す
        del final_hand["Cさん"]                # 古いキーを削除
        final_hand["西"] = final_hand["Dさん"]  # 新しいキーに値を移す
        del final_hand["Dさん"]                # 古いキーを削除
        final_hand["Aさん"] = final_hand["東"]  # 新しいキーに値を移す
        del final_hand["東"]                # 古いキーを削除
        final_hand["Bさん"] = final_hand["南"]  # 新しいキーに値を移す
        del final_hand["南"]                # 古いキーを削除
        final_hand["Cさん"] = final_hand["西"]  # 新しいキーに値を移す
        del final_hand["西"]                # 古いキーを削除
        final_hand["Dさん"] = final_hand["北"]  # 新しいキーに値を移す
        del final_hand["北"]                # 古いキーを削除

    initial_shanten = 0
    player_names = ["Aさん", "Bさん", "Cさん", "Dさん"]
    for i in range(4):
        player_name = player_names[i]

        # プレイヤー個別のデータを取得・処理
        raw_initial_hand = log_data[4 + i * 3]
        raw_draws_and_naki = log_data[5 + i * 3]
        raw_actions = log_data[6 + i * 3]

        initial_shanten = get_shanten_correct(raw_initial_hand)

        draws = []
        naki_count = 0
        for item in raw_draws_and_naki:
            if isinstance(item, int):
                draws.append(_decode_pai(item))
            elif isinstance(item, str):
                draws.append(_decode_pai(item))
                naki_count += 1
          
        discards = []
        riichi_turn = 0
        turn_counter = 0
        for action in raw_actions:
            if isinstance(action, int):
                turn_counter += 1
                discards.append(_decode_pai(action))
            elif isinstance(action, str) and action.startswith('r'):
                if riichi_turn == 0:
                    riichi_turn = turn_counter + 1
             
        tsumogiri_count = raw_actions.count(60)
    
                # カウント対象
        targets = dora + ura_dora + ['0m', '0p', '0s']

        hand = raw_initial_hand + draws
        # 各牌の枚数をカウント
        counter = Counter(hand)
   
        # ドラ・裏ドラ・赤ドラの合計
        dora_count = sum(counter[tile] for tile in dora)
        ura_dora_count = sum(counter[tile] for tile in ura_dora)
        red_dora_count = counter['0m'] + counter['0p'] + counter['0s']

            # --- ★ 変更点3: 全てのデータをまとめて登録 ---
        summary[player_name] = {
                'kyoku_id': kyoku_id,
                'initial_point':initial_point[i],
                'dora':dora,
                'ura_dora':ura_dora,
                'initial_hand':convert_hand_to_string_notation(raw_initial_hand),
                'initial_shanten': initial_shanten,
                'draws': draws,
                'discards': discards,
                'riichi_turn': -1 if riichi_turn == 0 else riichi_turn,
                'naki_count': -1 if naki_count == 0 else naki_count,
                'tsumogiri_count': tsumogiri_count,
                'dora_count': dora_count,
                'ura_dora_count': ura_dora_count,
                'red_dora_count': red_dora_count,
                'points_gained': point_changes[i], # 先に取得した結果から対応する値を取得
                'final_hand': final_hand
            }

    return summary

# ==============================================================================
# 3. 実行と表示
# ==============================================================================
if __name__ == '__main__':
    final_hands = []
    game_info = []
    all_game_info = []

    df = pd.read_csv('game_records.csv')
    for index, row in df.iterrows():
        tile_list_str = row['tile_List']
        wall_str = parse_tile_list_string(tile_list_str)
        gamelog_json = row['gamelog_json']
        wall_ints = [to_int(t) for t in wall_str if t]
        # 完全な牌山(136枚)の先頭から、配牌分(52枚)を単純に取り除く
        remaining_wall = wall_ints[52:]
        kyoku_id = row['game_ID']
        game_info = simulate_rounds_fixed(gamelog_json, remaining_wall, kyoku_id)
        all_game_info.append(game_info)
    
headers = [
    'kyoku_id', 'player', 'initial_point', 'dora', 'ura_dora', 'initial_hand', 'initial_shanten',
    'draws', 'discards', 'riichi_turn', 'naki_count', 'tsumogiri_count', 
    'dora_count', 'ura_dora_count', 'red_dora_count', 'points_gained', 'final_hand', 'fuuro_list','final_shanten','waits_list'
]

# データを格納するリスト
rows_to_write = []

# 元のデータをループして、CSVの各行を作成
for round_data in all_game_info:
    for player, player_data in round_data.items():
        final_hand_data = player_data.get('final_hand', {})
        row = {
            'kyoku_id': player_data.get('kyoku_id'),
            'player': player,
            'initial_point': player_data.get('initial_point'),
            'dora': ','.join(map(str, player_data.get('dora', []))),
            'ura_dora': ','.join(map(str, player_data.get('ura_dora', []))),
            'initial_hand': ','.join(map(str, player_data.get('initial_hand', []))),
            'initial_shanten': player_data.get('initial_shanten'),
            'draws': ','.join(map(str, player_data.get('draws', []))),
            'discards': ','.join(map(str, player_data.get('discards', []))),
            'riichi_turn': player_data.get('riichi_turn'),
            'naki_count': player_data.get('naki_count'),
            'tsumogiri_count': player_data.get('tsumogiri_count'),
            'dora_count': player_data.get('dora_count'),
            'ura_dora_count': player_data.get('ura_dora_count'),
            'red_dora_count': player_data.get('red_dora_count'),
            'points_gained': player_data.get('points_gained'),
            'final_hand': ','.join(map(str, final_hand_data.get(player, {}).get('final_hand', []))),
            'fuuro_list': ','.join(map(str, final_hand_data.get(player, {}).get('fuuro_list', []))),
            'final_shanten': str(final_hand_data.get(player, {}).get('final_shanten', 8)),
            'waits_list': final_hand_data.get(player, {}).get('waits_list', '') # デフォルトは空文字に
              }
        rows_to_write.append(row)

# CSVファイルへの書き込み
output_filename = 'mahjong_log.csv'
try:
    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows_to_write)
    print(f"データは正常に '{output_filename}' として保存されました。")
except IOError:
    print(f"エラー: ファイル '{output_filename}' に書き込めませんでした。")
