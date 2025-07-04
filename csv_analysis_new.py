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
from tqdm import tqdm
from scipy.stats import binomtest

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

def convert_tile_counts_34(tiles):
    # 赤ドラを通常の5牌として扱う変換ルール
    red_tile_map = {'0m': '5m', '0p': '5p', '0s': '5s'}

    # 赤ドラの置き換え処理
    normalized_tiles = [red_tile_map.get(tile, tile) for tile in tiles]

    # 34種類の牌の順序定義（標準麻雀の牌種）
    tile_order = [
        '1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m',
        '1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p',
        '1s', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s',
        '1z', '2z', '3z', '4z', '5z', '6z', '7z'
    ]

    # カウント処理（欠けてる牌種も0で埋める）
    count_map = Counter(normalized_tiles)
    tile_counts_34 = [count_map.get(tile, 0) for tile in tile_order]

    return tile_counts_34

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
FULL_DECK_INT_34 = sum([[i] * 4 for i in range(34)], [])

def analyze_composition_monte_carlo(observed_counts: list, total_tiles: int, n_sims: int = 1000):
    """
    モンテカルロ法を用いて、観測された牌構成の偏り（経験的p値）を評価する。

    Args:
        observed_counts (list): 実際に観測された34種の牌のカウントリスト。
        total_tiles (int): 観測された牌の合計枚数 (例: 31)。
        n_sims (int): シミュレーションの繰り返し回数。

    Returns:
        dict: {'p_empirical': 経験的p値}
    """
    if sum(observed_counts) != total_tiles:
        return {'p_empirical': None, 'error': '観測カウントと合計牌数が一致しません'}

    # 1. 期待度数の計算
    # 完全にランダムな場合、各牌の期待枚数を計算
    expected_counts = np.full(34, total_tiles * (4 / 136))
    
    # 2. 実際に観測されたデータのカイ二乗値（χ²値）を計算
    # この値が、観測データの「偏りの大きさ」を示す
    try:
        chi2_observed, _ = chisquare(f_obs=observed_counts, f_exp=expected_counts)
    except ValueError:
        return {'p_empirical': None, 'error': 'カイ二乗値の計算中にエラーが発生しました'}

    # 3. モンテカルロシミュレーションを実行
    simulated_chi2_values = []
    for _ in range(n_sims):
        # ランダムなサンプルをデッキから抽出
        sample_indices = random.sample(range(136), total_tiles)
        sample_tiles = [FULL_DECK_INT_34[i] for i in sample_indices]
        
        # サンプルの観測度数を計算
        simulated_obs = [0] * 34
        for tile_idx in sample_tiles:
            simulated_obs[tile_idx] += 1
        
        # シミュレーションサンプルのカイ二乗値を計算
        chi2_sim, _ = chisquare(f_obs=simulated_obs, f_exp=expected_counts)
        simulated_chi2_values.append(chi2_sim)
        
    # 4. 経験的p値を計算
    # 観測されたカイ二乗値が、シミュレーションで得られた多数のカイ二乗値と比べて
    # どのくらい極端な値だったのかを割合で示す
    p_empirical = np.sum(np.array(simulated_chi2_values) >= chi2_observed) / n_sims

    return {'p_empirical': p_empirical}

def calculate_cramers_v(observed_counts: list, expected_counts: list) -> float:
    """
    観測度数と期待度数からクラメールのVを算出する。
    値は0（関連なし）から1（完全な関連）の範囲。
    """
    if sum(observed_counts) == 0:
        return 0.0
        
    # カイ二乗値を計算
    chi2, _ = chisquare(f_obs=observed_counts, f_exp=expected_counts)
    n = sum(observed_counts)
    
    # 自由度 (k-1)
    dof = len(observed_counts) - 1
    if dof == 0:
        return 0.0
        
    # クラメールのVを計算
    v = np.sqrt(chi2 / (n * dof))
    return v

def analyze_conditional_entropy(player_sequence: list[int]):
    """
    プレイヤーに渡った牌のシーケンス（例：配牌13枚＋ツモ18枚）を受け取り、
    その並び順の乱雑さを条件付きエントロピーで評価します。
    """
    if len(player_sequence) < 2:
        return {'entropy_bits': None, 'error': 'Sequence too short'}

    # 連続する牌のペアの出現回数をカウント
    pair_counts = Counter(zip(player_sequence[:-1], player_sequence[1:]))
    # 個々の牌の出現回数をカウント
    tile_counts = Counter(player_sequence[:-1])
    
    num_pairs = len(player_sequence) - 1
    conditional_entropy = 0.0

    # H(Y|X) = -Σ p(x,y) * log2(p(y|x))
    for pair, count in pair_counts.items():
        x, y = pair
        p_xy = count / num_pairs
        p_y_given_x = count / tile_counts[x]
        conditional_entropy -= p_xy * np.log2(p_y_given_x)
        
    return {'entropy_bits': conditional_entropy}

def analyze_markov_chain(player_sequence: list[int]):
    """
    プレイヤーに渡った牌のシーケンス（例：配牌13枚＋ツモ18枚）を受け取り、
    牌種（萬筒索字）の遷移に統計的な偏り（癖）がないか検定します。
    """
    if len(player_sequence) < 10: # 十分な長さがない場合はスキップ
        return {'p_value': None, 'error': 'Sequence too short for Markov analysis'}

    def tile_to_state(tile_int):
        # 牌を4つの状態（0:萬, 1:筒, 2:索, 3:字）に分類
        if 10 < tile_int < 20 or tile_int == 51: return 0
        if 20 < tile_int < 30 or tile_int == 52: return 1
        if 30 < tile_int < 40 or tile_int == 53: return 2
        if 40 < tile_int < 50: return 3
        return -1 # エラー

    # 4x4の遷移回数行列を観測
    observed_matrix = np.zeros((4, 4), dtype=int)
    for i in range(len(player_sequence) - 1):
        state_from = tile_to_state(player_sequence[i])
        state_to = tile_to_state(player_sequence[i+1])
        if state_from != -1 and state_to != -1:
            observed_matrix[state_from, state_to] += 1
            
    # カイ二乗独立性検定を実行
    # 帰無仮説：行（遷移元）と列（遷移先）は独立している（＝前の牌と次の牌に関連性はない）
    # p値が低い場合、帰無仮説は棄却され、遷移に依存性（癖）があると判断
    try:
        chi2, p_value, dof, expected = stats.chi2_contingency(observed_matrix)
        return {'p_value': p_value, 'observed_matrix': observed_matrix.tolist()}
    except ValueError:
        # データが少なすぎて検定できない場合
        return {'p_value': None, 'error': 'Could not perform chi2 contingency test'}

def analyze_effective_tiles(haipai: list[int], draw_sequence: list[int], remaining_wall: list[int]):
    """
    プレイヤーの配牌（13枚）とツモ牌のシーケンス、そして残りの牌山を受け取り、
    手牌にとって価値のある「有効牌」を偶然以上に引いていないかを評価します。
    """
    
    # --- ここから先は、高度な麻雀AIの知識を要する「手牌評価関数」の実装が必要です ---
    # 今回は、その関数の存在を前提とした「設計」を示します。
    def get_effective_tiles_for_hand(hand: list[int]):
        """
        【要実装】手牌を受け取り、手を進める有効牌のリストを返すヘルパー関数。
        - シャンテン数を計算する。
        - シャンテン数を減らす、または受け入れ枚数を増やす牌を全てリストアップする。
        - 例：[1m, 2m] があれば、有効牌として [3m] を返す。
        """
        # この部分は非常に複雑なため、今回はダミーの値を返します
        # print(f"手牌評価を実行中: {hand}") # デバッグ用
        return [13, 23, 33] # ダミーの有効牌リスト

    effective_draws_count = 0
    current_hand = list(haipai)

    for i, draw in enumerate(draw_sequence):
        # 1. 現時点の有効牌を取得
        effective_tiles = get_effective_tiles_for_hand(current_hand)
        
        # 2. ツモが有効牌だったか判定
        if draw in effective_tiles:
            effective_draws_count += 1
        
        # 3. 手牌を更新（ツモ牌を加え、何を切るかはここでは単純化）
        current_hand.append(draw)
        # 実際の打牌シミュレーションはさらに複雑なため、ここでは省略
        
    # 4. 有効牌のツモ率を計算
    total_draws = len(draw_sequence)
    effective_draw_rate = (effective_draws_count / total_draws) * 100 if total_draws > 0 else 0

    # NOTE: 本来はこの「有効牌ツモ率」が理論値からどれだけ乖離しているかを
    # 超幾何分布検定などで評価しますが、今回は率の計算までを実装とします。
    
    return {'effective_draw_rate_percent': effective_draw_rate}

def verify_odd_even_bias(wall_after_haipai: list[int], num_draws_to_check: int = 20):
    """
    配牌後の牌山を受け取り、序盤のツモ牌における奇数・偶数の偏りを二項検定で検証する。
    """
    # 序盤の指定された枚数を取得
    target_tiles = wall_after_haipai[:num_draws_to_check]

    # 数牌の数字のみを抽出 (赤ドラは5として扱う)
    numbers = []
    for tile in target_tiles:
        if 10 <= tile < 40: # 通常の数牌 (1m-9s)
            numbers.append(tile % 10)
        elif 51 <= tile <= 53: # 赤ドラ
            numbers.append(5)

    n_suited_tiles = len(numbers)
    # 検定に十分なサンプルサイズがあるか確認
    if n_suited_tiles < 10:
        return 0

    # 奇数(1,3,5,7,9)の数をカウント
    n_odds = sum(1 for num in numbers if num % 2 != 0)

    # 二項検定を実行
    # 帰無仮説：ある数牌が奇数である確率は 5/9 である
    p_expected = 5/9
    result = binomtest(k=n_odds, n=n_suited_tiles, p=p_expected, alternative='two-sided')
    p_value = result.pvalue

    return p_value

def verify_MonteCarloSimulation_randomness(obs_real: list[int]):
    # 牌の種類を定義
    tile_order = [
        '1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m',
        '1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p',
        '1s', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s',
        '1z', '2z', '3z', '4z', '5z', '6z', '7z'
    ]

    # --- 変更点 1: 検証する牌の枚数を設定 ---
    n_tiles = 31  # 13枚 + 18枚
    return_value = 0
    # シミュレーション設定
    n_sims = 10000  # シミュレーション回数
    deck = sum([[i]*4 for i in range(34)], [])  # 136枚の麻雀牌の山

    # --- 変更点 2: 期待値を新しい枚数で計算 ---
    exp_count_per_category = n_tiles * (4 / 136)
    f_exp_array = np.full(34, exp_count_per_category)

    # 観測データのカイ二乗値（χ²値）を計算
    chi2_obs, p_value_theoretical = chisquare(obs_real, f_exp=f_exp_array)

    # モンテカルロシミュレーションの開始
    chi2_sims = []
    for _ in range(n_sims):
        sample = random.sample(deck, n_tiles)
        obs = [sample.count(i) for i in range(34)]

        # シミュレーション結果のカイ二乗値を計算
        chi2, _ = chisquare(obs, f_exp=f_exp_array)
        chi2_sims.append(chi2)

    # --- 変更点 4: 最終結果の計算と表示 ---
    # シミュレーション結果の中から、観測値以上のχ²値がどれくらいの割合で出現したか（経験的p値）を計算
    p_empirical = sum(c >= chi2_obs for c in chi2_sims) / n_sims

    return p_empirical

def verify_sequence_randomness(tile_sequence: list[int]):
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

    return p_runs

# ==============================================================================
# 配牌とツモリストの生成
# ==============================================================================
def random_analyze_hand_drawn_tile(wall_ints):
    """
    シャッフルされた内部表現の牌山 (intのリスト) から、
    各プレイヤーの初期手牌とツモ牌リストを生成し、
    分析関数が期待する「文字列リスト」形式で返す。
    """
    return_value = 0
    alpha = 0.0025
    # --- 配牌とツモリストの生成 ---
    # players辞書は内部で使用するが、最終的に分析用の全牌リストを返す
    players = {i: {'name': name} for i, name in enumerate(['東家', '南家', '西家', '北家'])}

    current_wall = list(wall_ints) # 入力が整数のリスト wall_ints であることを想定

    # 各プレイヤーの初期手牌とツモ牌を結合したリスト (分析用)
    player_tiles_int = []
    player_draw_int = []
    obs_real = []

    # 配牌
    for i in range(4):
        players[i]['hand'] = []
        players[i]['hand_draws'] = []
        players[i]['draws'] = []

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

    for i in range(4):
       #print(f"1 {players[i]['hand_draws']}")
       players[i]['hand_draws'] = [15 if x == 51 else x for x in players[i]['hand_draws']]
       players[i]['hand_draws'] = [25 if x == 52 else x for x in players[i]['hand_draws']]
       players[i]['hand_draws'] = [35 if x == 53 else x for x in players[i]['hand_draws']]
       players[i]['hand_draws'].sort()
       #print(f"2 {players[i]['hand_draws']}")

    for _ in range(18): # 18巡分
        for i in range(4):
            if not current_wall: break
            drawn_tile = current_wall.pop(0)
            players[i]['hand_draws'].append(drawn_tile)
            players[i]['hand_draws'] = [15 if x == 51 else x for x in players[i]['hand_draws']]
            players[i]['hand_draws'] = [25 if x == 52 else x for x in players[i]['hand_draws']]
            players[i]['hand_draws'] = [35 if x == 53 else x for x in players[i]['hand_draws']]
  
    p_runs = 0

    for i in range(4):
         player = players[i]
         player_hand_draws_int = player['hand_draws']
         p_runs = verify_sequence_randomness(player_hand_draws_int)
         
         if p_runs < alpha:
             return_value = 1

    return return_value

def analyze_hand_drawn_tile(row_index,wall_ints,analyze_str):
    """
    シャッフルされた内部表現の牌山 (intのリスト) から、
    各プレイヤーの初期手牌とツモ牌リストを生成し、
    分析関数が期待する「文字列リスト」形式で返す。
    """
    analyzed_results = []
    # --- 配牌とツモリストの生成 ---
    # players辞書は内部で使用するが、最終的に分析用の全牌リストを返す
    players = {i: {'name': name} for i, name in enumerate(['東家', '南家', '西家', '北家'])}

    current_wall = list(wall_ints) # 入力が整数のリスト wall_ints であることを想定

    # 各プレイヤーの初期手牌とツモ牌を結合したリスト (分析用)
    player_tiles_int = []
    player_draw_int = []
    obs_real = []

    # 配牌
    for i in range(4):
        players[i]['hand'] = []
        players[i]['hand_draws'] = []
        players[i]['draws'] = []

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

    for i in range(4):
       #print(f"1 {players[i]['hand_draws']}")
       players[i]['hand_draws'] = [15 if x == 51 else x for x in players[i]['hand_draws']]
       players[i]['hand_draws'] = [25 if x == 52 else x for x in players[i]['hand_draws']]
       players[i]['hand_draws'] = [35 if x == 53 else x for x in players[i]['hand_draws']]
       players[i]['hand_draws'].sort()
       #print(f"2 {players[i]['hand_draws']}")

    for _ in range(18): # 18巡分
        for i in range(4):
            if not current_wall: break
            drawn_tile = current_wall.pop(0)
            players[i]['draws'].append(drawn_tile)
            players[i]['hand_draws'].append(drawn_tile)
            players[i]['hand_draws'] = [15 if x == 51 else x for x in players[i]['hand_draws']]
            players[i]['hand_draws'] = [25 if x == 52 else x for x in players[i]['hand_draws']]
            players[i]['hand_draws'] = [35 if x == 53 else x for x in players[i]['hand_draws']]
    
    wall_after_haipai = current_wall

    p_runs = 0
    p_empirical = 0
    p_value = 0

    for i in range(4):
         player = players[i]
         player_hand_int = player['hand']
         player_hand_draws_int = player['hand_draws']
         player_draw_int = player['draws']
         p_runs = verify_sequence_randomness(player_hand_draws_int)
         obs_real = convert_tile_counts_34([to_str(t) for t in(player_hand_draws_int)])
         p_empirical = verify_MonteCarloSimulation_randomness(obs_real)
         p_value = verify_odd_even_bias(player_draw_int)
         entropy_result = analyze_conditional_entropy(player_hand_draws_int)
         markov_result = analyze_markov_chain(player_hand_draws_int)
         obs_counts = convert_tile_counts_34([to_str(t) for t in player_hand_draws_int])
         exp_counts = [len(player_hand_draws_int) * (4 / 136)] * 34
         comp_res = analyze_composition_monte_carlo(obs_counts, len(player_hand_draws_int))
         comp_res['cramers_v'] = calculate_cramers_v(obs_counts, exp_counts)
         #effective_tile_result = analyze_effective_tiles(player_hand_int, player_draw_int, wall_after_haipai)
          
         analyzed_results.append({
             "index":            row_index,
             "seat":             player['name'],
             "p_runs":           p_runs,
             "p_empirical":      p_empirical,
             "p_value":          p_value,
             "entropy":          entropy_result,
             "markov":           markov_result,
             "cramers_v":        comp_res['cramers_v']
         })

    return analyzed_results

# ==============================================================================
#CSV存在チェック
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

# ==============================================================================
#結果表示
# ==============================================================================
def generate_summary_report(all_game_results: list, alpha: float = 0.05):
    """
    全対局の分析結果リストを受け取り、3層アプローチに基づいた総合評価レポートを出力する。
    """
    print("\n" + "="*80)
    print("【総合評価レポート】")
    print("="*80)

    if not all_game_results:
        print("分析対象のデータがありません。")
        return

    # 生データをPandas DataFrameに変換して扱いやすくする
    flat_results = [item for sublist in all_game_results for item in sublist]
    df = pd.DataFrame(flat_results)

    p_value_source_map = {
        'p_runs': 'p_runs_val',
        'p_empirical': 'p_comp_val'
    }
    for source_col, target_col in p_value_source_map.items():
        if source_col in df.columns:
            df[target_col] = pd.to_numeric(df[source_col], errors='coerce')
        else:
            df[target_col] = np.nan # 列が存在しない場合はNaNで作成

    # 辞書型の'markov'列も同様に安全に処理
    if 'markov' in df.columns:
        # .get()を使い、キーが存在しなくてもエラーにならないようにする
        df['p_markov_val'] = df['markov'].apply(
            lambda x: x.get('p_value') if isinstance(x, dict) else np.nan
        ).astype(float)
    else:
        df['p_markov_val'] = np.nan

    # ——————————————
    # FDR補正をかけて、補正後も有意だったかのフラグ列を作成
    from statsmodels.stats.multitest import multipletests

    df = apply_fdr_correction(df, alpha)

    # 1) p値列を用意
    df['p_runs_val']   = pd.to_numeric(df['p_runs'],   errors='coerce')
    df['p_markov_val'] = df['markov'].apply(lambda x: x.get('p_value') if isinstance(x, dict) else np.nan)
    df['p_comp_val']   = pd.to_numeric(df['p_empirical'], errors='coerce')

    # 2) 各検定群ごとに BH 法で補正 → フラグ列を作成
    for src, flag in [
        ('p_runs_val',   'runs_anomaly_fdr'),
        ('p_markov_val', 'markov_anomaly_fdr'),
        ('p_comp_val',   'comp_anomaly_fdr'),
    ]:
        pvals = df[src]
        mask  = pvals.notna()
        reject, _, _, _ = multipletests(pvals[mask], alpha=alpha, method='fdr_bh')
        df.loc[mask, flag] = reject

    # 3) （従来の未補正フラグも残す場合は以下）
    df['runs_anomaly']   = df['p_runs_val']   < alpha
    df['markov_anomaly'] = df['p_markov_val'] < alpha
    df['comp_anomaly']   = df['p_comp_val']   < alpha

    # --- 第1層：全体サマリー（ダッシュボード） ---
    print("\n## 第1層：全体サマリー（異常度ダッシュボード）\n")
    
    # --- 修正後（FDR フラグを集計）
    summary = df.groupby('index').agg(
        runs_anomalies   = ('p_runs_val_reject_fdr',   'sum'),
        markov_anomalies = ('p_markov_val_reject_fdr', 'sum'),
        comp_anomalies   = ('p_comp_val_reject_fdr',   'sum'),
    ).reset_index()
    
    summary['total_anomalies'] = summary.runs_anomalies + summary.markov_anomalies + summary.comp_anomalies

    # --- 第3層：偏りのプロファイリング ---
    # サマリーテーブルにプロファイルを追加
    profiles = []
    for _, row in summary.iterrows():
        if row['runs_anomalies'] >= 3:
            profiles.append("A: 順序性の崩壊")
        else:
            # 1人のプレイヤーが複数の検定で異常を示した場合
            game_df = df[df['index'] == row['index']]
            player_total_anomalies = game_df.groupby('seat')[['runs_anomaly', 'markov_anomaly', 'comp_anomaly']].sum().sum(axis=1)
            if (player_total_anomalies >= 2).any():
                profiles.append("B: 複合的な局所偏り")
            else:
                profiles.append("-")
    summary['profile'] = profiles
    
    # 整形して出力
    summary.columns = ['対局ID', '順序偏り数', '遷移偏り数', '構成偏り数', '合計異常度', 'プロファイル']
    print(summary.to_string(index=False))

    # --- 第2層：個別対局ディープダイブ（ケースファイル） ---
    print("\n\n" + "="*50)
    print("## 第2層：個別対局ディープダイブ（特異ケース）")
    print("="*50)

    # 合計異常度が高い順に上位3ケースをピックアップ
    top_anomalous_games = summary.sort_values(by='合計異常度', ascending=False).head(3)

    def format_p_value(p_val, alpha=0.05):
        return f"{p_val:.4f} *" if pd.notna(p_val) and p_val < alpha else f"{p_val:.4f}"

    for _, game_row in top_anomalous_games.iterrows():
        game_id = game_row['対局ID']
        print(f"\n--- ケースファイル: 対局ID {game_id} (合計異常度: {game_row['合計異常度']}) ---")
        print(f"    推定プロファイル: {game_row['プロファイル']}")
        
        case_df = df[df['index'] == game_id].copy()
        
        # 結果を整形
        report_df = pd.DataFrame()
        report_df['席'] = case_df['seat']
        report_df['ランズ検定(p値)'] = case_df['p_runs_val'].apply(format_p_value)
        report_df['マルコフ連鎖(p値)'] = case_df['p_markov_val'].apply(format_p_value)  
        # 構成偏りのp値と効果量(V)を並べて表示
        report_df['構成(p値)'] = case_df['p_comp_val'].apply(format_p_value)
        # 'cramers_v' 列がデータに含まれていることを前提とする
        report_df['効果量(V)'] = case_df['cramers_v'].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "N/A")
        report_df['ランズ検定(FDR)']   = case_df['runs_anomaly_fdr'].map({True:'*', False:''})
        report_df['マルコフ連鎖(FDR)'] = case_df['markov_anomaly_fdr'].map({True:'*', False:''})
        report_df['構成偏り(FDR)']     = case_df['comp_anomaly_fdr'].map({True:'*', False:''})
    
        # 表示する列のリストを更新
        display_columns = ['席', 'ランズ検定(p値)', 'マルコフ連鎖(p値)', '構成(p値)', '効果量(V)','ランズ検定(FDR)','マルコフ連鎖(FDR)','構成偏り(FDR)']
        print(report_df[display_columns].to_string(index=False))

    print("\n" + "="*80)

from statsmodels.stats.multitest import multipletests
import numpy as np

def apply_fdr_correction(df: pd.DataFrame, alpha: float = 0.05):
    """
    df 中の各 p 値列に Benjamini-Hochberg FDR 補正をかけ、
    補正後 p 値と補正後も有意かのフラグ列を追加します。
    """
    # 補正対象の p 値列リスト
    p_cols = ['p_runs_val', 'p_markov_val', 'p_comp_val']
    
    for col in p_cols:
        if col not in df.columns:
            continue

        # NaN を除いた生 p 値配列
        mask = df[col].notna()
        orig_p = df.loc[mask, col].values
        
        if len(orig_p) == 0:
            # 補正対象がなければフラグ列だけ作っておく
            df[col + '_p_adj'] = np.nan
            df[col + '_reject_fdr'] = False
            continue
        
        # multipletests で補正
        reject, pvals_corrected, _, _ = multipletests(orig_p, alpha=alpha, method='fdr_bh')
        
        # 補正後 p 値列
        df.loc[mask, col + '_p_adj'] = pvals_corrected
        # 補正後も有意かどうかのフラグ列
        df.loc[mask, col + '_reject_fdr'] = reject

    # レポート用に件数を表示するだけなら以下
    num_before = int((df['p_runs_val']   < alpha).sum()
                 + (df['p_markov_val'] < alpha).sum()
                 + (df['p_comp_val']   < alpha).sum())
    num_after  = int((df['p_runs_val_reject_fdr']   == True).sum()
                 + (df['p_markov_val_reject_fdr'] == True).sum()
                 + (df['p_comp_val_reject_fdr']   == True).sum())

    print(f"--- 多重比較補正(FDR)結果 ---")
    print(f"補正前の有意検定数 (p < {alpha}): {num_before} 件")
    print(f"FDR補正後の有意検定数: {num_after} 件")
    print("--------------------------")


    return df

# ==============================================================================
# 実行
# ==============================================================================

if __name__ == '__main__':
     num_iterations = 10000 # 例として1000回に設定
     analyze_str = ""

     total_ng_counts = defaultdict(int)
     all_results = []
     analysis_output_for_one_game = []
     total_iterations = 0
     row_index = 0
     wall_list = read_csv()

     if wall_list == "":
        analyze_str = "random"
        for i in tqdm(range(num_iterations), desc="Analyzing Random Walls"):
           row_index = row_index + 1
           wall_ints = generate_mahjong_wall()

           if wall_ints is None:
               print(f"Iteration {i + 1}: エラー: 牌山の生成に失敗しました。スキップします。")
               continue # 次の繰り返しに進む
           total_iterations =  total_iterations + random_analyze_hand_drawn_tile(wall_ints)

           # ループ完了後の処理（必要に応じて）
           print(f"\nAnalysis for {total_iterations / num_iterations} iterations.")
           print(f"{analyze_str}")

     else:
         analyze_str = "csv"
         df = pd.read_csv('game_records.csv')
         num_iterations = 0
         pbar = tqdm(df.iterrows(),
                desc="処理中",           # 前につく説明文字列
                unit="局",               # 単位表示（it, items, bytes…）
                ncols=80,                # バーの幅（文字数）
                ascii=True
                )
         for index, row in pbar:
             tile_list_str = row['tile_List']          
             wall_str = parse_tile_list_string(tile_list_str)
             wall = [to_int(t) for t in wall_str if t]
             analysis_output_for_one_game = analyze_hand_drawn_tile(row_index,wall,analyze_str)
             all_results.append(analysis_output_for_one_game)
             row_index = row_index + 1
             num_iterations = num_iterations + 1

         pbar.close()

         generate_summary_report(all_results)