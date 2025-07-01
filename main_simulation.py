# =================================================================
#
#          統合麻雀分析エンジン (Integrated Mahjong Analysis Engine)
#                     【最終決定版 Ver.22.0】
#
# =================================================================
import pandas as pd
import numpy as np
import traceback
import json
import re
from collections import Counter
import os
from tqdm import tqdm
import hashlib
import random
from typing import Any, Dict, List

try:
    from mahjong.shanten import Shanten
    from mahjong.tile import TilesConverter
    from mahjong.hand_calculating.hand import HandCalculator
    from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
    from mahjong.meld import Meld
except Exception as e:
    print(f"インポートエラー: {e}")
    exit()

# =================================================================
# SECTION 1: ヘルパー関数群
# =================================================================

def get_tile_sort_key(tile_str: str) -> tuple:
    if not isinstance(tile_str, str) or len(tile_str) < 2: return (99, 99)
    suit_order = {'m': 0, 'p': 1, 's': 2, 'z': 3}
    num_char, suit_char = tile_str[0], tile_str[1]
    tile_num = 5 if num_char == '0' else int(num_char)
    return (suit_order.get(suit_char, 99), tile_num)

def hand_strings_to_34_array(hand_strings: list[str]) -> np.ndarray:
    man, pin, sou, honors = [], [], [], []
    for tile in hand_strings:
        if not isinstance(tile, str) or len(tile) != 2: continue
        num, suit = tile[0], tile[1]
        if num == '0': num = '5'
        if suit == 'm': man.append(num)
        elif suit == 'p': pin.append(num)
        elif suit == 's': sou.append(num)
        elif suit == 'z': honors.append(num)
    return TilesConverter.string_to_34_array(man="".join(man), pin="".join(pin), sou="".join(sou), honors="".join(honors))

def calculate_shanten(hand_strings: list[str], open_melds: list = None) -> int:
    try:
        all_tiles = list(hand_strings)
        if open_melds:
            for meld in open_melds:
                all_tiles.extend(meld['tiles'])
        valid_counts = [1, 2, 4, 5, 7, 8, 10, 11, 13, 14]
        if len(all_tiles) not in valid_counts: return 8
        tiles_34 = hand_strings_to_34_array(all_tiles)
        return Shanten().calculate_shanten(tiles_34)
    except Exception: return 8

def calculate_effective_tiles(hand_13_tiles: list) -> tuple[int, list]:
    """
    手牌の有効牌の数とその種類を計算します。
    """
    base_shanten = calculate_shanten(hand_13_tiles)
    if base_shanten <= -1: return 0, []
    effective_tiles, count = [], 0
    all_tile_types = [f'{i}{s}' for s in 'mpsz' for i in range(1, 10 if s != 'z' else 8)]
    current_hand_counter = Counter(hand_13_tiles)
    for tile in all_tile_types:
        if current_hand_counter[tile] < 4:
            if calculate_shanten(hand_13_tiles + [tile]) < base_shanten:
                num_available = 4 - current_hand_counter[tile]
                count += num_available
                effective_tiles.append(tile)
    return count, effective_tiles

def get_winning_tiles(hand_13_tiles: list, open_melds: list = None) -> set:
    if calculate_shanten(hand_13_tiles, open_melds) != 0: return set()
    winning_tiles = set()
    all_tile_types = [f'{i}{s}' for s in 'mpsz' for i in range(1, 10 if s != 'z' else 8)]
    current_hand_and_melds = list(hand_13_tiles)
    if open_melds:
        for meld in open_melds:
            current_hand_and_melds.extend(meld['tiles'])
    current_hand_counter = Counter(current_hand_and_melds)
    for tile in all_tile_types:
        if current_hand_counter.get(tile, 0) < 4:
            if calculate_shanten(hand_13_tiles + [tile], open_melds) == -1:
                winning_tiles.add(tile)
    return winning_tiles

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

# =================================================================
# SECTION 2: ゲームログパーサー
# =================================================================
class GameLogParser:
    def __init__(self, gamelog_json_string: str):
        if not gamelog_json_string or not isinstance(gamelog_json_string, str):
            raise ValueError("無効なJSONログ文字列")
        self.data = json.loads(gamelog_json_string.replace("'", "\""))
        self.log_data = self.data.get('log', [[]])[0]
    def _decode_pai(self, code) -> str:
        if not isinstance(code, int): return str(code)
        if code == 51: return '0m'
        if code == 52: return '0p'
        if code == 53: return '0s'
        suit, num = code // 10, code % 10
        if suit == 1: return f"{num}m"
        if suit == 2: return f"{num}p"
        if suit == 3: return f"{num}s"
        if suit == 4: return f"{num}z"
        return f'UNK({code})'
    def get_initial_conditions(self) -> dict:
        initial_conditions = {}
        if not self.log_data: return {'actual_game_events': []}
        initial_conditions['names'] = self.data.get('name', [f'P{i}' for i in range(4)])
        kyoku_info = self.log_data[0]
        initial_conditions['kyoku_info'] = {'kyoku': kyoku_info[0], 'honba': kyoku_info[1], 'riichi_bou': kyoku_info[2]}
        initial_conditions['dora_indicators'] = [self._decode_pai(p) for p in self.log_data[2]]
        initial_hands = {i: sorted([self._decode_pai(p) for p in self.log_data[4 + i * 3]], key=get_tile_sort_key) for i in range(4)}
        initial_conditions['initial_hands'] = initial_hands
        events = []
        base_event_template = {'player': -1, 'turn': 0, 'detail': None, 'tile': None, 'from_player': None, 'is_tsumogiri': None, 'current_hand_before_action': []}
        deal_event = base_event_template.copy()
        deal_event.update({'action': 'deal', 'hands': initial_hands, 'scores': self.log_data[1], 'turn': 0})
        events.append(deal_event)
        all_draws_by_player = {i: self.log_data[5 + i * 3] for i in range(4)}
        all_actions_by_player = {i: self.log_data[6 + i * 3] for i in range(4)}
        player_draw_cursors, player_action_cursors = Counter(), Counter()
        current_player = initial_conditions['kyoku_info']['kyoku'] % 4
        turn_counter = 0; loop_guard = 0
        while loop_guard < 150:
            loop_guard += 1
            if all(player_action_cursors[i] >= len(all_actions_by_player[i]) for i in range(4)): break
            is_naki_turn = events and events[-1]['action'] == 'naki'
            if not is_naki_turn:
                draw_cursor = player_draw_cursors[current_player]
                if draw_cursor >= len(all_draws_by_player[current_player]):
                    if all(player_action_cursors[i] >= len(all_actions_by_player[i]) for i in range(4)): break
                    current_player = (current_player + 1) % 4
                    continue
                draw_tile_code = all_draws_by_player[current_player][draw_cursor]
                if isinstance(draw_tile_code, int):
                    turn_counter += 1
                    draw_event = base_event_template.copy()
                    draw_event.update({'action': 'draw', 'player': current_player, 'tile': self._decode_pai(draw_tile_code), 'turn': turn_counter})
                    events.append(draw_event)
                player_draw_cursors[current_player] += 1
            action_cursor = player_action_cursors[current_player]
            if action_cursor >= len(all_actions_by_player[current_player]):
                if all(player_action_cursors[i] >= len(all_actions_by_player[i]) for i in range(4)): break
                current_player = (current_player + 1) % 4
                continue
            action_data = all_actions_by_player[current_player][action_cursor]
            player_action_cursors[current_player] += 1
            action_event = base_event_template.copy()
            action_event.update({'player': current_player, 'turn': turn_counter})
            next_player_after_discard = (current_player + 1) % 4
            if isinstance(action_data, int):
                action_event.update({'action': 'discard', 'tile': self._decode_pai(action_data), 'is_tsumogiri': (action_data == 60)})
                events.append(action_event)
                naki_found = False
                for i in range(1, 4):
                    p_to_check = (current_player + i) % 4
                    cursor_check = player_action_cursors[p_to_check]
                    if cursor_check < len(all_actions_by_player[p_to_check]):
                        next_action_data = all_actions_by_player[p_to_check][cursor_check]
                        if isinstance(next_action_data, str) and next_action_data[0] in 'cpm':
                            player_action_cursors[p_to_check] += 1
                            naki_event = base_event_template.copy()
                            naki_event.update({'action': 'naki', 'player': p_to_check, 'from_player': current_player, 'detail': next_action_data, 'turn': turn_counter})
                            events.append(naki_event)
                            current_player = p_to_check
                            naki_found = True
                            break
                if not naki_found: current_player = next_player_after_discard
            elif isinstance(action_data, str):
                if action_data.startswith('r'):
                    action_event.update({'action': 'riichi', 'tile': self._decode_pai(int(action_data[1:]))})
                    events.append(action_event)
                    current_player = next_player_after_discard
                else:
                    action_event.update({'action': 'naki', 'from_player': current_player, 'detail': action_data})
                    events.append(action_event)
                    if not (action_event.get('from_player') is not None and action_event.get('detail', '') and action_event['detail'][0] != 'a'):
                        current_player = next_player_after_discard
        end_game_event = base_event_template.copy()
        end_game_event.update({'action': 'end_game', 'detail': self.log_data[-1], 'turn': turn_counter, 'player': -1})
        events.append(end_game_event)
        initial_conditions['actual_game_events'] = events
        initial_conditions['parser'] = self
        return initial_conditions

# =================================================================
# SECTION 3: AIとPlayerController
# =================================================================
NUM_PLAYERS = 4
class DiscardAI:
    def _find_safe_tiles(self, hand: list, game_context: dict, my_pid: int) -> list:
        """ 安全な捨て牌を優先度順にリストアップする """
        safe_tiles_ranked = []

        # リーチ者情報を特定
        riichi_players = [pid for pid, p_info in game_context['players'].items() if p_info['is_riichi'] and pid != my_pid]
        if not riichi_players: return []

        riichi_discards = set()
        for r_pid in riichi_players:
            riichi_discards.update(game_context['players'][r_pid]['discards'])

        # レベル1：現物
        genbutsu = [tile for tile in hand if tile in riichi_discards]
        safe_tiles_ranked.extend(sorted(list(set(genbutsu)), key=get_tile_sort_key))

        # レベル2：スジ (1-4-7, 2-5-8, 3-6-9)
        suji_map = {1: [4], 2: [5], 3: [6], 4: [1, 7], 5: [2, 8], 6: [3, 9], 7: [4], 8: [5], 9: [6]}
        suji_candidates = set()
        for discard in riichi_discards:
            if discard[1] in 'mps':
                num = int(discard[0].replace('0', '5'))
                if num in suji_map:
                    for suji_num in suji_map[num]:
                        suji_candidates.add(f"{suji_num}{discard[1]}")

        suji_in_hand = [tile for tile in hand if tile in suji_candidates and tile not in safe_tiles_ranked]
        safe_tiles_ranked.extend(sorted(list(set(suji_in_hand)), key=get_tile_sort_key))

        return safe_tiles_ranked

    def choose_action(self, hand: list[str], open_melds: list, is_menzen: bool, kan_candidates: list[str], game_context: dict, my_pid: int, game_turn: int) -> dict:
        if not hand: return None

        # 状況評価：オリるべきか判断
        is_betaori_mode = False
        my_shanten = calculate_shanten(hand, open_melds)

        # 他家にリーチ者がいて、自分が2シャンテン以上ならオリる
        if any(p['is_riichi'] for pid, p in game_context['players'].items() if pid != my_pid):
            if my_shanten >= 2:
                is_betaori_mode = True

        # ベタオリモードの処理
        if is_betaori_mode:
            safe_tiles = self._find_safe_tiles(hand, game_context, my_pid)
            if safe_tiles:
                return {'action': 'discard', 'tile': safe_tiles[0]}
            else:
                # 安全牌がない場合、最も不要な字牌から捨てる (簡易ロジック)
                honor_tiles = sorted([t for t in hand if t[1] == 'z'], key=get_tile_sort_key, reverse=True)
                if honor_tiles: return {'action': 'discard', 'tile': honor_tiles[0]}
                # それもなければ、最も内側の数牌から捨てる
                return {'action': 'discard', 'tile': sorted(hand, key=lambda x: (abs(int(x[0].replace('0','5')) - 5)))[0]}

        # --- 以下は、オリないと判断した場合の攻撃ロジック (変更なし) ---
        if is_menzen and kan_candidates: return {'action': 'ankan', 'tile': kan_candidates[0]}
        if is_menzen and my_shanten == 0:
             return {'action': 'riichi', 'tile': hand[-1]}

        candidates = []
        for tile in list(set(hand)):
            temp_hand = list(hand); temp_hand.remove(tile)
            s = calculate_shanten(temp_hand, open_melds)
            _, eff_tiles = calculate_effective_tiles(temp_hand)
            ukeire_count = len(eff_tiles)
            candidates.append({'tile': tile, 'shanten': s, 'ukeire': ukeire_count})

        if not candidates: return {'action': 'discard', 'tile': hand[-1]}
        candidates.sort(key=lambda x: (x['shanten'], -x['ukeire']))
        return {'action': 'discard', 'tile': candidates[0]['tile']}

class CallAI:
    def evaluate(self, hand: list[str], tile: str, discarder_pos_relative: int, player_wind: int, round_wind: int, existing_melds: list) -> dict or None:
        hand_counter = Counter(hand)
        if hand_counter[tile] == 3: return {'action': 'daiminkan', 'tiles': [tile] * 3}
        if hand_counter[tile] >= 2: return {'action': 'pon', 'tiles': [tile] * 2}
        if discarder_pos_relative == 3 and tile[1] in 'mps':
            try:
                s, r = tile[1], int(tile[0].replace('0','5'))
                hand_ranks = {int(t[0].replace('0','5')) for t in hand if t[1] == s}
                patterns = [{'tiles':[f"{r-2}{s}",f"{r-1}{s}"],'ranks':{r-2,r-1}},{'tiles':[f"{r-1}{s}",f"{r+1}{s}"],'ranks':{r-1,r+1}},{'tiles':[f"{r+1}{s}",f"{r+2}{s}"],'ranks':{r+1,r+2}}]
                for p in patterns:
                    if p['ranks'].issubset(hand_ranks) and all(1 <= rank <= 9 for rank in p['ranks']):
                        return {'action': 'chi', 'tiles': p['tiles']}
            except (ValueError, IndexError): pass
        return None

class PlayerController:
    def __init__(self, player_id):
        self.player_id, self.hand, self.melds = player_id, [], []
        self.is_riichi, self.riichi_turn, self.is_ippatsu = False, -1, False
        self.is_tenpai, self.winning_tiles = False, set()
        self.discard_ai, self.call_ai = DiscardAI(), CallAI()
        self.turn_to_tenpai = -1

    @property
    def is_menzen(self): return len(self.melds) == 0
    def set_hand(self, hand_list): self.hand = sorted(hand_list, key=get_tile_sort_key)

    def update_tenpai_state(self):
        shanten = calculate_shanten(self.hand, self.melds)
        if shanten == 0:
            if not self.is_tenpai:
                self.winning_tiles = get_winning_tiles(self.hand, self.melds)
                self.is_tenpai = True
        else:
            if self.is_tenpai: self.is_tenpai, self.winning_tiles = False, set()

    def draw(self, tile): self.hand.append(tile); self.hand.sort(key=get_tile_sort_key)

    def discard(self, game_turn: int, game_context: dict) -> dict:
        kan_candidates = [tile for tile, count in Counter(self.hand).items() if count == 4]
        action = self.discard_ai.choose_action(self.hand, self.melds, self.is_menzen, kan_candidates, game_context, self.player_id, game_turn)

        if not action: return None
        tile_to_discard = action.get('tile')

        if action['action'] == 'ankan':
            if tile_to_discard and self.hand.count(tile_to_discard) == 4:
                 for _ in range(4): self.hand.remove(tile_to_discard)
                 self.melds.append({'type': 'ankan', 'tiles': [tile_to_discard] * 3})
                 self.hand.sort(key=get_tile_sort_key); return action
            else: return None

        if tile_to_discard in self.hand:
            self.hand.remove(tile_to_discard)
        else:
            return {'action': 'discard', 'tile': self.hand[-1]} if self.hand else None

        self.hand.sort(key=get_tile_sort_key)
        if action['action'] == 'riichi' and not self.is_riichi:
            self.is_riichi, self.riichi_turn, self.is_ippatsu = True, game_turn, True
            self.update_tenpai_state()
            if self.is_tenpai and self.turn_to_tenpai == -1:
                self.turn_to_tenpai = game_turn
        return action

    def check_call(self, tile, discarder_id, player_wind, round_wind, existing_melds):
        relative_pos = (self.player_id - discarder_id + NUM_PLAYERS) % NUM_PLAYERS
        return self.call_ai.evaluate(self.hand, tile, relative_pos, player_wind, round_wind, existing_melds)

    def perform_call(self, call_info, called_tile, from_who):
        for tile in call_info['tiles']:
            if tile in self.hand: self.hand.remove(tile)
        meld_type = call_info['action']
        display_tiles = sorted(call_info['tiles'] + [called_tile], key=get_tile_sort_key)
        record_tiles = display_tiles[:3] if 'kan' in meld_type else display_tiles
        self.melds.append({'type': meld_type, 'tiles': record_tiles})
        self.hand.sort(key=get_tile_sort_key); self.is_ippatsu = False

# =================================================================
# SECTION 4: デュアルシミュレーションエンジン
# =================================================================
class DualSimulator:
    def __init__(self, initial_conditions: dict):
        self.init_cond = initial_conditions
        self.ai_controllers = [PlayerController(i) for i in range(4)]
        self.metrics = self._initialize_metrics()
    def _initialize_metrics(self) -> dict:
        metrics = {}
        for p_idx in range(4):
            metrics[p_idx] = {'kyoku_id':self.init_cond['kyoku_info']['kyoku'],'player_id':p_idx,'player_name':self.init_cond['names'][p_idx],'is_winner':0,'win_type':'-','is_loser':0,'is_tenpai_at_draw':0,'draw_type':'-','times_reached_tenpai':0,'tenpai_grab_victim_count':0,'riichi_count':0,'ippatsu_win':0,'kan_count':0,'rinshan_win':0,'chankan_win':0,'initial_shanten':8,'initial_effective_tiles':0,'effective_tiles_in_wall':0,'game_date':'','analysis_target':'','actual_winner':'N/A','actual_loser_or_win_type':'N/A','simulation_winner':'N/A','simulation_loser_or_win_type':'N/A','fulfillment_rate':0.0,'jigoku_potential':0,'jigoku_stall':0.0,'shanten_adv':0.0,'tsumo_adv':0,'advantage_score':0.0,'is_double_riichi':0,'actual_tenpai_turn':-1,'ai_tenpai_turn':-1,'last_discard_is_deal_in':0,'had_bad_call':0,'total_draw_count':0,'drew_potential_win_tile_count':0, 'tsumo_win_count': 0, 'status': 'N/A', 'ma_tsumo_adv': 0, 'tenpai_grab_victim_count_ma': 0, 'float_discard_deal_in_count': 0, 'bad_call_count': 0, 'winning_tile_drawn_by_others_count': 0}
        return metrics
    def run(self) -> list:
        try:
            self.init_cond['parser'] = self.init_cond.get('parser', GameLogParser("{\"log\":[[]]}"))
            if 'reconstructed_wall' not in self.init_cond or not self.init_cond['reconstructed_wall']:
                self.init_cond['reconstructed_wall'] = self._generate_wall()
            self._calculate_initial_metrics()
            actual_history = self._run_actual_replay()
            ai_history = self._run_ai_simulation()
            self._calculate_final_metrics(actual_history, ai_history)
            return list(self.metrics.values())
        except Exception:
            traceback.print_exc()
            return []
    def _generate_wall(self) -> List[str]:
        tiles = ([f"{n}{s}" for s in "mps" for n in range(1,10)]*4 + [f"{z}z" for z in range(1,8)]*4)
        red_dora = ['0m', '0p', '0s']
        for r_d in red_dora:
            regular_tile = f"5{r_d[1]}"
            try:
                tiles[tiles.index(regular_tile)] = r_d
            except ValueError: pass
        seed_str = f"{self.init_cond.get('server_seed', '')}|{self.init_cond.get('client_seed', '')}"
        seed = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16)
        random.Random(seed).shuffle(tiles)
        return tiles
    def _calculate_initial_metrics(self):
        hands = self.init_cond['initial_hands']
        wall = self.init_cond.get('reconstructed_wall', [])
        wall_counter = Counter(wall[sum(len(h) for h in hands.values()):])
        for p_idx, hand_list in hands.items():
            self.metrics[p_idx]['initial_shanten'] = calculate_shanten(hand_list)
            eff_count, eff_list = calculate_effective_tiles(hand_list)
            self.metrics[p_idx]['initial_effective_tiles'] = eff_count
            self.metrics[p_idx]['effective_tiles_in_wall'] = sum(wall_counter[tile] for tile in eff_list)

    def _run_actual_replay(self) -> dict:
        hands = {p: Counter(h) for p, h in self.init_cond['initial_hands'].items()}
        history = {'shanten': {p: [self.metrics[p]['initial_shanten']] for p in range(4)}, 'tsumo_adv': Counter(), 'riichi': Counter(), 'is_in_riichi_state': {p: False for p in range(4)}, 'kan': Counter(), 'tenpai_turn': {p: -1 for p in range(4)}, 'winning_tiles_at_tenpai': {p: set() for p in range(4)}, 'drew_potential_win_tile': Counter(), 'turn_count': Counter(), 'last_discards': {p: None for p in range(4)}, 'draw_final_info': {p: False for p in range(4)}, 'all_events_df': pd.DataFrame(), 'last_drawn_tile_by_player': {p: None for p in range(4)}, 'open_melds': {p: [] for p in range(4)}}
        ma_events_data, turn_id_counter, last_event = [], 0, None
        temp_parser = self.init_cond['parser']
        for event in self.init_cond['actual_game_events']:
            p_idx = event.get('player', -1)
            event_action = event['action']
            current_turn_in_kyoku = event.get('turn', 0)
            ma_event_row = {'round_id': self.init_cond['kyoku_info']['kyoku'], 'player': p_idx, 'action': event_action, 'tile': event.get('tile'), 'from_who': event.get('from_player'), 'is_tsumogiri': event.get('is_tsumogiri'), 'timestamp': turn_id_counter, 'turn': current_turn_in_kyoku, 'detail': event.get('detail'), 'current_hand_before_action': []}
            if event_action == 'deal':
                for pid, h in event['hands'].items(): hands[pid] = Counter(h)
                if p_idx != -1: ma_event_row['current_hand_before_action'] = list(hands[p_idx].elements())
                ma_events_data.append(ma_event_row)
            elif event_action == 'draw' and p_idx != -1:
                melds = history['open_melds'][p_idx]; hand_before = list(hands[p_idx].elements())
                prev_shanten = calculate_shanten(hand_before, melds)
                if prev_shanten == 0 and history['tenpai_turn'].get(p_idx, -1) == -1:
                    history['tenpai_turn'][p_idx] = current_turn_in_kyoku
                    if not history['is_in_riichi_state'][p_idx]: history['winning_tiles_at_tenpai'][p_idx] = get_winning_tiles(hand_before, melds)
                drawn_tile = event.get('tile')
                if drawn_tile is not None:
                    hands[p_idx][drawn_tile] += 1; history['last_drawn_tile_by_player'][p_idx] = drawn_tile
                    for any_pid in range(4):
                        if history['tenpai_turn'][any_pid] != -1 and drawn_tile in history['winning_tiles_at_tenpai'][any_pid]:
                            history['drew_potential_win_tile'][p_idx] += 1; break
                current_shanten = calculate_shanten(list(hands[p_idx].elements()), melds)
                if current_shanten < prev_shanten: history['tsumo_adv'][p_idx] += 1
                history['shanten'][p_idx].append(current_shanten); history['turn_count'][p_idx] += 1
                ma_event_row['current_hand_before_action'] = list(hands[p_idx].elements()); ma_events_data.append(ma_event_row)
            elif event_action in ['discard', 'riichi'] and p_idx != -1:
                ma_event_row['current_hand_before_action'] = list(hands[p_idx].elements())
                tile = event.get('tile')
                if event.get('is_tsumogiri'): tile = history['last_drawn_tile_by_player'][p_idx]
                if event_action == 'riichi': history['riichi'][p_idx] += 1; history['is_in_riichi_state'][p_idx] = True
                if tile and hands[p_idx].get(tile, 0) > 0: hands[p_idx][tile] -= 1
                melds = history['open_melds'][p_idx]; hand_after = list(hands[p_idx].elements())
                shanten = calculate_shanten(hand_after, melds)
                history['shanten'][p_idx].append(shanten)
                if shanten == 0:
                    if history['tenpai_turn'][p_idx] == -1: history['tenpai_turn'][p_idx] = current_turn_in_kyoku
                    if not history['is_in_riichi_state'][p_idx]: history['winning_tiles_at_tenpai'][p_idx] = get_winning_tiles(hand_after, melds)
                history['last_discards'][p_idx] = tile; ma_event_row['tile'] = tile; ma_events_data.append(ma_event_row)
            elif event_action == 'naki' and p_idx != -1:
                detail = event.get('detail', ''); ma_event_row['current_hand_before_action'] = list(hands[p_idx].elements())
                if isinstance(detail, str) and detail:
                    naki_type = detail[0]; meld_to_add = None
                    try:
                        codes = [int(c) for c in re.findall(r'\d\d', detail)]; tiles = [temp_parser._decode_pai(c) for c in codes]
                        called_tile = last_event.get('tile') if last_event else None
                        if naki_type in ['p','m','a']:
                            n_rm = {'p':2,'m':3,'a':4}.get(naki_type,0); meld_t = tiles[0]
                            for _ in range(n_rm):
                                if hands[p_idx].get(meld_t, 0) > 0: hands[p_idx][meld_t] -= 1
                            meld_to_add = {'type':naki_type, 'tiles':[meld_t]*3}
                        elif naki_type == 'k':
                            kan_t = tiles[0]
                            if kan_t and hands[p_idx].get(kan_t, 0) > 0:
                                hands[p_idx][kan_t] -= 1
                                for m in history['open_melds'][p_idx]:
                                    if m.get('type')=='p' and m.get('tiles',[""])[0]==kan_t: m['type']='k'; break
                        elif naki_type == 'c':
                            if called_tile and len(tiles)==3:
                                hand_tiles = [t for t in tiles if t!=called_tile]
                                for t in hand_tiles:
                                    if hands[p_idx].get(t,0)>0: hands[p_idx][t] -= 1
                                meld_to_add = {'type':'c', 'tiles':sorted(tiles, key=get_tile_sort_key)}
                        if meld_to_add: history['open_melds'][p_idx].append(meld_to_add)
                    except(ValueError,IndexError): pass
                if isinstance(detail,str) and detail.startswith(('m','a','k')): history['kan'][p_idx]+=1
                melds = history['open_melds'][p_idx]; shanten = calculate_shanten(list(hands[p_idx].elements()), melds)
                history['shanten'][p_idx].append(shanten)
                if shanten == 0 and history['tenpai_turn'][p_idx] == -1: history['tenpai_turn'][p_idx] = current_turn_in_kyoku
                ma_events_data.append(ma_event_row)
            elif event_action == 'end_game':
                detail = event.get('detail')
                if detail:
                    history.update({'final_event': event, 'final_scores': detail[1]})
                    if detail[0]=='和了':
                        winner_id = detail[2][0]
                        if history['tenpai_turn'].get(winner_id, -1)==-1: history['tenpai_turn'][winner_id]=current_turn_in_kyoku
                ma_events_data.append(ma_event_row)
                break
            turn_id_counter += 1; last_event = event
        history['all_events_df'] = pd.DataFrame(ma_events_data)
        return history

    def _run_ai_simulation(self) -> dict:
        for i, c in enumerate(self.ai_controllers):
            c.set_hand(self.init_cond['initial_hands'][i]); c.is_riichi, c.riichi_turn, c.is_ippatsu = False, -1, False; c.is_tenpai, c.winning_tiles = False, set(); c.turn_to_tenpai = -1

        current_player_idx = self.init_cond['kyoku_info']['kyoku'] % 4
        wall = list(self.init_cond.get('reconstructed_wall', []))
        if not wall: return {'winner': -1, 'score_change': [0]*4, 'end_type': '牌山データなし', 'loser': -1, 'furikomi_pushed_count': Counter()}

        for p_hand in self.init_cond['initial_hands'].values():
            for tile in p_hand:
                if tile in wall: wall.remove(tile)

        turn = 0
        simulation_history = {'player_tenpai_turn': {i: -1 for i in range(4)}, 'winner': -1, 'score_change': [0]*4, 'end_type': 'N/A', 'loser': -1, 'furikomi_pushed_count': Counter()}
        discard_piles = {p:[] for p in range(4)}

        while wall and turn < 150:
            turn += 1
            controller = self.ai_controllers[current_player_idx]

            if not wall:
                simulation_history['end_type'] = '流局';
                for p_idx in range(4):
                    self.ai_controllers[p_idx].update_tenpai_state()
                    if self.ai_controllers[p_idx].is_tenpai: self.metrics[p_idx]['is_tenpai_at_draw'] = 1
                break

            draw_pai = wall.pop(0); controller.draw(draw_pai)

            controller.update_tenpai_state()
            if controller.is_tenpai and controller.turn_to_tenpai == -1:
                controller.turn_to_tenpai = turn
                simulation_history['player_tenpai_turn'][current_player_idx] = turn

            if calculate_shanten(controller.hand, controller.melds) == -1:
                simulation_history.update({'winner': current_player_idx, 'end_type': 'ツモ'}); return simulation_history

            game_context = {
                'dora_indicators': self.init_cond.get('dora_indicators', []),
                'players': {
                    pid: {'discards': discard_piles[pid], 'is_riichi': c.is_riichi}
                    for pid, c in enumerate(self.ai_controllers)
                }
            }
            action = controller.discard(turn, game_context)
            if not action or 'tile' not in action: current_player_idx = (current_player_idx + 1) % 4; continue

            discarded_tile = action.get('tile')
            discard_piles[current_player_idx].append(discarded_tile)

            if action['action'] == 'ankan': continue

            naki_found = False; next_player_after_discard = (current_player_idx + 1) % 4
            for i in range(1, 4):
                p_to_check_idx = (current_player_idx + i) % 4
                other_controller = self.ai_controllers[p_to_check_idx]
                other_controller.update_tenpai_state()
                if other_controller.is_tenpai and discarded_tile in other_controller.winning_tiles:
                    simulation_history['furikomi_pushed_count'][current_player_idx] += 1
                    simulation_history.update({'winner': p_to_check_idx, 'loser': current_player_idx, 'end_type': 'ロン'}); return simulation_history

                # ★★★【ここからが修正部分】★★★
                # 鳴き判断に必要な情報を計算
                player_wind = (p_to_check_idx - (self.init_cond['kyoku_info']['kyoku'] % 4) + 4) % 4
                round_wind = self.init_cond['kyoku_info']['kyoku'] // 4

                # 正しい引数を渡してcheck_callを呼び出す
                call_info = other_controller.check_call(
                    discarded_tile,
                    current_player_idx,
                    player_wind,
                    round_wind,
                    other_controller.melds
                )
                # ★★★【ここまでが修正部分】★★★

                if call_info:
                    other_controller.perform_call(call_info, discarded_tile, current_player_idx)
                    current_player_idx = p_to_check_idx
                    naki_found = True
                    break

            if not naki_found: current_player_idx = next_player_after_discard

        if simulation_history['winner'] == -1:
            simulation_history['end_type'] = '流局'

        return simulation_history

    def _calculate_final_metrics(self, actual_history, ai_history):
        actual_end_detail = actual_history.get('final_event', {}).get('detail')
        if not actual_end_detail: return
        end_type_actual = actual_end_detail[0]; actual_scores = actual_end_detail[1]
        actual_winner_id = -1; actual_loser_id = -1; actual_loser_or_win_type_str = 'N/A'
        if end_type_actual == '和了':
            win_info = actual_end_detail[2]; actual_winner_id = win_info[0]
            losers = [i for i, score in enumerate(actual_scores) if score < 0]
            if len(losers) == 1:
                win_type_str = 'ロン'; actual_loser_id = losers[0]; self.metrics[actual_loser_id]['is_loser'] = 1
                actual_loser_or_win_type_str = self.init_cond['names'][actual_loser_id]
            else:
                win_type_str = 'ツモ'; actual_loser_or_win_type_str = 'ツモ和了'
                self.metrics[actual_winner_id]['tsumo_win_count'] = 1
            self.metrics[actual_winner_id]['is_winner'] = 1; self.metrics[actual_winner_id]['win_type'] = win_type_str
        elif end_type_actual == '流局':
            actual_loser_or_win_type_str = '流局'
            for p_idx, score in enumerate(actual_scores):
                if score > 0: self.metrics[p_idx]['is_tenpai_at_draw'] = 1
            for p_idx in range(4): self.metrics[p_idx]['draw_type'] = '流局'
        ai_winner_id = ai_history['winner']; ai_scores = ai_history['score_change']; end_type_sim = ai_history['end_type']
        sim_loser_or_win_type_str = 'N/A'
        if end_type_sim == 'ツモ': sim_loser_or_win_type_str = 'ツモ和了'
        elif end_type_sim == 'ロン' and ai_history['loser'] != -1: sim_loser_or_win_type_str = self.init_cond['names'][ai_history['loser']]
        elif end_type_sim == '流局': sim_loser_or_win_type_str = '流局'
        for p_idx in range(4):
            my_avg_shanten = np.mean(actual_history['shanten'][p_idx]) if actual_history['shanten'][p_idx] else 8
            others_shanten_flat = [s for i, h in actual_history['shanten'].items() if i != p_idx for s in h]
            others_avg_shanten = np.mean(others_shanten_flat) if others_shanten_flat else my_avg_shanten
            self.metrics[p_idx].update({
                'times_reached_tenpai': 1 if actual_history['tenpai_turn'].get(p_idx, -1) != -1 else 0,
                'riichi_count': actual_history['riichi'].get(p_idx, 0),
                'kan_count': actual_history['kan'].get(p_idx, 0),
                'tsumo_adv': actual_history['tsumo_adv'].get(p_idx, 0),
                'shanten_adv': others_avg_shanten - my_avg_shanten,
                'actual_winner': self.init_cond['names'][actual_winner_id] if actual_winner_id != -1 else "流局",
                'actual_loser_or_win_type': actual_loser_or_win_type_str,
                'simulation_winner': self.init_cond['names'][ai_winner_id] if ai_winner_id != -1 else "流局",
                'simulation_loser_or_win_type': sim_loser_or_win_type_str,
                'fulfillment_rate': actual_scores[p_idx] - ai_scores[p_idx],
                'actual_tenpai_turn': actual_history['tenpai_turn'].get(p_idx, -1),
                'ai_tenpai_turn': ai_history['player_tenpai_turn'].get(p_idx, -1),
                'total_draw_count': sum(actual_history['turn_count'].values()),
                'tenpai_grab_victim_count': ai_history['furikomi_pushed_count'].get(p_idx, 0),
                'drew_potential_win_tile_count': actual_history['drew_potential_win_tile'].get(p_idx, 0)
            })
            self.metrics[p_idx]['advantage_score'] = (self.metrics[p_idx]['shanten_adv'] * 10) + (self.metrics[p_idx]['tsumo_adv'] * 5) + (self.metrics[p_idx]['fulfillment_rate'] / 1000)
        if not actual_history['all_events_df'].empty:
            ma_analyzer = MahjongAnalyzer(actual_history['all_events_df'])
            ma_analysis_results = ma_analyzer.run_full_analysis()
            kyoku_id = self.init_cond['kyoku_info']['kyoku']
            for p_idx in range(4):
                status_row = ma_analysis_results.get('status', pd.DataFrame())
                if not status_row.empty:
                    res = status_row[(status_row['round_id'] == kyoku_id) & (status_row['player'] == p_idx)]
                    if not res.empty: self.metrics[p_idx]['status'] = res.iloc[0]['status']

# =================================================================
# SECTION 5: MahjongAnalyzer
# =================================================================
class MahjongAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._preprocess()
    def _preprocess(self):
        if 'action' not in self.df.columns: self.df['action'] = 'unknown'
    def run_full_analysis(self) -> Dict[str, pd.DataFrame]:
        results = {'status': self.identify_status()}
        return results
    def identify_status(self) -> pd.DataFrame:
        status = []
        for (rid, ply), grp in self.df.groupby(['round_id','player']):
            s = 'fold'
            if 'riichi' in set(grp['action']): s = 'riichi'
            end_event = grp[grp['action'] == 'end_game'].iloc[0] if not grp[grp['action'] == 'end_game'].empty else None
            if end_event is not None and end_event['detail']:
                detail = end_event['detail']
                if detail and len(detail) > 2:
                    if detail[0] == '和了' and detail[2][0] == ply: s = 'win'
                    elif detail[0] == '流局' and len(detail[1]) > ply and detail[1][ply] > 0: s = 'draw_tenpai'
            status.append({'round_id': rid, 'player': ply, 'status': s})
        return pd.DataFrame(status)

# =================================================================
# SECTION 6: 分析統括 & 実行ブロック
# =================================================================
class AnalysisOrchestrator:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.results = []
    def run(self):
        print("全対局の統合分析を開始します...")
        for index, row in tqdm(self.df.iterrows(), total=self.df.shape[0], desc="対局分析中"):
            gamelog_json = row.get('gamelog_json')
            if pd.isna(gamelog_json): continue
            try:
                parser = GameLogParser(gamelog_json)
                initial_conditions = parser.get_initial_conditions()
                if not initial_conditions.get('actual_game_events'): continue
                tile_list_str = row.get('tile_List')
                if pd.notna(tile_list_str): initial_conditions['reconstructed_wall'] = parse_tile_list_string(tile_list_str)
                simulator = DualSimulator(initial_conditions)
                analysis_result = simulator.run()
                if analysis_result:
                    for i in range(len(analysis_result)):
                        analysis_result[i].update({'game_date': row.get('game_date', ''), 'analysis_target': row.get('analysis_target', '')})
                    self.results.extend(analysis_result)
            except Exception:
                traceback.print_exc()
        if not self.results: print("\n有効な分析結果が生成されませんでした。"); return
        self.results_df = pd.DataFrame(self.results)
        # Final cleanup for CSV output
        all_cols = ['kyoku_id', 'player_id', 'player_name', 'game_date', 'analysis_target', 'is_winner', 'win_type', 'is_loser', 'is_tenpai_at_draw', 'draw_type', 'times_reached_tenpai', 'drew_potential_win_tile_count', 'tsumo_win_count', 'tenpai_grab_victim_count', 'riichi_count', 'ippatsu_win', 'kan_count', 'rinshan_win', 'chankan_win', 'initial_shanten', 'initial_effective_tiles', 'effective_tiles_in_wall', 'jigoku_potential', 'jigoku_stall', 'shanten_adv', 'tsumo_adv', 'advantage_score', 'actual_winner', 'actual_loser_or_win_type', 'simulation_winner', 'simulation_loser_or_win_type', 'fulfillment_rate', 'is_double_riichi', 'actual_tenpai_turn', 'ai_tenpai_turn', 'last_discard_is_deal_in', 'had_bad_call', 'total_draw_count', 'status', 'ma_tsumo_adv', 'tenpai_grab_victim_count_ma', 'float_discard_deal_in_count', 'bad_call_count']
        for col in all_cols:
            if col not in self.results_df.columns: self.results_df[col] = 0 if 'count' in col or 'is' in col else '-'
        self.results_df = self.results_df[all_cols]
        output_path = 'comprehensive_analysis_results.csv'
        self.results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n分析が完了しました。全指標を '{output_path}' に保存しました。")
        self.display_summary_results()
    def display_summary_results(self):
        print("\n" + "="*70 + "\n【分析結果サマリー】\n" + "="*70)
        if not hasattr(self, 'results_df') or self.results_df.empty:
            print("表示する分析結果がありません。"); return
        df = self.results_df
        if 'player_name' in df.columns:
            for name in sorted(df['player_name'].unique()):
                player_df = df[df['player_name'] == name]
                if player_df.empty: continue
                print(f"\n--- {name} ---")
                player_summary = player_df.select_dtypes(include=np.number).mean(numeric_only=True)
                for col_name, value in player_summary.items():
                    if col_name in ['kyoku_id', 'player_id']: continue
                    print(f"  {col_name:<35}: {value:+.2f}")
                if 'status' in player_df.columns:
                    status_counts = player_df['status'].value_counts(normalize=True) * 100
                    print(f"  {'Status Distribution':<35}:")
                    for status_type, percentage in status_counts.items():
                        print(f"    - {status_type:<31}: {percentage:.2f}%")
        print("\n" + "="*70)

if __name__ == '__main__':
    CSV_FILE_PATH = 'game_records.csv'
    print("\n" + "#"*70 + "\n統合麻雀分析エンジンへようこそ。\n" + "#"*70)
    print(f"\nマスターファイル: '{CSV_FILE_PATH}'")
    if not os.path.exists(CSV_FILE_PATH):
        print(f"エラー: {CSV_FILE_PATH} が見つかりません。")
    else:
        print("分析を開始します...")
        orchestrator = AnalysisOrchestrator(CSV_FILE_PATH)
        orchestrator.run()