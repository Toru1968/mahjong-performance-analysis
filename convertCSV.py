import pandas as pd
import json
import collections
from tqdm import tqdm
import csv
from collections import Counter

# =================================================================
# SECTION 1: ヘルパー関数群 (変更なし)
# =================================================================
def get_tile_sort_key(tile_str: str) -> tuple:
    if not isinstance(tile_str, str) or len(tile_str) < 2: return (99, 99)
    suit_order = {'m': 0, 'p': 1, 's': 2, 'z': 3}
    num_char, suit_char = tile_str[0], tile_str[1]
    tile_num = 5 if num_char == '0' else int(num_char)
    return (suit_order.get(suit_char, 99), tile_num)

def to_str(tile_code: int) -> str:
    if not isinstance(tile_code, int): return str(tile_code)
    if tile_code == 51: return '0m'
    if tile_code == 52: return '0p'
    if tile_code == 53: return '0s'
    suits = {1: 'm', 2: 'p', 3: 's', 4: 'z'}
    suit = suits.get(tile_code // 10)
    num = tile_code % 10
    if suit and num is not None: return f"{num}{suit}"
    return f"UNK({tile_code})"

def string_to_tile_list(tile_str: str) -> list[str]:
    if not isinstance(tile_str, str) or len(tile_str) % 2 != 0: return []
    return [f"{tile_str[i]}{tile_str[i+1]}" for i in range(0, len(tile_str), 2)]

def parse_kyoku_index(kyoku_index: str):
    """
    "東1-0" のような文字列を
      wind   = "東"
      kyoku  = 1   (局番号)
      honba  = 0   (本場)
    に分解して返す。
    """
    wind = kyoku_index[0]            # '東','南','西','北'
    rest = kyoku_index[1:]           # "1-0"
    kyoku_str, honba_str = rest.split('-')
    return wind, int(kyoku_str), int(honba_str)

# =================================================================
# SECTION 2: ゲームログパーサー (ご指示に基づき修正)
# =================================================================
class GameLogParser:
    """
    天鳳の牌譜JSONログから特定の情報を抽出するパーサー。
    配牌と親の最初のツモ牌を取得する機能に特化。
    """
    def __init__(self, gamelog_json_string: str):
        """
        コンストラクタ。JSON文字列を読み込み、logデータを準備する。
        """
        if not gamelog_json_string or not isinstance(gamelog_json_string, str):
            raise ValueError("無効または空のJSONログ文字列が渡されました。")
        
        try:
            # 天鳳ログはシングルクオートを含むことがあるため、ダブルクオートに置換
            self.data = json.loads(gamelog_json_string.replace("'", "\""))
            # 実際のゲーム進行データは 'log' キーの最初の要素に格納されている
            self.log_data = self.data.get('log', [[]])[0]
        except (json.JSONDecodeError, IndexError) as e:
            raise ValueError(f"JSONログの解析に失敗しました: {e}")

    def get_kyoku_id(self) -> int:
        """
        局ID (0=東1, 1=東2, ..., 4=南1) を取得する。
        """
        if not self.log_data or len(self.log_data) < 1:
            # ログデータが不正な場合はデフォルトで東1局(0)を返すか、エラーを発生させる
            # ここでは処理継続を試みるため0を返す
            return 0
        # log_data[0][0] に局IDが格納されている
        return self.log_data[0][0]

    def get_initial_hands_as_codes(self) -> dict[int, list[int]]:
        """
        全プレイヤーの配牌を牌コードのリストとして取得する。
        戻り値: {0: [牌コード,...], 1: [牌コード,...], ...}
        """
        if not self.log_data or len(self.log_data) < 17: # 4人分の配牌データが存在するかチェック
            # データが不足している場合は、4人分の空のリストを返す
            return {i: [] for i in range(4)}
        
        # 配牌は log_data[4], log_data[7], log_data[10], log_data[13] に格納されている
        hands = {i: self.log_data[4 + i * 3] for i in range(4)}
        return hands

    def get_dealer_first_tsumo_code(self) -> int or None:
        """
        その局の親(ディーラー)の最初のツモ牌をコード形式で取得する。
        """
        if not self.log_data:
            return None
        
        # 局IDを4で割った余りが、その局の親のプレイヤーID (0-3) になる
        dealer_id = self.get_kyoku_id() % 4
        
        # 親のツモ牌リストのインデックスを計算 (配牌+1)
        draw_list_index = 5 + dealer_id * 3
        
        if len(self.log_data) <= draw_list_index:
            return None # ツモリスト自体が存在しない場合

        draw_list = self.log_data[draw_list_index]
        
        # ツモリストが空でなければ、最初の要素が最初のツモ牌
        return draw_list[0] if draw_list else None

# =================================================================
# SECTION 3: 牌山復元関数 (2つのうち、より堅牢なロジックを選択)
# =================================================================
def reconstruct_initial_wall(
    incomplete_wall_tiles: list[str],
    initial_hands_by_player: dict[int, list[int]],
    dealer_first_tsumo: int,
    dealer_id: int
) -> list[str]:
    """
    ご指示のチャンク単位ロジックで配牌ブロックを構築し、完全な牌山を復元する。
    """
    haipai_block = []
    haipai_block_code = []
    # pop(0)で先頭から抜き出すため、各手牌リストをコピーして使用
    player_hands_copy = {p: list(hand) for p, hand in initial_hands_by_player.items()}
    # 1. 4枚ずつのブロックを3回繰り返す
    for i in range(3):
         if dealer_id == 1:
             haipai_block.extend(player_hands_copy[0][:4])
             player_hands_copy[0] = player_hands_copy[0][4:]
             haipai_block.extend(player_hands_copy[1][:4])
             player_hands_copy[1] = player_hands_copy[1][4:]
             haipai_block.extend(player_hands_copy[2][:4])
             player_hands_copy[2] = player_hands_copy[2][4:]
             haipai_block.extend(player_hands_copy[3][:4])
             player_hands_copy[3] = player_hands_copy[3][4:]
         if dealer_id == 2:
             haipai_block.extend(player_hands_copy[1][:4])
             player_hands_copy[1] = player_hands_copy[1][4:]
             haipai_block.extend(player_hands_copy[2][:4])
             player_hands_copy[2] = player_hands_copy[2][4:]
             haipai_block.extend(player_hands_copy[3][:4])
             player_hands_copy[3] = player_hands_copy[3][4:]
             haipai_block.extend(player_hands_copy[0][:4])
             player_hands_copy[0] = player_hands_copy[0][4:]
         if dealer_id == 3:
             haipai_block.extend(player_hands_copy[2][:4])
             player_hands_copy[2] = player_hands_copy[2][4:]
             haipai_block.extend(player_hands_copy[3][:4])
             player_hands_copy[3] = player_hands_copy[3][4:]
             haipai_block.extend(player_hands_copy[0][:4])
             player_hands_copy[0] = player_hands_copy[0][4:]
             haipai_block.extend(player_hands_copy[1][:4])
             player_hands_copy[1] = player_hands_copy[1][4:]
         if dealer_id == 4:
             haipai_block.extend(player_hands_copy[3][:4])
             player_hands_copy[3] = player_hands_copy[3][4:]
             haipai_block.extend(player_hands_copy[0][:4])
             player_hands_copy[0] = player_hands_copy[0][4:]
             haipai_block.extend(player_hands_copy[1][:4])
             player_hands_copy[1] = player_hands_copy[1][4:]
             haipai_block.extend(player_hands_copy[2][:4])
             player_hands_copy[2] = player_hands_copy[2][4:]

     # 2. 最後の1枚を各プレイヤーから追加
    if dealer_id == 1:
         haipai_block.append(player_hands_copy[0].pop(0)) 
         haipai_block.append(player_hands_copy[1].pop(0)) 
         haipai_block.append(player_hands_copy[2].pop(0)) 
         haipai_block.append(player_hands_copy[3].pop(0)) 
    if dealer_id == 2:
         haipai_block.append(player_hands_copy[1].pop(0)) 
         haipai_block.append(player_hands_copy[2].pop(0)) 
         haipai_block.append(player_hands_copy[3].pop(0)) 
         haipai_block.append(player_hands_copy[0].pop(0)) 
    if dealer_id == 3:
         haipai_block.append(player_hands_copy[2].pop(0)) 
         haipai_block.append(player_hands_copy[3].pop(0)) 
         haipai_block.append(player_hands_copy[0].pop(0)) 
         haipai_block.append(player_hands_copy[1].pop(0)) 
    if dealer_id == 4:
         haipai_block.append(player_hands_copy[3].pop(0)) 
         haipai_block.append(player_hands_copy[0].pop(0)) 
         haipai_block.append(player_hands_copy[1].pop(0)) 
         haipai_block.append(player_hands_copy[2].pop(0)) 

    try:
        # 3. 復元した配牌ブロックと、残りの牌山を結合
        haipai_block.append(dealer_first_tsumo)
        haipai_block_code = [to_str(code) for code in haipai_block]
        full_initial_wall_codes = haipai_block_code + incomplete_wall_tiles
        #print(f"復元された牌山とカウント:{len(full_initial_wall_codes)} {full_initial_wall_codes}")

    except Exception as e:
        print(f"エラー{e}")
        exit
    
    # 5. 全てを文字列形式に変換して返却
    return full_initial_wall_codes

# =================================================================
# SECTION 4: メイン処理ブロック (変更なし)
# =================================================================
if __name__ == '__main__':
    INPUT_CSV_PATH = 'game_records_bak.csv'
    OUTPUT_CSV_PATH = 'game_records.csv'
    
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"エラー: ファイル '{INPUT_CSV_PATH}' が見つかりませんでした。")
        exit()

    reconstructed_walls = []
    
    print("\n牌山の復元処理を開始します...")
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            # CSVの 'tile_List' はゲーム終了時に残っていた牌山(不完全)
            incomplete_wall_str = row.get('tile_List', '')
            gamelog_json_str = row.get('gamelog_json', '')

            if not gamelog_json_str or pd.isna(gamelog_json_str):
                raise ValueError("gamelog_jsonが空です。")

            # === ここで修正したパーサーを使用 ===
            parser = GameLogParser(gamelog_json_str)
            wind,dealer_id,honba = parse_kyoku_index(row.get('game_ID', ''))

            # 全プレイヤーの配牌をコードで取得
            initial_hands_codes = parser.get_initial_hands_as_codes()
            
            # その局の「親」の最初のツモをコードで取得
            dealer_first_tsumo = parser.get_dealer_first_tsumo_code()
            # === ここまで ===

            incomplete_wall_list = string_to_tile_list(incomplete_wall_str)

            reconstructed_wall_list = reconstruct_initial_wall(
                incomplete_wall_tiles=incomplete_wall_list,
                initial_hands_by_player=initial_hands_codes,
                dealer_first_tsumo=dealer_first_tsumo,
                dealer_id=dealer_id
            )
            
            reconstructed_walls.append("".join(reconstructed_wall_list))

        except Exception as e:
            print(f"\n警告: {index}行目でエラーが発生しました。この行は更新されません。エラー: {e}")
            reconstructed_walls.append(row.get('tile_List', '')) # エラー時は元の値を維持

    df['tile_List'] = reconstructed_walls
    df.to_csv(OUTPUT_CSV_PATH, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8-sig')

    print(f"\n処理が完了しました。復元された牌山を含む新しいCSVファイル '{OUTPUT_CSV_PATH}' が作成されました。")