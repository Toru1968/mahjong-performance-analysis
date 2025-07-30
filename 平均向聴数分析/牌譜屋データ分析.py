import requests
import json
import csv
import time
from datetime import datetime

def format_string_to_unix_timestamp(date_string: str) -> int:
    """
    Formats a date string in 'YYYY-MM-DD HH:MM:SS' format to a Unix timestamp (seconds).

    Args:
        date_string: The date and time string to convert.

    Returns:
        The Unix timestamp as an integer, or None if the string format is incorrect.
    """
    try:
        # Parse the string into a datetime object
        dt_object = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
        # Get the Unix timestamp (float) and convert to integer
        unix_timestamp = int(dt_object.timestamp())
        return unix_timestamp
    except (ValueError, TypeError) as e:
        print(f"Error converting string '{date_string}': {e}")
        return None

def get_player_stats(player_id: int, start_date: str, end_date: str, mode: int = 12):
    """
    雀魂牌譜屋のAPIから指定したプレイヤーの集計済み統計データを取得する。

    :param player_id: プレイヤーID
    :param start_date: 統計開始日 (例: "2020-01-01")
    :param end_date: 統計終了日 (例: "2025-07-26")
    :param mode: ゲームモード (玉の間など)
    :return: 統計データの辞書、またはエラー時にNone
    """
    # 日付文字列をミリ秒単位のUnixタイムスタンプに変換
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    # 終了日はその日の終わりまでを含む
    end_ts = int(datetime.strptime(f"{end_date} 23:59:59", "%Y-%m-%d %H:%M:%S").timestamp() * 1000)

    # APIのURLを構築
    # 注: tagパラメータは変更される可能性があるため、ここでは省略するか、必要に応じて指定します
    url = f"https://5-data.amae-koromo.com/api/v2/pl4/player_extended_stats/{player_id}/{start_ts}/{end_ts}?mode={mode}"

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()  # HTTPエラーがあれば例外を発生させる
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"エラーが発生しました: {e}")
        return None

# --- ここからがメインの処理 ---
if __name__ == "__main__":
    input_csv_file = 'game_data_flat.csv'
    try:
        df_input = pd.read_csv(input_csv_file)
        
        # accountId, nickname, level の3つの列で、重複しないプレイヤーのリストを作成
        unique_players_df = df_input[['accountId', 'modeId','nickname', 'level']].drop_duplicates(subset=['accountId']).reset_index(drop=True)
        
        print(f"'{input_csv_file}' から {len(unique_players_df)} 人のユニークなプレイヤー情報を抽出しました。")
    except Exception as e:
        print(f"'{input_csv_file}' の読み込みでエラーが発生しました: {e}")
        unique_players_df = pd.DataFrame() # エラー時は空のDataFrameを作成

# この後の処理（APIを呼び出すループなど）で、
# 作成された `unique_player_ids` リストをそのまま使用できます。

if input_csv_file:
    all_stats_data = {}
    period_start = "2010-01-01"
    period_end = "2025-07-26"
    mode_id = 0
    print("\n各プレイヤーの統計データ取得を開始します...")
    for index, player_row in unique_players_df.iterrows():
        player_id = player_row['accountId']
        nickname = player_row['nickname']
        level = player_row['level']
        mode_id = player_row['modeId']

        # APIを呼び出して統計データを取得
        stats = get_player_stats(player_id, period_start, period_end, mode_id)

        if stats:
            # FIX 2: Use the player_id variable as the key
            all_stats_data[player_id] = {
                    'nickname': nickname,
                    'level': level,
                    'data': stats
                    }
        time.sleep(1)

    print(f"\n分析を開始します...")
    headers = [
    'player_id',
    'nickname',
    'level',    
    '対局数',
    '和了率',
    'ツモ率',
    'ダマ率',
    '放銃率',
    '副露率',
    '立直率',
    '平均和了点',
    '最大連荘数',
    '平均和了巡目',
    '平均放銃点',
    '流局率',
    '流局時テンパイ率',
    '一発率',
    '裏ドラ率',
    '被親倍満以上率', # 「被炸率」は一般的に満貫以上の手に放銃した割合を指すことが多いです
    '平均被弾点数',   # 「平均被炸点数」は満貫以上の手に放銃した際の平均失点
    '放銃時立直率',
    '放銃時副露率',
    '立直後放銃率',
    '立直後非即時放銃率',
    '副露後放銃率',
    '立直後和了率',
    '副露後和了率',
    '立直後流局率',
    '副露後流局率',
    '立直への放銃回数',
    '副露への放銃回数',
    'ダマへの放銃回数',
    '立直和了回数',
    '副露和了回数',
    'ダマ和了回数',
    '平均立直巡目',
    '立直収支',
    '立直収入',
    '立直支出',
    '先制立直率',
    '追っかけ立直率',
    '被追っかけ立直率',
    'フリテン立直率',
    '立直時良形率',
    '立直時多面待ち率',
    '立直時良形率2',
    '役満回数',
    '最大合計翻数',
    'ダブル立直回数',
    '打点効率',
    '放銃損失',
    '純打点効率',
    '平均配牌シャンテン数',
    '親番平均配牌シャンテン数',
    '子番平均配牌シャンテン数'
    ]
    output_file = 'player_stats.csv'
    print(f"\n全データの取得が完了しました。'{output_file}' に書き出します...")
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for player_id, player_stats in all_stats_data.items():
            basic = player_stats.get('data', {})
            writer.writerow({
                'player_id': player_id,
                'nickname': player_stats.get('nickname'),
                'level': player_stats.get('level'),
                '対局数': basic.get('count'),
                '和了率': basic.get('和牌率'),
                'ツモ率': basic.get('自摸率'),
                'ダマ率': basic.get('默听率'),
                '放銃率': basic.get('放铳率'),
                '副露率': basic.get('副露率'),
                '立直率': basic.get('立直率'),
                '平均和了点': basic.get('平均打点'),
                '最大連荘数': basic.get('最大连庄'),
                '平均和了巡目': basic.get('和了巡数'),
                '平均放銃点': basic.get('平均铳点'),
                '流局率': basic.get('流局率'),
                '流局時テンパイ率': basic.get('流听率'),
                '一発率': basic.get('一发率'),
                '裏ドラ率': basic.get('里宝率'),
                '被親倍満以上率': basic.get('被炸率'),
                '平均被弾点数': basic.get('平均被炸点数'),
                '放銃時立直率': basic.get('放铳时立直率'),
                '放銃時副露率': basic.get('放铳时副露率'),
                '立直後放銃率': basic.get('立直后放铳率'),
                '立直後非即時放銃率': basic.get('立直后非瞬间放铳率'),
                '副露後放銃率': basic.get('副露后放铳率'),
                '立直後和了率': basic.get('立直后和牌率'),
                '副露後和了率': basic.get('副露后和牌率'),
                '立直後流局率': basic.get('立直后流局率'),
                '副露後流局率': basic.get('副露后流局率'),
                '立直への放銃回数': basic.get('放铳至立直'),
                '副露への放銃回数': basic.get('放铳至副露'),
                'ダマへの放銃回数': basic.get('放铳至默听'),
                '立直和了回数': basic.get('立直和了'),
                '副露和了回数': basic.get('副露和了'),
                'ダマ和了回数': basic.get('默听和了'),
                '平均立直巡目': basic.get('立直巡目'),
                '立直収支': basic.get('立直收支'),
                '立直収入': basic.get('立直收入'),
                '立直支出': basic.get('立直支出'),
                '先制立直率': basic.get('先制率'),
                '追っかけ立直率': basic.get('追立率'),
                '被追っかけ立直率': basic.get('被追率'),
                'フリテン立直率': basic.get('振听立直率'),
                '立直時良形率': basic.get('立直好型'),
                '立直時多面待ち率': basic.get('立直多面'),
                '立直時良形率2': basic.get('立直好型2'),
                '役満回数': basic.get('役满'),
                '最大合計翻数': basic.get('最大累计番数'),
                'ダブル立直回数': basic.get('W立直'),
                '打点効率': basic.get('打点效率'),
                '放銃損失': basic.get('铳点损失'),
                '純打点効率': basic.get('净打点效率'),
                '平均配牌シャンテン数': basic.get('平均起手向听'),
                '親番平均配牌シャンテン数': basic.get('平均起手向听亲'),
                '子番平均配牌シャンテン数': basic.get('平均起手向听子'),
            })

    print("書き出しが完了しました。")
else:
    print("CSVに出力するデータがありませんでした。")