import json
import pandas as pd
from datetime import datetime

# --- ここからがメインの処理 ---
if __name__ == "__main__":
    with open('taglist7.txt', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # フラットなリストを格納するための空のリストを用意
    flattened_data = []

    # 各対局データをループ処理
    for game in data:
        start_ts = game.get("startTime")
        end_ts = game.get("endTime")

        # FIX: Convert integer timestamps (assuming seconds) to datetime objects and format them for display/storage
        start_time_formatted = None
        if isinstance(start_ts, int):
            # Convert seconds timestamp to datetime
            try:
                start_dt = datetime.fromtimestamp(start_ts) # Removed / 1000
                start_time_formatted = start_dt.strftime("%Y-%m-%d %H:%M:%S") # Format as string
            except (TypeError, ValueError) as e:
                #print(f"Warning: Could not convert startTime {start_ts} to datetime: {e}. Skipping game.")
                continue
        else:
            #print(f"Warning: Unexpected startTime format: {start_ts}. Skipping game.")
            continue

        end_time_formatted = None
        if isinstance(end_ts, int):
            # Convert seconds timestamp to datetime
            try:
                end_dt = datetime.fromtimestamp(end_ts) # Removed / 1000
                end_time_formatted = end_dt.strftime("%Y-%m-%d %H:%M:%S") # Format as string
            except (TypeError, ValueError) as e:
                 #print(f"Warning: Could not convert endTime {end_ts} to datetime: {e}. Skipping game.")
                 continue
        else:
             #print(f"Warning: Unexpected endTime format: {end_ts}. Skipping game.")
             continue


        # 対局ごとの共通情報を取得
        game_info = {
            "_id": game.get("_id"),
            "modeId": game.get("modeId"),
            "uuid": game.get("uuid"),
            "startTime": start_time_formatted, # Store formatted string
            "endTime": end_time_formatted,   # Store formatted string
        }

        # 対局内の各プレイヤー情報をループ処理
        for player in game.get("players", []):
            # 共通情報とプレイヤー情報を結合して1行のデータを作成
            row = game_info.copy()
            row.update({
                "accountId": player.get("accountId"),
                "nickname": player.get("nickname"),
                "level": player.get("level"),
                "score": player.get("score"),
                "gradingScore": player.get("gradingScore"),
            })
            flattened_data.append(row)

    # pandasのDataFrameを作成
    df = pd.DataFrame(flattened_data)

    # 列の順序を指定通りに整理
    column_order = [
        "_id", "modeId", "uuid", "startTime", "endTime",
        "accountId", "nickname", "level", "score", "gradingScore"
    ]
    df = df[column_order]

    # CSVファイルとして保存
    output_filename = 'game_data_flat.csv'
    df.to_csv(output_filename, index=False, encoding='utf-8-sig')

    print(f"✅ データは正常にフラット化され、'{output_filename}' に保存されました。")
    #print("\n--- CSVファイルの内容プレビュー ---")
    #print(df.to_string())