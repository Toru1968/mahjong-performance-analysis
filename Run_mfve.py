import pandas as pd
from io import StringIO

def run_mfve(metrics_csv_data, config):
    """
    指標値リストと設定に基づき、オンライン麻雀の公平性を総合的に判定するエンジン。

    Args:
        metrics_csv_data (str): 論文の表1に相当する指標値リストのCSV形式文字列。
        config (dict): 判定基準を定義した設定。

    Returns:
        dict: 判定結果と詳細を含む辞書。
    """
    metrics_df = pd.read_csv(StringIO(metrics_csv_data))
    alpha = config['alpha']

    # --- モジュールA, B, C の評価 ---
    # p値を持つ全てのメトリクスを抽出し、有意水準と比較
    p_value_metrics = metrics_df.dropna(subset=['p_value'])
    abc_results = {}
    abc_passed = True
    for _, row in p_value_metrics.iterrows():
        metric = row['metric_name']
        p_value = row['p_value']
        is_pass = p_value > alpha
        abc_results[metric] = {'p_value': p_value, 'passed': is_pass}
        if not is_pass:
            abc_passed = False

    # C判定: モジュールA, B, Cのいずれかが不合格の場合
    if not abc_passed:
        return {
            'rating': 'C',
            'title': '要改善',
            'reason': '確率生成プロセスにバイアスが存在する可能性が示唆されました。',
            'details': abc_results
        }

    # --- モジュールD の評価 ---
    hand_conf = config['module_d']['vpc_hand']
    player_conf = config['module_d']['vpc_player']

    hand_val = metrics_df.loc[metrics_df['metric_name'] == 'vpc_hand_estimate', 'value'].iloc[0]
    player_val = metrics_df.loc[metrics_df['metric_name'] == 'vpc_player_estimate', 'value'].iloc[0]

    details_d = {
        'VPC_Hand (運)': {'value': hand_val, 'config': hand_conf},
        'VPC_Player (実力)': {'value': player_val, 'config': player_conf}
    }

    # A判定: 厳密基準を満たすか
    hand_in_strict = hand_conf['strict_range'][0] <= hand_val <= hand_conf['strict_range'][1]
    player_in_strict = player_conf['strict_range'][0] <= player_val <= player_conf['strict_range'][1]
    if hand_in_strict and player_in_strict:
        return {
            'rating': 'A',
            'title': '極めて公平',
            'reason': '確率生成は健全であり、ゲームバランスも理想的な範囲内です。',
            'details': details_d
        }

    # B判定: 緩和基準を満たすか
    hand_in_relaxed = hand_conf['relaxed_range'][0] <= hand_val <= hand_conf['relaxed_range'][1]
    player_in_relaxed = player_conf['relaxed_range'][0] <= player_val <= player_conf['relaxed_range'][1]
    if hand_in_relaxed and player_in_relaxed:
        return {
            'rating': 'B',
            'title': '実用上公平',
            'reason': '確率生成は健全であり、ゲームバランスも許容範囲内です。',
            'details': details_d
        }

    # D判定: 緩和基準からも逸脱
    return {
        'rating': 'D',
        'title': '不公平',
        'reason': '確率操作の痕跡はありませんが、ゲームバランスが著しく損なわれています。',
        'details': details_d
    }


def generate_report(result):
    """判定結果を整形してレポートとして出力する"""
    report = f"""
==================================================
  Mahjong Fairness Verification Engine (MFVE)
            - 総合評価レポート -
==================================================

[ 総合判定 ] {result['rating']}判定 ({result['title']})

[ 判定理由 ]
{result['reason']}

--------------------------------------------------
[ 詳細データ ]
"""
    if result['rating'] == 'C':
        report += "▼ 確率生成プロセスの検定 (モジュールA, B, C)\n"
        for metric, res in result['details'].items():
            status = "✓ 合格" if res['passed'] else "✗ 不合格"
            report += f"  - {metric:<25} p値={res['p_value']:.4f}  -> {status}\n"
    else:
        report += "▼ 確率生成プロセスの検定 (モジュールA, B, C)\n  - 全ての項目で統計的有意差は認められず、問題ありませんでした。\n\n"
        report += "▼ ゲームバランスの検定 (モジュールD)\n"
        for component, data in result['details'].items():
            value = data['value']
            conf = data['config']
            status_strict = "✓" if conf['strict_range'][0] <= value <= conf['strict_range'][1] else "✗"
            status_relaxed = "✓" if conf['relaxed_range'][0] <= value <= conf['relaxed_range'][1] else "✗"
            report += (
                f"  - {component:<20} 実測値: {value:.2%}\n"
                f"    - 厳密基準 [{conf['strict_range'][0]:.0%}, {conf['strict_range'][1]:.0%}] -> {status_strict}\n"
                f"    - 緩和基準 [{conf['relaxed_range'][0]:.0%}, {conf['relaxed_range'][1]:.0%}] -> {status_relaxed}\n"
            )

    report += "=================================================="
    return report

# --- メイン処理 ---

# 1. 判定基準の定義
config_data = {
    "alpha": 0.01,
    "module_d": {
        "vpc_hand": {
            "strict_range": [0.30, 0.45],
            "relaxed_range": [0.29, 0.46]
        },
        "vpc_player": {
            "strict_range": [0.40, 0.55],
            "relaxed_range": [0.39, 0.56]
        }
    }
}

# 2. 論文のケーススタディの指標値リストを入力
# (本来は外部CSVファイルから読み込む)
metrics_input = """metric_name,value,ci_lower,ci_upper,p_value
hqs_ks_statistic,0.0012,,0.085,
shanten_chi2_statistic,12.5,,0.091,
tsumo_acf_lag,-0.0005,,0.452,
tsumo_runs_z_score,0.89,,0.373,
glmm_rate_coefficient,0.0001,-0.0002,0.0004,0.812
glmm_seat_coefficient,-0.0003,-0.0008,0.0002,0.215
vpc_hand_estimate,0.47,0.46,0.48,
vpc_player_estimate,0.38,0.37,0.39,
"""

# 3. 判定エンジンを実行
final_result = run_mfve(metrics_input, config_data)

# 4. レポートを生成して表示
report = generate_report(final_result)
print(report)