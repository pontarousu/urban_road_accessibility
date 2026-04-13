"""
Phase 3 (Experiment): evaluate_potential_flow.py
大域的ポテンシャル場における単一パケットの流体力学的な経路探索シミュレーション

[理論]
1. 目的地を起点として、コスト・ラスタ(Cost Raster)上でFast Marchingを実行する。
2. 計算された「目的地への到達コスト表面」は、局所解を持たない大域的なポテンシャル場となる（スリバチ状の斜面）。
3. ユーザーパケットは斜面の傾き（勾配ベクトル）をリアルタイムに計算し、最も急な下り坂（最急降下方向）へ向かって転がり落ちる。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import osmnx as ox
import rasterio.transform
from skimage.graph import MCP_Geometric

# v2フォルダ内にある自作の汎用スキルモジュールをインポート
import urban_network_tools as unt

# 日本語フォント設定
plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# パラメータ設定
# ==========================================
CENTER = (36.5613, 136.6562) # 金沢市中心部（香林坊など）
DIST   = 2000                # 分析半径 (高速化のために2kmに縮小)
RES    = 10                  # ピクセル解像度(m)

# シミュレーション・パラメータ
STEP_SIZE = 1.0     # 1回の更新で進むピクセル数（滑らかに動かすための粒度）
MAX_STEPS = 5000    # 到達できなかった場合の安全装置

# ==========================================
# 関数の定義
# ==========================================
def get_lowest_neighbor(costs, r, c):
    """
    現在地(r, c)の周囲8ピクセルの中で、最もポテンシャル（コスト）が低い座標を返す。
    """
    h, w = costs.shape
    r0, c0 = int(round(r)), int(round(c))
    
    best_r, best_c = r0, c0
    min_cost = costs[r0, c0]
    
    # 8方向を探索
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
                
            nr, nc = r0 + dr, c0 + dc
            if 0 <= nr < h and 0 <= nc < w:
                if costs[nr, nc] < min_cost:
                    min_cost = costs[nr, nc]
                    best_r, best_c = nr, nc
                    
    return best_r, best_c

def simulate_packet_descent_discrete(cost_surface, start_r, start_c, dest_r, dest_c):
    """
    目的地(Goal)から計算されたポテンシャル場(cost_surface)の上を、
    8方向の最も低いマスを選びながらパケットが滑り落ちる。
    """
    trajectory_r = [start_r]
    trajectory_c = [start_c]
    energy_consumed = 0.0
    
    current_r, current_c = start_r, start_c
    
    for _ in range(MAX_STEPS):
        # 目的地付近（2ピクセル以内）に到達したら終わり
        dist_to_goal = np.sqrt((current_r - dest_r)**2 + (current_c - dest_c)**2)
        if dist_to_goal <= 2.0:
            print("🚀 ゴールに到達しました！")
            break
            
        # 最も低い隣接マスを取得
        next_r, next_c = get_lowest_neighbor(cost_surface, current_r, current_c)
        
        # どの方向にも下れない（スタックした）場合
        if next_r == current_r and next_c == current_c:
            print("⚠️ 周囲にこれ以上低い場所がありません。局所解でスタックしました。")
            break
            
        current_r, current_c = float(next_r), float(next_c)
        
        # 移動距離ベースのエネルギー消費
        energy_consumed += 1.0
        
        trajectory_r.append(current_r)
        trajectory_c.append(current_c)
        
    return np.array(trajectory_r), np.array(trajectory_c), energy_consumed


# ==========================================
# メイン処理
# ==========================================
def main():
    print("1. 路上ネットワークと水域のダウンロード...")
    G = ox.graph_from_point(CENTER, dist=DIST, network_type='drive')
    G_proj = ox.project_graph(G)
    nodes_proj, edges_proj = ox.graph_to_gdfs(G_proj)
    
    try:
        water = ox.features_from_point(CENTER, tags={'natural': 'water', 'waterway': 'riverbank'}, dist=DIST)
        water_proj = water.to_crs(G_proj.graph['crs']) if len(water) > 0 else None
    except Exception:
        water_proj = None

    print("2. マスター・コストラスタの生成（作っておいたモジュールが活躍！）...")
    bounds = edges_proj.total_bounds
    master_cost, transform = unt.build_cost_surface(edges_proj, water_proj, bounds, RES)
    
    print("3. 目的地 (Destination) を設定し、大域的ポテンシャル場を構築...")
    # 中心点（香林坊のど真ん中）を目的地とする
    center_px_x = (bounds[2] + bounds[0]) / 2.0
    center_px_y = (bounds[3] + bounds[1]) / 2.0
    col_d, row_d = ~transform * (center_px_x, center_px_y)
    dest_r, dest_c = int(row_d), int(col_d)
    
    # 目的地から波及する最小コスト計算（これが即ち大域的ポテンシャル場！）
    mcp = MCP_Geometric(master_cost)
    potential_surface, _ = mcp.find_costs(starts=[(dest_r, dest_c)])
    
    print("4. 出発地 (Origin) を設定し、パケットを流し込む...")
    # 南東にある山間部や川の近くをスタート地点にする（少し離れた場所）
    start_x = bounds[0] + (bounds[2] - bounds[0]) * 0.8
    start_y = bounds[1] + (bounds[3] - bounds[1]) * 0.2
    col_s, row_s = ~transform * (start_x, start_y)
    start_r, start_c = float(row_s), float(col_s)
    
    # シミュレーション実行
    traj_r, traj_c, energy = simulate_packet_descent_discrete(potential_surface, start_r, start_c, dest_r, dest_c)
    print(f"   => シミュレーション終了: {len(traj_r)} steps. 消費エネルギー: {energy:.1f}")

    # ピクセル座標を実際の地理空間座標（メートル）に変換
    traj_x, traj_y = rasterio.transform.xy(transform, traj_r, traj_c)
    
    print("5. プロットして可視化...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    
    # 描画範囲 (地理空間座標)
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
    
    # ==========================
    # 左図：大域的ポテンシャル場
    # ==========================
    plot_pot = np.copy(potential_surface)
    plot_pot[plot_pot > 9999] = np.nan
    
    im = ax1.imshow(plot_pot, cmap='viridis_r', origin='upper', alpha=0.9, extent=extent)
    plt.colorbar(im, ax=ax1, label='Potential Energy (Time Cost)', fraction=0.046, pad=0.04)
    
    # 軌跡と点の描画
    ax1.plot(traj_x, traj_y, color='crimson', linewidth=3, zorder=5, label='パケット軌跡')
    ax1.scatter([start_x], [start_y], c='blue', s=150, zorder=6, label='出発地')
    ax1.scatter([bounds[0] + (bounds[2]-bounds[0])/2], [bounds[1] + (bounds[3]-bounds[1])/2], c='red', marker='*', s=300, zorder=6, label='目的地')
    
    ax1.set_title('左図: 大域的ポテンシャル場上の斜面降下', fontsize=18)
    ax1.legend()
    ax1.axis('off')
    
    # ==========================
    # 右図：実際の道路・障害物ネットワーク
    # ==========================
    # 道路エッジを描画
    edges_proj.plot(ax=ax2, color='gray', linewidth=0.5, alpha=0.6)
    
    # 水域（障害物）を描画
    if water_proj is not None:
        water_proj.plot(ax=ax2, color='lightblue', alpha=1.0)
        
    # 軌跡と点の描画（透明度を上げて道が見えるようにする）
    ax2.plot(traj_x, traj_y, color='crimson', linewidth=5, zorder=5, alpha=0.5, label='パケット軌跡 (流体力学)')
    ax2.scatter([start_x], [start_y], c='blue', s=150, zorder=6, label='出発地')
    ax2.scatter([bounds[0] + (bounds[2]-bounds[0])/2], [bounds[1] + (bounds[3]-bounds[1])/2], c='red', marker='*', s=300, zorder=6, label='目的地')
    
    ax2.set_xlim(bounds[0], bounds[2])
    ax2.set_ylim(bounds[1], bounds[3])
    ax2.set_title('右図: 実際の現実空間ネットワーク上の経路', fontsize=18)
    ax2.legend()
    ax2.axis('off')
    
    out_path = '/Users/pontarousu/Q1zemi/demand_model_v2/potential_descent_result.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved → {out_path}")

if __name__ == '__main__':
    main()
