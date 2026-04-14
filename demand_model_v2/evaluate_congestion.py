# [未使用] このファイルは Step 3 の静的検証プロットです。現行のシミュレーションは simulate_equilibrium.py を参照してください。
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import copy
from assign_capacity import calculate_capacity, apply_dynamic_weights

plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False

CENTER = (36.5613, 136.6562) # 金沢市中心部
DIST = 1000 # 少し広く取ることで迂回の余地（別のルート選択肢）を持たせる

def main():
    print("1. 路上ネットワーク（グラフ）のダウンロード (半径1000m)...")
    G = ox.graph_from_point(CENTER, dist=DIST, network_type='drive')
    G_proj = ox.project_graph(G)
    
    print("2. グラフの初期設定 (Capacity, Volume, 制限速度に基づくFree Flow Time)...")
    for u, v, k, data in G_proj.edges(keys=True, data=True):
        # 容量(Capacity)の推測とセット
        data['capacity'] = calculate_capacity(data)
        data['volume'] = 0.0
        
        # Free Flow Time (自由流状態での所要時間: s) = 距離(m) / 制限速度(m/s)
        length = data.get('length', 10.0)
        speed_kph = data.get('maxspeed', 40.0) # デフォルト40km/h
        if isinstance(speed_kph, list):
            speed_kph = speed_kph[0]
        try:
            speed_mps = float(speed_kph) * 1000 / 3600
        except ValueError:
            speed_mps = 40.0 * 1000 / 3600 # 数値変換できない場合のフォールバック
            
        data['free_flow_time'] = length / speed_mps
        
    # 初期重み(bpr_weight)の計算 (最初は皆volume=0なので free_flow_time と一致する)
    G_proj = apply_dynamic_weights(G_proj)
    
    # 比較検証のための出発・目的ノード設定 (南西から北東へ向かう対角線)
    nodes_gdf = ox.graph_to_gdfs(G_proj, edges=False)
    minx, miny, maxx, maxy = nodes_gdf.total_bounds
    start_node = ox.distance.nearest_nodes(G_proj, minx + (maxx-minx)*0.2, miny + (maxy-miny)*0.2)
    end_node = ox.distance.nearest_nodes(G_proj, maxx - (maxx-minx)*0.2, maxy - (maxy-miny)*0.2)
    
    print("3. 【交通量ゼロ状態】通常時の最短経路 (Path A) を探索...")
    route_A = nx.shortest_path(G_proj, start_node, end_node, weight='bpr_weight')
    cost_A_free = sum(G_proj.edges[u,v,0]['bpr_weight'] for u, v in zip(route_A[:-1], route_A[1:]))
    print(f"  => 最短ルート所要時間: {cost_A_free:.1f} 秒")
    
    print("4. 上記の通常ルート上に大量の交通需要 (Volume) を発生させる（大渋滞を起こす）...")
    # BPR関数の検証：キャパシティを大幅に超えるパケット(1500台)を流し込む
    MASSIVE_TRAFFIC = 1500
    for u, v in zip(route_A[:-1], route_A[1:]):
        G_proj.edges[u,v,0]['volume'] += MASSIVE_TRAFFIC
        
    print("5. BPR関数を用いてネットワーク全体の重み（所要時間ペナルティ）を再計算...")
    G_proj = apply_dynamic_weights(G_proj)
    
    cost_A_jammed = sum(G_proj.edges[u,v,0]['bpr_weight'] for u, v in zip(route_A[:-1], route_A[1:]))
    print(f"  => （参考）もし渋滞したPath Aをそのまま突っ切った場合の所要時間: {cost_A_jammed:.1f} 秒 に激増！")
    
    print("6. 【大渋滞発生下】後続パケットが渋滞を避けるための迂回ルート (Path B) を探索...")
    route_B = nx.shortest_path(G_proj, start_node, end_node, weight='bpr_weight')
    cost_B_jammed = sum(G_proj.edges[u,v,0]['bpr_weight'] for u, v in zip(route_B[:-1], route_B[1:]))
    print(f"  => 迂回ルートの所要時間: {cost_B_jammed:.1f} 秒 (渋滞に突っ込むよりはるかにマシ！)")
    
    print("7. 比較プロットの生成...")
    routes = [route_A, route_B]
    route_colors = ['none', 'dodgerblue']  # route_Aは後で赤ハイライトするためここでは一旦透明に近い色か後回しにする
    
    # plot_graph_routesで描画
    fig, ax = ox.plot_graph_routes(
        G_proj,
        routes,
        route_colors=['crimson', 'dodgerblue'],
        route_linewidth=3,
        edge_linewidth=1.0,
        node_size=0,
        bgcolor='white',
        edge_color='lightgray',
        show=False,
        close=False,
        figsize=(12, 12)
    )
    
    # 渋滞しているエッジ（volumeが大きなところ）を「太い赤マーカー」で背後にハイライト描画する
    for u, v in zip(route_A[:-1], route_A[1:]):
        edge_data = G_proj.edges[u,v,0]
        geom = edge_data.get('geometry')
        if geom:
            x, y = geom.xy
            ax.plot(x, y, color='red', linewidth=8, alpha=0.3, zorder=1)
        else:
            x = [G_proj.nodes[u]['x'], G_proj.nodes[v]['x']]
            y = [G_proj.nodes[u]['y'], G_proj.nodes[v]['y']]
            ax.plot(x, y, color='red', linewidth=8, alpha=0.3, zorder=1)
            
    # スタートとゴールのマーカー
    orig_point = G_proj.nodes[start_node]
    dest_point = G_proj.nodes[end_node]
    ax.scatter(orig_point['x'], orig_point['y'], c='blue', s=80, zorder=6, label='出発地')
    ax.scatter(dest_point['x'], dest_point['y'], c='purple', marker='*', s=200, zorder=6, label='目的地')
    
    # 凡例用のダミープロット
    ax.plot([], [], color='crimson', linewidth=3, label=f'本来の最短ルート (約 {cost_A_free/60:.1f}分)')
    ax.plot([], [], color='red', linewidth=8, alpha=0.3, label='キャパオーバーによる大渋滞発生')
    ax.plot([], [], color='dodgerblue', linewidth=3, label=f'渋滞回避の迂回ルート (約 {cost_B_jammed/60:.1f}分)')
    ax.legend(fontsize=12, loc='lower right', facecolor='white', framealpha=0.9)
    
    ax.set_title('Step 3: 渋滞ペナルティ(BPR関数)による自動迂回行動のシミュレーション\n(対象: 金沢市中心部 半径1km網)', fontsize=16)
    
    out_path = '/Users/pontarousu/Q1zemi/demand_model_v2/congestion_detour_result.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved → {out_path}")
    plt.close(fig)

if __name__ == '__main__':
    main()
