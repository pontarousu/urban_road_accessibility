"""
Demand Model v2 (Mesoscopic Model) - Step 1: グラフ上の単一パケット・ナビゲーション
ラスタ（ピクセル）からグラフ（ネットワーク）への転換実験

[理論]
1. osmnxを用いて道路ネットワークを有向グラフG(V,E)として取得する。
2. 出発地の座標と目的地の座標から、グラフ上の最寄りノード(Nearest Node)を特定する。
3. NetworkXに実装されたダイクストラ法を用いて、最短経路を算出する。
4. 計算された経路（エッジの列）をマップ上に可視化する。
"""

import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 日本語フォント設定
plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# パラメータ設定
# ==========================================
CENTER = (36.5613, 136.6562) # 金沢市中心部（香林坊など）
DIST   = 2000                # 分析半径 (2km)

def main():
    print("1. 路上ネットワーク（グラフ）のダウンロード...")
    # driveネットワークを取得。ラスタ法の時のような水域データの除外は不要。
    # グラフには最初から「車が通れない川や建物」は含まれていないからである。
    G = ox.graph_from_point(CENTER, dist=DIST, network_type='drive')
    
    # 距離計算のためにメートル単位の座標系に投影
    G_proj = ox.project_graph(G)
    
    print("2. 出発地 (Origin) と目的地 (Destination) ノードの特定...")
    # ノードの座標群を取り出す
    nodes_gdf = ox.graph_to_gdfs(G_proj, edges=False)
    minx, miny, maxx, maxy = nodes_gdf.total_bounds
    
    # 目的地（マップの中心 = 香林坊）の最寄りノード
    # G_proj.graph にはプロジェクションされた中心座標が入ることが多いが、安全のため境界の中央を計算
    center_x = (maxx + minx) / 2.0
    center_y = (maxy + miny) / 2.0
    dest_node = ox.distance.nearest_nodes(G_proj, center_x, center_y)
    
    # 出発地（南東の端っこ = 寺町周辺）の最寄りノード
    orig_x = minx + (maxx - minx) * 0.8
    orig_y = miny + (maxy - miny) * 0.2
    orig_node = ox.distance.nearest_nodes(G_proj, orig_x, orig_y)
    
    print(f"   => 出発ノード: {orig_node}")
    print(f"   => 目的ノード: {dest_node}")

    print("3. 最短経路（Dijkstra法）の探索...")
    # エッジの 'length' (長さ: m) をコスト（重み）として経路探索
    try:
        route = nx.shortest_path(G_proj, source=orig_node, target=dest_node, weight='length')
        # 通過するエッジの長さを合計して総距離を計算
        route_length = sum(G_proj.edges[u, v, 0].get('length', 0) for u, v in zip(route[:-1], route[1:]))
        print(f"   => 探索成功！ パケットは {len(route)} 個の交差点（ノード）を通過します。")
        print(f"   => 総移動距離: {route_length:.1f} m")
    except nx.NetworkXNoPath:
        print("   ❌ 到達不可能なノードの組み合わせです。（川などでネットワークが完全に分断されている）")
        return

    print("4. 結果のプロット...")

    fig, ax = ox.plot_graph_route(
        G_proj, 
        route, 
        route_color='crimson', 
        route_linewidth=1.5,      # ベースの道幅にピッタリ重ねるために細くする
        node_size=0, 
        bgcolor='white', 
        edge_color='lightgray',
        show=False, 
        close=False
    )
    
    # スタートとゴールにマーカーを追加
    orig_point = nodes_gdf.loc[orig_node]
    dest_point = nodes_gdf.loc[dest_node]
    ax.scatter(orig_point.geometry.x, orig_point.geometry.y, c='blue', s=50, zorder=6, label='出発地')
    ax.scatter(dest_point.geometry.x, dest_point.geometry.y, c='red', marker='*', s=150, zorder=6, label='目的地')
    
    ax.set_title('新しいStep 1: グラフ（メゾスコピック）モデル上でのパケットの最短経路', fontsize=14, color='black')
    
    out_path = '/Users/pontarousu/Q1zemi/demand_model_v2/graph_route_result.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved → {out_path}")

if __name__ == '__main__':
    main()
