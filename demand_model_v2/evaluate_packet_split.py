import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from shapely.geometry import LineString

# 日本語フォント設定
plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False
# HTMLアニメーション出力の上限容量を拡張（デフォルト20MBで途切れる問題への対処）
plt.rcParams['animation.embed_limit'] = 50.0 

CENTER = (36.5613, 136.6562) # 金沢市中心部
DIST = 500 

def get_smooth_path_padded(G, route, target_frames, max_frames):
    """
    指定されたtarget_framesの数だけ等速で進み、その後は終点に留まる（max_framesまでパディングする）
    """
    if len(route) < 2:
        pt = (G.nodes[route[0]]['x'], G.nodes[route[0]]['y'])
        return [pt] * max_frames
        
    points = []
    for u, v in zip(route[:-1], route[1:]):
        edge_data = G.edges[u, v, 0]
        if 'geometry' in edge_data:
            coords = list(edge_data['geometry'].coords)
        else:
            coords = [(G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y'])]
        
        if len(points) == 0:
            points.extend(coords)
        else:
            points.extend(coords[1:])
            
    line = LineString(points)
    
    # target_frames で等分割して移動
    if target_frames > 1:
        distances = np.linspace(0, line.length, target_frames)
        path = [line.interpolate(d).coords[0] for d in distances]
    else:
        path = [line.coords[-1]]
        
    # 残りのフレーム（max_framesとの差分）はそのままゴール地点に留まる
    padding = [path[-1]] * (max_frames - len(path))
    return path + padding

def main():
    print("1. 路上ネットワーク（グラフ）のダウンロード (半径500m)...")
    G = ox.graph_from_point(CENTER, dist=DIST, network_type='drive')
    G_proj = ox.project_graph(G)
    
    nodes_gdf = ox.graph_to_gdfs(G_proj, edges=False)
    minx, miny, maxx, maxy = nodes_gdf.total_bounds
    
    # 真のフォークノードを算出するため、Start, EndA, EndB を定義
    start_node = ox.distance.nearest_nodes(G_proj, (maxx+minx)/2, miny + (maxy-miny)*0.1)
    end_a_node = ox.distance.nearest_nodes(G_proj, minx + (maxx-minx)*0.2, maxy - (maxy-miny)*0.1)
    end_b_node = ox.distance.nearest_nodes(G_proj, maxx - (maxx-minx)*0.2, maxy - (maxy-miny)*0.1)
    
    print("2. 経路探索と「真の分岐点 (True Fork)」の自動特定...")
    route_A = nx.shortest_path(G_proj, start_node, end_a_node, weight='length')
    route_B = nx.shortest_path(G_proj, start_node, end_b_node, weight='length')
    
    # Startから出発し、道が共通している間は1つのパケットとして扱う（被り判定）
    shared_path = []
    for u, v in zip(route_A, route_B):
        if u == v:
            shared_path.append(u)
        else:
            break
            
    true_fork_node = shared_path[-1] # ルートが枝分かれする直前のノード！
    
    route_stage1 = shared_path
    route_stage2A = route_A[route_A.index(true_fork_node):]
    route_stage2B = route_B[route_B.index(true_fork_node):]
    
    # 各ルートの物理距離をメートルで取得
    def get_len(r):
        return sum(G_proj.edges[u,v,0].get('length',0) for u,v in zip(r[:-1], r[1:]))

    dist_1 = get_len(route_stage1)
    dist_2A = get_len(route_stage2A)
    dist_2B = get_len(route_stage2B)
    
    # 速度を完全一定に保つ（例: 1フレームあたり10m）
    SPEED_PER_FRAME = 10.0
    FRAMES_1 = max(int(dist_1 / SPEED_PER_FRAME), 1)
    FRAMES_2A = max(int(dist_2A / SPEED_PER_FRAME), 1)
    FRAMES_2B = max(int(dist_2B / SPEED_PER_FRAME), 1)
    
    MAX_FRAMES_2 = max(FRAMES_2A, FRAMES_2B)
    
    print(f"3. アニメーション生成（一定速度:{SPEED_PER_FRAME}m/frame, 真の分岐ノード:{true_fork_node}）...")
    coords_stage1 = get_smooth_path_padded(G_proj, route_stage1, FRAMES_1, FRAMES_1)
    coords_stage2A = get_smooth_path_padded(G_proj, route_stage2A, FRAMES_2A, MAX_FRAMES_2)
    coords_stage2B = get_smooth_path_padded(G_proj, route_stage2B, FRAMES_2B, MAX_FRAMES_2)
    
    # プロット準備
    fig, ax = ox.plot_graph(G_proj, show=False, close=False, edge_linewidth=1.5, edge_color='lightgray', node_size=0, bgcolor='white', figsize=(10,10))
    ax.set_title('メゾスコピックモデルにおける交通需要（パケット）の分岐シミュレーション\n(対象エリア: 金沢市 香林坊周辺)', fontsize=16)
    
    # スタート・ゴールノードのスタティック描画
    ax.scatter([G_proj.nodes[start_node]['x']], [G_proj.nodes[start_node]['y']], c='blue', s=100, marker='o', label='Start')
    ax.scatter([G_proj.nodes[end_a_node]['x']], [G_proj.nodes[end_a_node]['y']], c='green', s=200, marker='*', label='End A')
    ax.scatter([G_proj.nodes[end_b_node]['x']], [G_proj.nodes[end_b_node]['y']], c='purple', s=200, marker='*', label='End B')
    ax.legend(loc='lower right')
    
    # 需要データ
    TOTAL_USERS = 100
    RATIO_A = 0.6
    RATIO_B = 0.4
    SCALE = 7 
    
    scatter = ax.scatter([], [], c='crimson', alpha=0.8, edgecolors='darkred', zorder=10)
    # 動的に数値を表示するためのテキストオブジェクト
    text1 = ax.text(-1, -1, "", color='white', fontsize=11, ha='center', va='center', fontweight='bold', zorder=11)
    text2A = ax.text(-1, -1, "", color='white', fontsize=11, ha='center', va='center', fontweight='bold', zorder=11)
    text2B = ax.text(-1, -1, "", color='white', fontsize=11, ha='center', va='center', fontweight='bold', zorder=11)
    
    def update(frame):
        if frame < FRAMES_1:
            x, y = coords_stage1[frame]
            scatter.set_offsets([[x, y]])
            scatter.set_sizes([TOTAL_USERS * SCALE])
            text1.set_position((x, y))
            text1.set_text(str(int(TOTAL_USERS)))
            text2A.set_text("")
            text2B.set_text("")
        else:
            sub_frame = frame - FRAMES_1
            axA, ayA = coords_stage2A[sub_frame]
            axB, ayB = coords_stage2B[sub_frame]
            scatter.set_offsets([[axA, ayA], [axB, ayB]])
            scatter.set_sizes([(TOTAL_USERS * RATIO_A) * SCALE, (TOTAL_USERS * RATIO_B) * SCALE])
            
            text1.set_text("")
            text2A.set_position((axA, ayA))
            text2A.set_text(str(int(TOTAL_USERS * RATIO_A)))
            text2B.set_position((axB, ayB))
            text2B.set_text(str(int(TOTAL_USERS * RATIO_B)))
            
        return scatter, text1, text2A, text2B
        
    print("4. レンダリング開始...")
    ani = animation.FuncAnimation(fig, update, frames=FRAMES_1 + MAX_FRAMES_2, interval=50, blit=True)
    
    # GIFとして保存
    gif_path = '/Users/pontarousu/Q1zemi/demand_model_v2/packet_split.gif'
    ani.save(gif_path, writer=animation.PillowWriter(fps=20))
    print(f"Saved GIF → {gif_path}")
    
    # インタラクティブHTML
    html_path = '/Users/pontarousu/Q1zemi/demand_model_v2/packet_split_interactive.html'
    jshtml = ani.to_jshtml()
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(jshtml)
    print(f"Saved HTML → {html_path}")
    
    plt.close(fig)

if __name__ == '__main__':
    main()
