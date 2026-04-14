# [未使用] このファイルは simulate_equilibrium.py に置き換えられました（2026-04-14）。詳細は theory_discussion_log.md §4 を参照。
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as pe
import numpy as np
import imageio_ffmpeg
from shapely.geometry import LineString
from assign_capacity import calculate_capacity, apply_dynamic_weights

plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()

CENTER = (36.5613, 136.6562)
DIST = 1000

def get_smooth_path_padded(G, route, target_frames, max_frames):
    if len(route) < 2:
        return [(G.nodes[route[0]]['x'], G.nodes[route[0]]['y'])] * max_frames
        
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
    if target_frames > 1:
        distances = np.linspace(0, line.length, target_frames)
        path = [line.interpolate(d).coords[0] for d in distances]
    else:
        path = [line.coords[-1]]
    padding = [path[-1]] * (max_frames - len(path))
    return path + padding

def main():
    print("1. グラフロードとCapacityセットアップ...")
    G = ox.graph_from_point(CENTER, dist=DIST, network_type='drive')
    G_proj = ox.project_graph(G)
    
    for u, v, k, data in G_proj.edges(keys=True, data=True):
        data['capacity'] = calculate_capacity(data)
        data['volume'] = 0.0
        length = data.get('length', 10.0)
        speed_kph = data.get('maxspeed', 40.0)
        if isinstance(speed_kph, list): speed_kph = speed_kph[0]
        try: speed_mps = float(speed_kph) * 1000 / 3600
        except: speed_mps = 40.0 * 1000 / 3600
        data['free_flow_time'] = length / speed_mps
        
    G_proj = apply_dynamic_weights(G_proj)
    
    nodes_gdf = ox.graph_to_gdfs(G_proj, edges=False)
    minx, miny, maxx, maxy = nodes_gdf.total_bounds
    start_node = ox.distance.nearest_nodes(G_proj, minx + (maxx-minx)*0.2, miny + (maxy-miny)*0.2)
    end_node = ox.distance.nearest_nodes(G_proj, maxx - (maxx-minx)*0.2, maxy - (maxy-miny)*0.2)
    
    print("2. 最初は全員（1500人）が1つの塊として最短ルートを進む...")
    TOTAL_TRAFFIC = 1500
    route_origin = nx.shortest_path(G_proj, start_node, end_node, weight='free_flow_time')
    
    # 意図的に開始から25%進んだノードを「ボトルネックの交差点（分裂地点: Node X）」に指定する
    split_idx = min(len(route_origin)-1, max(1, len(route_origin)//4))
    split_node = route_origin[split_idx]
    
    # スタートからNode X までの「共通の道のり」
    common_route = route_origin[:split_idx+1]
    
    print(f" => 交差点 (Node ID: {split_node}) に到達。ここでミクロな意思決定の集約（分裂）が起こる！")
    
    # 3. Node X から先の最短ルート（本来のルート）への評価
    route_A = nx.shortest_path(G_proj, split_node, end_node, weight='free_flow_time')
    # 本来のルートの最初の道のキャパシティを確認
    next_node = route_A[1] if len(route_A) > 1 else route_A[0]
    edge_cap = G_proj.edges[split_node, next_node, 0]['capacity'] if split_node != next_node else 1000
    
    # 先頭の500人はそのまま進入、残りの1000人は「渋滞を察知して迂回」するようグループ分け
    TRAFFIC_A = min(int(edge_cap), 500)
    TRAFFIC_B = TOTAL_TRAFFIC - TRAFFIC_A
    
    # 迂回路（route_B）を確実に発生させるため、仮想的にルートAを大渋滞(Volume=5000)状態にしてDijkstraを走らせる
    for u, v in zip(route_A[:-1], route_A[1:]):
        G_proj.edges[u,v,0]['volume'] += 5000 
    G_proj = apply_dynamic_weights(G_proj)
    
    route_B = nx.shortest_path(G_proj, split_node, end_node, weight='bpr_weight')
    
    # ダミーの渋滞を取り除き、実際の物理量(TRAFFIC_A/B)をあてがい、正しい所要時間を算出する
    for u, v in zip(route_A[:-1], route_A[1:]):
        G_proj.edges[u,v,0]['volume'] = TRAFFIC_A
    for u, v in zip(route_B[:-1], route_B[1:]):
        G_proj.edges[u,v,0]['volume'] = TRAFFIC_B
    G_proj = apply_dynamic_weights(G_proj)
    
    print("4. フレーム数の算出（Mid-trip Splittingアニメーション用）...")
    FRAMES_COMMON = 50 # 交差点までの到達フレーム
    path_common = get_smooth_path_padded(G_proj, common_route, FRAMES_COMMON, FRAMES_COMMON)
    
    cost_A = sum(G_proj.edges[u,v,0]['bpr_weight'] for u, v in zip(route_A[:-1], route_A[1:]))
    cost_B = sum(G_proj.edges[u,v,0]['bpr_weight'] for u, v in zip(route_B[:-1], route_B[1:]))
    
    BASE_FRAMES = 120
    FRAMES_A = max(int(BASE_FRAMES * (cost_A / min(cost_A, cost_B))), 1)
    FRAMES_B = max(int(BASE_FRAMES * (cost_B / min(cost_A, cost_B))), 1)
    frames_after_split = max(FRAMES_A, FRAMES_B)
    
    MAX_FRAMES = FRAMES_COMMON + frames_after_split
    
    path_A = get_smooth_path_padded(G_proj, route_A, FRAMES_A, frames_after_split)
    path_B = get_smooth_path_padded(G_proj, route_B, FRAMES_B, frames_after_split)
    
    # 座標マッピング
    coords_main = path_common + [path_common[-1]] * frames_after_split
    coords_A = [path_common[0]] * FRAMES_COMMON + path_A
    coords_B = [path_common[0]] * FRAMES_COMMON + path_B
    
    print("5. プロットとアニメーションのセットアップ...")
    fig, ax = ox.plot_graph(G_proj, show=False, close=False, edge_linewidth=1.0, edge_color='lightgray', node_size=0, bgcolor='white', figsize=(10,10))
    ax.set_title("Step 3.5: 「ミクロな個人の意思決定」の総和としてのメゾスコピック自発分裂\n交差点で『混んでるな』と悟った後続組が一斉に迂回を始める", fontsize=14)
    
    orig = G_proj.nodes[start_node]
    dest = G_proj.nodes[end_node]
    split_node_data = G_proj.nodes[split_node]
    ax.scatter([orig['x']], [orig['y']], c='green', s=200, zorder=6, label='START')
    ax.scatter([split_node_data['x']], [split_node_data['y']], c='yellow', marker='X', s=300, zorder=6, label='SPLIT NODE (ボトルネック交差点)')
    ax.scatter([dest['x']], [dest['y']], c='purple', marker='*', s=400, zorder=6, label='GOAL')
    ax.legend(fontsize=11, loc='lower right', facecolor='white', framealpha=0.9)

    trail_main, = ax.plot([], [], color='darkviolet', linewidth=8, alpha=0.5, zorder=1)
    trail_A, = ax.plot([], [], color='crimson', linewidth=6, alpha=0.4, zorder=1)
    trail_B, = ax.plot([], [], color='dodgerblue', linewidth=6, alpha=0.4, zorder=1)

    scat_main = ax.scatter([], [], c='darkviolet', edgecolors='black', s=TOTAL_TRAFFIC*0.8, zorder=10)
    scat_A = ax.scatter([], [], c='crimson', edgecolors='darkred', s=TRAFFIC_A*0.8, zorder=11, alpha=0.0)
    scat_B = ax.scatter([], [], c='dodgerblue', edgecolors='blue', s=TRAFFIC_B*0.8, zorder=11, alpha=0.0)
    
    text_outline = [pe.withStroke(linewidth=3, foreground='black')]
    text_main = ax.text(-1, -1, "", color='white', fontsize=12, ha='center', va='center', fontweight='bold', zorder=12, path_effects=text_outline)
    text_A = ax.text(-1, -1, "", color='white', fontsize=12, ha='center', va='center', fontweight='bold', zorder=12, path_effects=text_outline)
    text_B = ax.text(-1, -1, "", color='white', fontsize=12, ha='center', va='center', fontweight='bold', zorder=12, path_effects=text_outline)
    
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=14, fontweight='bold', color='black')
    
    def update(frame):
        if frame < FRAMES_COMMON:
            # 分裂前：巨大パケットが移動
            trail_main.set_data([coords_main[i][0] for i in range(frame+1)], [coords_main[i][1] for i in range(frame+1)])
            
            scat_main.set_alpha(1.0)
            scat_A.set_alpha(0.0)
            scat_B.set_alpha(0.0)
            
            axM, ayM = coords_main[frame]
            scat_main.set_offsets([[axM, ayM]])
            text_main.set_position((axM, ayM))
            text_main.set_text(str(TOTAL_TRAFFIC))
            
            trail_A.set_data([], [])
            trail_B.set_data([], [])
            text_A.set_text("")
            text_B.set_text("")
            
            time_text.set_text(f"Simulation Frame: {frame}/{MAX_FRAMES} (1500人の集団が進行中...)")
            
        else:
            # 分裂後：ミクロの意思決定に基づくマクロなパケットの自己組織化(分裂)
            trail_main.set_data([coords_main[i][0] for i in range(FRAMES_COMMON)], [coords_main[i][1] for i in range(FRAMES_COMMON)])
            
            trail_A.set_data([coords_A[i][0] for i in range(FRAMES_COMMON, frame+1)], [coords_A[i][1] for i in range(FRAMES_COMMON, frame+1)])
            trail_B.set_data([coords_B[i][0] for i in range(FRAMES_COMMON, frame+1)], [coords_B[i][1] for i in range(FRAMES_COMMON, frame+1)])
            
            scat_main.set_alpha(0.0)
            text_main.set_text("")
            
            scat_A.set_alpha(1.0)
            scat_B.set_alpha(1.0)
            
            axA, ayA = coords_A[frame]
            axB, ayB = coords_B[frame]
            
            scat_A.set_offsets([[axA, ayA]])
            scat_B.set_offsets([[axB, ayB]])
            
            text_A.set_position((axA, ayA))
            text_A.set_text(str(TRAFFIC_A))
            text_B.set_position((axB, ayB))
            text_B.set_text(str(TRAFFIC_B))
            
            time_text.set_text(f"Simulation Frame: {frame}/{MAX_FRAMES} (交差点での意思決定と迂回分裂!)")
            
        return scat_main, scat_A, scat_B, trail_main, trail_A, trail_B, text_main, text_A, text_B, time_text

    print(f"6. MP4動画ファイル（.mp4）としてレンダリング中... (全 {MAX_FRAMES} フレーム)")
    ani = animation.FuncAnimation(fig, update, frames=MAX_FRAMES, interval=33, blit=True)
    
    mp4_path = '/Users/pontarousu/Q1zemi/demand_model_v2/dynamic_equilibrium_split.mp4'
    writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='Demand Model AI'), bitrate=2000)
    ani.save(mp4_path, writer=writer)
    
    print(f"Saved MP4 → {mp4_path}")
    plt.close(fig)

if __name__ == '__main__':
    main()
