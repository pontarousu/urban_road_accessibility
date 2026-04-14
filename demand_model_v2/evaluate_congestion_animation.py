# [未使用] このファイルは Step 3 の検証用アニメーションです。現行のシミュレーションは simulate_equilibrium.py を参照してください。
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
    
    print("2. 経路A（大需要パケット）の生成と渋滞発生...")
    route_A = nx.shortest_path(G_proj, start_node, end_node, weight='bpr_weight')
    
    MASSIVE_TRAFFIC_A = 1500
    for u, v in zip(route_A[:-1], route_A[1:]):
        G_proj.edges[u,v,0]['volume'] += MASSIVE_TRAFFIC_A
    G_proj = apply_dynamic_weights(G_proj)
    
    print("3. 経路B（後続の迂回パケット）の生成...")
    route_B = nx.shortest_path(G_proj, start_node, end_node, weight='bpr_weight')
    TRAFFIC_B = 100  # 迂回するパケットの需要サイズ
    
    print("4. BPR所要時間をベースにした移動フレーム数（速度差）の計算...")
    cost_A = sum(G_proj.edges[u,v,0]['bpr_weight'] for u, v in zip(route_A[:-1], route_A[1:]))
    cost_B = sum(G_proj.edges[u,v,0]['bpr_weight'] for u, v in zip(route_B[:-1], route_B[1:]))
    
    # 迂回ルートBが滑らかに動く最低フレーム（例：100フレーム）を基準にする
    FRAMES_B = 100 
    FRAMES_A = int(FRAMES_B * (cost_A / cost_B))  # 時間比率を維持して渋滞Aの遅さを出す
    MAX_FRAMES = max(FRAMES_A, FRAMES_B)
        
    coords_A = get_smooth_path_padded(G_proj, route_A, FRAMES_A, MAX_FRAMES)
    coords_B = get_smooth_path_padded(G_proj, route_B, FRAMES_B, MAX_FRAMES)
    
    print("5. プロットとアニメーションのセットアップ...")
    fig, ax = ox.plot_graph(G_proj, show=False, close=False, edge_linewidth=1.0, edge_color='lightgray', node_size=0, bgcolor='white', figsize=(10,10))
    ax.set_title('Step 3アニメーション: 渋滞にはまる先行者(赤)と迂回する軽快な後続(青)', fontsize=16)
    
    # スタート・ゴールノードの追加
    orig = G_proj.nodes[start_node]
    dest = G_proj.nodes[end_node]
    ax.scatter([orig['x']], [orig['y']], c='green', s=200, zorder=6, label='START')
    ax.scatter([dest['x']], [dest['y']], c='purple', marker='*', s=400, zorder=6, label='GOAL')
    ax.legend(fontsize=12, loc='lower right', facecolor='white', framealpha=0.9)

    # 動的な赤い渋滞軌跡（最初は空）
    red_trail, = ax.plot([], [], color='red', linewidth=8, alpha=0.3, zorder=1)

    scat_A = ax.scatter([], [], c='crimson', edgecolors='darkred', zorder=10)
    scat_B = ax.scatter([], [], c='dodgerblue', edgecolors='blue', zorder=11)
    
    # ノード内の需要数字ラベル
    text_A = ax.text(-1, -1, "", color='white', fontsize=10, ha='center', va='center', fontweight='bold', zorder=12)
    text_B = ax.text(-1, -1, "", color='white', fontsize=10, ha='center', va='center', fontweight='bold', zorder=12)
    
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=14, fontweight='bold', color='black')
    
    def update(frame):
        axA, ayA = coords_A[frame]
        axB, ayB = coords_B[frame]
        
        # 渋滞の軌跡：パケットAが通過した軌跡をフレーム毎に動的に描いていく
        trail_x = [coords_A[i][0] for i in range(frame+1)]
        trail_y = [coords_A[i][1] for i in range(frame+1)]
        red_trail.set_data(trail_x, trail_y)
        
        # パケットの移動
        scat_A.set_offsets([[axA, ayA]])
        scat_A.set_sizes([MASSIVE_TRAFFIC_A * 0.8])
        
        scat_B.set_offsets([[axB, ayB]])
        scat_B.set_sizes([max(TRAFFIC_B * 0.8, 150)])
        
        # テキストの更新
        text_A.set_position((axA, ayA))
        text_A.set_text(str(MASSIVE_TRAFFIC_A))
        text_B.set_position((axB, ayB))
        text_B.set_text(str(TRAFFIC_B))
        
        time_text.set_text(f"Simulation Frame: {frame}/{MAX_FRAMES} (所要時間経過)")
        return scat_A, scat_B, red_trail, text_A, text_B, time_text

    print(f"6. MP4動画ファイル（.mp4）としてレンダリング中... (全 {MAX_FRAMES} フレーム)")
    # 30fpsにすることで滑らかさを確保
    ani = animation.FuncAnimation(fig, update, frames=MAX_FRAMES, interval=33, blit=True)
    
    mp4_path = '/Users/pontarousu/Q1zemi/demand_model_v2/congestion_detour.mp4'
    writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='Demand Model AI'), bitrate=2000)
    ani.save(mp4_path, writer=writer)
    
    print(f"Saved MP4 → {mp4_path}")
    plt.close(fig)

if __name__ == '__main__':
    main()
