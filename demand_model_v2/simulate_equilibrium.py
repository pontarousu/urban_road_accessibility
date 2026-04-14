"""
Demand Model v2 - Step 3.5: 逐次シミュレーション・エンジン (v3: 段階的投入版)
============================================================
交差点ごとに「次のエッジの貯留容量 vs 自分の人数」を判定し、
容量を超過する場合はドライバーの集団意思決定として
パケットが自発的に分裂（迂回）するシミュレーション。

[v3での追加修正]
- 段階的投入（Gradual Injection）: 1500台を一度に生成するのではなく、
  injection_rateステップあたりの台数で段階的にSTARTノードへ投入する
  → 先行バッチが道を空けてから後続が来るため、物理的に自然な交通流が実現

[v2での修正点（継続）]
1. Volume退出処理: パケットがエッジを離れたら volume を減算する
2. 次元の整合: 分裂判定に「貯留容量 C_storage（台）」を使用
"""

import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as pe
import numpy as np
import imageio_ffmpeg
from shapely.geometry import LineString
from assign_capacity import (
    calculate_capacity,
    calculate_storage_capacity,
    apply_dynamic_weights,
)

# 描画設定
plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()

# 対象エリア: 金沢市中心部
CENTER = (36.5613, 136.6562)
DIST = 1000

# パケットの配色パレット（投入バッチごとに色を循環）
COLORS = [
    'darkviolet',    # 0: 初期
    'crimson',       # 1: 分裂子 - 直進組
    'dodgerblue',    # 2: 分裂子 - 迂回組
    'forestgreen',   # 3
    'darkorange',    # 4
    'deeppink',      # 5
    'teal',          # 6
    'gold',          # 7
]


# ============================================================
# 1. グラフ初期化
# ============================================================
def setup_graph():
    """道路ネットワークの読み込みとCapacity/BPRの初期化"""
    print("1. グラフロードとCapacityセットアップ...")
    G = ox.graph_from_point(CENTER, dist=DIST, network_type='drive')
    G_proj = ox.project_graph(G)

    for u, v, k, data in G_proj.edges(keys=True, data=True):
        data['capacity'] = calculate_capacity(data)
        data['storage_capacity'] = calculate_storage_capacity(data)
        data['volume'] = 0.0
        length = data.get('length', 10.0)
        speed_kph = data.get('maxspeed', 40.0)
        if isinstance(speed_kph, list):
            speed_kph = speed_kph[0]
        try:
            speed_mps = float(speed_kph) * 1000 / 3600
        except (ValueError, TypeError):
            speed_mps = 40.0 * 1000 / 3600
        data['free_flow_time'] = length / speed_mps

    apply_dynamic_weights(G_proj, capacity_key='storage_capacity')

    # スタート・ゴールをネットワークの対角線上に配置
    nodes_gdf = ox.graph_to_gdfs(G_proj, edges=False)
    minx, miny, maxx, maxy = nodes_gdf.total_bounds
    start_node = ox.distance.nearest_nodes(
        G_proj, minx + (maxx - minx) * 0.2, miny + (maxy - miny) * 0.2)
    end_node = ox.distance.nearest_nodes(
        G_proj, maxx - (maxx - minx) * 0.2, maxy - (maxy - miny) * 0.2)

    # 貯留容量の統計情報を表示
    storage_caps = [d['storage_capacity']
                    for _, _, _, d in G_proj.edges(keys=True, data=True)]
    print(f"   貯留容量の統計: min={min(storage_caps)}, "
          f"median={sorted(storage_caps)[len(storage_caps)//2]}, "
          f"max={max(storage_caps)}, edges={len(storage_caps)}")

    return G_proj, start_node, end_node


# ============================================================
# 2. Volume管理ユーティリティ
# ============================================================
def reset_volumes(G):
    """全エッジのvolumeをゼロにリセットする"""
    for u, v, k, d in G.edges(keys=True, data=True):
        d['volume'] = 0.0


def recompute_volumes_from_packets(G, packets):
    """
    全パケットの occupied_edge から、エッジ上の交通量を再計算する。
    各ステップの冒頭で呼ぶことで、volumeの累積ドリフトを完全に防止する。
    """
    reset_volumes(G)
    for p in packets:
        edge = p.get('occupied_edge')
        if edge and not p['arrived']:
            eu, ev = edge
            try:
                G.edges[eu, ev, 0]['volume'] += p['size']
            except KeyError:
                pass


# ============================================================
# 3. 逐次シミュレーション・エンジン
# ============================================================
def run_simulation(G, start_node, end_node,
                   total_users=1500, injection_rate=100):
    """
    交差点ごとに判定する逐次シミュレーション・エンジン（v3: 段階的投入版）。

    Parameters:
        total_users: 投入する総ユーザー数
        injection_rate: 1ステップあたりの投入台数（段階的投入）

    [v3での変更点]
    - 初期パケットを1つ作るのではなく、ステップごとに injection_rate 人ずつ
      STARTノードへ新規パケットとして投入する（段階的投入）
    - 先行バッチが道を空けてから後続が到着するため、
      貯留容量の小さい住宅地道路でも自然に流れる
    """
    injection_steps = (total_users + injection_rate - 1) // injection_rate
    print(f"2. 逐次シミュレーション開始")
    print(f"   総ユーザー: {total_users}人")
    print(f"   投入レート: {injection_rate}人/ステップ × {injection_steps}ステップ")

    # 全エッジのvolumeをリセット
    reset_volumes(G)
    apply_dynamic_weights(G, capacity_key='storage_capacity')

    # パケットID管理
    pid_counter = [0]
    def next_pid():
        pid = pid_counter[0]
        pid_counter[0] += 1
        return pid

    packets = []
    history = []
    injected_total = 0
    MAX_STEPS = 500

    for step in range(MAX_STEPS):
        # === v3: 段階的投入 ===
        if injected_total < total_users:
            batch_size = min(injection_rate, total_users - injected_total)
            # 現在の渋滞状況を反映した最短ルートを計算
            route = nx.shortest_path(
                G, start_node, end_node, weight='bpr_weight')
            new_packet = {
                'id': next_pid(),
                'node': start_node,
                'prev_node': start_node,
                'dest': end_node,
                'size': batch_size,
                'route': list(route),
                'arrived': False,
                'color_idx': 0,  # 投入時は darkviolet
                'occupied_edge': None,
            }
            packets.append(new_packet)
            injected_total += batch_size
            print(f"  Step {step}: {batch_size}人を投入 "
                  f"(累計: {injected_total}/{total_users})")

        # === 毎ステップ冒頭でVolumeを再計算 ===
        recompute_volumes_from_packets(G, packets)
        apply_dynamic_weights(G, capacity_key='storage_capacity')

        # === スナップショット記録 ===
        snapshot = []
        for p in packets:
            snapshot.append({
                'id': p['id'],
                'node': p['node'],
                'prev_node': p['prev_node'],
                'size': p['size'],
                'arrived': p['arrived'],
                'color_idx': p['color_idx'],
            })
        history.append(snapshot)

        # 全パケット到着チェック（投入完了後のみ判定）
        if injected_total >= total_users and all(p['arrived'] for p in packets):
            print(f"  => 全パケットがゴールに到達（{step}ステップ）")
            break

        # === 各パケットの処理 ===
        next_packets = []
        for p in packets:
            if p['arrived']:
                p['prev_node'] = p['node']
                next_packets.append(p)
                continue

            if len(p['route']) < 2:
                p['arrived'] = True
                p['prev_node'] = p['node']
                p['occupied_edge'] = None
                next_packets.append(p)
                continue

            u = p['route'][0]
            v = p['route'][1]

            try:
                edge_data = G.edges[u, v, 0]
                cap = edge_data['storage_capacity']
            except KeyError:
                p['arrived'] = True
                p['prev_node'] = p['node']
                p['occupied_edge'] = None
                next_packets.append(p)
                continue

            # ■ 貯留容量判定 ■
            if p['size'] <= cap:
                # 全員通過可能 → 1ホップ前進
                p['prev_node'] = u
                p['node'] = v
                p['route'] = p['route'][1:]
                p['occupied_edge'] = (u, v)
                if v == p['dest']:
                    p['arrived'] = True
                    p['occupied_edge'] = None
                next_packets.append(p)
            else:
                # ■ 分裂発生 ■
                size_a = int(cap)
                size_b = p['size'] - size_a

                if step < 30 or step % 20 == 0:  # ログ量を制限
                    print(f"  Step {step}: Node {u} で分裂！ "
                          f"{p['size']}人 → {size_a}人(直進) + {size_b}人(迂回)"
                          f" [貯留容量={cap}]")

                # グループA: 直進
                route_a = p['route'][1:]
                pa = {
                    'id': next_pid(),
                    'node': v,
                    'prev_node': u,
                    'dest': p['dest'],
                    'size': size_a,
                    'route': route_a,
                    'arrived': (v == p['dest']),
                    'color_idx': 1,
                    'occupied_edge': (u, v) if v != p['dest'] else None,
                }
                next_packets.append(pa)

                # Aの進入を反映してBの迂回路を探索
                edge_data['volume'] += size_a
                apply_dynamic_weights(G, capacity_key='storage_capacity')

                try:
                    new_route = nx.shortest_path(
                        G, u, p['dest'], weight='bpr_weight')
                except nx.NetworkXNoPath:
                    new_route = [u]

                # グループB: 迂回
                pb = {
                    'id': next_pid(),
                    'node': u,
                    'prev_node': u,
                    'dest': p['dest'],
                    'size': size_b,
                    'route': new_route,
                    'arrived': False,
                    'color_idx': 2,
                    'occupied_edge': None,
                }
                next_packets.append(pb)

        packets = next_packets

    # === 結果サマリ ===
    final = history[-1]
    arrived_packets = [p for p in final if p['arrived']]
    moving_packets = [p for p in final if not p['arrived']]
    total_arrived = sum(p['size'] for p in arrived_packets)
    sizes = [p['size'] for p in final]

    print(f"\n=== シミュレーション結果 ===")
    print(f"総ステップ数: {len(history)}")
    print(f"最終パケット数: {len(final)} "
          f"(到着: {len(arrived_packets)}, 移動中: {len(moving_packets)})")
    print(f"到着済み人数: {total_arrived}/{total_users}")
    print(f"パケットサイズ: min={min(sizes)}, max={max(sizes)}, "
          f"median={sorted(sizes)[len(sizes)//2]}")

    return history


# ============================================================
# 4. アニメーションフレーム生成
# ============================================================
def get_node_xy(G, node_id):
    """ノードの(x, y)座標を返す"""
    return (G.nodes[node_id]['x'], G.nodes[node_id]['y'])


def interpolate_on_edge(G, u, v, fraction):
    """エッジ(u,v)上でfraction(0~1)に対応する座標を返す"""
    try:
        edge_data = G.edges[u, v, 0]
        if 'geometry' in edge_data:
            line = edge_data['geometry']
        else:
            line = LineString([get_node_xy(G, u), get_node_xy(G, v)])
    except KeyError:
        x1, y1 = get_node_xy(G, u)
        x2, y2 = get_node_xy(G, v)
        return (x1 + (x2 - x1) * fraction, y1 + (y2 - y1) * fraction)

    point = line.interpolate(fraction * line.length)
    return (point.x, point.y)


def build_animation_frames(G, history):
    """シミュレーション履歴をアニメーション用フレーム配列に変換する"""
    FRAMES_PER_STEP = 12  # 1ホップあたり12フレーム（30fps → 0.4秒/ホップ）
    frames = []

    # 冒頭: 初期状態を0.5秒間表示
    for _ in range(15):
        frame_data = []
        for p in history[0]:
            x, y = get_node_xy(G, p['node'])
            frame_data.append({
                'x': x, 'y': y,
                'size': p['size'],
                'color': COLORS[p['color_idx'] % len(COLORS)],
                'label': str(p['size']),
            })
        frames.append(frame_data)

    # ステップ間の補間
    for step_idx in range(len(history) - 1):
        current = {p['id']: p for p in history[step_idx]}
        nxt = {p['id']: p for p in history[step_idx + 1]}

        cur_ids = set(current.keys())
        nxt_ids = set(nxt.keys())

        continuing = cur_ids & nxt_ids
        born = nxt_ids - cur_ids
        died = cur_ids - nxt_ids

        for f in range(FRAMES_PER_STEP):
            frac = f / max(FRAMES_PER_STEP - 1, 1)
            frame_data = []

            for pid in continuing:
                c = current[pid]
                n = nxt[pid]
                if c['node'] != n['node']:
                    pos = interpolate_on_edge(G, c['node'], n['node'], frac)
                else:
                    pos = get_node_xy(G, c['node'])
                frame_data.append({
                    'x': pos[0], 'y': pos[1],
                    'size': c['size'],
                    'color': COLORS[c['color_idx'] % len(COLORS)],
                    'label': str(c['size']),
                })

            for pid in died:
                if frac < 0.15:
                    c = current[pid]
                    pos = get_node_xy(G, c['node'])
                    frame_data.append({
                        'x': pos[0], 'y': pos[1],
                        'size': c['size'],
                        'color': COLORS[c['color_idx'] % len(COLORS)],
                        'label': str(c['size']),
                    })

            for pid in born:
                if frac >= 0.15:
                    n = nxt[pid]
                    prev = n.get('prev_node', n['node'])
                    if prev != n['node']:
                        pos = interpolate_on_edge(G, prev, n['node'], frac)
                    else:
                        pos = get_node_xy(G, n['node'])
                    frame_data.append({
                        'x': pos[0], 'y': pos[1],
                        'size': n['size'],
                        'color': COLORS[n['color_idx'] % len(COLORS)],
                        'label': str(n['size']),
                    })

            frames.append(frame_data)

    # 末尾: 最終状態を1秒間表示
    for _ in range(30):
        frame_data = []
        for p in history[-1]:
            x, y = get_node_xy(G, p['node'])
            frame_data.append({
                'x': x, 'y': y,
                'size': p['size'],
                'color': COLORS[p['color_idx'] % len(COLORS)],
                'label': str(p['size']),
            })
        frames.append(frame_data)

    return frames


# ============================================================
# 5. MP4レンダリング
# ============================================================
def dot_size(n_people):
    """パケットの人数からドットの描画サイズ（ポイント²）を計算する"""
    # 1人 = 30pt², 100人 = 300pt² にスケーリング（視認性確保）
    return max(n_people * 3, 30)


def render_mp4(G, frames, start_node, end_node, output_path,
               total_users=1500, injection_rate=100):
    """フレームデータからMP4アニメーションを生成する"""
    print(f"4. MP4レンダリング中... (全 {len(frames)} フレーム)")

    fig, ax = ox.plot_graph(
        G, show=False, close=False,
        edge_linewidth=1.0, edge_color='lightgray',
        node_size=0, bgcolor='white', figsize=(12, 10))

    ax.set_title(
        'Step 3.5 v3: 逐次シミュレーション（段階的投入 + 貯留容量ベース分裂）\n'
        f'総ユーザー: {total_users}人 | 投入レート: {injection_rate}人/step',
        fontsize=13)

    # スタート・ゴールマーカー
    sx, sy = get_node_xy(G, start_node)
    ex, ey = get_node_xy(G, end_node)
    ax.scatter([sx], [sy], c='green', s=200, zorder=6,
               marker='o', label='START')
    ax.scatter([ex], [ey], c='purple', s=400, zorder=6,
               marker='*', label='GOAL')

    # === 凡例パネル（右側に配置） ===
    # ドットサイズの凡例
    legend_elements = [
        plt.scatter([], [], s=dot_size(1), c='gray', edgecolors='black',
                    linewidths=0.5, label='1人'),
        plt.scatter([], [], s=dot_size(10), c='gray', edgecolors='black',
                    linewidths=0.5, label='10人'),
        plt.scatter([], [], s=dot_size(50), c='gray', edgecolors='black',
                    linewidths=0.5, label='50人'),
        plt.scatter([], [], s=dot_size(100), c='gray', edgecolors='black',
                    linewidths=0.5, label='100人'),
    ]
    # 色の凡例
    color_legend = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkviolet',
                    markersize=8, label='新規投入'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='crimson',
                    markersize=8, label='直進組'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='dodgerblue',
                    markersize=8, label='迂回組'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                    markersize=10, label='START'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='purple',
                    markersize=12, label='GOAL'),
    ]

    leg1 = ax.legend(handles=legend_elements, title='ドットサイズ',
                     loc='upper right', fontsize=9, title_fontsize=10,
                     facecolor='white', framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=color_legend, title='色の意味',
              loc='lower right', fontsize=9, title_fontsize=10,
              facecolor='white', framealpha=0.9)

    text_outline = [pe.withStroke(linewidth=3, foreground='black')]

    scatter = ax.scatter([], [], zorder=10)
    texts = []
    info_text = ax.text(
        0.02, 0.97, '', transform=ax.transAxes,
        fontsize=11, fontweight='bold', color='black',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))

    def update(frame_idx):
        nonlocal texts

        for t in texts:
            t.remove()
        texts = []

        frame_data = frames[frame_idx]

        if not frame_data:
            scatter.set_offsets(np.empty((0, 2)))
            info_text.set_text(f"Frame {frame_idx}/{len(frames)}")
            return [scatter, info_text]

        xs = [d['x'] for d in frame_data]
        ys = [d['y'] for d in frame_data]
        sizes = [dot_size(d['size']) for d in frame_data]
        colors = [d['color'] for d in frame_data]

        offsets = np.column_stack([xs, ys])
        scatter.set_offsets(offsets)
        scatter.set_sizes(sizes)
        scatter.set_facecolors(colors)
        scatter.set_edgecolors('black')
        scatter.set_linewidths(0.8)

        # パケット数が少ないときだけ人数ラベルを表示
        if len(frame_data) <= 15:
            for d in frame_data:
                if d['size'] >= 5:  # 5人以上のパケットだけラベル表示
                    t = ax.text(
                        d['x'], d['y'], d['label'],
                        color='white', fontsize=8,
                        ha='center', va='center', fontweight='bold',
                        zorder=12, path_effects=text_outline)
                    texts.append(t)

        n_packets = len(frame_data)
        total_people = sum(d['size'] for d in frame_data)
        arrived = sum(1 for d in frame_data
                      if d['size'] == 0)  # 到着済みはフレームから消える
        info_text.set_text(
            f"パケット数: {n_packets} | "
            f"総人数: {total_people} | "
            f"Frame: {frame_idx}/{len(frames)}")

        return [scatter, info_text] + texts

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=33, blit=False)

    writer = animation.FFMpegWriter(
        fps=30, metadata=dict(artist='Demand Model v2'), bitrate=2500)
    ani.save(output_path, writer=writer)

    print(f"  => Saved → {output_path}")
    plt.close(fig)


# ============================================================
# メイン
# ============================================================
def main():
    G, start_node, end_node = setup_graph()

    # v3: 100人/ステップ × 15ステップ = 1500人を段階的に投入
    history = run_simulation(
        G, start_node, end_node,
        total_users=1500, injection_rate=100)

    print("3. アニメーションフレーム生成中...")
    frames = build_animation_frames(G, history)

    output_path = '/Users/pontarousu/Q1zemi/demand_model_v2/dynamic_equilibrium_split.mp4'
    render_mp4(G, frames, start_node, end_node, output_path,
               total_users=1500, injection_rate=100)


if __name__ == '__main__':
    main()
