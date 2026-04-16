"""
Demand Model v2 - Step 3.5: 逐次シミュレーション・エンジン (v4: 境界流入版)
============================================================
交差点ごとに「次のエッジの貯留容量 vs 自分の人数」を判定し、
容量を超過する場合はドライバーの集団意思決定として
パケットが自発的に分裂（迂回）するシミュレーション。

[v4] 境界流入（Boundary Injection）: マップの外周ノードから流入
[v3] 段階的投入（Gradual Injection）
[v2] Volume退出処理 + 貯留容量（次元統一）
"""

import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
import numpy as np
import imageio_ffmpeg
from matplotlib.collections import LineCollection
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

# パケットの配色パレット
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
def setup_graph(n_entry_points=6):
    """
    道路ネットワークの読み込みとCapacity/BPRの初期化。
    マップの外周（境界）ノードを検出し、流入地点として返す。
    """
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

    # === 境界ノードの検出 ===
    nodes_gdf = ox.graph_to_gdfs(G_proj, edges=False)
    minx, miny, maxx, maxy = nodes_gdf.total_bounds
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2

    node_dists = []
    for node_id, row in nodes_gdf.iterrows():
        dx = row.geometry.x - cx
        dy = row.geometry.y - cy
        dist = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        node_dists.append((node_id, dist, angle))

    # 角度でセクター分割し、各セクター内で最も遠いノードを選択
    sector_size = 2 * np.pi / n_entry_points
    entry_nodes = []
    for i in range(n_entry_points):
        sector_min = -np.pi + i * sector_size
        sector_max = sector_min + sector_size
        candidates = [(nid, d, a) for nid, d, a in node_dists
                       if sector_min <= a < sector_max]
        if candidates:
            best = max(candidates, key=lambda x: x[1])
            entry_nodes.append(best[0])

    dest_node = ox.distance.nearest_nodes(G_proj, cx, cy)

    # 到達可能でないエントリーポイントを除外
    reachable = [en for en in entry_nodes
                 if nx.has_path(G_proj, en, dest_node)]
    entry_nodes = reachable

    print(f"   流入地点: {len(entry_nodes)}箇所（マップ外周）")
    print(f"   目的地: Node {dest_node}（都市中心部）")

    storage_caps = [d['storage_capacity']
                    for _, _, _, d in G_proj.edges(keys=True, data=True)]
    print(f"   貯留容量の統計: min={min(storage_caps)}, "
          f"median={sorted(storage_caps)[len(storage_caps)//2]}, "
          f"max={max(storage_caps)}")

    return G_proj, entry_nodes, dest_node


# ============================================================
# 2. Volume管理ユーティリティ
# ============================================================
def reset_volumes(G):
    """全エッジのvolumeをゼロにリセットする"""
    for u, v, k, d in G.edges(keys=True, data=True):
        d['volume'] = 0.0


def recompute_volumes_from_packets(G, packets):
    """全パケットのoccupied_edgeからvolumeを再計算する"""
    reset_volumes(G)
    for p in packets:
        edge = p.get('occupied_edge')
        if edge and not p['arrived']:
            eu, ev = edge
            try:
                G.edges[eu, ev, 0]['volume'] += p['size']
            except KeyError:
                pass


def snapshot_edge_volumes(G):
    """現在の全エッジのvolume/storage_capacityをスナップショットとして返す"""
    volumes = {}
    for u, v, k, data in G.edges(keys=True, data=True):
        vol = data.get('volume', 0.0)
        cap = data.get('storage_capacity', 1)
        if vol > 0:
            volumes[(u, v)] = vol / max(cap, 1)
    return volumes


# ============================================================
# 3. 逐次シミュレーション・エンジン
# ============================================================
def run_simulation(G, entry_nodes, dest_node,
                   total_users=1500, injection_rate=100):
    """
    交差点ごとに判定する逐次シミュレーション・エンジン（v4: 境界流入版）。

    Returns:
        history: ステップごとのパケットスナップショット
        volume_history: ステップごとのエッジ混雑率スナップショット
        split_counts: ステップごとの分裂（迂回選択）回数
    """
    n_entries = len(entry_nodes)
    per_entry_rate = max(injection_rate // n_entries, 1)
    actual_rate = per_entry_rate * n_entries

    print(f"2. 逐次シミュレーション開始")
    print(f"   総ユーザー: {total_users}人")
    print(f"   流入地点: {n_entries}箇所")
    print(f"   投入レート: {per_entry_rate}人/地点/step × {n_entries}地点 = {actual_rate}人/step")

    reset_volumes(G)
    apply_dynamic_weights(G, capacity_key='storage_capacity')

    pid_counter = [0]
    def next_pid():
        pid = pid_counter[0]
        pid_counter[0] += 1
        return pid

    packets = []
    history = []
    volume_history = []  # エッジ混雑率のスナップショット
    split_counts = []    # ステップごとの分裂回数
    injected_total = 0
    MAX_STEPS = 500

    for step in range(MAX_STEPS):
        # === 境界ノード群から同時投入 ===
        if injected_total < total_users:
            step_injected = 0
            for entry_node in entry_nodes:
                remaining = total_users - injected_total - step_injected
                if remaining <= 0:
                    break
                batch_size = min(per_entry_rate, remaining)
                try:
                    route = nx.shortest_path(
                        G, entry_node, dest_node, weight='bpr_weight')
                except nx.NetworkXNoPath:
                    continue
                packets.append({
                    'id': next_pid(),
                    'node': entry_node,
                    'prev_node': entry_node,
                    'dest': dest_node,
                    'size': batch_size,
                    'route': list(route),
                    'arrived': False,
                    'color_idx': 0,
                    'occupied_edge': None,
                })
                step_injected += batch_size
            injected_total += step_injected
            if step_injected > 0:
                print(f"  Step {step}: {step_injected}人を投入 "
                      f"({n_entries}地点) [累計: {injected_total}/{total_users}]")

        # === Volume再計算 ===
        recompute_volumes_from_packets(G, packets)
        apply_dynamic_weights(G, capacity_key='storage_capacity')

        # === スナップショット ===
        snapshot = []
        for p in packets:
            snapshot.append({
                'id': p['id'], 'node': p['node'],
                'prev_node': p['prev_node'], 'size': p['size'],
                'arrived': p['arrived'], 'color_idx': p['color_idx'],
            })
        history.append(snapshot)
        volume_history.append(snapshot_edge_volumes(G))

        # 全パケット到着チェック
        if injected_total >= total_users and all(p['arrived'] for p in packets):
            print(f"  => 全パケットがゴールに到達（{step}ステップ）")
            break

        # === 各パケットの処理 ===
        next_packets = []
        step_splits = 0

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

            if p['size'] <= cap:
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
                step_splits += 1

                if step < 30 or step % 20 == 0:
                    print(f"  Step {step}: Node {u} 分裂 "
                          f"{p['size']}→{size_a}+{size_b} [cap={cap}]")

                # グループA: 直進
                pa = {
                    'id': next_pid(), 'node': v, 'prev_node': u,
                    'dest': p['dest'], 'size': size_a,
                    'route': p['route'][1:],
                    'arrived': (v == p['dest']),
                    'color_idx': 1,
                    'occupied_edge': (u, v) if v != p['dest'] else None,
                }
                next_packets.append(pa)

                edge_data['volume'] += size_a
                apply_dynamic_weights(G, capacity_key='storage_capacity')

                try:
                    new_route = nx.shortest_path(
                        G, u, p['dest'], weight='bpr_weight')
                except nx.NetworkXNoPath:
                    new_route = [u]

                # グループB: 迂回
                pb = {
                    'id': next_pid(), 'node': u, 'prev_node': u,
                    'dest': p['dest'], 'size': size_b,
                    'route': new_route, 'arrived': False,
                    'color_idx': 2,
                    'occupied_edge': None,
                }
                next_packets.append(pb)

        packets = next_packets
        split_counts.append(step_splits)

    # === 結果サマリ ===
    final = history[-1]
    arrived = [p for p in final if p['arrived']]
    sizes = [p['size'] for p in final]
    total_arrived = sum(p['size'] for p in arrived)
    total_splits = sum(split_counts)

    print(f"\n=== シミュレーション結果 ===")
    print(f"総ステップ数: {len(history)}")
    print(f"最終パケット数: {len(final)} (到着: {len(arrived)})")
    print(f"到着済み人数: {total_arrived}/{total_users}")
    print(f"パケットサイズ: min={min(sizes)}, max={max(sizes)}, "
          f"median={sorted(sizes)[len(sizes)//2]}")
    print(f"総分裂回数: {total_splits}")

    return history, volume_history, split_counts


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


def build_edge_collection(G):
    """
    グラフの全エッジの座標を取得し、LineCollectionの描画データを構築する。
    Returns: (segments, edge_keys) — segmentsはLineCollection用の座標リスト
    """
    segments = []
    edge_keys = []
    for u, v, k, data in G.edges(keys=True, data=True):
        if 'geometry' in data:
            coords = list(data['geometry'].coords)
        else:
            x1, y1 = get_node_xy(G, u)
            x2, y2 = get_node_xy(G, v)
            coords = [(x1, y1), (x2, y2)]
        segments.append(coords)
        edge_keys.append((u, v))
    return segments, edge_keys


def build_animation_frames(G, history, volume_history):
    """シミュレーション履歴をアニメーション用フレーム配列に変換する"""
    FRAMES_PER_STEP = 12
    frames = []  # パケットフレーム
    volume_frames = []  # 各フレームに対応するエッジ混雑率

    # 冒頭: 初期状態を0.5秒間表示
    for _ in range(15):
        frame_data = []
        for p in history[0]:
            x, y = get_node_xy(G, p['node'])
            frame_data.append({
                'x': x, 'y': y, 'size': p['size'],
                'color': COLORS[p['color_idx'] % len(COLORS)],
                'label': str(p['size']),
            })
        frames.append(frame_data)
        volume_frames.append(volume_history[0] if volume_history else {})

    # ステップ間の補間
    for step_idx in range(len(history) - 1):
        current = {p['id']: p for p in history[step_idx]}
        nxt = {p['id']: p for p in history[step_idx + 1]}
        cur_ids = set(current.keys())
        nxt_ids = set(nxt.keys())
        continuing = cur_ids & nxt_ids
        born = nxt_ids - cur_ids
        died = cur_ids - nxt_ids

        # このステップ間のエッジ混雑率（補間なし、ステップ単位）
        vol_snap = volume_history[min(step_idx + 1, len(volume_history) - 1)]

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
                    'x': pos[0], 'y': pos[1], 'size': c['size'],
                    'color': COLORS[c['color_idx'] % len(COLORS)],
                    'label': str(c['size']),
                })

            for pid in died:
                if frac < 0.15:
                    c = current[pid]
                    pos = get_node_xy(G, c['node'])
                    frame_data.append({
                        'x': pos[0], 'y': pos[1], 'size': c['size'],
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
                        'x': pos[0], 'y': pos[1], 'size': n['size'],
                        'color': COLORS[n['color_idx'] % len(COLORS)],
                        'label': str(n['size']),
                    })

            frames.append(frame_data)
            volume_frames.append(vol_snap)

    # 末尾: 最終状態を1秒間表示
    for _ in range(30):
        frame_data = []
        for p in history[-1]:
            x, y = get_node_xy(G, p['node'])
            frame_data.append({
                'x': x, 'y': y, 'size': p['size'],
                'color': COLORS[p['color_idx'] % len(COLORS)],
                'label': str(p['size']),
            })
        frames.append(frame_data)
        volume_frames.append(volume_history[-1] if volume_history else {})

    return frames, volume_frames


# ============================================================
# 5. MP4レンダリング
# ============================================================
def dot_size(n_people):
    """パケットの人数からドットの描画サイズを計算する"""
    return max(n_people * 3, 30)


def render_mp4(G, frames, volume_frames, entry_nodes, dest_node, output_path,
               total_users=1500, injection_rate=100):
    """フレームデータからMP4アニメーション（渋滞ヒートマップ付き）を生成する"""
    print(f"4. MP4レンダリング中... (全 {len(frames)} フレーム)")

    fig, ax = ox.plot_graph(
        G, show=False, close=False,
        edge_linewidth=0.5, edge_color='#e0e0e0',
        node_size=0, bgcolor='white', figsize=(12, 10))

    n_entries = len(entry_nodes)
    ax.set_title(
        f'逐次シミュレーション v4（境界流入 + 渋滞ヒートマップ）\n'
        f'総ユーザー: {total_users}人 | {n_entries}箇所から流入',
        fontsize=13)

    # === 渋滞ヒートマップ用のエッジLineCollection ===
    segments, edge_keys = build_edge_collection(G)
    # 初期状態: 全エッジ透明
    initial_colors = [(0, 0, 0, 0)] * len(segments)
    congestion_lc = LineCollection(
        segments, colors=initial_colors, linewidths=4, alpha=0.7, zorder=2)
    ax.add_collection(congestion_lc)

    # 渋滞カラーマップ: 緑（空き）→黄→赤（混雑）
    cmap_congestion = plt.cm.RdYlGn_r

    # === 流入地点マーカー（マップ外縁に配置） ===
    for i, en in enumerate(entry_nodes):
        ex, ey = get_node_xy(G, en)
        # 三角マーカーを少し外側にオフセットして表示
        nodes_gdf = ox.graph_to_gdfs(G, edges=False)
        minx, miny, maxx, maxy = nodes_gdf.total_bounds
        cx_map = (minx + maxx) / 2
        cy_map = (miny + maxy) / 2
        dx = ex - cx_map
        dy = ey - cy_map
        dist = np.sqrt(dx**2 + dy**2)
        if dist > 0:
            offset = 50  # 50m外側にオフセット
            ox_offset = ex + dx / dist * offset
            oy_offset = ey + dy / dist * offset
        else:
            ox_offset, oy_offset = ex, ey

        ax.annotate(
            f'IN {i+1}', xy=(ex, ey), xytext=(ox_offset, oy_offset),
            fontsize=8, fontweight='bold', color='darkgreen',
            ha='center', va='center',
            arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen',
                      alpha=0.8, edgecolor='darkgreen'),
            zorder=7)

    # 目的地マーカー
    dx_d, dy_d = get_node_xy(G, dest_node)
    ax.scatter([dx_d], [dy_d], c='purple', s=400, zorder=8,
               marker='*', edgecolors='white', linewidths=1.5)
    ax.annotate(
        'GOAL', xy=(dx_d, dy_d), xytext=(dx_d, dy_d + 80),
        fontsize=10, fontweight='bold', color='purple',
        ha='center', va='bottom',
        arrowprops=dict(arrowstyle='->', color='purple', lw=1.5),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8d5f5',
                  alpha=0.9, edgecolor='purple'),
        zorder=9)

    # 渋滞度のカラーバー
    sm = plt.cm.ScalarMappable(cmap=cmap_congestion,
                                norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, shrink=0.6)
    cbar.set_label('渋滞度 (Volume / 貯留容量)', fontsize=10)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(['空き', '混雑', '飽和'])

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
        vol_snap = volume_frames[frame_idx]

        # === 渋滞ヒートマップ更新 ===
        edge_colors = []
        edge_widths = []
        for u, v in edge_keys:
            ratio = vol_snap.get((u, v), 0.0)
            if ratio < 0.01:
                edge_colors.append((0, 0, 0, 0))  # 透明
                edge_widths.append(0)
            else:
                r = min(ratio, 1.0)
                edge_colors.append(cmap_congestion(r))
                edge_widths.append(2 + r * 6)  # 混雑度に応じて太く
        congestion_lc.set_colors(edge_colors)
        congestion_lc.set_linewidths(edge_widths)

        # === パケット描画 ===
        if not frame_data:
            scatter.set_offsets(np.empty((0, 2)))
            info_text.set_text(f"Frame {frame_idx}/{len(frames)}")
            return [scatter, info_text, congestion_lc]

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

        # パケット数が少ないときだけラベル表示
        if len(frame_data) <= 15:
            for d in frame_data:
                if d['size'] >= 5:
                    t = ax.text(
                        d['x'], d['y'], d['label'],
                        color='white', fontsize=8,
                        ha='center', va='center', fontweight='bold',
                        zorder=12, path_effects=text_outline)
                    texts.append(t)

        n_packets = len(frame_data)
        total_people = sum(d['size'] for d in frame_data)
        info_text.set_text(
            f"パケット数: {n_packets} | 総人数: {total_people} | "
            f"Frame: {frame_idx}/{len(frames)}")

        return [scatter, info_text, congestion_lc] + texts

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=33, blit=False)
    writer = animation.FFMpegWriter(
        fps=30, metadata=dict(artist='Demand Model v2'), bitrate=2500)
    ani.save(output_path, writer=writer)
    print(f"  => Saved → {output_path}")
    plt.close(fig)


# ============================================================
# 6. 迂回選択数グラフ
# ============================================================
def plot_split_counts(split_counts, output_path, frames_per_step=12,
                      initial_frames=15):
    """ステップごとの迂回選択（分裂）回数をプロットする（x軸はフレーム数）"""
    print(f"5. 迂回選択数グラフ生成中...")

    fig, ax = plt.subplots(figsize=(12, 5))

    # ステップ → フレームに変換（冒頭の静止フレーム分をオフセット）
    frame_positions = [initial_frames + s * frames_per_step
                       for s in range(len(split_counts))]
    bar_width = frames_per_step * 0.8

    ax.bar(frame_positions, split_counts, width=bar_width,
           color='dodgerblue', alpha=0.7,
           edgecolor='navy', linewidth=0.5, label='分裂回数/ステップ')

    # 移動平均線
    if len(split_counts) > 5:
        window = 5
        ma = np.convolve(split_counts, np.ones(window)/window, mode='valid')
        ma_frames = [initial_frames + s * frames_per_step
                     for s in range(window - 1, len(split_counts))]
        ax.plot(ma_frames, ma, color='crimson', linewidth=2.5,
                label=f'移動平均（{window}ステップ）', zorder=5)

    ax.set_xlabel('フレーム数（30fps）', fontsize=13)
    ax.set_ylabel('迂回選択数（分裂回数）', fontsize=13)
    ax.set_title('各フレームにおける迂回選択パケット数の推移', fontsize=15)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # 投入フェーズを背景色で表示
    inject_end_frame = initial_frames + 15 * frames_per_step
    ax.axvspan(0, min(inject_end_frame, max(frame_positions)),
               alpha=0.08, color='green')
    ax.text(initial_frames + 10, max(split_counts) * 0.95 if split_counts else 1,
            '← 投入中', fontsize=9, color='green', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  => Saved → {output_path}")
    plt.close(fig)


# ============================================================
# メイン
# ============================================================
def main():
    G, entry_nodes, dest_node = setup_graph(n_entry_points=6)

    history, volume_history, split_counts = run_simulation(
        G, entry_nodes, dest_node,
        total_users=1500, injection_rate=120)

    print("3. アニメーションフレーム生成中...")
    frames, volume_frames = build_animation_frames(G, history, volume_history)

    mp4_path = '/Users/pontarousu/Q1zemi/demand_model_v2/dynamic_equilibrium_split.mp4'
    render_mp4(G, frames, volume_frames, entry_nodes, dest_node, mp4_path,
               total_users=1500, injection_rate=120)

    graph_path = '/Users/pontarousu/Q1zemi/demand_model_v2/split_counts_graph.png'
    plot_split_counts(split_counts, graph_path)


if __name__ == '__main__':
    main()
