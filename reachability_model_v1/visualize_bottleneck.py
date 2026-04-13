"""
ボトルネック地点の可視化スクリプト
Phase 3のStep 1で抽出された「IoU下位30%ile」のサンプル地点を地図上に表示する
"""

import osmnx as ox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import numpy.ma as ma
import rasterio.transform
from rasterio.features import rasterize
from skimage.graph import MCP_Geometric
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

# ---- 日本語フォント設定（macOS: Hiragino Sans） ----
plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False

# ---- パラメータ（optimize_road.py と同一） ----
CENTER      = (36.5613, 136.6562)
DIST        = 5000
RES         = 10
SAMPLE_PITCH = 500
TRIP_LIMIT  = 1500
BOTTLENECK_PERCENTILE = 30

def iou_for_point(px, py, cost_surface, transform, limit_cost, euclidean_area, pixel_area):
    col, row = ~transform * (px, py)
    r, c = int(row), int(col)
    h, w = cost_surface.shape
    if not (0 <= r < h and 0 <= c < w):
        return 0.0
    if cost_surface[r, c] > 9999:
        return 0.0
    mcp = MCP_Geometric(cost_surface)
    costs, _ = mcp.find_costs(starts=[(r, c)])
    reached = np.sum(costs <= limit_cost) * pixel_area
    return min(reached / euclidean_area, 1.0)

def main():
    limit_cost     = TRIP_LIMIT / RES
    pixel_area     = RES * RES
    euclidean_area = np.pi * (TRIP_LIMIT ** 2)

    print("1. Downloading road network...")
    G = ox.graph_from_point(CENTER, dist=DIST, network_type='drive')
    G_proj = ox.project_graph(G)
    nodes_proj, edges_proj = ox.graph_to_gdfs(G_proj)

    try:
        water = ox.features_from_point(CENTER, tags={'natural': 'water', 'waterway': 'riverbank'}, dist=DIST)
        water_proj = water.to_crs(G_proj.graph['crs']) if len(water) > 0 else None
    except Exception:
        water_proj = None

    print("2. Building cost raster...")
    bounds = edges_proj.total_bounds
    minx, miny, maxx, maxy = bounds
    width  = int(np.ceil((maxx - minx) / RES))
    height = int(np.ceil((maxy - miny) / RES))
    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)

    cost = np.full((height, width), 20.0, dtype=np.float32)
    road_shapes = [(geom, 1.0) for geom in edges_proj.geometry]
    cost = rasterize(road_shapes, out_shape=(height, width), transform=transform,
                     fill=20.0, default_value=1.0, dtype=np.float32)
    if water_proj is not None and not water_proj.empty:
        ws = [(g, 99999.0) for g in water_proj.geometry if g.geom_type in ('Polygon','MultiPolygon')]
        if ws:
            cost = rasterize(ws, out=cost, transform=transform, default_value=99999.0)

    print("3. Computing baseline IoU...")
    x_coords = np.arange(minx + 500, maxx - 500, SAMPLE_PITCH)
    y_coords = np.arange(miny + 500, maxy - 500, SAMPLE_PITCH)
    mesh_x, mesh_y = np.meshgrid(x_coords, y_coords)
    pts_flat = np.column_stack((mesh_x.ravel(), mesh_y.ravel()))

    from joblib import Parallel, delayed
    baseline_iou = np.array(Parallel(n_jobs=-1)(
        delayed(iou_for_point)(px, py, cost, transform, limit_cost, euclidean_area, pixel_area)
        for px, py in pts_flat
    ))

    threshold = np.percentile(baseline_iou[baseline_iou > 0], BOTTLENECK_PERCENTILE)
    bottleneck_mask = (baseline_iou > 0) & (baseline_iou <= threshold)
    normal_mask     = baseline_iou > threshold
    zero_mask       = baseline_iou == 0.0

    print(f"  Threshold (30th pct): IoU <= {threshold:.4f}")
    print(f"  Bottleneck points: {bottleneck_mask.sum()}")

    # ---- プロット ----
    extent = [x_coords[0] - SAMPLE_PITCH/2, x_coords[-1] + SAMPLE_PITCH/2,
              y_coords[0] - SAMPLE_PITCH/2, y_coords[-1] + SAMPLE_PITCH/2]

    iou_grid = baseline_iou.reshape((len(y_coords), len(x_coords)))

    fig, axes = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'wspace': 0.05})

    # --- 左：IoUヒートマップ上にボトルネック点を重ねる ---
    ax = axes[0]
    edges_proj.plot(ax=ax, color='lightgray', linewidth=0.3, alpha=0.4, zorder=1)
    if water_proj is not None and not water_proj.empty:
        water_proj.plot(ax=ax, color='lightblue', alpha=0.5, zorder=2)
    ax.imshow(ma.masked_where(iou_grid == 0, iou_grid),
              extent=extent, origin='lower', cmap='RdYlGn',
              alpha=0.5, vmin=0.0, vmax=0.7, zorder=3)
    edges_proj.plot(ax=ax, color='dimgray', linewidth=0.35, alpha=0.5, zorder=4)

    # ボトルネック：赤い × マーク
    ax.scatter(pts_flat[bottleneck_mask, 0], pts_flat[bottleneck_mask, 1],
               c='crimson', marker='x', s=120, linewidths=2.0, zorder=6,
               label=f'Bottleneck (IoU ≤ {threshold:.3f}, N={bottleneck_mask.sum()})')
    # 正常点：半透明の緑の点
    ax.scatter(pts_flat[normal_mask, 0], pts_flat[normal_mask, 1],
               c='limegreen', marker='o', s=30, alpha=0.5, zorder=5,
               label=f'Normal (IoU > {threshold:.3f}, N={normal_mask.sum()})')

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_axis_off()
    ax.set_title('IoUヒートマップ + ボトルネック地点（赤×）\n'
                 f'金沢市 - 下位{BOTTLENECK_PERCENTILE}%ile (IoU ≤ {threshold:.3f})',
                 fontsize=13)
    ax.legend(loc='lower left', fontsize=9, framealpha=0.85)

    # --- 右：ボトルネック地点のみを強調 ---
    ax2 = axes[1]
    edges_proj.plot(ax=ax2, color='lightgray', linewidth=0.3, alpha=0.4, zorder=1)
    if water_proj is not None and not water_proj.empty:
        water_proj.plot(ax=ax2, color='lightblue', alpha=0.6, zorder=2)
    edges_proj.plot(ax=ax2, color='dimgray', linewidth=0.35, alpha=0.5, zorder=3)

    # IoUの値に基づいてボトルネック点をグラデーションで表示
    bx = pts_flat[bottleneck_mask, 0]
    by = pts_flat[bottleneck_mask, 1]
    biou = baseline_iou[bottleneck_mask]
    sc = ax2.scatter(bx, by, c=biou, cmap='YlOrRd_r', s=300,
                     vmin=0.0, vmax=threshold, zorder=5,
                     edgecolors='black', linewidths=0.8, marker='s')

    cbar = fig.colorbar(sc, ax=ax2, shrink=0.6, pad=0.02)
    cbar.set_label('IoU Score (Bottleneck only)', fontsize=10)

    # ボトルネックの重心を表示
    cx, cy = pts_flat[bottleneck_mask, 0].mean(), pts_flat[bottleneck_mask, 1].mean()
    ax2.scatter(cx, cy, c='blue', marker='*', s=400, zorder=7,
               label=f'重心 (ノードプール中心)')
    ax2.legend(loc='lower left', fontsize=9, framealpha=0.85)

    ax2.set_xlim(extent[0], extent[1])
    ax2.set_ylim(extent[2], extent[3])
    ax2.set_axis_off()
    ax2.set_title(f'ボトルネック地点の詳細（IoU値グラデーション）\n'
                  f'★ = ボトルネック重心（候補ノードプールの基準点）',
                  fontsize=13)

    fig.suptitle('金沢市 道路ネットワーク ボトルネック地点の可視化\n'
                 f'（Phase 3 Step 1 – 下位{BOTTLENECK_PERCENTILE}%ile: IoU ≤ {threshold:.3f}, {bottleneck_mask.sum()}地点）',
                 fontsize=14, y=1.01)
    fig.subplots_adjust(left=0.01, right=0.98, top=0.93, bottom=0.01)

    out_path = '/Users/pontarousu/Q1zemi/bottleneck_visualization.png'
    plt.savefig(out_path, dpi=250, bbox_inches='tight')
    print(f"\nSaved → {out_path}")

if __name__ == '__main__':
    main()
