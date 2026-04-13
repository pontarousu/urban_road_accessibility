"""
Phase 3: New Road Optimization Algorithm
========================================
Objective Function: ROI = sum(delta_IoU of nearby sample points) / edge_length
Find the virtual edge (bridge/road) with maximum ROI.

Key steps:
  1. Load/recompute city graph + cost raster + baseline IoU scores
  2. Extract bottleneck sample points (low IoU)
  3. For each bottleneck, find nearby disconnected node pairs (short Euclidean, long network distance)
  4. For each candidate edge: add to graph, rebuild cost raster pixel, recompute IoU for affected points
  5. Compute ROI, rank candidates, visualize best options
"""

import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import rasterio.transform
from rasterio.features import rasterize
from skimage.graph import MCP_Geometric
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# PARAMETERS (adjust here to experiment)
# =====================================================================
CITY_NAME   = 'Kanazawa (Ishikawa, Japan)'
CENTER      = (36.5613, 136.6562)
DIST        = 5000          # network download radius (m)
RES         = 10            # raster resolution (m)
SAMPLE_PITCH = 500          # sampling grid pitch (m)
TRIP_LIMIT  = 1500          # isodistance limit (m)

# Phase 3 specific
BOTTLENECK_PERCENTILE = 30  # sample points below this IoU percentile are "bottlenecks"
MAX_EUCLIDEAN_GAP  = 300    # max Euclidean distance between node pair to consider as candidate (m)
MIN_NETWORK_RATIO  = 3.0    # must be at least this many times longer than Euclidean (detour factor)
ROI_INFLUENCE_RADIUS = 1500 # only sample points within this radius of the new edge are counted (m)
MAX_CANDIDATES     = 50     # cap candidate edges for feasibility
N_BEST_TO_SHOW     = 5      # how many top-ROI edges to visualise

OUTPUT_DIR = '/Users/pontarousu/Q1zemi'

# =====================================================================
# HELPERS
# =====================================================================
def build_cost_surface(edges_proj, water_proj, minx, miny, maxx, maxy, width, height, transform):
    cost = np.full((height, width), 20.0, dtype=np.float32)
    road_shapes = [(geom, 1.0) for geom in edges_proj.geometry]
    cost = rasterize(road_shapes, out_shape=(height, width), transform=transform,
                     fill=20.0, default_value=1.0, dtype=np.float32)
    if water_proj is not None and not water_proj.empty:
        water_shapes = [(g, 99999.0) for g in water_proj.geometry
                        if g.geom_type in ('Polygon', 'MultiPolygon')]
        if water_shapes:
            cost = rasterize(water_shapes, out=cost, transform=transform, default_value=99999.0)
    return cost


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


def batch_iou(points_xy, cost_surface, transform, limit_cost, euclidean_area, pixel_area):
    """Run IoU for a list of (x, y) tuples, return array of IoU values."""
    results = Parallel(n_jobs=-1)(
        delayed(iou_for_point)(px, py, cost_surface, transform,
                               limit_cost, euclidean_area, pixel_area)
        for px, py in points_xy
    )
    return np.array(results)


def add_virtual_edge_to_cost(cost_surface, transform, node_a_xy, node_b_xy):
    """Burn a line between two projected (x,y) points into the cost surface with cost=1."""
    import copy
    new_cost = cost_surface.copy()
    line = LineString([node_a_xy, node_b_xy])
    line_shapes = [(line, 1.0)]
    new_cost = rasterize(line_shapes, out=new_cost, transform=transform, default_value=1.0)
    return new_cost


# =====================================================================
# MAIN
# =====================================================================
def main():
    limit_cost     = TRIP_LIMIT / RES
    pixel_area     = RES * RES
    euclidean_area = np.pi * (TRIP_LIMIT ** 2)

    # ------------------------------------------------------------------
    # 1. Download data
    # ------------------------------------------------------------------
    print(f"=== Phase 3 Optimization: {CITY_NAME} ===")
    print("1. Downloading road network & obstacles...")
    G = ox.graph_from_point(CENTER, dist=DIST, network_type='drive')
    G_proj = ox.project_graph(G)
    nodes_proj, edges_proj = ox.graph_to_gdfs(G_proj)

    try:
        water = ox.features_from_point(CENTER, tags={'natural': 'water', 'waterway': 'riverbank'}, dist=DIST)
        water_proj = water.to_crs(G_proj.graph['crs']) if len(water) > 0 else None
    except Exception as e:
        print(f"  (water fetch failed: {e})")
        water_proj = None

    # ------------------------------------------------------------------
    # 2. Build master cost raster
    # ------------------------------------------------------------------
    print("2. Building master cost raster...")
    bounds = edges_proj.total_bounds
    minx, miny, maxx, maxy = bounds
    width  = int(np.ceil((maxx - minx) / RES))
    height = int(np.ceil((maxy - miny) / RES))
    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)
    base_cost = build_cost_surface(edges_proj, water_proj, minx, miny, maxx, maxy, width, height, transform)

    # ------------------------------------------------------------------
    # 3. Generate sampling grid & compute baseline IoU
    # ------------------------------------------------------------------
    print("3. Computing baseline IoU across city grid...")
    x_coords = np.arange(minx + 500, maxx - 500, SAMPLE_PITCH)
    y_coords = np.arange(miny + 500, maxy - 500, SAMPLE_PITCH)
    mesh_x, mesh_y = np.meshgrid(x_coords, y_coords)
    pts_flat = np.column_stack((mesh_x.ravel(), mesh_y.ravel()))

    baseline_iou = batch_iou(pts_flat, base_cost, transform, limit_cost, euclidean_area, pixel_area)
    print(f"  Median IoU = {np.median(baseline_iou):.3f}, Min = {np.min(baseline_iou):.3f}")

    # ------------------------------------------------------------------
    # 4. Identify bottleneck sample points
    # ------------------------------------------------------------------
    threshold = np.percentile(baseline_iou[baseline_iou > 0], BOTTLENECK_PERCENTILE)
    bottleneck_mask = (baseline_iou > 0) & (baseline_iou <= threshold)
    bottleneck_pts  = pts_flat[bottleneck_mask]
    print(f"4. Bottleneck points (below {BOTTLENECK_PERCENTILE}th pct, IoU≤{threshold:.3f}): {len(bottleneck_pts)}")

    # ------------------------------------------------------------------
    # 5. Find candidate virtual edges
    # ------------------------------------------------------------------
    print(f"5. Searching candidate node pairs (Euclidean ≤ {MAX_EUCLIDEAN_GAP}m, detour ≥ {MIN_NETWORK_RATIO}x)...")
    # Build projected node list
    node_ids = list(G_proj.nodes)
    node_xy  = np.array([(G_proj.nodes[n]['x'], G_proj.nodes[n]['y']) for n in node_ids])

    # Filter to nodes near any bottleneck point (speed up search)
    bot_center_x = bottleneck_pts[:, 0].mean()
    bot_center_y = bottleneck_pts[:, 1].mean()
    # Use only nodes within DIST/2 of any bottleneck (rough filter)
    dists_to_center = np.sqrt((node_xy[:, 0] - bot_center_x)**2 + (node_xy[:, 1] - bot_center_y)**2)
    local_mask = dists_to_center < (DIST * 0.8)
    local_ids  = [node_ids[i] for i in range(len(node_ids)) if local_mask[i]]
    local_xy   = node_xy[local_mask]

    print(f"  Candidate node pool: {len(local_ids)} nodes")

    # Check pairs where Euclidean is small but network path length is large
    candidates = []  # list of (nodeA_id, nodeB_id, eucl_dist, network_dist)
    n = len(local_ids)

    # Random sampling of pairs for performance (exhaustive would be O(n^2))
    rng = np.random.default_rng(42)
    sample_count = min(n, 300)
    sampled_idx = rng.choice(n, size=sample_count, replace=False)

    for ii in sampled_idx:
        nA = local_ids[ii]
        xA, yA = local_xy[ii]
        for jj in sampled_idx:
            if jj <= ii:
                continue
            nB = local_ids[jj]
            xB, yB = local_xy[jj]
            eucl = np.sqrt((xA - xB)**2 + (yA - yB)**2)
            if eucl > MAX_EUCLIDEAN_GAP or eucl < 50:
                continue
            # Network distance (undirected)
            try:
                nd = nx.shortest_path_length(G_proj.to_undirected(), nA, nB, weight='length')
            except nx.NetworkXNoPath:
                nd = float('inf')
            if nd == float('inf') or nd / eucl >= MIN_NETWORK_RATIO:
                candidates.append((nA, nB, eucl, nd, xA, yA, xB, yB))

            if len(candidates) >= MAX_CANDIDATES:
                break
        if len(candidates) >= MAX_CANDIDATES:
            break

    print(f"  Found {len(candidates)} candidate virtual edges")
    if len(candidates) == 0:
        print("  No candidates found. Try relaxing MAX_EUCLIDEAN_GAP or MIN_NETWORK_RATIO.")
        return

    # ------------------------------------------------------------------
    # 6. Evaluate ROI for each candidate edge
    # ------------------------------------------------------------------
    print("6. Evaluating ROI for each candidate (recomputing IoU with virtual edge)...")
    roi_results = []

    # Identify which sample points are within influence radius of each candidate
    for idx, (nA, nB, eucl, nd, xA, yA, xB, yB) in enumerate(candidates):
        # Mid-point of proposed edge
        mid_x, mid_y = (xA + xB) / 2, (yA + yB) / 2
        dists_from_edge = np.sqrt((pts_flat[:, 0] - mid_x)**2 + (pts_flat[:, 1] - mid_y)**2)
        affected_mask = dists_from_edge <= ROI_INFLUENCE_RADIUS

        if not np.any(affected_mask):
            continue

        # Burn virtual edge into cost surface
        new_cost = add_virtual_edge_to_cost(base_cost, transform, (xA, yA), (xB, yB))

        # Recompute IoU only for affected points
        affected_pts  = pts_flat[affected_mask]
        new_iou       = batch_iou(affected_pts.tolist(), new_cost, transform,
                                  limit_cost, euclidean_area, pixel_area)
        old_iou       = baseline_iou[affected_mask]
        delta_iou_sum = np.sum(np.maximum(new_iou - old_iou, 0))

        roi = delta_iou_sum / eucl if eucl > 0 else 0.0
        roi_results.append({
            'nA': nA, 'nB': nB,
            'xA': xA, 'yA': yA, 'xB': xB, 'yB': yB,
            'eucl_m': eucl, 'network_m': nd,
            'delta_iou_sum': delta_iou_sum,
            'roi': roi,
            'new_cost_surface': new_cost,
            'baseline_iou_affected': old_iou,
            'new_iou_affected': new_iou,
            'affected_mask': affected_mask,
        })
        print(f"  [{idx+1}/{len(candidates)}] edge={eucl:.0f}m, ΔIoU_sum={delta_iou_sum:.4f}, ROI={roi:.6f}")

    if not roi_results:
        print("  No ROI results computed.")
        return

    roi_results.sort(key=lambda r: r['roi'], reverse=True)

    # ------------------------------------------------------------------
    # 7. Visualise top-N candidates
    # ------------------------------------------------------------------
    print(f"7. Visualising top {N_BEST_TO_SHOW} candidates...")
    fig, axes = plt.subplots(1, N_BEST_TO_SHOW, figsize=(7 * N_BEST_TO_SHOW, 9))
    if N_BEST_TO_SHOW == 1:
        axes = [axes]

    import numpy.ma as ma
    iou_grid_base = baseline_iou.reshape((len(y_coords), len(x_coords)))

    extent = [x_coords[0] - SAMPLE_PITCH/2, x_coords[-1] + SAMPLE_PITCH/2,
              y_coords[0] - SAMPLE_PITCH/2, y_coords[-1] + SAMPLE_PITCH/2]

    for rank, (ax, res) in enumerate(zip(axes, roi_results[:N_BEST_TO_SHOW])):
        # After IoUグリッドを構築（影響エリアのみ更新）
        iou_grid_after = baseline_iou.copy()
        iou_grid_after[res['affected_mask']] = np.maximum(
            res['new_iou_affected'], baseline_iou[res['affected_mask']])
        iou_grid_after = iou_grid_after.reshape((len(y_coords), len(x_coords)))

        # delta（改善量）グリッド
        delta_grid = iou_grid_after - iou_grid_base

        # ① ベースラインのIoUヒートマップ（背景・薄く）
        edges_proj.plot(ax=ax, color='lightgray', linewidth=0.3, alpha=0.4, zorder=1)
        ax.imshow(ma.masked_where(iou_grid_base == 0, iou_grid_base),
                  extent=extent, origin='lower', cmap='RdYlGn',
                  alpha=0.4, vmin=0.0, vmax=0.7, zorder=2)

        # ② 改善エリアの表示: deltaが正の場所のみ「After IoU値」を強調表示（無色にならないようAbsoluteで描く）
        # vminをdeltaの実際の最小正値に合わせてダイナミック化
        after_highlight = ma.masked_where(delta_grid <= 0.001, iou_grid_after)
        vmax_dyn = min(iou_grid_after.max(), 0.7)
        im_after = ax.imshow(after_highlight, extent=extent, origin='lower', cmap='Blues',
                             alpha=0.80, vmin=0.0, vmax=vmax_dyn, zorder=3)

        # ③ 道路ネットワーク（最前面）
        edges_proj.plot(ax=ax, color='dimgray', linewidth=0.4, alpha=0.5, zorder=4)

        # ④ 提案路線（赤線）
        ax.plot([res['xA'], res['xB']], [res['yA'], res['yB']],
                color='crimson', linewidth=3.5, zorder=6, solid_capstyle='round')
        ax.scatter([res['xA'], res['xB']], [res['yA'], res['yB']],
                   color='crimson', s=90, zorder=7)

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_axis_off()

        n_improved = int(np.sum(delta_grid > 0.001))
        ax.set_title(
            f"Rank #{rank+1}\nROI={res['roi']:.5f}\n"
            f"Length={res['eucl_m']:.0f}m  ΔIoU_sum={res['delta_iou_sum']:.3f}\n"
            f"Improved cells: {n_improved}",
            fontsize=10, pad=6
        )

    fig.suptitle(
        f"Top {N_BEST_TO_SHOW} Virtual Road Candidates — {CITY_NAME}\n"
        f"Crimson = proposed road | Blue cells = IoU-improved area (After IoU absolute value)",
        fontsize=13, y=1.01
    )
    fig.subplots_adjust(wspace=0.05, top=0.88)
    out_path = f'{OUTPUT_DIR}/optimization_result.png'
    plt.savefig(out_path, dpi=250, bbox_inches='tight')
    print(f"  Saved \u2192 {out_path}")

    # ------------------------------------------------------------------
    # 8. Print ranking table
    # ------------------------------------------------------------------
    print("\n====== TOP CANDIDATES RANKING ======")
    print(f"{'Rank':>4} | {'ROI':>10} | {'Length(m)':>10} | {'ΔIoU_sum':>10} | {'Detour ratio':>13}")
    print("-" * 60)
    for rank, r in enumerate(roi_results[:10], 1):
        dr = r['network_m'] / r['eucl_m'] if r['eucl_m'] > 0 else float('inf')
        print(f"{rank:>4} | {r['roi']:>10.6f} | {r['eucl_m']:>10.1f} | {r['delta_iou_sum']:>10.4f} | {dr:>13.1f}x")

    # ------------------------------------------------------------------
    # 9. Single best candidate: Before/After comparison
    # ------------------------------------------------------------------
    best = roi_results[0]
    print(f"\n8. Generating before/after map for best candidate...")

    # Before/After の共通extentは既に計算済み
    fig2, axes2 = plt.subplots(1, 2, figsize=(22, 10),
                               gridspec_kw={'wspace': 0.06})
    ax_b, ax_a = axes2

    before_grid = iou_grid_base
    after_arr   = baseline_iou.copy()
    after_arr[best['affected_mask']] = np.maximum(
        best['new_iou_affected'], baseline_iou[best['affected_mask']])
    after_grid = after_arr.reshape((len(y_coords), len(x_coords)))

    for ax, g, label in [(ax_b, before_grid, 'BEFORE'), (ax_a, after_grid, 'AFTER')]:
        edges_proj.plot(ax=ax, color='lightgray', linewidth=0.3, alpha=0.4, zorder=1)
        im2 = ax.imshow(ma.masked_where(g == 0, g), extent=extent, origin='lower',
                        cmap='RdYlGn', alpha=0.65, vmin=0.0, vmax=0.7, zorder=2)
        edges_proj.plot(ax=ax, color='dimgray', linewidth=0.4, alpha=0.55, zorder=3)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_axis_off()

        med_iou = np.median(best['baseline_iou_affected'] if label == 'BEFORE'
                            else best['new_iou_affected'])
        ax.set_title(f'{label}\nMedian IoU (affected area) = {med_iou:.3f}',
                     fontsize=15, pad=10)

        if label == 'AFTER':
            ax.plot([best['xA'], best['xB']], [best['yA'], best['yB']],
                    color='crimson', linewidth=5, zorder=5)
            ax.scatter([best['xA'], best['xB']], [best['yA'], best['yB']],
                       color='crimson', s=150, zorder=6)
            ax.annotate('Proposed road',
                        xy=((best['xA']+best['xB'])/2, (best['yA']+best['yB'])/2),
                        fontsize=11, color='crimson', fontweight='bold',
                        xytext=(20, 20), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='crimson'))

    # カラーバーを図の右端に独立して配置（画像と被らないように）
    cbar_ax = fig2.add_axes([0.92, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
    cbar = fig2.colorbar(im2, cax=cbar_ax)
    cbar.set_label('IoU Score', fontsize=12)

    fig2.suptitle(
        f'Best Candidate Before/After — {CITY_NAME}\n'
        f'Proposed length: {best["eucl_m"]:.0f}m  |  '
        f'ROI: {best["roi"]:.5f}  |  ΔIoU_sum: {best["delta_iou_sum"]:.3f}',
        fontsize=15, y=1.02
    )
    fig2.subplots_adjust(left=0.02, right=0.91, top=0.93, bottom=0.02)

    out_path2 = f'{OUTPUT_DIR}/optimization_best_beforeafter.png'
    plt.savefig(out_path2, dpi=250, bbox_inches='tight')
    print(f"  Saved \u2192 {out_path2}")
    print("\nPhase 3 complete!")


if __name__ == '__main__':
    main()
