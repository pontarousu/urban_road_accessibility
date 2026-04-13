import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import numpy as np
from rasterio.features import rasterize
import rasterio.transform
from skimage.graph import MCP_Geometric
from joblib import Parallel, delayed
import time
import warnings
warnings.filterwarnings('ignore')

def calculate_iou_for_point(idx, px, py, cost_surface, transform, limit_cost, euclidean_area, pixel_area):
    """
    1箇所のサンプリング地点におけるIoUを計算するワーカー関数
    """
    col, row = ~transform * (px, py)
    r, c = int(row), int(col)
    
    height, width = cost_surface.shape
    if not (0 <= r < height and 0 <= c < width):
        return idx, px, py, 0.0 # 範囲外
        
    # 水のど真ん中などコスト無限大からのスタートはIoU 0とする
    if cost_surface[r, c] > 9999:
        return idx, px, py, 0.0
        
    mcp = MCP_Geometric(cost_surface)
    # 起点からの最小コスト配列を取得
    costs, _ = mcp.find_costs(starts=[(r, c)])
    
    # 到達可能なピクセル数をカウント
    reached_pixels = np.sum(costs <= limit_cost)
    reached_area = reached_pixels * pixel_area
    
    # 実際はユークリッド面積を超えないように上限をクリップ
    iou = min(reached_area / euclidean_area, 1.0)
    
    return idx, px, py, iou

def main():
    start_time = time.time()
    
    city_name = 'Kanazawa (Ishikawa, Japan)'
    center_point = (36.5613, 136.6562) # 金沢市中心部（兼六園・中心街寄り）
    dist = 5000 # 半径5km（10km x 10kmのエリア）
    res = 10 # 10mメッシュ
    sample_pitch = 500 # 500m間隔で測定
    trip_distance_limit = 1500 # 1.5km（約20分徒歩圏内）の広がりを評価
    
    limit_cost = trip_distance_limit / res
    pixel_area = res * res
    euclidean_area = np.pi * (trip_distance_limit ** 2)

    print(f"--- Generating Heatmap for {city_name} ---")
    
    print("1. Downloading Network and Obstacles...")
    G = ox.graph_from_point(center_point, dist=dist, network_type='drive')
    G_proj = ox.project_graph(G)
    nodes_proj, edges_proj = ox.graph_to_gdfs(G_proj)
    
    try:
        water = ox.features_from_point(center_point, tags={'natural': 'water', 'waterway': 'riverbank'}, dist=dist)
        if len(water) > 0:
            water_proj = water.to_crs(G_proj.graph['crs'])
        else:
            water_proj = None
    except Exception as e:
        print(f"  Water fetch failed: {e}")
        water_proj = None

    print(f"2. Building Master Cost Raster (Resolution: {res}m)...")
    bounds = edges_proj.total_bounds
    minx, miny, maxx, maxy = bounds
    width = int(np.ceil((maxx - minx) / res))
    height = int(np.ceil((maxy - miny) / res))
    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)
    
    cost_surface = np.full((height, width), 20.0, dtype=np.float32)
    road_shapes = [(geom, 1.0) for geom in edges_proj.geometry]
    cost_surface = rasterize(road_shapes, out_shape=(height, width), transform=transform, 
                             fill=20.0, default_value=1.0, dtype=np.float32)
                             
    if water_proj is not None and not water_proj.empty:
        water_shapes = []
        for geom in water_proj.geometry:
            if geom.geom_type in ['Polygon', 'MultiPolygon']:
                water_shapes.append((geom, 99999.0))
        if water_shapes:
            cost_surface = rasterize(water_shapes, out=cost_surface, transform=transform, default_value=99999.0)

    print("3. Generating Sampling Points...")
    x_coords = np.arange(minx + 500, maxx - 500, sample_pitch)
    y_coords = np.arange(miny + 500, maxy - 500, sample_pitch)
    mesh_x, mesh_y = np.meshgrid(x_coords, y_coords)
    points_flat = np.column_stack((mesh_x.ravel(), mesh_y.ravel()))
    
    print(f"  Total points to evaluate: {len(points_flat)}")
    
    print("4. Running Batch Simulation (Parallel Joblib)...")
    # joblib.Parallelを使用して並列処理を実行 (n_jobs=-1で利用可能なすべてのコアを利用)
    # ※ cost_surface はメモリマップ経由でワーカープロセスに共有されます
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(calculate_iou_for_point)(
            i, pt[0], pt[1], cost_surface, transform, limit_cost, euclidean_area, pixel_area
        ) for i, pt in enumerate(points_flat)
    )
    
    print("5. Plotting Heatmap...")
    # 結果の集計とGeoDataFrame化
    res_x = [r[1] for r in results]
    res_y = [r[2] for r in results]
    res_iou = [r[3] for r in results]
    res_geometry = [Point(x, y) for x, y in zip(res_x, res_y)]
    gdf_results = gpd.GeoDataFrame({'IoU': res_iou}, geometry=res_geometry, crs=G_proj.graph['crs'])
    
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # 背景の水域
    if water_proj is not None and not water_proj.empty:
        water_proj.plot(ax=ax, color='lightblue', alpha=0.5, zorder=2)
        
    # ヒートマップ（グリッド画像として隙間なくプロット）
    iou_grid = np.array(res_iou).reshape((len(y_coords), len(x_coords)))
    # IoUが0.0の地点（完全に中心が水域などの無効地点）は透過させるためマスクする
    import numpy.ma as ma
    iou_grid_masked = ma.masked_where(iou_grid == 0.0, iou_grid)
    
    extent = [
        x_coords[0] - sample_pitch/2, 
        x_coords[-1] + sample_pitch/2, 
        y_coords[0] - sample_pitch/2, 
        y_coords[-1] + sample_pitch/2
    ]
    # ヒートマップを最背面（zorder=1）に配置し、透明度をやや高めに設定
    sc = ax.imshow(iou_grid_masked, extent=extent, origin='lower', cmap='RdYlGn', 
                   alpha=0.55, vmin=0.0, vmax=0.7, zorder=1)
                   
    # 道路ネットワークを最前面（zorder=3）に描画し、道路の形状がはっきり見えるようにする
    edges_proj.plot(ax=ax, color='dimgray', linewidth=0.4, alpha=0.7, zorder=3)
    
    # ヒートマップの範囲外の川や道が表示されないよう、分析対象エリア（extent）でトリミング
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    
    plt.colorbar(sc, ax=ax, label='Network Efficiency Score (IoU)', shrink=0.7)
    
    ax.set_title(f'Urban Connectivity Heatmap - {city_name}\n(IoU based on Cost Raster Simulation - Radius {trip_distance_limit}m)', fontsize=16)
    ax.set_axis_off()
    
    output_path = "/Users/pontarousu/Q1zemi/heatmap_kanazawa.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    elapsed = time.time() - start_time
    print(f"\nAll done in {elapsed:.1f} seconds! Heatmap saved to {output_path}")

if __name__ == "__main__":
    main()
