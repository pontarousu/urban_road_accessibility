import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio.transform
from rasterio.features import rasterize
from skimage.graph import MCP_Geometric
import warnings
warnings.filterwarnings('ignore')

def run_cost_raster_simulation(ax, title, center_point, dist, output_limit):
    print(f"\n--- Processing {title} ---")
    
    # 1. 道路網の取得
    print("  Downloading road network...")
    try:
        G = ox.graph_from_point(center_point, dist=dist, network_type='drive')
        G_proj = ox.project_graph(G)
        nodes_proj, edges_proj = ox.graph_to_gdfs(G_proj)
    except Exception as e:
        print(f"  Failed getting road network: {e}")
        return
        
    # 2. 水域（障害物）の取得
    print("  Downloading water bodies (obstacles)...")
    try:
        water = ox.features_from_point(center_point, tags={'natural': 'water', 'waterway': 'riverbank'}, dist=dist)
        if len(water) > 0:
            water_proj = water.to_crs(G_proj.graph['crs'])
        else:
            water_proj = None
    except Exception as e:
        print(f"  No water bodies found or failed: {e}")
        water_proj = None

    # 3. ラスタライズの設定（10m解像度）
    print("  Setting up Raster...")
    bounds = edges_proj.total_bounds
    minx, miny, maxx, maxy = bounds
    res = 10 # 10m pixel
    
    width = int(np.ceil((maxx - minx) / res))
    height = int(np.ceil((maxy - miny) / res))
    
    # Affine transform for rasterio
    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)
    
    # 4. コストサーフェスの構築
    # デフォルトのオフロードコスト（徒歩: 道路の20倍遅いとする）
    cost_surface = np.full((height, width), 20.0, dtype=np.float32)
    
    # 道路をコスト1で上書き
    road_shapes = [(geom, 1.0) for geom in edges_proj.geometry]
    cost_surface = rasterize(road_shapes, out_shape=(height, width), transform=transform, 
                             fill=20.0, default_value=1.0, dtype=np.float32)
                             
    # 障害物をコスト無限大で上書き
    if water_proj is not None and not water_proj.empty:
        water_shapes = []
        for geom in water_proj.geometry:
            if geom.geom_type in ['Polygon', 'MultiPolygon']:
                water_shapes.append((geom, 99999.0))
        if water_shapes:
            cost_surface = rasterize(water_shapes, out=cost_surface, transform=transform, default_value=99999.0)

    # 5. MCP (ダイクストラ) の実行
    print("  Running Fast Marching (Cost Raster)...")
    # 中心点のピクセル座標を計算
    center_y, center_x = ~transform * (center_point[1], center_point[0]) # 注意: transformは(lon, lat)など元の座標系が必要なので、G_projの座標系での中心を使う
    
    # 投影座標系での中心点を見つける（nearest nodeの座標を使うのが確実）
    center_node_id = ox.distance.nearest_nodes(G, X=center_point[1], Y=center_point[0])
    center_pt_proj = nodes_proj.loc[center_node_id].geometry
    col, row = ~transform * (center_pt_proj.x, center_pt_proj.y)
    
    row, col = int(row), int(col)
    
    # コスト計算（距離1あたりのコスト × 距離。解像度をかける必要があるがMCP_Geometricは格子間隔1として計算するため、後でresをかける）
    mcp = MCP_Geometric(cost_surface)
    cumulative_costs, _ = mcp.find_costs(starts=[(row, col)])
    
    # 到達限界：道路上を output_limit メートル進める距離に設定（10mメッシュでコスト1なので、リミットは output_limit / 10）
    limit_cost = output_limit / res
    reached_mask = cumulative_costs <= limit_cost
    
    # 6. 結果のプロット表示
    print("  Plotting...")
    # 背景
    edges_proj.plot(ax=ax, color='lightgray', linewidth=0.5, alpha=0.5)
    
    # 障害物
    if water_proj is not None and not water_proj.empty:
        water_proj.plot(ax=ax, color='lightblue', alpha=0.8)
        
    # ラスタの結果をポリゴン化して描画すると重いので、imshowを使う
    extent = [minx, maxx, miny, maxy]
    # マスクされた領域のみ色を付ける
    display_img = np.zeros((*reached_mask.shape, 4))
    display_img[reached_mask] = [1.0, 0.5, 0.0, 0.6] # Orange with transparency
    
    ax.imshow(display_img, extent=extent, origin='upper', zorder=2)
    
    # 起点
    ax.scatter(center_pt_proj.x, center_pt_proj.y, color='red', marker='*', s=200, zorder=5)
    
    ax.set_title(title, fontsize=14)
    ax.set_axis_off()

def main():
    scenarios = [
        {'name': 'Kyoto (Grid Urban)', 'center': (35.0111, 135.7599), 'dist': 3000, 'limit': 2000},
        {'name': 'Tokyo (Obstacle Urban)', 'center': (35.6812, 139.7671), 'dist': 3000, 'limit': 2000},
        {'name': 'Hakone (Mountain)', 'center': (35.2324, 139.1069), 'dist': 3000, 'limit': 2000}
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    for i, s in enumerate(scenarios):
        run_cost_raster_simulation(axes[i], s['name'], s['center'], s['dist'], s['limit'])
        
    plt.tight_layout()
    output_path = "/Users/pontarousu/Q1zemi/cost_raster_evaluation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nAll done! Plot saved to {output_path}")

if __name__ == "__main__":
    main()
