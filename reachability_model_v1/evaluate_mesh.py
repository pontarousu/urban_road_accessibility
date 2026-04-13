import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point, MultiPoint, box
import matplotlib.pyplot as plt
import alphashape
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def create_grid(gdf, cell_size=50):
    """バウンディングボックスを指定されたサイズのメッシュで分割する関数"""
    bounds = gdf.total_bounds # [minx, miny, maxx, maxy]
    minx, miny, maxx, maxy = bounds
    
    # xとyの座標配列を作成
    x_coords = np.arange(minx, maxx, cell_size)
    y_coords = np.arange(miny, maxy, cell_size)
    
    # メッシュ（四角形ポリゴン）のリストを作成
    polymesh = []
    for x in x_coords:
        for y in y_coords:
            polymesh.append(box(x, y, x + cell_size, y + cell_size))
            
    # GeoDataFrameにして返す
    grid = gpd.GeoDataFrame({'geometry': polymesh}, crs=gdf.crs)
    return grid

def main():
    print("Downloading Tokyo map data (around Tokyo Station)...")
    # 東京駅周辺、広めに半径3000m（皇居が確実に入るように）
    center_point = (35.6812, 139.7671) # 東京駅
    G = ox.graph_from_point(center_point, dist=3000, network_type='drive')
    
    # グラフの投影変換前に中心ノードのIDを取得する
    center_node = ox.distance.nearest_nodes(G, X=center_point[1], Y=center_point[0])
    
    # メートル単位系(UTM)に変換
    G_proj = ox.project_graph(G)
    
    print("Calculating IsoDistance...")
    # 到達距離（2000メートル：皇居の裏あたりまで到達するかどうかの距離）
    trip_distance = 2000
    
    subgraph_nodes = nx.single_source_dijkstra_path_length(G_proj, center_node, weight='length', cutoff=trip_distance)
    subgraph = G_proj.subgraph(subgraph_nodes)
    
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(subgraph)
    G_bg_nodes, G_bg_edges = ox.graph_to_gdfs(G_proj)
    
    print("Generating Polygons & Mesh...")
    # 1. Convex Hull
    points = MultiPoint(nodes_gdf.geometry.tolist())
    convex_hull = points.convex_hull
    
    # 2. Alpha Shape (凹包)
    try:
        coords = [(p.x, p.y) for p in nodes_gdf.geometry]
        alpha_polygon = alphashape.alphashape(coords, 0.001)
    except Exception as e:
        print(f"Alpha shape failed: {e}")
        alpha_polygon = convex_hull
        
    # 3. Mesh (Grid Rasterization)
    print("  Creating base grid...")
    # 計算対象のグラフ全体のバウンディングボックスに対して50mメッシュを作成
    grid = create_grid(G_bg_edges, cell_size=50)
    
    print("  Intersecting grid with reached edges...")
    # 到達したエッジと交差するメッシュセルだけを抽出（＝実際の面的到達域）
    # sjoinを使ってエッジと重なるメッシュを取得
    reached_grid = gpd.sjoin(grid, edges_gdf, how='inner', predicate='intersects')
    # 重複を削除
    reached_grid = reached_grid.drop_duplicates(subset='geometry')
    
    # 表示用にメッシュ全体を結合して一つのマルチポリゴンにする（重いのを防ぐため）
    mesh_polygon = reached_grid.unary_union
    
    print("Plotting results...")
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    titles = ['Network Edges', 'Convex Hull', 'Alpha Shape', 'Grid Mesh (50m)']
    polygons = [None, convex_hull, alpha_polygon, mesh_polygon]
    
    for i, ax in enumerate(axes):
        # 背景（皇居の空き地も可視化される）
        G_bg_edges.plot(ax=ax, color='lightgray', linewidth=0.5, alpha=0.5)
        
        # 中心点（東京駅）
        ax.scatter(nodes_gdf.loc[center_node].geometry.x, nodes_gdf.loc[center_node].geometry.y, 
                   color='red', marker='*', s=200, zorder=5)
                   
        if i == 0:
            edges_gdf.plot(ax=ax, color='blue', linewidth=1)
            nodes_gdf.plot(ax=ax, color='black', markersize=2)
        else:
            poly = polygons[i]
            if poly.geom_type in ['MultiPolygon', 'GeometryCollection']:
                gpd.GeoSeries(list(poly.geoms)).plot(ax=ax, alpha=0.5, color='orange', edgecolor='none')
            else:
                gpd.GeoSeries([poly]).plot(ax=ax, alpha=0.5, color='orange', edgecolor='none')
                
        ax.set_title(titles[i], fontsize=14)
        ax.set_axis_off()
        
    plt.tight_layout()
    output_path = "/Users/pontarousu/Q1zemi/mesh_evaluation_tokyo.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
