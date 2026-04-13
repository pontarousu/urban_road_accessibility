import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point, MultiPoint, LineString
import matplotlib.pyplot as plt
import alphashape
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Downloading map data...")
    # 京都市の中心部（烏丸御池周辺）を対象に、半径2500m圏内の道路網を取得
    center_point = (35.0111, 135.7599) # 烏丸御池
    G = ox.graph_from_point(center_point, dist=2500, network_type='drive')
    
    # グラフの投影変換前に中心ノードのIDを取得する（緯度経度ベースで検索するため）
    center_node = ox.distance.nearest_nodes(G, X=center_point[1], Y=center_point[0])
    
    # 距離の計算を正確にするため、投影系をメートル単位系(UTM)に変換
    G_proj = ox.project_graph(G)
    
    print("Calculating IsoDistance...")
    # 到達距離（メートル）
    trip_distance = 1500
    
    # 中心ノードからtrip_distance以内のノードを取得
    subgraph_nodes = nx.single_source_dijkstra_path_length(G_proj, center_node, weight='length', cutoff=trip_distance)
    subgraph = G_proj.subgraph(subgraph_nodes)
    
    # ノードとエッジをGeoDataFrame化
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(subgraph)
    
    print("Generating Polygons...")
    # 1. Convex Hull
    points = MultiPoint(nodes_gdf.geometry.tolist())
    convex_hull = points.convex_hull
    
    # 2. Alpha Shape (凹包)
    # alphaパラメータの調整。メートル単位なので小さい値とする（例: 0.005）
    # 値が大きいほどタイトになり、小さすぎるとConvex Hullに近づく
    try:
        coords = [(p.x, p.y) for p in nodes_gdf.geometry]
        alpha_polygon = alphashape.alphashape(coords, 0.001) # alpha 0.005はきつすぎる可能性があるため0.001に調整
    except Exception as e:
        print(f"Alpha shape failed: {e}")
        alpha_polygon = convex_hull # fallback
        
    # 3. Edge Buffer
    # 全てのエッジを結合してバッファをかける（例:道路の太さを考慮して20m）
    edges_union = edges_gdf.unary_union
    edge_buffer = edges_union.buffer(20)
    
    print("Plotting results...")
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    titles = ['Network Nodes & Edges', 'Convex Hull', 'Alpha Shape (alpha=0.005)', 'Edge Buffer (20m)']
    polygons = [None, convex_hull, alpha_polygon, edge_buffer]
    
    # ベースの道路網全体も描画用に取得（背景用）
    G_bg_nodes, G_bg_edges = ox.graph_to_gdfs(G_proj)
    
    for i, ax in enumerate(axes):
        # 背景のグレーの道路網
        G_bg_edges.plot(ax=ax, color='lightgray', linewidth=0.5, alpha=0.5)
        
        # 中心点
        ax.scatter(nodes_gdf.loc[center_node].geometry.x, nodes_gdf.loc[center_node].geometry.y, 
                   color='red', marker='*', s=200, zorder=5)
                   
        if i == 0:
            # ネットワーク自体
            edges_gdf.plot(ax=ax, color='blue', linewidth=1)
            nodes_gdf.plot(ax=ax, color='black', markersize=2)
        else:
            # ポリゴンを描画
            poly = polygons[i]
            if poly.geom_type == 'MultiPolygon':
                gpd.GeoSeries(list(poly.geoms)).plot(ax=ax, alpha=0.4, color='orange', edgecolor='red')
            else:
                gpd.GeoSeries([poly]).plot(ax=ax, alpha=0.4, color='orange', edgecolor='red')
                
        ax.set_title(titles[i], fontsize=14)
        ax.set_axis_off()
        
    plt.tight_layout()
    output_path = "/Users/pontarousu/Q1zemi/polygon_evaluation_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
