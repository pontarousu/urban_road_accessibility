"""
Urban Network Analysis Skills (Generic Utilities)
汎用的に使える「コスト・ラスタ歩行シミュレーション」等の関数群
需要分析など異なるプロジェクトでもインポートして使い回せます。
"""

import numpy as np
import rasterio.transform
from rasterio.features import rasterize
from skimage.graph import MCP_Geometric

def build_cost_surface(edges_proj, water_proj, bounds, resolution=10):
    """
    道路・障害物からマスター・コストラスタ（移動の「しにくさ」の面）を生成する。
    """
    minx, miny, maxx, maxy = bounds
    width  = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))
    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)

    # デフォルト: 道なき平地 (コスト=20)
    cost = np.full((height, width), 20.0, dtype=np.float32)

    # 道路: 高速移動可能 (コスト=1)
    road_shapes = [(geom, 1.0) for geom in edges_proj.geometry]
    cost = rasterize(road_shapes, out_shape=(height, width), transform=transform,
                     fill=20.0, default_value=1.0, dtype=np.float32)

    # 障害物: 侵入不可 (コスト=99999)
    if water_proj is not None and not water_proj.empty:
        water_shapes = [(g, 99999.0) for g in water_proj.geometry
                        if g.geom_type in ('Polygon', 'MultiPolygon')]
        if water_shapes:
            cost = rasterize(water_shapes, out=cost, transform=transform, default_value=99999.0)

    return cost, transform

def calculate_reachability_iou(px, py, cost_surface, transform, limit_cost, euclidean_area, pixel_area):
    """
    Fast Marching Method(ダイクストラ)を用いて、1点からの到達圏面積(IoU)を計算する。
    """
    col, row = ~transform * (px, py)
    r, c = int(row), int(col)
    h, w = cost_surface.shape
    if not (0 <= r < h and 0 <= c < w):
        return 0.0
    if cost_surface[r, c] > 9999: # 障害物の中心の場合は到達不能
        return 0.0

    mcp = MCP_Geometric(cost_surface)
    costs, _ = mcp.find_costs(starts=[(r, c)])
    reached_pixels = np.sum(costs <= limit_cost)
    reached_area = reached_pixels * pixel_area
    return min(reached_area / euclidean_area, 1.0)

# 今後はこのファイルを import urban_network_tools として各実験スクリプトから呼び出せます。
