import osmnx as ox
import networkx as nx

def calculate_capacity(edge_data):
    """
    OSMnxのエッジデータから車線数(lanes)と道路種別(highway)を取得し、
    その道が「1時間あたり何台の車を通せるか（Base Capacity）」を推測して返す。
    デフォルト値（車線数不明の場合の推測値）を持つ。
    """
    # 道路種別ごとの1車線あたりの概算キャパシティ (例: 車/時間)
    # これらは交通工学上の大雑把な仮定値（メゾスコピックモデル用）
    CAPACITY_PER_LANE = {
        'motorway': 2000,
        'trunk': 1500,
        'primary': 1000,
        'secondary': 800,
        'tertiary': 600,
        'residential': 300,
        'unclassified': 300,
        'living_street': 100,
    }
    
    # 欠落している場合の推測車線数
    DEFAULT_LANES = {
        'motorway': 2,
        'trunk': 2,
        'primary': 2,
        'secondary': 1,
        'tertiary': 1,
        'residential': 1,
        'unclassified': 1,
        'living_street': 1,
    }
    
    # highway種別の取得（リストの場合があるため最初の要素を取る）
    hw = edge_data.get('highway', 'unclassified')
    if isinstance(hw, list):
        hw = hw[0]
        
    base_cap_per_lane = CAPACITY_PER_LANE.get(hw, 300)
    
    # lanesの取得
    lanes = edge_data.get('lanes')
    if lanes is None:
        num_lanes = DEFAULT_LANES.get(hw, 1)
    else:
        # lanesは '2' などの文字列、あるいは ['2', '3'] のリストの場合がある
        if isinstance(lanes, list):
            lanes = lanes[0]
        try:
            num_lanes = int(lanes)
        except ValueError:
            num_lanes = DEFAULT_LANES.get(hw, 1)
            
    # キャパシティ = 1車線のキャパシティ × 車線数
    return base_cap_per_lane * num_lanes


# 渋滞時の車間距離込みの1台あたりの道路専有長（メートル）
JAM_SPACING = 7.5

def calculate_storage_capacity(edge_data):
    """
    道路の物理的な貯留容量（同時に何台の車が収まるか）を算出する。

    C_storage = (道路長 / 車間距離) × 車線数

    これにより「この道に今何台入れるか」という絶対数（台）が得られる。
    フローベース容量 (veh/h) とは次元が異なり、パケット人数 N との
    比較に使うことで次元の整合性を保つ。
    """
    # 道路長の取得（デフォルト50m）
    length = edge_data.get('length', 50.0)

    # 車線数の推定（calculate_capacityと同じロジック）
    DEFAULT_LANES = {
        'motorway': 2, 'trunk': 2, 'primary': 2,
        'secondary': 1, 'tertiary': 1, 'residential': 1,
        'unclassified': 1, 'living_street': 1,
    }
    hw = edge_data.get('highway', 'unclassified')
    if isinstance(hw, list):
        hw = hw[0]

    lanes = edge_data.get('lanes')
    if lanes is None:
        num_lanes = DEFAULT_LANES.get(hw, 1)
    else:
        if isinstance(lanes, list):
            lanes = lanes[0]
        try:
            num_lanes = int(lanes)
        except ValueError:
            num_lanes = DEFAULT_LANES.get(hw, 1)

    # 貯留容量 = 道路長 / 車間距離 × 車線数（最低1台は入れる）
    return max(int(length / JAM_SPACING * num_lanes), 1)


def extract_bpr_weight(volume, capacity, free_flow_time, alpha=0.15, beta=4.0):
    """
    BPR (Bureau of Public Roads) 関数に基づく動的所要時間の計算
    
    Parameters:
        volume (float): 現在の道路上の交通量（パケットサイズ）
        capacity (float): 道路の許容量
        free_flow_time (float): 誰もいない時の所要時間（長さ / 制限速度等）
        alpha, beta: BPR関数のパラメータ（標準的には α=0.15, β=4.0）
        
    Returns:
        float: 渋滞ペナルティを加味した実際の所要時間（これが新たなエッジコストとなる）
    """
    if capacity <= 0:
        capacity = 1.0 # ゼロ除算防止
        
    return free_flow_time * (1.0 + alpha * ((volume / capacity) ** beta))

def apply_dynamic_weights(G, capacity_key='capacity'):
    """
    グラフ内の全エッジに対して、現在のvolumeに基づいてBPR関数で動的コスト(bpr_weight)を適用する。

    Parameters:
        G: NetworkXグラフ
        capacity_key: BPRの分母に使う容量属性キー。
                      'capacity' = フローベース(veh/h)、
                      'storage_capacity' = 貯留ベース(veh)。
    """
    for u, v, k, data in G.edges(keys=True, data=True):
        volume = data.get('volume', 0.0)
        capacity = data.get(capacity_key, data.get('capacity', 300.0))
        fft = data.get('free_flow_time', 1.0)  # 自由流時間のデフォルト

        # 新たな重みを計算
        new_weight = extract_bpr_weight(volume, capacity, fft)
        data['bpr_weight'] = new_weight

    return G
