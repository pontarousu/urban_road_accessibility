import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from assign_capacity import calculate_capacity

plt.rcParams['font.family'] = 'Hiragino Sans' # for Mac
plt.rcParams['axes.unicode_minus'] = False

CENTER = (36.5613, 136.6562) # 金沢市中心部
DIST = 1000

def main():
    print("1. グラフのダウンロード...")
    G = ox.graph_from_point(CENTER, dist=DIST, network_type='drive')
    G_proj = ox.project_graph(G)

    print("2. Capacityの算定と色分け...")
    edge_colors = []
    edge_linewidths = []
    
    # カラーマップの準備 (Capacity 100 ~ 4000)
    cmap = plt.cm.get_cmap('plasma') # 視認性の高いカラーマップ
    vmin = 100
    vmax = 3000
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for u, v, k, data in G_proj.edges(keys=True, data=True):
        cap = calculate_capacity(data)
        data['capacity'] = cap
        
        # 容量に応じた色と線の太さを決定
        rgba = cmap(norm(cap))
        edge_colors.append(rgba)
        
        # 太い道(大容量)は線も太くする
        width = 1.0 + (cap / 1000.0)
        edge_linewidths.append(width)

    print("3. プロット生成...")
    fig, ax = ox.plot_graph(
        G_proj,
        edge_color=edge_colors,
        edge_linewidth=edge_linewidths,
        node_size=0,
        bgcolor='white',
        show=False,
        close=False,
        figsize=(12, 12)
    )
    
    ax.set_title('各道路エッジの許容量（Capacity）マップ\n色が明るく線が太いほど、大量の車を裁ける幹線道路', fontsize=16)
    
    # カラーバーの追加
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Capacity (車/時間)', fontsize=12)

    out_path = '/Users/pontarousu/Q1zemi/demand_model_v2/capacity_map.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved → {out_path}")
    plt.close(fig)

if __name__ == '__main__':
    main()
