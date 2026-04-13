# Urban Road Network Accessibility & Optimization 🛣️

<p align="center">
  <strong>都市道路ネットワーク構造の定量分析と、新規道路建設地点の最適化</strong>
</p>

> [!NOTE]
> **Background / Disclaimer**
> 本リポジトリは、大学のゼミナール（Q1zemi）における研究プロジェクトのプロトタイプとして開発されたものです。OpenStreetMap (OSM) を活用し、物理的な障害物（川など）を考慮したコスト・ラスタによる到達圏解析（Cost Raster & Fast Marching Method）の実証実験を目的としています。試験的なコードが含まれており、現在も需要モデル（Demand Model）を取り入れた検証に向けて開発を継続中のリポジトリです。

---

## 概要 (Overview)
任意の都市における「道路ネットワークの交通効率」を定量化し、交通効率が著しく低い（分断されている）ボトルネック地点を特定します。さらに、**最も短い道路・橋を最も少ない投資で建設することで、最大の効率改善をもたらす最適な建設場所**を自動的に提案する最適化アルゴリズムの実装です。

金沢市をテストベッドとし、中心部の密な街区、川による分断、山間部の過疎エリアという対照的な特徴に対するアルゴリズムの有効性を実証しました。

## プロジェクトの3大成果 (Key Achievements)

### 1. 普遍的な到達面積算出手法の確立
既存のConvex Hull（凸包）やAlpha Shapeが抱えていた「川や山などの到達不能領域を跨いでしまう問題」を解決するため、**コスト・ラスタ表面上の歩行シミュレーション（Fast Marching Method）**を導入しました。これにより、都市・山間・障害物エリアの全てで物理的に矛盾のない到達圏（IoU）を算出することに成功しました。

### 2. 都市全体の交通効率（IoU）ヒートマップの生成
金沢市全域（10km×10km）を500mメッシュでサンプリングし、並列処理によって高速にIoUヒートマップを生成。ネットワーク上の「見えないバリア」を可視化し、下位30パーセンタイル（IoU ≤ 0.076）の108地点を交通ボトルネックとして抽出しました。

### 3. ROIに基づく新規道路建設の最適化提案
$$\text{ROI} = \frac{\sum \Delta\text{IoU}}{\text{新しい道路の長さ [m]}}$$
上記の目的関数に基づき、「物理的には近いがネットワーク上は遠い（迂回を強いられている）」地点を探索。金沢市において、わずか116mの橋を架けるだけで9.6倍の迂回ルートを劇的に解消できる最適候補地をアルゴリズムによって自動同定しました。

## ディレクトリ構成 (Directory Structure)

*   **`reachability_model_v1/`** ... 到達可能性を中心とした基本モデルのソースコード、検証ログ、図表（Phase 1~3 の全アーカイブ）
*   **`demand_model_v2/`** ... 居住人口や交通量などの「道路需要」を目的関数に組み込む次期バージョンの実験ディレクトリ
    *   `urban_network_tools.py` : コスト・ラスタ計算を他プロジェクトでも再利用するための汎用モジュール

## 技術スタック (Tech Stack)
*   **Language:** Python 3.9
*   **Geospatial & Network:** `osmnx`, `networkx`, `geopandas`, `shapely`
*   **Raster & Surface Analysis:** `rasterio`, `scikit-image` (`MCP_Geometric`)
*   **Parallel Computing:** `joblib`
*   **Visualization:** `matplotlib`
