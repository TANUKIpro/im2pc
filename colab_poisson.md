# Poisson Surface Reconstruction ガイド（Google Colab）

点群からメッシュを生成するための Screened Poisson Surface Reconstruction 実行ガイドです。

**要件**: T4 GPU で十分（CPU 処理がメイン）。DiffCD と異なり A100 は不要です。

## DiffCD との比較

| 観点 | Poisson（本ガイド） | DiffCD |
|------|---------------------|--------|
| GPU 要件 | T4 で十分（CPU メイン） | A100 推奨 |
| 処理時間 | 数秒〜数分 | 10分〜1時間 |
| 入力要件 | 点群＋法線（自動推定可） | 点群のみ |
| 出力特性 | 滑らか、watertight | 詳細保持 |
| 適した用途 | 滑らかな表面、高速プレビュー | 複雑な形状、高精度 |

## 1. GPU ランタイムの設定

1. メニューから **ランタイム → ランタイムのタイプを変更** を開く
2. ハードウェアアクセラレータで **GPU (T4)** を選択して保存

> Poisson 再構成は主に CPU 処理のため、T4 で十分です。Open3D の一部処理で GPU を活用します。

## 2. 環境セットアップ

```python
# GPU 確認（T4 で OK）
!nvidia-smi

# 依存パッケージ（JAX/PyTorch 不要）
!pip install -q open3d trimesh plotly
```

> DiffCD と異なり、深層学習フレームワークは不要です。セットアップが軽量・高速です。

## 3. 点群のアップロード

### 方法 A: Google Drive からマウント

```python
from google.colab import drive
drive.mount('/content/drive')

# 点群ファイルをコピー
!cp /content/drive/MyDrive/path/to/object_denoised.ply /content/
```

### 方法 B: ブラウザから直接アップロード

```python
from google.colab import files
uploaded = files.upload()  # object_denoised.ply を選択
```

## 4. 点群の読み込みと確認

```python
import open3d as o3d
import numpy as np

# 読み込み
INPUT_PLY = "object_denoised.ply"
pcd = o3d.io.read_point_cloud(INPUT_PLY)
print(f"Points: {len(pcd.points):,}")
print(f"Has normals: {pcd.has_normals()}")
print(f"Has colors: {pcd.has_colors()}")

# バウンディングボックス
bbox = pcd.get_axis_aligned_bounding_box()
extent = bbox.get_extent()
print(f"Bounding box: {extent[0]:.3f} x {extent[1]:.3f} x {extent[2]:.3f}")
```

## 5. 法線推定（Poisson 必須の前処理）

Poisson 再構成には法線が必須です。点群に法線がない場合、または品質を向上させたい場合は推定します。

```python
def estimate_normals_auto(pcd, k_neighbors=50):
    """
    点群の密度に基づいて適切な radius を自動計算し、法線を推定する。

    Args:
        pcd: Open3D PointCloud
        k_neighbors: 近傍点数（デフォルト: 50）

    Returns:
        法線付き点群
    """
    # 平均点間距離を計算
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)

    # radius を平均距離の 2〜3 倍に設定
    radius = avg_dist * 2.5
    print(f"Average point distance: {avg_dist:.5f}")
    print(f"Using radius: {radius:.5f}, k_neighbors: {k_neighbors}")

    # 法線推定
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=k_neighbors
        )
    )

    # 法線の向きを一貫させる（表面が外向きになるように）
    pcd.orient_normals_consistent_tangent_plane(k=k_neighbors)

    return pcd

# 法線推定を実行
pcd = estimate_normals_auto(pcd)
print(f"Normals estimated: {pcd.has_normals()}")
```

### 法線の可視化（検証用）

```python
# 法線を可視化（サンプリングして表示）
pcd_vis = pcd.uniform_down_sample(every_k_points=max(1, len(pcd.points) // 10000))
o3d.visualization.draw_geometries([pcd_vis], point_show_normal=True)
```

> Colab では `draw_geometries` が動作しない場合があります。その場合は後述の Plotly 可視化を使用してください。

## 6. Poisson Surface Reconstruction 実行

```python
# Poisson 再構成
# depth: 解像度（6-7: プレビュー、8: 標準、9-10: 高精度）
DEPTH = 9

mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd,
    depth=DEPTH,
    scale=1.1,
    linear_fit=False
)

print(f"Raw mesh - Vertices: {len(mesh.vertices):,}, Faces: {len(mesh.triangles):,}")
```

### depth パラメータガイド

| depth | 解像度 | 用途 | 処理時間 |
|-------|--------|------|----------|
| 6-7 | 低 | 高速プレビュー | 数秒 |
| 8 | 中 | 標準的な用途 | 10-30秒 |
| 9 | 高 | 高品質出力 | 1-2分 |
| 10+ | 超高 | 最高精度（メモリ注意） | 数分〜 |

## 7. メッシュのトリミング（低密度領域の除去）

Poisson 再構成は凸包を埋めるため、点群がない領域にも面が生成されます。密度に基づいてトリミングします。

```python
# 密度の統計
densities = np.asarray(densities)
print(f"Density range: {densities.min():.2f} - {densities.max():.2f}")
print(f"Density mean: {densities.mean():.2f}, std: {densities.std():.2f}")

# 低密度頂点を除去（パーセンタイルで閾値設定）
DENSITY_QUANTILE = 0.01  # 下位 1% を除去
density_threshold = np.quantile(densities, DENSITY_QUANTILE)
print(f"Removing vertices with density < {density_threshold:.2f}")

# 密度でフィルタリング
vertices_to_remove = densities < density_threshold
mesh.remove_vertices_by_mask(vertices_to_remove)

print(f"Trimmed mesh - Vertices: {len(mesh.vertices):,}, Faces: {len(mesh.triangles):,}")
```

## 8. メッシュの後処理

### 8.1 孤立メッシュの除去

```python
# 連結成分を取得し、最大のものだけを保持
triangle_clusters, cluster_n_triangles, cluster_area = (
    mesh.cluster_connected_triangles()
)
triangle_clusters = np.asarray(triangle_clusters)
cluster_n_triangles = np.asarray(cluster_n_triangles)

# 最大クラスタのみ保持
largest_cluster_idx = cluster_n_triangles.argmax()
triangles_to_remove = triangle_clusters != largest_cluster_idx
mesh.remove_triangles_by_mask(triangles_to_remove)
mesh.remove_unreferenced_vertices()

print(f"After removing isolated parts - Vertices: {len(mesh.vertices):,}, Faces: {len(mesh.triangles):,}")
```

### 8.2 平滑化（オプション）

```python
# Taubin 平滑化（形状を保持しながら平滑化）
mesh_smooth = mesh.filter_smooth_taubin(number_of_iterations=10)
mesh_smooth.compute_vertex_normals()

# または Laplacian 平滑化（より強い平滑化）
# mesh_smooth = mesh.filter_smooth_laplacian(number_of_iterations=5)
```

### 8.3 穴埋め（Trimesh 使用）

```python
import trimesh

# Open3D → Trimesh 変換
mesh_trimesh = trimesh.Trimesh(
    vertices=np.asarray(mesh_smooth.vertices),
    faces=np.asarray(mesh_smooth.triangles)
)

# 穴埋め
if not mesh_trimesh.is_watertight:
    print("Mesh has holes, attempting to fill...")
    trimesh.repair.fill_holes(mesh_trimesh)
    print(f"After fill_holes - Is watertight: {mesh_trimesh.is_watertight}")
```

## 9. メッシュの保存

```python
# PLY 形式で保存
OUTPUT_PLY = "object_mesh_poisson.ply"
mesh_trimesh.export(OUTPUT_PLY)
print(f"Saved: {OUTPUT_PLY}")

# OBJ 形式でも保存（オプション）
# mesh_trimesh.export("object_mesh_poisson.obj")
```

## 10. 検証（Success Criteria）

```python
import trimesh

mesh = trimesh.load(OUTPUT_PLY)
print(f"Vertices: {len(mesh.vertices):,}")
print(f"Faces: {len(mesh.faces):,}")
print(f"Is watertight: {mesh.is_watertight}")

# バウンディングボックス
bounds = mesh.bounds
extent = bounds[1] - bounds[0]
print(f"Bounding box: {extent[0]:.3f} x {extent[1]:.3f} x {extent[2]:.3f}")

# 表面積・体積（watertight の場合のみ有効）
if mesh.is_watertight:
    print(f"Surface area: {mesh.area:.3f}")
    print(f"Volume: {mesh.volume:.3f}")
```

## 11. 3D ビジュアライゼーション

```python
import numpy as np
import plotly.graph_objects as go

# メッシュ読み込み
mesh = trimesh.load(OUTPUT_PLY)
vertices = mesh.vertices
faces = mesh.faces

# 元の点群も読み込み（比較用）
pcd = o3d.io.read_point_cloud(INPUT_PLY)
pcd_points = np.asarray(pcd.points)

# ダウンサンプリング（表示用）
MAX_POINTS = 50_000
if len(pcd_points) > MAX_POINTS:
    rng = np.random.default_rng(seed=42)
    idx = rng.choice(len(pcd_points), size=MAX_POINTS, replace=False)
    pcd_points = pcd_points[idx]
    print(f"Point cloud downsampled to {MAX_POINTS:,} for visualization")

# Plotly 描画
fig = go.Figure()

# メッシュ
fig.add_trace(go.Mesh3d(
    x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
    color='lightblue',
    opacity=0.7,
    name='Mesh (Poisson)',
))

# 点群（半透明で重ねる）
fig.add_trace(go.Scatter3d(
    x=pcd_points[:, 0], y=pcd_points[:, 1], z=pcd_points[:, 2],
    mode='markers',
    marker=dict(size=1, color='red', opacity=0.3),
    name='Point Cloud',
))

fig.update_layout(
    scene=dict(aspectmode='data'),
    width=900, height=700,
    title='Poisson Mesh vs Point Cloud Comparison',
)
fig.show()
```

### 法線の可視化（Plotly）

```python
# 法線を矢印で表示（サンプリング）
pcd_normals = o3d.io.read_point_cloud(INPUT_PLY)
estimate_normals_auto(pcd_normals)

points = np.asarray(pcd_normals.points)
normals = np.asarray(pcd_normals.normals)

# サンプリング
SAMPLE_SIZE = 1000
rng = np.random.default_rng(seed=42)
idx = rng.choice(len(points), size=min(SAMPLE_SIZE, len(points)), replace=False)
pts = points[idx]
nrm = normals[idx]

# 矢印の長さ
arrow_scale = np.mean(pcd_normals.compute_nearest_neighbor_distance()) * 5

fig = go.Figure()

# 点
fig.add_trace(go.Scatter3d(
    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
    mode='markers',
    marker=dict(size=2, color='blue'),
    name='Points',
))

# 法線（線として表示）
for i in range(len(pts)):
    fig.add_trace(go.Scatter3d(
        x=[pts[i, 0], pts[i, 0] + nrm[i, 0] * arrow_scale],
        y=[pts[i, 1], pts[i, 1] + nrm[i, 1] * arrow_scale],
        z=[pts[i, 2], pts[i, 2] + nrm[i, 2] * arrow_scale],
        mode='lines',
        line=dict(color='red', width=1),
        showlegend=False,
    ))

fig.update_layout(
    scene=dict(aspectmode='data'),
    width=900, height=700,
    title='Point Cloud with Normals',
)
fig.show()
```

## 12. 結果のダウンロード

### 方法 A: Google Drive にコピー

```python
!cp object_mesh_poisson.ply /content/drive/MyDrive/im2pc_output/
```

### 方法 B: ブラウザで直接ダウンロード

```python
from google.colab import files
files.download("object_mesh_poisson.ply")
```

## トラブルシューティング

### 法線が反転している（メッシュが裏返し）

法線の向きが一貫していない場合、メッシュの内外が反転することがあります。

```python
# 法線を反転
pcd.normals = o3d.utility.Vector3dVector(-np.asarray(pcd.normals))

# または、Trimesh で面の向きを修正
mesh_trimesh.fix_normals()
```

### メッシュに穴や膜がある

1. **密度トリミングの閾値を調整**:
```python
DENSITY_QUANTILE = 0.05  # 下位 5% を除去（より積極的）
```

2. **depth を下げる**（粗い再構成で膜を減らす）:
```python
DEPTH = 8  # 9 → 8 に下げる
```

3. **手動で穴埋め**:
```python
trimesh.repair.fill_holes(mesh_trimesh)
```

### メモリ不足（depth が高すぎる）

```
MemoryError または Killed
```

1. `depth` を下げる（10 → 9 → 8）
2. 点群をダウンサンプリング:
```python
pcd_down = pcd.voxel_down_sample(voxel_size=0.01)
```

### 点群の密度が不均一で品質が悪い

1. **事前にダウンサンプリング**して均一化:
```python
pcd = pcd.voxel_down_sample(voxel_size=0.005)
```

2. **統計的外れ値除去**:
```python
pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
```

### 法線推定が遅い・失敗する

点群が大きすぎる場合:
```python
# ダウンサンプリングしてから法線推定
pcd_down = pcd.voxel_down_sample(voxel_size=0.01)
estimate_normals_auto(pcd_down)

# 法線を元の点群に伝播
pcd_tree = o3d.geometry.KDTreeFlann(pcd_down)
normals = []
for point in np.asarray(pcd.points):
    _, idx, _ = pcd_tree.search_knn_vector_3d(point, 1)
    normals.append(np.asarray(pcd_down.normals)[idx[0]])
pcd.normals = o3d.utility.Vector3dVector(np.array(normals))
```

---

## 参考リンク

- [Open3D Poisson Reconstruction](http://www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.html)
- [Screened Poisson Surface Reconstruction (論文)](https://www.cs.jhu.edu/~misha/MyPapers/ToG13.pdf)
- [Open3D Python API](http://www.open3d.org/docs/latest/python_api/index.html)
