# DiffCD メッシュ生成ガイド（Google Colab）

点群からメッシュを生成するための DiffCD（ECCV 2024）実行ガイドです。

**要件**: A100 GPU（40GB VRAM）推奨。T4/V100 でも動作しますが、点群サイズの制限が厳しくなります。

## 1. GPU ランタイムの設定

1. メニューから **ランタイム → ランタイムのタイプを変更** を開く
2. ハードウェアアクセラレータで **GPU (A100)** を選択して保存

> A100 は 40GB VRAM を搭載しており、100万点規模の点群を処理できます。

## 2. 環境セットアップ

```python
# GPU 確認
!nvidia-smi

# DiffCD クローン
!git clone https://github.com/Linusnie/diffcd.git
%cd diffcd

# 依存パッケージ（JAX + CUDA 12）
!pip install -r requirements.txt
!pip install --upgrade "jax[cuda12_pip]==0.4.14" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 点群処理用
!pip install open3d trimesh
```

## 3. 点群のアップロード

### 方法 A: Google Drive からマウント

```python
from google.colab import drive
drive.mount('/content/drive')

# 点群ファイルをコピー
!cp /content/drive/MyDrive/path/to/object_denoised.ply .
```

### 方法 B: ブラウザから直接アップロード

```python
from google.colab import files
uploaded = files.upload()  # object_denoised.ply を選択
```

## 4. 点群の前処理（ダウンサンプリング + NPY 変換）

DiffCD は NPY 形式（XYZ 座標のみ）を入力とします。大規模点群はダウンサンプリングが必要です。

```python
import open3d as o3d
import numpy as np

# 読み込み
INPUT_PLY = "object_denoised.ply"
pcd = o3d.io.read_point_cloud(INPUT_PLY)
print(f"Original: {len(pcd.points):,} points")

# ダウンサンプリング（目標: 約100万点）
# voxel_size を調整して点数を制御
TARGET_POINTS = 1_000_000
current_points = len(pcd.points)

if current_points > TARGET_POINTS:
    # 推定 voxel size を計算
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox_extent = bbox.get_extent()
    volume = np.prod(bbox_extent)
    target_density = TARGET_POINTS / volume
    voxel_size = (1 / target_density) ** (1/3) * 0.9

    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

    # 微調整（目標の ±10% に収める）
    while len(pcd_down.points) > TARGET_POINTS * 1.1:
        voxel_size *= 1.1
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    while len(pcd_down.points) < TARGET_POINTS * 0.9 and voxel_size > 1e-6:
        voxel_size *= 0.9
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

    print(f"Downsampled: {len(pcd_down.points):,} points (voxel_size={voxel_size:.5f})")
else:
    pcd_down = pcd
    print(f"No downsampling needed: {len(pcd_down.points):,} points")

# NPY 保存（XYZ 座標のみ、float32）
points = np.asarray(pcd_down.points, dtype=np.float32)
np.save("object_points.npy", points)
print(f"Saved: object_points.npy ({points.shape})")
```

## 5. DiffCD 実行

### 5.1 学習（Implicit Surface Fitting）

```python
# A100 向け推奨パラメータ
# NOTE: --method diff-cd を明示的に指定（tyro CLI の型解決エラー回避）
!python fit_implicit.py \
    --output-dir outputs/object_mesh \
    --dataset.path ../object_points.npy \
    --method diff-cd \
    --method.alpha 100.0 \
    --n-batches 30000 \
    --final-mesh-points-per-axis 512
```

### パラメータ調整ガイド

| パラメータ | デフォルト | A100推奨 | T4/V100 | 説明 |
|-----------|-----------|----------|---------|------|
| `--method` | diff-cd | diff-cd | diff-cd | メソッド指定（必須） |
| `--method.alpha` | 100.0 | 100.0 | 100.0 | 損失関数の重み（float必須） |
| `--batch-size` | 5000 | 5000 | 3000 | バッチあたりの点数 |
| `--n-batches` | 40000 | 30000 | 20000 | 学習イテレーション |
| `--final-mesh-points-per-axis` | 512 | 512 | 256 | メッシュ解像度 |

> **メモリ不足時**: `--batch-size 3000` や `--final-mesh-points-per-axis 256` に下げてください。
>
> **tyro エラー対策**: `--method diff-cd --method.alpha 100.0` は必須です。DiffCD のデフォルト値に型の問題があるため、明示的に指定する必要があります。

### 5.2 進捗確認

学習中は損失値が出力されます。損失が収束したら Ctrl+C で中断しても最終メッシュは生成されます。

```
Step 1000: loss = 0.0234
Step 2000: loss = 0.0187
...
```

## 6. メッシュの後処理

生成されたメッシュを平滑化し、最終出力として保存します。

```python
import trimesh

# 生成メッシュを読み込み
mesh = trimesh.load("outputs/object_mesh/meshes/mesh_final.ply")
print(f"Vertices: {len(mesh.vertices):,}")
print(f"Faces: {len(mesh.faces):,}")
print(f"Is watertight: {mesh.is_watertight}")

# ラプラシアン平滑化（オプション）
mesh_smooth = trimesh.smoothing.filter_laplacian(mesh, iterations=2)

# 保存
mesh_smooth.export("object_mesh_final.ply")
print("Saved: object_mesh_final.ply")
```

## 7. 検証（Success Criteria）

```python
import trimesh

mesh = trimesh.load("object_mesh_final.ply")
print(f"Vertices: {len(mesh.vertices):,}")
print(f"Faces: {len(mesh.faces):,}")
print(f"Is watertight: {mesh.is_watertight}")

# バウンディングボックス
bounds = mesh.bounds
extent = bounds[1] - bounds[0]
print(f"Bounding box: {extent[0]:.3f} x {extent[1]:.3f} x {extent[2]:.3f}")
```

## 8. 3D ビジュアライゼーション

```python
import numpy as np
import plotly.graph_objects as go

# メッシュ読み込み
mesh = trimesh.load("object_mesh_final.ply")
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
    name='Mesh',
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
    title='Mesh vs Point Cloud Comparison',
)
fig.show()
```

## 9. 結果のダウンロード

### 方法 A: Google Drive にコピー

```python
!cp object_mesh_final.ply /content/drive/MyDrive/im2pc_output/
!cp -r outputs/object_mesh /content/drive/MyDrive/im2pc_output/diffcd_outputs/
```

### 方法 B: ブラウザで直接ダウンロード

```python
from google.colab import files
files.download("object_mesh_final.ply")
```

## トラブルシューティング

### tyro CLI エラー（Invalid input / alpha has invalid default）

```
Field alpha has invalid default
Default value 100 with type int does not match type <class 'float'>
```

DiffCD のデフォルト値に型の問題があります。`--method diff-cd --method.alpha 100.0` を明示的に指定してください：

```python
!python fit_implicit.py \
    --output-dir outputs/object_mesh \
    --dataset.path ../object_points.npy \
    --method diff-cd \
    --method.alpha 100.0 \
    --n-batches 30000
```

### JAX CUDA エラー

JAX と CUDA のバージョン不一致の場合：

```python
!pip install --upgrade "jax[cuda12_pip]==0.4.14" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### CUDA out of memory

1. `--batch-size` を下げる（5000 → 3000 → 2000）
2. `--final-mesh-points-per-axis` を下げる（512 → 256 → 128）
3. 点群をさらにダウンサンプリング（100万点 → 50万点）

### メッシュに穴がある

1. `--n-batches` を増やす（30000 → 50000）
2. 入力点群の密度が低い部分がないか確認
3. `trimesh.fill_holes()` で後処理

```python
mesh.fill_holes()
mesh.export("object_mesh_filled.ply")
```

### 形状が元の点群と一致しない

1. NPY 変換時にスケールが変わっていないか確認
2. 点群の正規化が必要な場合：

```python
# 重心を原点に移動、スケールを正規化
points = points - points.mean(axis=0)
scale = np.abs(points).max()
points = points / scale
np.save("object_points_normalized.npy", points)
```

---

## 参考リンク

- [DiffCD GitHub](https://github.com/Linusnie/diffcd)
- [DiffCD Paper (ECCV 2024)](https://arxiv.org/abs/2312.13311)
