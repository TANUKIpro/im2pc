# Google Colab セットアップガイド

Google Colab 上で動画から 3D 点群を生成する手順です。

- **Pi3X のみ (pi3x_cli.py)**: シーン全体の点群を生成
- **SAM2 + Pi3X (pi3x_sam2_cli.py)**: 特定のオブジェクトだけの点群を生成（推奨）

## 1. GPU ランタイムの設定

1. メニューから **ランタイム → ランタイムのタイプを変更** を開く
2. ハードウェアアクセラレータで **GPU (T4)** を選択して保存

> T4 は 16GB VRAM を搭載しており、Pi3X の推論に十分です。

## 2. リポジトリのクローン

```python
!git clone --recursive https://github.com/TANUKIpro/im2pc.git
%cd im2pc
```

## 3. 依存パッケージのインストール

Colab にプリインストールされている PyTorch (2.x + CUDA) をそのまま使います。`repos/pi3/requirements.txt` が `torch==2.5.1` 等を要求しますが、API 互換性があるため Colab 版で問題ありません。固定バージョンを上書きしないよう、必要なパッケージだけ個別にインストールします。

```python
!pip install plyfile huggingface_hub safetensors einops timm hydra-core iopath
!cd repos/sam2 && SAM2_BUILD_CUDA=0 pip install --no-deps --no-build-isolation -e .
```

> `pillow`, `opencv-python`, `numpy` は Colab にプリインストール済みです。
> SAM2 は Hydra ベースのモデル構築を使用するため、`pip install -e` による正式インストールが必要です（`sys.path` 追加だけでは Hydra のクラス解決が失敗します）。`--no-deps` で PyTorch のバージョン競合を回避し、`SAM2_BUILD_CUDA=0` で CUDA 拡張ビルドをスキップします。

## 4. 動画の準備（Google Drive マウント）

```python
from google.colab import drive
drive.mount('/content/drive')
```

Google Drive 上に処理したい動画（`.mp4`）を配置しておきます。

## 5. pi3x_cli.py の実行

```python
!python host/pi3x_cli.py /content/drive/MyDrive/path/to/video.mp4 \
    --output-dir output \
    --confidence-threshold 0.1 \
    --frame-interval 10 \
    --max-frames 50
```

### 主要オプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--confidence-threshold` | `0.1` | 点群フィルタリングの信頼度閾値 |
| `--frame-interval` | `10` | N フレームごとに抽出 |
| `--max-frames` | `50` | Pi3X に入力する最大フレーム数 |
| `--pixel-limit` | `255000` | フレームあたりの最大ピクセル数（リサイズ制御） |
| `--edge-rtol` | `0.03` | 深度エッジフィルタリングの相対許容値 |

> GPU メモリが不足する場合は `--max-frames` を減らすか `--pixel-limit` を小さくしてください。

## 6. 結果のダウンロード

### 方法 A: Google Drive にコピー

```python
!cp -r output /content/drive/MyDrive/im2pc_output
```

### 方法 B: ブラウザで直接ダウンロード

```python
from google.colab import files
files.download('output/object.ply')
files.download('output/camera_poses.json')
```

### 出力ファイル

- `output/object.ply` — 3D 点群（PLY 形式）
- `output/camera_poses.json` — 各フレームのカメラ姿勢（4x4 行列）

## 7. 3D ビジュアライゼーション（インタラクティブ）

生成した点群とカメラ軌跡を Colab 上で直接確認できます。Plotly はプリインストール済みのため追加インストールは不要です。

```python
import json
import numpy as np
from plyfile import PlyData
import plotly.graph_objects as go

# --- PLY 読み込み ---
ply = PlyData.read("output/object.ply")
verts = ply["vertex"]
x = np.array(verts["x"], dtype=np.float32)
y = np.array(verts["y"], dtype=np.float32)
z = np.array(verts["z"], dtype=np.float32)
r = np.array(verts["red"], dtype=np.uint8)
g = np.array(verts["green"], dtype=np.uint8)
b = np.array(verts["blue"], dtype=np.uint8)

# --- ダウンサンプリング（200K 点超の場合） ---
MAX_POINTS = 200_000
n_pts = len(x)
if n_pts > MAX_POINTS:
    rng = np.random.default_rng(seed=42)
    idx = rng.choice(n_pts, size=MAX_POINTS, replace=False)
    idx.sort()
    x, y, z, r, g, b = x[idx], y[idx], z[idx], r[idx], g[idx], b[idx]
    print(f"ダウンサンプリング: {n_pts:,} → {MAX_POINTS:,} 点")
else:
    print(f"点群: {n_pts:,} 点（ダウンサンプリング不要）")

# --- カメラ姿勢読み込み ---
with open("output/camera_poses.json") as f:
    cam_data = json.load(f)
poses = np.array(cam_data["poses"])            # (N, 4, 4)
frame_indices = cam_data["frame_indices"]       # [0, 10, 20, ...]

cam_pos = poses[:, :3, 3]                       # カメラ位置
cam_dir = -poses[:, :3, 2]                      # 視線方向（-Z 軸）

# --- コーンスケール（シーンサイズの 5%） ---
scene_range = max(x.max() - x.min(), y.max() - y.min(), z.max() - z.min())
cone_size = scene_range * 0.05

# --- Plotly 描画 ---
colors = [f"rgb({ri},{gi},{bi})" for ri, gi, bi in zip(r, g, b)]

fig = go.Figure()

# 点群
fig.add_trace(go.Scatter3d(
    x=x, y=y, z=z,
    mode="markers",
    marker=dict(size=1, color=colors),
    hoverinfo="skip",
    name="Point Cloud",
))

# カメラ位置
fig.add_trace(go.Scatter3d(
    x=cam_pos[:, 0], y=cam_pos[:, 1], z=cam_pos[:, 2],
    mode="markers+text",
    marker=dict(size=5, color="red", symbol="diamond"),
    text=[str(fi) for fi in frame_indices],
    textposition="top center",
    textfont=dict(size=8, color="red"),
    name="Camera Positions",
))

# カメラ軌跡（線）
fig.add_trace(go.Scatter3d(
    x=cam_pos[:, 0], y=cam_pos[:, 1], z=cam_pos[:, 2],
    mode="lines",
    line=dict(color="red", width=2),
    hoverinfo="skip",
    showlegend=False,
))

# カメラ視線方向（コーン）
fig.add_trace(go.Cone(
    x=cam_pos[:, 0], y=cam_pos[:, 1], z=cam_pos[:, 2],
    u=cam_dir[:, 0], v=cam_dir[:, 1], w=cam_dir[:, 2],
    sizemode="absolute",
    sizeref=cone_size,
    colorscale=[[0, "red"], [1, "red"]],
    showscale=False,
    name="View Direction",
))

fig.update_layout(
    scene=dict(aspectmode="data"),
    width=900, height=700,
    title="3D Point Cloud & Camera Trajectory",
    legend=dict(x=0, y=1),
)
fig.show()
```

> **操作方法**
> - マウス左ドラッグ: 回転
> - 右ドラッグ / Ctrl+ドラッグ: パン
> - スクロール: ズーム
>
> 点群が 200,000 点を超える場合は自動的にダウンサンプリングされます。全点を表示したい場合はコード中の `MAX_POINTS` を増やしてください（ブラウザが重くなる可能性があります）。

## トラブルシューティング

### CUDA out of memory
`--max-frames` や `--pixel-limit` の値を下げてください。

```python
!python host/pi3x_cli.py /content/drive/MyDrive/video.mp4 \
    --output-dir output \
    --max-frames 20 \
    --pixel-limit 150000
```

### "Need at least 2 frames" エラー
動画が短すぎるか `--frame-interval` が大きすぎます。`--frame-interval` を小さくしてください。

---

# SAM2 + Pi3X パイプライン (pi3x_sam2_cli.py)

動画から**特定のオブジェクトだけ**を切り出して 3D 点群を生成します。SAM2 でオブジェクトマスクを作成し、Pi3X はフル画像で推論（姿勢推定の精度を維持）した後、マスクをポストフィルタとして適用します。

## セットアップ

GPU ランタイム、リポジトリクローン、Google Drive マウントは上記の Pi3X のみの手順と同じです。依存パッケージに `ipympl`（インタラクティブ click 用）を追加します。

```python
!pip install plyfile huggingface_hub safetensors einops timm ipympl hydra-core iopath
!cd repos/sam2 && SAM2_BUILD_CUDA=0 pip install --no-deps --no-build-isolation -e .
```

## セル 1: フレーム抽出

```python
import sys
sys.path.insert(0, ".")

from host.pi3x_sam2_cli import extract_frames

VIDEO_PATH = "/content/drive/MyDrive/path/to/video.mp4"  # 動画パスを変更
OUTPUT_DIR = "output_sam2"

frames_dir = extract_frames(
    VIDEO_PATH,
    frame_interval=10,
    max_frames=50,
    output_dir=OUTPUT_DIR,
)
```

## セル 2: 初回フレーム表示

```python
import cv2
import matplotlib.pyplot as plt

first_frame = cv2.imread(str(frames_dir / "00000.jpg"))
first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 8))
plt.imshow(first_frame_rgb)
plt.title("First frame - note pixel coordinates for click prompts")
plt.axis("on")
plt.show()
print(f"Frame size: {first_frame.shape[1]}x{first_frame.shape[0]}")
```

## セル 3: インタラクティブ click でオブジェクト選択

`ipympl` を使ってフレーム上でクリックし、対象オブジェクトの座標を取得します。
- **左クリック**: positive point（オブジェクト上）
- **右クリック**: negative point（背景）

```python
%matplotlib widget
import matplotlib.pyplot as plt
import cv2

first_frame = cv2.imread(str(frames_dir / "00000.jpg"))
first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

clicks = []  # [(x, y, label), ...]

fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(first_frame_rgb)
ax.set_title("Left click = positive (green), Right click = negative (red)")

def onclick(event):
    if event.inaxes != ax:
        return
    x, y = event.xdata, event.ydata
    if event.button == 1:  # left click = positive
        label = 1
        ax.plot(x, y, "g+", markersize=15, markeredgewidth=3)
    elif event.button == 3:  # right click = negative
        label = 0
        ax.plot(x, y, "rx", markersize=15, markeredgewidth=3)
    else:
        return
    clicks.append((x, y, label))
    fig.canvas.draw()
    print(f"  Point ({x:.0f}, {y:.0f}) label={label}")

fig.canvas.mpl_connect("button_press_event", onclick)
plt.show()
```

クリックが完了したら次のセルで座標を確認します。

```python
# clicks リストから points と labels を分離
points = [[c[0], c[1]] for c in clicks]
labels = [c[2] for c in clicks]
print(f"Points: {points}")
print(f"Labels: {labels}")
```

> **`ipympl` が使えない場合の代替**: 上のセル 2 で表示されたフレームの座標を目視で読み取り、手動で指定できます。
>
> ```python
> points = [[512, 384]]  # オブジェクトの中心付近のピクセル座標
> labels = [1]           # 1=positive
> ```

## セル 4: SAM2 マスク生成・伝播

```python
from host.pi3x_sam2_cli import run_sam2_segmentation

mask_dir = run_sam2_segmentation(
    str(frames_dir),
    points=points,
    labels=labels,
    output_dir=OUTPUT_DIR,
    prompt_frame=0,
    model_type="large",
)
```

### マスクプレビュー

```python
import cv2
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
mask_files = sorted(mask_dir.glob("*.png"))
preview_indices = np.linspace(0, len(mask_files) - 1, 4, dtype=int)

for ax, idx in zip(axes, preview_indices):
    frame = cv2.imread(str(frames_dir / f"{idx:05d}.jpg"))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(mask_files[idx]), cv2.IMREAD_GRAYSCALE)
    # Resize mask to frame size if needed
    if mask.shape[:2] != frame_rgb.shape[:2]:
        mask = cv2.resize(mask, (frame_rgb.shape[1], frame_rgb.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    overlay = frame_rgb.copy()
    overlay[mask > 127] = (overlay[mask > 127] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    ax.imshow(overlay)
    ax.set_title(f"Frame {idx}")
    ax.axis("off")

plt.suptitle("SAM2 Mask Propagation Preview")
plt.tight_layout()
plt.show()
```

## セル 5: Pi3X 推論

```python
from host.pi3x_sam2_cli import run_pi3x_inference

results, imgs = run_pi3x_inference(
    str(frames_dir),
    pixel_limit=255000,
    max_frames=50,
)
```

## セル 6: マスク適用 → PLY 保存

```python
from host.pi3x_sam2_cli import filter_and_save

ply_path = filter_and_save(
    results, imgs, str(mask_dir),
    output_dir=OUTPUT_DIR,
    conf_threshold=0.1,
    edge_rtol=0.03,
)
```

## セル 7: 3D ビジュアライゼーション

```python
import json
import numpy as np
from plyfile import PlyData
import plotly.graph_objects as go

# --- PLY 読み込み ---
ply = PlyData.read(str(ply_path))
verts = ply["vertex"]
x = np.array(verts["x"], dtype=np.float32)
y = np.array(verts["y"], dtype=np.float32)
z = np.array(verts["z"], dtype=np.float32)
r = np.array(verts["red"], dtype=np.uint8)
g = np.array(verts["green"], dtype=np.uint8)
b = np.array(verts["blue"], dtype=np.uint8)

# --- ダウンサンプリング（200K 点超の場合） ---
MAX_POINTS = 200_000
n_pts = len(x)
if n_pts > MAX_POINTS:
    rng = np.random.default_rng(seed=42)
    idx = rng.choice(n_pts, size=MAX_POINTS, replace=False)
    idx.sort()
    x, y, z, r, g, b = x[idx], y[idx], z[idx], r[idx], g[idx], b[idx]
    print(f"Downsampled: {n_pts:,} -> {MAX_POINTS:,} points")
else:
    print(f"Points: {n_pts:,} (no downsampling)")

# --- カメラ姿勢 ---
with open(f"{OUTPUT_DIR}/camera_poses.json") as f:
    cam_data = json.load(f)
poses = np.array(cam_data["poses"])
cam_pos = poses[:, :3, 3]
cam_dir = -poses[:, :3, 2]

scene_range = max(x.max() - x.min(), y.max() - y.min(), z.max() - z.min())
cone_size = scene_range * 0.05

colors = [f"rgb({ri},{gi},{bi})" for ri, gi, bi in zip(r, g, b)]
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=x, y=y, z=z, mode="markers",
    marker=dict(size=1, color=colors),
    hoverinfo="skip", name="Point Cloud",
))
fig.add_trace(go.Scatter3d(
    x=cam_pos[:, 0], y=cam_pos[:, 1], z=cam_pos[:, 2],
    mode="markers+text",
    marker=dict(size=5, color="red", symbol="diamond"),
    text=[str(i) for i in range(len(cam_pos))],
    textposition="top center", textfont=dict(size=8, color="red"),
    name="Camera Positions",
))
fig.add_trace(go.Scatter3d(
    x=cam_pos[:, 0], y=cam_pos[:, 1], z=cam_pos[:, 2],
    mode="lines", line=dict(color="red", width=2),
    hoverinfo="skip", showlegend=False,
))
fig.add_trace(go.Cone(
    x=cam_pos[:, 0], y=cam_pos[:, 1], z=cam_pos[:, 2],
    u=cam_dir[:, 0], v=cam_dir[:, 1], w=cam_dir[:, 2],
    sizemode="absolute", sizeref=cone_size,
    colorscale=[[0, "red"], [1, "red"]],
    showscale=False, name="View Direction",
))

fig.update_layout(
    scene=dict(aspectmode="data"),
    width=900, height=700,
    title="Object Point Cloud & Camera Trajectory",
    legend=dict(x=0, y=1),
)
fig.show()
```

## セル 8: 結果のダウンロード

```python
# Google Drive にコピー
!cp -r {OUTPUT_DIR} /content/drive/MyDrive/im2pc_output_sam2

# またはブラウザで直接ダウンロード
from google.colab import files
files.download(str(ply_path))
files.download(f"{OUTPUT_DIR}/camera_poses.json")
```

## CLI でのワンショット実行

Colab セルを使わず、CLI で一括実行することも可能です。

```python
!python host/pi3x_sam2_cli.py /content/drive/MyDrive/video.mp4 \
    --point 512,384 \
    --output-dir output_sam2 \
    --frame-interval 10 \
    --max-frames 50
```

### 主要オプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--point x,y` | (必須) | オブジェクト上の positive click 座標（複数指定可） |
| `--neg-point x,y` | - | 背景の negative click 座標（複数指定可） |
| `--box x1,y1,x2,y2` | - | バウンディングボックス指定 |
| `--prompt-frame` | `0` | SAM2 プロンプトを指定するフレーム番号 |
| `--sam2-model` | `large` | SAM2 モデル (`tiny`/`small`/`base`/`large`) |
| `--confidence-threshold` | `0.1` | 点群フィルタリングの信頼度閾値 |
| `--frame-interval` | `10` | N フレームごとに抽出 |
| `--max-frames` | `50` | 最大フレーム数 |
| `--pixel-limit` | `255000` | フレームあたりの最大ピクセル数 |
| `--edge-rtol` | `0.03` | 深度エッジフィルタリングの相対許容値 |

## トラブルシューティング (SAM2 + Pi3X)

### CUDA out of memory
SAM2 と Pi3X は順次実行されるため、同時にメモリを消費しません。それでも不足する場合は `--max-frames` や `--pixel-limit` を下げてください。`--sam2-model tiny` にするとメモリ使用量が大幅に減ります。

### マスクが全フレームに伝播しない
初回フレーム上のクリック位置を確認してください。オブジェクトの中心付近を positive click し、背景を negative click すると精度が上がります。

### 点群が空になる
`Filtering stats` のログを確認してください。SAM2 マスクが小さすぎる場合は `--confidence-threshold` を下げるか、クリック位置を見直してください。
