# テクスチャベイキングパイプライン 詳細設計

## 概要

DiffCD等で生成したgeometry onlyメッシュに、多視点RGB画像からテクスチャを焼き付ける。
パイプラインは2つのフェーズで構成される。

```
Phase 1: カメラ内部パラメータ推定 (extract_intrinsics.py)
Phase 2: UV生成 + テクスチャ焼付  (texture_mesh.py)
```

---

## Phase 1: カメラ内部パラメータ推定

### 背景

Pi3Xは推論結果として cam-to-world 外部パラメータのみを出力し、内部パラメータ（焦点距離・主点）を明示的に保存しない。内部的には `rays` テンソルとして暗黙的に予測しているが、`filter_and_save()` でディスクに書き出されない。

### 手法

RGB付き点群を各カメラ空間に変換し、候補 K でフレームに投影した際の「投影位置の画素色」と「点群の色」の一致度から K を推定する。

#### アルゴリズム

```
1. 点群からランダム50,000点をサンプル
2. 10フレームを等間隔に選択（評価用）
3. FOV 35°〜84°（1°刻み）でグリッドサーチ
   - 各FOVからfxを算出、fy=fx（正方ピクセル仮定）
   - cx=W/2, cy=H/2（主点中心仮定）
   - 全評価フレームへ点群を投影し、色のMSEを計算
   - MSEが最小となるFOVを選択
4. ベストFOVを初期値としてNelder-Mead最適化
   - パラメータ: [fx, fy, cx, cy]（4次元）
   - 目的関数: 全評価フレームの加重平均MSE（最小化）
```

#### 色一致度スコア

各フレーム i について:
```
score_i = -MSE(frame_color[projected_uv], pointcloud_color) × count_i
```
マスク内に投影された点のみを使用。全フレームの加重平均が最終スコア。

### 入出力

```
入力:
  - object.ply          RGB付き点群 (2.66M点)
  - camera_poses.json   93個のcam-to-world 4x4行列
  - frames/*.jpg        1080x1920 JPEG画像
  - masks/*.png         2値マスク (0/255)

出力:
  - intrinsics.json     推定結果
    {
      "fx": 1824.74,  "fy": 1791.95,
      "cx": 540.29,   "cy": 961.53,
      "image_width": 1080, "image_height": 1920,
      "fov_horizontal_deg": 32.97,
      "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    }

  - reprojections/*.jpg  検証用再投影画像（--visualize時）
```

### CLIオプション

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--point-cloud` | (必須) | 色付き点群PLY |
| `--poses` | (必須) | camera_poses.json |
| `--frames-dir` | (必須) | フレーム画像ディレクトリ |
| `--masks-dir` | (必須) | マスク画像ディレクトリ |
| `--output` | `intrinsics.json` | 出力JSONパス |
| `--num-eval-frames` | `10` | 評価に使うフレーム数 |
| `--subsample` | `50000` | サンプル点数 |
| `--visualize` | `false` | 再投影画像を保存 |
| `--vis-dir` | `data/output_sam2/reprojections` | 再投影画像の保存先 |

---

## Phase 2: テクスチャ焼付

### 処理フロー

```
                  ┌─────────────┐
                  │ メッシュ(PLY) │
                  └──────┬──────┘
                         │
                  ┌──────▼──────┐
             ┌────┤ xatlas UV生成 ├────┐
             │    └─────────────┘    │
      新頂点配列              UV座標(V_new, 2)
      新面配列               [0,1]正規化
             │                      │
             └──────────┬───────────┘
                        │
                 ┌──────▼──────┐
                 │ テクセルマッピング │  テクスチャ空間の各ピクセルを
                 │ (UV→Face+Bary)│  メッシュ面+重心座標に対応付け
                 └──────┬──────┘
                        │
            ┌───────────▼───────────┐
            │  テクセル3D位置・法線計算  │
            │  pos3d = w0*v0+w1*v1+w2*v2│
            └───────────┬───────────┘
                        │
         ┌──────────────▼──────────────┐
         │ 各カメラ視点からの色サンプリング │  ×93視点
         │                              │
         │  1. 法線→カメラ方向のcosθ計算  │
         │  2. cosθ > 0.1 のテクセルを選択 │
         │  3. 3D位置を画像面に投影       │
         │  4. 画像範囲内＋マスク内を確認   │
         │  5. バイリニア補間で色取得       │
         │  6. cosθを重みとして蓄積        │
         └──────────────┬──────────────┘
                        │
                 ┌──────▼──────┐
                 │  加重平均正規化  │  color = Σ(w_i × c_i) / Σ(w_i)
                 └──────┬──────┘
                        │
                 ┌──────▼──────┐
                 │ シームパディング │  UVアイランド境界を
                 │  (8反復膨張)   │  隣接テクセル平均で拡張
                 └──────┬──────┘
                        │
              ┌─────────▼─────────┐
              │ OBJ + MTL + PNG出力 │
              └───────────────────┘
```

### 各ステップの詳細

#### 1. UV Atlas生成 (xatlas)

```python
vmapping, new_faces, uvs = xatlas.parametrize(vertices, faces)
```

- 入力メッシュの頂点をUVシーム境界で分割（648k → 864k頂点）
- 面数は不変（1,297,514面）
- UV座標は [0,1] に正規化済み
- 処理時間: 約5-8分（Docker CPU）

#### 2. テクセルマッピング

テクスチャ画像の各ピクセルがどのメッシュ面に対応するかを算出。

```
方法: OpenCV fillConvexPoly でUV三角形をint32バッファにラスタライズ
      → 各テクセルの面IDを取得
      → 面の3頂点UV座標から重心座標をベクトル演算で算出
```

- 2048×2048テクスチャで約290万テクセルが有効（69.2%）
- 残りはUVアイランド間の空白領域

#### 3. 可視性判定

深度バッファを使わず、法線ベースの可視性判定を採用:

```
visible = (cos(face_normal, view_direction) > 0.1) AND (mask[projected_pixel] > 0)
```

| 手法 | 精度 | 速度 | 適用条件 |
|------|------|------|---------|
| 深度バッファ | 高 | 遅 (CPU) | 任意形状 |
| **法線+マスク** | **中-高** | **速** | **凸に近い物体** |
| 法線のみ | 低 | 最速 | 完全凸物体 |

ボトルのような概ね凸な物体では法線+マスクで十分な品質が得られる。
凹部では自己遮蔽が起きうるが、SAM2マスクが背景混入を防ぐため実用上問題ない。

#### 4. 色サンプリングとブレンディング

```
各テクセルの最終色 = Σ(cos_angle_i × color_i) / Σ(cos_angle_i)
```

- `cos_angle`: 面法線と視線方向のなす角の余弦値（0〜1）
- 正面から見た視点ほど高い重み → 斜め視点のぼやけを抑制
- バイリニア補間でサブピクセル精度の色取得

#### 5. シームパディング

UVアイランド境界の隙間をテクスチャフィルタリング時のアーティファクト防止のため埋める。

```
kernel = [[0,1,0],[1,0,1],[0,1,0]]  (4近傍)

repeat 8回:
  空テクセルのうち、有効な隣接テクセルがあるものを
  隣接テクセルの平均色で埋める
```

#### 6. OBJ出力

- 頂点座標: `v x y z`
- UV座標: `vt u v`
- 面: `f v1/vt1 v2/vt2 v3/vt3`（1-indexed）
- テクスチャ画像はV軸反転（OBJ規約: V=0が下端）

### CLIオプション

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--mesh` | (必須) | geometry only PLY |
| `--intrinsics` | (必須) | intrinsics.json |
| `--poses` | (必須) | camera_poses.json |
| `--frames-dir` | (必須) | フレーム画像ディレクトリ |
| `--masks-dir` | (必須) | マスク画像ディレクトリ |
| `--output-dir` | `data/output_textured` | 出力ディレクトリ |
| `--tex-size` | `2048` | テクスチャ解像度 |
| `--max-views` | `None`(全て) | 使用する最大視点数 |
| `--pad-iterations` | `8` | シームパディング反復回数 |

---

## Docker環境

### Dockerfile.texture

```dockerfile
FROM python:3.11-slim-bookworm
RUN apt-get install -y g++ cmake libgl1-mesa-glx libglib2.0-0
RUN pip install numpy opencv-python-headless plyfile scipy xatlas trimesh
```

- `g++` / `cmake`: xatlasのビルドに必要
- `libgl1-mesa-glx`: OpenCVのGUI非依存描画に必要
- PyTorch不要（CPU numpy処理のみ）

### ビルド・実行

```bash
# ビルド
docker build -f docker/Dockerfile.texture -t im2pc-texture .

# 実行
docker run --rm -v $(pwd):/app -w /app im2pc-texture \
    python -u host/texture_mesh.py [引数...]
```

---

## 座標系

### Pi3X出力 (OpenCV慣例)
- X: 右, Y: 下, Z: 奥（カメラ正面方向）
- cam-to-world行列: 4x4同次変換行列

### 投影計算
```
world → camera:  p_cam = W2C @ p_world  (W2C = inv(C2W))
camera → pixel:  u = fx * x/z + cx,  v = fy * y/z + cy
```

### OBJ出力
- テクスチャV軸: V=0が画像下端（上下反転して保存）

---

## パフォーマンス実測値

テスト環境: Docker Desktop (Apple Silicon, CPU only)

| 処理 | 648k頂点 / 1.3M面 | 備考 |
|------|-------------------|------|
| xatlas UV生成 | 5-8分 | 面数に比例 |
| テクセルマッピング | 2-3分 | fillConvexPoly ×1.3M回 |
| 1視点の色サンプリング | ~1秒 | 290万テクセルのベクトル演算 |
| 93視点の全処理 | ~2分 | 投影+サンプリング |
| 合計 | ~15分 | xatlasが支配的 |

### テクスチャカバレッジ

| 視点数 | カバレッジ |
|--------|-----------|
| 5視点 | 86.4% |
| 93視点 | 92.2% |

残り約8%はメッシュ底面や内側など、どの視点からも見えない領域。

---

## 制約と改善候補

### 現在の制約

| 項目 | 内容 |
|------|------|
| 自己遮蔽 | 凹部で背面の色が混入しうる（マスクが緩和） |
| CPU速度 | xatlas / テクセルマッピングのPythonループが律速 |
| 透過物体 | ガラスなど透過素材はrefraction考慮なし |

### 改善候補（今後）

1. **GPU深度バッファ**: PyTorch3D `MeshRasterizer` による正確な遮蔽判定
2. **微分可能レンダリング**: テクスチャをgradient descentで最適化しシーム改善
3. **メッシュ簡略化**: xatlas前にdecimationで処理時間短縮
4. **Cython/Numba**: テクセルマッピングループの高速化
