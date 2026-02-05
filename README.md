# Pi3X + SAM2 Multi-View Reconstruction

動画から物体をセグメントし、3D点群を生成するパイプラインです。

## アーキテクチャ

```
┌──────────────────────────────────────────────────┐
│                  macOS Host                       │
│  ┌────────────────────────────────────────────┐  │
│  │     Docker Container (CPU only)             │  │
│  │  [Gradio UI] [Video Processing]             │  │
│  │          ↓ HTTP (localhost:5050) ↓          │  │
│  └────────────────────────────────────────────┘  │
│                        ↕                          │
│  ┌────────────────────────────────────────────┐  │
│  │     venv環境 (MPS GPU)                      │  │
│  │  [SAM2 Video Predictor] [Pi3X Inference]   │  │
│  │  Flask Server @ port 5050                   │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

## セットアップ

### 1. ホスト環境の構築

```bash
# venv環境を作成し、依存関係をインストール
./host/setup.sh
```

これにより以下が実行されます：
- Python venv環境の作成 (`.venv/`)
- SAM2リポジトリのクローン (`repos/sam2/`)
- Pi3リポジトリのクローン (`repos/pi3/`)
- 依存関係のインストール
- SAM2チェックポイントのダウンロード

### 2. 推論サーバーの起動

```bash
# venvをアクティベート
source .venv/bin/activate

# 推論サーバーを起動
python host/inference_server.py
```

サーバーが `http://localhost:5050` で起動します。

### 3. Docker UIの起動

別のターミナルで：

```bash
cd docker
docker-compose up --build
```

### 4. ブラウザでアクセス

http://localhost:7860 を開きます。

## 使い方

1. **動画を選択**: ドロップダウンから `data/` ディレクトリ内の動画を選択
2. **対象物をクリック**: 1フレーム目で対象物をクリック（緑=含める、赤=除外）
3. **マスク生成**: 「Generate Mask Preview」ボタンでマスクを確認
4. **マスク伝播**: 「Propagate Masks」ボタンで全フレームにマスクを適用
5. **3D再構成**: 「Run Reconstruction」ボタンでPi3X推論を実行
6. **PLYダウンロード**: 生成された点群ファイルをダウンロード

## CLI ツール（SAM2なし）

SAM2を使わず、Pi3Xモデルのみで動画から直接3D点群を生成するCLIツールです。

```bash
source .venv/bin/activate

python host/pi3x_cli.py <video_path> [options]
```

### 引数

| 引数 | 必須 | デフォルト | 説明 |
|------|------|-----------|------|
| `video_path` | ○ | — | 入力動画のパス |
| `--confidence-threshold` | — | `0.1` | 信頼度閾値 (0.0〜1.0) |
| `--frame-interval` | — | `10` | フレーム間引き間隔 |
| `--output-dir` | — | `data/output` | 出力ディレクトリ |
| `--max-frames` | — | `50` | Pi3Xに入力する最大フレーム数 |

### 実行例

```bash
python host/pi3x_cli.py data/IMG_1110.mp4 \
    --confidence-threshold 0.1 \
    --frame-interval 10 \
    --output-dir data/output \
    --max-frames 50
```

出力:
- `<output-dir>/object.ply` — 3D点群
- `<output-dir>/camera_poses.json` — カメラ姿勢列

## ポイントクラウド ノイズ削減ツール

生成した点群からノイズを除去するCLIツールです。DBSCAN（密度ベースクラスタリング）とSOR（Statistical Outlier Removal）を組み合わせて、浮遊ノイズを効果的に除去します。

```bash
source .venv/bin/activate

python host/denoise_ply.py <input_ply> [options]
```

### 引数

| 引数 | 必須 | デフォルト | 説明 |
|------|------|-----------|------|
| `input_ply` | ○ | — | 入力PLYファイルのパス |
| `-o, --output` | — | `<input>_denoised.ply` | 出力PLYファイルのパス |
| `--method` | — | `dbscan+sor` | ノイズ削減手法 (`dbscan`, `sor`, `dbscan+sor`) |
| `--dbscan-eps` | — | 自動計算 | DBSCANのイプシロン距離 |
| `--dbscan-min-points` | — | `10` | DBSCANのコア点に必要な最小点数 |
| `--sor-neighbors` | — | `20` | SORで使用する近傍点数 |
| `--sor-std-ratio` | — | `2.0` | SORの標準偏差倍率 |
| `--max-dbscan-points` | — | `500000` | DBSCANダウンサンプリング閾値 |
| `-q, --quiet` | — | — | 詳細出力を抑制 |

### 実行例

```bash
# 基本使用（DBSCAN + SOR）
python host/denoise_ply.py data/output/object.ply

# 出力ファイル指定
python host/denoise_ply.py data/output/object.ply -o data/output/clean.ply

# SORのみ（高速）
python host/denoise_ply.py data/output/object.ply --method sor

# パラメータ調整（より積極的なノイズ除去）
python host/denoise_ply.py data/output/object.ply \
    --sor-neighbors 25 \
    --sor-std-ratio 1.5
```

### アルゴリズム

1. **DBSCAN**: 密度ベースクラスタリングで最大クラスタを抽出し、離れたノイズ塊を除去
2. **SOR**: 各点の近傍点との平均距離を計算し、統計的外れ値を除去

大規模点群（50万点超）では、DBSCANの前にボクセルダウンサンプリングを自動適用してメモリ効率を改善します。

## ディレクトリ構造

```
pi3x/
├── host/
│   ├── setup.sh              # 環境セットアップスクリプト
│   ├── requirements.txt      # ホスト側依存関係
│   ├── inference_server.py   # GPU推論APIサーバー
│   ├── pi3x_cli.py           # Pi3X CLIツール（SAM2なし）
│   └── denoise_ply.py        # ポイントクラウドノイズ削減ツール
├── app/
│   ├── main.py               # Gradio WebUI
│   ├── api_client.py         # 推論サーバークライアント
│   └── video_processor.py    # フレーム処理ユーティリティ
├── docker/
│   ├── Dockerfile            # UIコンテナ定義
│   ├── docker-compose.yml    # サービス構成
│   └── requirements.txt      # コンテナ側依存関係
├── repos/                    # 外部リポジトリ（自動クローン）
│   ├── sam2/
│   └── pi3/
├── data/
│   ├── IMG_1110.mp4          # 入力動画
│   └── output/               # 出力PLY
│       ├── object.ply        # 生成された点群
│       ├── object_denoised.ply # ノイズ削減後の点群
│       ├── camera_poses.json
│       └── masks/
└── .venv/                    # Python仮想環境
```

## API エンドポイント

| Endpoint | Method | 説明 |
|----------|--------|------|
| `/health` | GET | ヘルスチェック |
| `/sam2/init_video` | POST | ビデオ読み込み・フレーム抽出 |
| `/sam2/add_prompt` | POST | クリック座標でマスク生成 |
| `/sam2/propagate` | POST | 全フレームへマスク伝播 |
| `/pi3x/reconstruct` | POST | 点群・カメラ姿勢生成 |
| `/download/ply` | GET | PLYファイルダウンロード |

## 要件

- macOS (Apple Silicon推奨)
- Docker Desktop for Mac
- Python 3.10以上
- PyTorch 2.0+ (MPS対応)

## トラブルシューティング

### サーバーに接続できない

```bash
# 推論サーバーが起動しているか確認
curl http://localhost:5050/health
```

### SAM2チェックポイントがない

```bash
cd repos/sam2/checkpoints
./download_ckpts.sh
```

### メモリ不足

- `frame_interval` を大きくしてフレーム数を減らす
- `recon_frame_interval` を大きくして再構成に使うフレーム数を減らす
