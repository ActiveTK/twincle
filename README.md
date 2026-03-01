# twincle

CUDA GPU で双子素数の個数と Brun の部分和を高速に計算するツールです。  
篩と集計を GPU に寄せ、ホイール分解とセグメント処理で VRAM 使用量を制御しつつ、誤差評価も併せて出力します。

## 概要

対象は `p` と `p+2` が共に素数となる双子素数対です。  
Brun の部分和 `sum_{p, p+2 <= N} (1/p + 1/(p+2))` を計算し、Hardy–Littlewood 型の外挿値 `B2*`（`C2` 使用）や誤差上限も出力します。  
マルチ GPU で並列実行します。

## 使い方

### ビルド

```bash
cargo build --release
```

前提:
- CUDA Toolkit（`nvcc` が `PATH` にあること）
- NVIDIA GPU
- Windows の場合は MSVC コンパイラ（`build.rs` で `-ccbin` を指定）

環境変数:
- `CUDA_PTX_ONLY=1`: PTX のみを生成
- `CUDA_SM120=1|0`: SM120 (RTX 5090) の PTX/SASS 追加を強制/無効
- `CUDA_SM100=1|0`: SM100 (B200) の PTX/SASS 追加を強制/無効

### 実行例

```bash
# N=10^13 まで
cargo run --release -- --limit 10000000000000

# N=10^14 まで
cargo run --release -- --exp 14

# ホイール最適化込み
cargo run --release -- --exp 14 --test-wheels --auto-tune-seg

# 10 分割のうち第 2 区間を実行
cargo run --release -- --exp 14 --splitmode --split 2

# 10 分割のうち第 7 区間を実行
cargo run --release -- --exp 14 --splitmode --split 7

# split の統合
python result_integrate.py
```

### オプション

- `--limit <u64>`: 探索上限（inclusive）
- `--exp <u32>`: 上限を `10^exp` に設定
- `--wheel <u32>`: ホイール法の法 `M`（`30` / `210` / `30030` など）
- `--splitmode`: 固定 10 分割モードを有効化
- `--split <u32>`: 10 分割のうち何番目を実行するか（`1..=10`）
- `--test-wheels`: 複数の `M` を 30 秒ずつ測り最速を採用
- `--segment-k <u64>`: 1 セグメントの `k` 数（0 で自動）
- `--segment-mem-frac <f64>`: VRAM 使用率（自動時、既定 0.25）
- `--auto-tune-seg`: 複数候補で `segment-k` を短時間チューニング
- `--benchmark`: フル探索ではなく一定時間の速度計測
- `--benchmark-seconds <u64>`: ベンチ時間
- `--benchmark-target <u64>`: 到達見積もり用の仮想上限

## アルゴリズム概要

### 1. ホイール分解による候補生成

`M` をホイール法の法とし、以下を満たす剰余 `r` を列挙します。

- `r` は奇数
- `gcd(r, M) = 1`
- `gcd((r+2) mod M, M) = 1`

候補は `p = M * k + r` として表現し、`r` の集合は CPU 側で前計算します。

### 2. 基底素数の生成と分類

`sqrt(limit)` までの素数を奇数篩で生成し、`M` の因子は除外します。  
さらに `p <= 2^15` を「小素数」、それ以上を「大素数」として分割し、異なる GPU カーネルを使います。

### 3. GPU 篩（2 つの合成判定ビットセット）

候補 `p` と `p+2` の合成判定を、ビットセット `comp_p` と `comp_p2` に保持します。

- 大素数: `(prime, residue)` のペアを並列化するカーネルで負荷分散
- 小素数: 1 ブロック=1 素数の prime-major 方式で高密度タイル処理

`p < q^2` の範囲は無視し、最小の `k` から効率的にマーキングします。

### 4. 双子素数の集計

`comp_p` と `comp_p2` が共に 0 のビットが双子素数候補です。  
GPU 側で `1/p + 1/(p+2)` を Kahan 加算で積算し、ブロックごとに縮約します。

### 5. ホイールの取りこぼし補正

ホイールは `gcd(p, M) = gcd(p+2, M) = 1` の候補のみを見るため、  
`M` の素因子に絡む小さな双子素数対を取りこぼします。  
`p <= M` の範囲を CPU 側で総当たりし、個数と和を補正します。

## 最適化の要点

- **ホイール法**: 30/210/30030 を選択可能。`--test-wheels` で実測選択。
- **セグメント処理**: `k` の範囲を分割し、ビットセットの VRAM 使用量を抑制。
- **VRAM 自動割り当て**: `segment_k` を VRAM と `segment_mem_frac` から推定。
- **負荷分散**: 小素数は prime-major、大素数は (prime, residue) ペア並列化。
- **ストリーム順序**: バッファ初期化を同一ストリームに載せて同期を最小化。
- **マルチ GPU**: `k` 範囲をアトミックに割り当てて並行処理。

## 精度と誤差評価

- **Kahan 加算**: GPU カーネル内と CPU 集約の両方で使用。
- **誤差上限の出力**:
  - 集積誤差上限（Higham 型の上限）
  - 1 項評価の丸め誤差上限（`gamma_n(5)`）
- **倍精度 (`f64`)** を使用。

## ログとチェックポイント

- `run.log`: 実行ログ
- `exp_log_e*.jsonl`: `--exp` 指定時のチェックポイント記録
- `result_partXof10.json`: `--splitmode --split X` 指定時の split ログ
- `result.json`: `python result_integrate.py` で生成される統合結果
- `10^(exp-3)` 刻みで `10^exp` まで（全 1000 点）を JSONL で保存
- コンソール出力は `exp>=4` のとき `10^(exp-3)`, `10^(exp-2)`, `10^(exp-1)` のみを要約表示

## split 実行と統合

`--splitmode --split s` を指定すると、`10^n` までの探索を 10 分割したうちの第 `s` 区間だけを探索します。  
`--exp n` のとき、第 `s` 区間は `((s-1)*10^(n-1), s*10^(n-1)]` を担当します。  
たとえば `--exp 17 --splitmode --split 2` は `(10^16, 2*10^16]` を計算します。

このときログファイル名は `result_part2of10.json` のようになり、記録される `sum` は「全体の途中和」ではなく「その split 自体の部分和」です。

`result_partXof10.json` の最後の `final` レコードには以下が入ります。

- その split の `twins`
- その split の `sum`
- `accum_err_bound`
- `term_eval_err_bound`
- `total_err_bound`
- `elapsed_secs`

複数 split を結合するときは次を実行します。

```bash
python result_integrate.py
```

必要ならファイルを明示指定できます。

```bash
python result_integrate.py result_part1of10.json result_part2of10.json
```

統合時には split の `sum` / `twins` / 誤差上限を足し合わせ、ホイールの取りこぼし補正を 1 回だけ加えた `result.json` を出力します。

---

必要に応じて、実験ログやベンチ結果のレポート形式も追加できます。
