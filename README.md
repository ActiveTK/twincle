# twincle

GPU（CUDA）およびCPUで双子素数の候補を高速に走査し、Brun 部分和（B2）の近似を計算する Rust 実装です。wheel（30 / 210 / 30030 など）による候補削減と、CUDA カーネルによるビットセットふるいを組み合わせ、進捗・速度・推定時間を表示しながら探索します。CPU モードも備えています。

**主な特徴**
- CUDA GPU と CPU の両対応（`--cpu` でCPUモード）
- wheel 法による候補削減（`--wheel` / `--test-wheels`）
- セグメント分割で VRAM 量に合わせた処理（`--segment-k` / `--segment-mem-frac` / `--auto-tune-seg`）
- Brun 部分和と推定値 `B2*` の出力
- 進捗・速度・ETA をログと標準出力へ

## 仕組み（概要）
- 候補は `p = M * k + r`（`gcd(p, M)=1` かつ `gcd(p+2, M)=1`）のみを対象にします。
- 各セグメントで `p` と `p+2` が合成数であるかをビットセットでマークし、両方が未マークなら双子素数候補とみなします。
- GPU では CUDA カーネルでふるいと集計を行い、CPU では同等の処理をスレッド分割で実行します。

## 必要要件
- Rust（edition 2024）
- CUDA Toolkit（`nvcc` が PATH にあること）
- NVIDIA GPU（CUDA 対応）

GPU がない環境でも `--cpu` で動作します。

## ビルド

```bash
cargo build --release
```

ビルド時に `src/kernel.cu` が `nvcc` でコンパイルされます。以下の環境変数を利用できます。

- `CUDA_PTX_ONLY=1`：PTX のみ生成（fatbin 生成を回避）
- `CUDA_SM120=1|0`：SM120（例: RTX 5090）向けコード生成を強制/無効化

Windows では MSVC のパスを `build.rs` で指定しています。環境に合わせて必要なら修正してください。

## 実行例

GPU モード（デフォルト）:

```bash
cargo run --release -- --limit 1000000000000 --wheel 30030
```

指数で指定:

```bash
cargo run --release -- --exp 12
```

CPU モード（自動スレッド数）:

```bash
cargo run --release -- --cpu
```

CPU スレッド数指定:

```bash
cargo run --release -- --cpu 12
```

ホイールサイズの自動選定（30 / 210 / 30030 を各30秒テスト）:

```bash
cargo run --release -- --test-wheels
```

ベンチマークのみ実行（指定秒数）:

```bash
cargo run --release -- --benchmark --benchmark-seconds 10
```

## 主要オプション（抜粋）
- `--limit <N>`: 探索上限（既定: `10,000,000,000,000`）
- `--exp <E>`: `limit = 10^E` を使用
- `--wheel <M>`: wheel modulus（30 / 210 / 30030）
- `--test-wheels`: 各 wheel を短時間テストして最速を選ぶ
- `--cpu [N]`: CPU モード（任意でスレッド数）
- `--segment-k <K>`: セグメントの k 幅（0 で自動）
- `--segment-mem-frac <F>`: VRAM の利用比率（自動時）
- `--auto-tune-seg`: セグメント自動チューニング
- `--benchmark`: ベンチマークのみ実行

## 出力
- 進捗バー、候補/秒、ETA
- twin 数、Brun 部分和、推定 `B2*`
- すべて `run.log` にも記録されます

## 構成
- `src/main.rs`: CLI、探索制御、GPU 実行、最終レポート
- `src/cpu.rs`: CPU 実装
- `src/kernel.cu`: CUDA カーネル
- `build.rs`: `nvcc` 呼び出し・アーキテクチャ設定

## 注意
- 非常に大きな `--limit` を指定すると時間・メモリ消費が大きくなります。
- GPU での最適パラメータは機種・VRAM に依存します。
