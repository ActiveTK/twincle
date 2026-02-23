以下は、vastaiを使うときに参考にすべき内容です。

・vastaiコマンドが利用可能
・インスタンスを立ち上げてDockerfileを使ってcargoをいれた上で git clone https://github.com/ActiveTK/twincle でこのプロジェクトを取得してビルド・実行。
・結果を取得して、それを一覧にまとめてほしい。
・いろんな種類のGPUを試してみてほしいけど、ちゃんとbenchmarkを実行したあとはインスタンスを削除すること。
・インスタンスは一つずつ実行し、終了時には確実に削除する。絶対に3分以上一つのインスタンスを実行し続けない。
・インスタンス操作はスクリプトを用いてはならず、確実に毎回vastaiコマンドを使用すること。
・Start-Sleep -Seconds 20とvastai logsを繰り返しながら状態を確認する。
・インスタンスごとに対応しているCUDAのバージョンは違う。
・ネットワークが極端に遅いものは選んではいけない。
・GFWなどによりgithubに接続できないなどネットワークが不安定なインスタンスが存在する。直ちに破棄せよ。
・そもそも設定不備により起動しないインスタンスも存在する。直ちに破棄せよ。

# Vast.ai 運用メモ（twincle）

## 決まりごと（ユーザー指示）
- **1台ずつ厳密に**「起動 → ベンチ → 削除」。並列起動しない。
- **3分超えたら即削除**（何があっても）。エラーが出た瞬間に確実に削除。
- **PS1禁止**。自分の制御外に出るスクリプトは使わず、都度コマンド実行。
- 起動が遅い・ネットが遅い・地域要因（GFWなど）が疑われる場合は、**別オファーに切り替える**。
- **Dockerイメージはローカルで構築**して Docker Hub に push。インスタンス側で `apt` などは極力使わない。

## Dockerイメージの構築と保存先
- Docker Hub リポジトリ: `fuckdocker42731/twincle`
- 使うタグ（CUDA別）:
  - `fuckdocker42731/twincle:cuda12.4`
  - `fuckdocker42731/twincle:cuda12.2`
  - `fuckdocker42731/twincle:cuda11.8`

### build / push コマンド（ローカル）
```powershell
# CUDA 12.4
 docker build -f Dockerfile.bench --build-arg CUDA_TAG=12.4.1 -t fuckdocker42731/twincle:cuda12.4 .
 docker push fuckdocker42731/twincle:cuda12.4

# CUDA 12.2
 docker build -f Dockerfile.bench --build-arg CUDA_TAG=12.2.2 -t fuckdocker42731/twincle:cuda12.2 .
 docker push fuckdocker42731/twincle:cuda12.2

# CUDA 11.8
 docker build -f Dockerfile.bench --build-arg CUDA_TAG=11.8.0 -t fuckdocker42731/twincle:cuda11.8 .
 docker push fuckdocker42731/twincle:cuda11.8
```

## SSHキーと接続方法
- **使う鍵**: `C:\Users\ActiveTK\.ssh\vastai`
- 権限が緩いと SSH が拒否されるので、必要なら:
```powershell
icacls C:\Users\ActiveTK\.ssh\vastai /inheritance:r /grant:r ActiveTK:F
```

接続コマンド例:
```powershell
# ssh-url で接続先確認
vastai ssh-url <instance_id>

# 直接接続してベンチ実行
ssh -i C:\Users\ActiveTK\.ssh\vastai -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@<IP> -p <PORT> "/app/twincle --benchmark --benchmark-seconds 10 --limit 1000000000000"
```

## ベンチの実行手順（1台ずつ）
1. **オファー検索**
```powershell
vastai search offers "gpu_name=RTX_4090" --raw --limit 3 -o "dph"
```
2. **インスタンス作成**（CUDAバージョンは `cuda_max_good` を見て選ぶ）
```powershell
vastai create instance <offer_id> --image fuckdocker42731/twincle:cuda12.4 --disk 20 --ssh --direct
```
3. **起動待ち**（`loading`→`running`になるまで短く待つ）
```powershell
vastai show instances
```
4. **SSHでベンチ実行**
```powershell
vastai ssh-url <instance_id>
ssh -i C:\Users\ActiveTK\.ssh\vastai -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@<IP> -p <PORT> "/app/twincle --benchmark --benchmark-seconds 10 --limit 1000000000000"
```
5. **即削除**
```powershell
vastai destroy instance <instance_id>
```

## 失敗時の対処メモ
- `PTX JIT compilation failed` が出たら **即削除**。
- `loading` が長引く／SSHが開かないなら **3分以内に削除**。
- SSH失敗時は **鍵が正しいか**（`.ssh\vastai`）と権限を確認。

## 注意点
- `vastai logs` は `No such container` が出ることがある（ssh準備前など）。
- 速度検証は **`/app/twincle --benchmark --benchmark-seconds 10 --limit 1000000000000`** で統一。
- 必ず `vastai show instances` で **稼働中ゼロ** を確認する。
