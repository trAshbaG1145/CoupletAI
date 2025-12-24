# CoupletAI 部署与运行指南

本文提供**完整**的“从 0 到可用”运行/部署步骤（以 Windows PowerShell 为主），涵盖：数据准备 → 预处理 → 训练 → 推理（CLI/Web/API）→ Docker。

---

## 0. 目录约定（重要）

本项目默认约定：

- 数据原始目录：`couplet/`（解压后应包含 `train/`、`test/`、`vocabs`）
- 预处理输出目录：`dataset/`（生成 `vocab.pkl/train.pkl/test.pkl`）
- 训练输出目录：`output/`（保存模型与训练状态）

你可以用参数覆盖：

- `preprocess.py --input/--output/--max_seq_len`
- `main.py --dir/--output/--logdir/--max_seq_len` 等

---

## 1. 本地运行（推荐）

### 1.1 环境准备

- Python：建议 3.9+
- PyTorch：按官方指引安装（CPU/CUDA 版本差异很大）：
  - https://pytorch.org/get-started/locally/

安装项目依赖（除 PyTorch 外）：

```bash
pip install -r requirements.txt
```

### 1.2 下载并准备数据集

数据集（couplet.tar.gz）：
- https://github.com/wb14123/couplet-dataset/releases/download/1.0/couplet.tar.gz

在仓库根目录执行（PowerShell 里也可直接运行）：

```powershell
Set-Location E:\Coding\CoupletAI
curl -L -o couplet.tar.gz https://github.com/wb14123/couplet-dataset/releases/download/1.0/couplet.tar.gz
# Windows 10/11 自带 tar
tar -xzf couplet.tar.gz
```

解压后应得到：

- `couplet/vocabs`
- `couplet/train/in.txt`、`couplet/train/out.txt`
- `couplet/test/in.txt`、`couplet/test/out.txt`

### 1.3 预处理（生成 dataset/*.pkl）

```powershell
python preprocess.py --input couplet --output dataset --max_seq_len 32
```

产物：

- `dataset/vocab.pkl`
- `dataset/train.pkl`
- `dataset/test.pkl`

### 1.4 训练（单卡/CPU）

最简训练：

```powershell
python main.py
```

常用参数示例：

```powershell
python main.py -m transformer --epochs 20 --batch_size 768 --max_seq_len 32 --dir dataset --output output --logdir runs
```

设备说明：
- 默认会在可用时使用 CUDA；如需强制 CPU：加 `--no_cuda`。

混合精度（仅 CUDA 生效）：

```powershell
python main.py --fp16
```

TensorBoard（可选）：

```powershell
tensorboard --logdir runs
```

### 1.5 多卡训练（DDP，推荐）

用 `torchrun` 启动（2 卡示例）：

```powershell
torchrun --standalone --nproc_per_node=2 main.py --ddp
```

说明：
- DDP 下只在 rank0 写日志/保存模型。

（可选）DataParallel（不推荐，兼容保留）：

```powershell
python main.py --dp
```

### 1.6 checkpoint、断点续训与早停

保存规则（见 `main.py`）：
- 每隔 `--save_epoch` 个 epoch 保存一次模型：`output/<ModelClass>_<epoch>.bin`
- 同时会写一个训练状态文件：`output/last_state.bin`（包含 optimizer/scheduler/scaler/epoch/global_step 等）

断点续训：

```powershell
python main.py --resume output\last_state.bin
```

Early Stopping（可选）：

```powershell
python main.py --early_stop_patience 3 --early_stop_min_delta 0.0
```

---

## 2. 推理

### 2.1 命令行（CLI）

使用训练生成的 checkpoint：

```powershell
python clidemo.py -p output\Transformer_10.bin
```

解码策略（可选）：

```powershell
# 原始逐位 argmax
python clidemo.py -p output\Transformer_10.bin --decode argmax

# 约束贪心（默认策略），可调 topk
python clidemo.py -p output\Transformer_10.bin --decode constrained --topk 20

# beam search
python clidemo.py -p output\Transformer_10.bin --decode beam --beam_size 5 --topk 20

# 可选约束开关（默认关闭，显式开启）
python clidemo.py -p output\Transformer_10.bin --no_copy --match_punct
```

如需 GPU 推理：

```powershell
python clidemo.py -p output\Transformer_10.bin --cuda
```

### 2.2 Web（页面 + API）

启动服务：

```powershell
python webdemo.py --model output\Transformer_10.bin --host 0.0.0.0 --port 5000
```

启用 GPU 推理（如果 CUDA 可用）：

```powershell
python webdemo.py --model output\Transformer_10.bin --cuda
```

浏览器访问：
- http://127.0.0.1:5000/

#### Web API

接口：`POST /predict`

PowerShell 调用示例：

```powershell
$body = @{ coupletup = "马齿草焉无马齿" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/predict -ContentType "application/json" -Body $body
```

返回示例：

```json
{"coupletdown":"..."}
```

---

## 3. Docker 部署

### 3.1 构建镜像

在仓库根目录：

```powershell
docker build -f docker/Dockerfile -t coupletai .
```

说明：
- 镜像不会在构建期下载数据集。
- 训练产物（`output/`）与数据集（`couplet/`、`dataset/`）建议通过 volume 挂载。

### 3.2 容器内启动 Web（推荐方式）

前提：你已经在宿主机训练好了模型，且 `output/` 中存在可用的 `*.bin`。

```powershell
docker run --rm -p 5000:5000 `
  -v ${PWD}\output:/app/output `
  coupletai `
  python webdemo.py --model /app/output/Transformer_10.bin --host 0.0.0.0 --port 5000
```

如果需要容器使用 GPU（取决于你的 Docker/驱动环境，例如 Docker Desktop + WSL2）：

```powershell
docker run --rm --gpus all -p 5000:5000 `
  -v ${PWD}\output:/app/output `
  coupletai `
  python webdemo.py --model /app/output/Transformer_10.bin --cuda --host 0.0.0.0 --port 5000
```

### 3.3 容器内训练（可选）

前提：你在宿主机准备好了 `dataset/`（运行过 `preprocess.py`）。

```powershell
docker run --rm `
  -v ${PWD}\dataset:/app/dataset `
  -v ${PWD}\output:/app/output `
  coupletai `
  python main.py --dir dataset --output output
```

---

## 4. 常见问题（FAQ）

1) `torchrun` 找不到？
- 通常是 PyTorch 未正确安装或环境未激活；确保 `python -c "import torch; print(torch.__version__)"` 能运行。

2) Web 返回 `model not loaded`？
- 启动 `webdemo.py` 时未传 `--model` 或路径不正确。

3) `coupletup too long`？
- Web 端默认 `--max_input_len 64`，可调整：
  - `python webdemo.py --model ... --max_input_len 128`

4) 想复现“原始解码行为”？
- CLI/Web 启动时显式设置：`--decode argmax`
