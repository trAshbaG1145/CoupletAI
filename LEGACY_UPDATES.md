# CoupletAI：LEGACY_NOTES 对照检查与已落地改进（2025-12-24）

本文对照 `LEGACY_NOTES.md` 的条目逐项核查当前仓库状态，并总结已经落地的现代化改进与仍残留/可继续迭代的点。

---

## 1) 混合精度训练：Apex AMP

- 原问题：`--fp16` 依赖 NVIDIA Apex AMP。
- 当前状态：已修复（不再依赖 Apex）。
- 已做改进：
  - `main.py` 使用 `torch.cuda.amp.autocast` + `torch.cuda.amp.GradScaler` 实现混合精度。
- 仍可继续：
  - `--fp16_opt_level` 目前已无实际作用（历史遗留参数，可后续清理）。

## 2) 多卡训练：DataParallel

- 原问题：检测多卡自动走 `torch.nn.DataParallel`。
- 当前状态：部分修复。
- 已做改进：
  - `main.py` 新增 `--ddp`，支持用 `torchrun` 启动 DDP（`DistributedSampler`、只在 rank0 写日志/保存）。
  - `README.md` 补充了 DDP 启动示例。
- 仍存在：
  - 未开启 `--ddp` 时，如果检测到多卡仍会走 `DataParallel`（兼容保留）。
- 建议：
  - 如果你希望“默认不再使用 DataParallel”，可以改成：默认单卡；显式 `--dp` 才启用 DataParallel。

## 3) README 依赖版本声明过老

- 原问题：README 写 `python 3.6+`、`pytorch 1.2+`。
- 当前状态：已修复。
- 已做改进：
  - `README.md` 更新为建议 `Python 3.9+`、`PyTorch 2.x`，并给出 PyTorch 官网安装指引。

## 4) 缺少依赖清单

- 原问题：无 `requirements.txt`/`pyproject.toml`。
- 当前状态：已修复（最低限度）。
- 已做改进：
  - 新增 `requirements.txt`（包含 `flask/nltk/tensorboard/tqdm`；PyTorch 仍建议按平台从官网安装）。
- 仍可继续：
  - 若追求更强可复现，可追加 `pyproject.toml` + 锁定工具（pip-tools/poetry/uv）。

## 5) Dockerfile：latest + git clone 外部仓库 + 构建期下载数据

- 原问题：不可复现、构建慢且与本地代码无关。
- 当前状态：已修复（按本地代码构建）。
- 已做改进：
  - `docker/Dockerfile`：
    - 固定基础镜像版本（`pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime`）。
    - `COPY . /app` 使用当前工作区代码。
    - `pip install -r requirements.txt` 安装完整依赖。
  - `README.md`：将 Docker 用法改为 `docker build -f docker/Dockerfile -t coupletai .`。
- 仍可继续：
  - 目前镜像为 CUDA runtime 版本；如需 CPU 环境，可再提供一个 CPU Dockerfile（或 ARG 切换）。
  - 数据集仍需用户按 README 下载/解压（这是更可控的方式）。

## 6) Web 服务：sys.argv[1] + app.run + 把输入放 URL path

- 原问题：参数不健壮、开发服务器、API 设计不佳。
- 当前状态：部分修复。
- 已做改进：
  - `webdemo.py`：
    - 改为 `argparse`：`python webdemo.py --model <path>`。
    - 新增 `POST /predict` JSON API（`{"coupletup": "..."}`），不再把输入放在 URL path。
    - 增加基本输入校验与错误返回。
- 仍存在：
  - 仍使用 Flask 内置开发服务器 `app.run(...)`（适合 demo，不适合生产）。
- 建议：
  - 若要生产部署：Windows 可用 `waitress`；Linux/容器可用 `gunicorn`。

## 7) 前端依赖：Bootstrap 3 CDN

- 原问题：Bootstrap 3 过老。
- 当前状态：已修复。
- 已做改进：
  - `templates/index.html` 升级到 Bootstrap 5 CDN。
- 仍可继续：
  - 目前仍为外链 CDN；如需离线/内网稳定性，可把 CSS vendoring 到 `static/`。

## 8) 训练工程习惯：无 seed、checkpoint 不含优化器状态

- 原问题：不可复现、无法无缝恢复训练。
- 当前状态：已修复（核心部分）。
- 已做改进：
  - `main.py`：新增 `--seed` 并设置 `random/torch` 种子。
  - checkpoint 额外保存：`optimizer/scheduler/scaler/epoch/global_step`。
  - 多卡保存：保存前剥离 `DataParallel/DDP` 的 `.module`，避免 key 前缀问题。
- 仍可继续：
  - 未设置 `cudnn.deterministic/benchmark`（是否开启取决于性能/确定性权衡）。
  - 未实现“从 checkpoint 恢复训练”的加载逻辑（只做了保存增强）。

## 9) 评估指标库：NLTK BLEU

- 原问题：NLTK 依赖偏重且 BLEU 对该任务未必最合适。
- 当前状态：仍存在（按 baseline 保留）。
- 已做改进：
  - `module/metric.py` 修正类型标注与返回类型（更易被类型检查/IDE 理解）。
  - `requirements.txt` 明确列出 `nltk`。
- 建议：
  - 如需更贴合对联任务，可补充 chrF/BERTScore/或对仗规则类指标；BLEU/Rouge-L 继续作为 baseline。

## 10) 其他“老味道”但不一定要改

- `TensorDataset` + `torch.save(train.pkl)`：仍在使用（可接受）。
- `ReduceLROnPlateau`：仍在使用（可接受）。

---

## 附：本次涉及的主要改动文件

- 依赖与文档：`requirements.txt`、`README.md`
- 训练：`main.py`
- Web：`webdemo.py`、`templates/index.html`
- Docker：`docker/Dockerfile`
- 指标/类型标注：`module/metric.py`

