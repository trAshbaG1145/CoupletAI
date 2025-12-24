# CoupletAI：当前项目 vs 原始项目（对比文档）

> 说明
> - “原始项目”指本次改造前的仓库状态，以 `PROJECT_REVIEW.md` 与 `LEGACY.md` 中描述的问题点为基线。
> - “当前项目”指本次会话内已落地的修改后的状态（以当前工作区代码为准）。

---

## 1) 一句话总结

- 原始项目：能跑通训练/CLI/Web demo，但存在关键正确性 bug、依赖/部署不可复现、训练工程能力不足、推理解码过于单一。
- 当前项目：修复了关键 bug，补齐依赖与更可复现的 Docker 构建，训练侧支持 AMP/DDP/seed/resume/early stopping，并新增可选的约束/搜索解码，且已接入 CLI/Web/训练 demo。

---

## 2) 依赖与可复现

- 原始项目
  - README 仅给出宽泛版本（`python 3.6+`、`pytorch 1.2+`），无依赖清单。
- 当前项目
  - 新增 `requirements.txt`（Flask/NLTK/TensorBoard/tqdm；PyTorch 按官网安装）。
  - `README.md` 更新为建议 `Python 3.9+`、`PyTorch 2.x`，并给出安装方式。

影响/注意：
- 依赖安装从“猜依赖”变为“可一键安装（除 PyTorch）”。

---

## 3) 训练：AMP / 多卡 / 可复现 / 断点

- 原始项目
  - fp16 依赖 NVIDIA Apex（安装门槛高）。
  - 多卡自动走 `DataParallel`。
  - 未设置随机种子。
  - checkpoint 仅保存模型与少量信息；无法恢复 optimizer/scheduler 等训练状态。
  - 无 early stopping。
- 当前项目
  - `main.py`：Apex → PyTorch AMP（`autocast` + `GradScaler`）。
  - `main.py`：新增 `--ddp` 支持 DDP（配合 `torchrun`）。
  - `main.py`：DataParallel 改为显式 `--dp` 才启用（兼容保留，但默认不再自动启用）。
  - `main.py`：新增 `--seed`。
  - `main.py`：增强 checkpoint（保存 optimizer/scheduler/scaler/epoch/global_step）。
  - `main.py`：新增 `--resume`（从训练态 checkpoint 恢复训练）。
  - `main.py`：新增 `--early_stop_patience/--early_stop_min_delta`。

影响/注意：
- 多卡推荐用 DDP：`torchrun --standalone --nproc_per_node=<N> main.py --ddp`。
- `--fp16_opt_level` 为历史遗留参数（当前已不影响 PyTorch AMP 行为）。

---

## 4) checkpoint 兼容性（tokenizer 键名）

- 原始项目
  - 存在 `tokenzier`（拼写错误）与 `tokenizer` 混用风险，导致不同入口可能 KeyError。
- 当前项目
  - 保存统一为 `tokenizer`（`main.py`）。
  - 读取端做兼容：优先 `tokenizer`，fallback `tokenzier`（`clidemo.py`、`webdemo.py`）。

影响/注意：
- 旧 checkpoint 仍可加载；新 checkpoint 字段统一为 `tokenizer`。

---

## 5) tokenizer 正确性修复

- 原始项目
  - `Tokenizer.convert_ids_to_tokens()` 的 `ignore_pad` 逻辑存在错误，可能导致 `decode()` 得到空串或异常行为。
- 当前项目
  - 修复为：仅在 `ignore_pad=True` 时跳过 `[PAD]`；未知 id 回退为 `[UNK]`。

影响/注意：
- 该修复会改变部分情况下的 `decode()` 输出（从“异常/空”变为“合理可读”）。

---

## 6) 推理策略（P2#6：逐位 argmax → 约束/搜索解码）

- 原始项目
  - 推理策略单一：逐位置 `argmax`。
- 当前项目
  - 新增 `module/decoding.py`：
    - `argmax`（保留原始行为）
    - `constrained`（带约束的贪心，默认）
    - `beam`（带约束的 beam search）
  - 约束项（可通过参数控制）：默认禁 PAD/UNK；no-copy/标点对齐为可选开关；并提供重复限制等约束。
  - 接入入口：`clidemo.py`、`webdemo.py`、`main.py` 的 `predict_demos()`。

影响/注意：
- 默认解码从“纯 argmax”变为“带约束的贪心”，输出风格可能变化。
- 如需完全复现原始行为：
  - CLI：`python clidemo.py -p <ckpt> --decode argmax`
  - Web：启动时加 `--decode argmax`

---

## 7) Web demo：参数、API、健壮性

- 原始项目
  - 用 `sys.argv[1]` 读取模型路径。
  - 把输入放 URL path（`/<coupletup>`），缺少长度限制/错误处理。
  - 推理固定 CPU。
- 当前项目
  - `webdemo.py` 改为 `argparse`：`python webdemo.py --model <path>`。
  - 新增 `POST /predict` JSON API：`{"coupletup": "..."}`。
  - 增加输入校验（空输入/长度限制）与错误返回。
  - 增加 `--cuda` 与 `--max_input_len`。
  - 模板 `templates/index.html` 增加错误提示区域。

影响/注意：
- 仍使用 Flask 开发服务器 `app.run(...)`（适合 demo；生产建议 waitress/gunicorn）。

---

## 8) Docker 构建方式

- 原始项目
  - `FROM pytorch/pytorch:latest`（不可复现）。
  - 构建期 `git clone` 外部仓库、下载数据集（与本地修改脱节、构建慢且易失败）。
  - 依赖安装不完整。
- 当前项目
  - `docker/Dockerfile` 固定基础镜像版本（`pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime`）。
  - `COPY . /app` 构建当前工作区代码。
  - `pip install -r requirements.txt` 安装依赖。

影响/注意：
- 数据集不再在构建期下载：按 `README.md` 手动下载/解压或用 volume 挂载更可控。

---

## 9) 文档与使用方式变化（摘要）

- `README.md` 新增/更新了：
  - 依赖安装（含 PyTorch 官方指引）
  - DDP/DataParallel 启动方式
  - resume/early stopping
  - 推理解码参数示例
  - Docker build 命令

---

## 10) 当前与原始的“行为变化”清单（便于验收）

- 推理默认解码：argmax → constrained（可用 `--decode argmax` 回退）。
- 多卡默认行为：自动 DataParallel → 默认不启用（需 `--dp` 或 `--ddp`）。
- checkpoint tokenizer 键：`tokenzier`（旧）→ `tokenizer`（新，读取端仍兼容旧）。
- Web API：`/<coupletup>` → `POST /predict` JSON。

