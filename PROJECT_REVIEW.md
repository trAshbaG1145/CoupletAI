# CoupletAI 项目评审（基于仓库现状的代码走读）

> 评审范围：`CoupletAI/` 目录下的源码与配置文件（`main.py`、`preprocess.py`、`clidemo.py`、`webdemo.py`、`module/*`、`docker/Dockerfile`、`README.md`、`LICENSE`、`templates/index.html`、`.gitignore`，以及随仓库附带的 `couplet/` 数据目录结构）。
>
> 评审目标：从 **算法/建模、工程结构、可复现性、鲁棒性、安全性、部署、文档** 等方面给出优点、不足与改进建议。

---

## 1. 项目概览

- **功能定位**：自动对对联（输入上联，生成下联）。
- **核心建模思路**：把“生成下联”转为 **序列标注/逐字预测**：用下联 token 去标注上联 token（上下联等长这一先验）。
- **主要入口**：
  - `preprocess.py`：将 `couplet/` 下的 `in.txt/out.txt` 处理成 `dataset/*.pkl`（含 `vocab.pkl`、`train.pkl`、`test.pkl`）。
  - `main.py`：训练与评估（TensorBoard 记录、周期性 demo 预测、BLEU/Rouge-L 评估、保存 checkpoint）。
  - `clidemo.py`：命令行交互式对联。
  - `webdemo.py`：Flask Web UI + 简单 API。

---

## 2. 数据与预处理

### 2.1 数据格式

- 仓库中存在 `couplet/` 目录，包含：
  - `couplet/train/in.txt`、`couplet/train/out.txt`
  - `couplet/test/in.txt`、`couplet/test/out.txt`
  - `couplet/vocabs`
- 数据文件行级对齐：每行由空格分隔的“字”组成，例如：
  - 上联：`晚 风 摇 树 树 还 挺`
  - 下联：`晨 露 润 花 花 更 红`

### 2.2 预处理流程（`preprocess.py`）

- `read_examples()`：从 `in.txt/out.txt` 读取 token 序列。
- `Tokenizer.build(vocab_file)`：从 `vocabs` 构建字表，内置 `[PAD]`、`[UNK]`。
- `convert_features_to_tensors()`：
  - 固定 `max_seq_len`（默认 32）
  - `input_ids/target_ids` pad 到定长
  - `masks` 用 bool 标记 padding 位置（实现是 `masks[i, :real_len]=0`，padding 位置为 1）

### 2.3 优点

- 数据格式简洁直观，且与“上下联等长”的问题结构匹配。
- 预处理逻辑清晰：读入→数值化→张量化→持久化，便于离线训练。

### 2.4 不足与风险

- `couplet/test/` 目录存在 `*.swp`（vim swap）文件：`.in.txt.swp`、`.out.txt.swp`，说明数据目录可能包含编辑器残留；容易污染发布与容器构建。
- `max_seq_len=32` 的默认设置对较长上联会截断：缺少截断策略说明/统计分析。
- token 化是“逐字”，没有词级信息、韵律/平仄特征，也没有对对联常见约束（对仗、词性/结构一致）做显式建模。

---

## 3. 模型与算法设计（`module/model.py`）

### 3.1 模型集合

项目提供了多种对比模型，便于实验：

- `BiLSTM`
- `Transformer`（Encoder-only + embedding weight tying 形式的输出投影）
- `CNN`
- `BiLSTMAttn`（BiLSTM + MultiheadAttention）
- `BiLSTMCNN`（BiLSTM + Conv1d）
- `BiLSTMConvAttRes`（BiLSTM + Conv + Attn + Residual + LayerNorm）

### 3.2 优点

- **模型多样**，对课程实验/论文对比友好。
- **模块化**：新增模型只需在 `module/model.py` 写类、在 `module/__init__.py:init_model_by_key()` 注册。
- 大多数模型共享“embedding → 编码 → 投影 → vocab logits”的统一接口，训练端 `main.py` 简洁。

### 3.3 不足与风险（工程/实现角度）

- `Transformer` 结构相对“轻量/简化”，更像“把 TransformerEncoder 当成特征提取器”的实现：
  - 缺少常见的 embedding/输出层 LayerNorm、某些位置编码处理的细节。
  - `embed_dim` 与 `hidden_dim` 的两层线性映射（`linear1/linear2`）能用，但解释性/可控性一般。
- 当前解码策略是逐位置 argmax（贪心），没有 beam search、约束解码、重排等策略；对对联这种强结构任务，生成质量上限会受限。

---

## 4. 训练与评估（`main.py`、`module/metric.py`）

### 4.1 训练流程

- `DataLoader` 读取 `train.pkl/test.pkl`
- 损失：`nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)`
- 优化器：Adam
- 学习率调度：`ReduceLROnPlateau(optimizer, mode='min')`（按 epoch 累积 loss 调整）
- 多卡：`torch.nn.DataParallel`
- 可选 fp16：依赖 NVIDIA Apex（`--fp16`）
- TensorBoard：每 100 step 记录 loss

### 4.2 指标

- `calc_bleu()`：调用 NLTK BLEU（带平滑）
- `calc_rouge_l()`：手写 LCS 版 Rouge-L

### 4.3 优点

- 训练脚本“一键跑通”，参数集中在 argparse，适合快速实验。
- 训练中 `predict_demos()` 提供定性观察，结合 BLEU/Rouge-L 定量评估，闭环完整。
- 忽略 `[PAD]` 的 loss 处理正确。

### 4.4 不足与风险

- **可复现性不足**：
  - 未设置随机种子（torch/cuda/python），不同运行可能波动。
  - 未记录/固化依赖版本（缺 `requirements.txt`/`pyproject.toml`）。
- **训练稳定性/工程特性不足**：
  - 没有 Early Stopping。
  - 没有断点续训（checkpoint 未保存 optimizer/scheduler 状态）。
  - `ReduceLROnPlateau` 用的是 epoch 累积 loss（`accu_loss`），但未做“平均到 batch/样本”的归一化，跨 batch_size 对比不严格。
- **fp16 依赖 Apex**：Apex 安装门槛高；缺少 PyTorch AMP（`torch.cuda.amp`）的替代路径。

---

## 5. 推理与服务（`clidemo.py`、`webdemo.py`、`templates/index.html`）

### 5.1 CLI Demo

- 交互式输入上联，输出下联。

### 5.2 Web Demo

- Flask Web 表单：`/`
- 简单 API：`/<coupletup>`（将上联放在 URL path 中）

### 5.3 优点

- 有“可用的 demo”，对展示/验收很加分。
- Web UI 极简，依赖少，上手快。

### 5.4 不足与风险

- **模型保存/加载键名不一致风险（高优先级）**：
  - `main.py:save_model()` 存的是 `tokenzier`（拼写错误）。
  - `clidemo.py` 读取的是 `tokenzier`（与保存一致）。
  - `webdemo.py` 当前读取的是 `tokenizer`（与保存不一致）。

  这意味着：同一个 checkpoint 在 CLI 可能能跑，但 Web 可能直接 KeyError 崩溃（或反之，取决于 checkpoint 字典键名）。

- **webdemo.py 存在重复/冗余导入**：
  - 同时 `from main import init_model_by_key` 与 `from module import ..., init_model_by_key`，容易引入维护混乱。

- **服务端输入未做限制**：
  - `/<coupletup>` 把用户输入放在 URL path，未做长度限制/编码处理；异常字符或超长输入可能导致 404、编码错误或性能问题。

- **设备选择固定**：
  - `webdemo.py` 直接 `cpu` 推理，未提供 `cuda` 选项或环境检测。

---

## 6. 工程结构与依赖管理

### 6.1 项目结构

- 顶层脚本清晰：训练/预处理/demo 分离。
- `module/` 拆分得当：模型、tokenizer、metric。

### 6.2 `.gitignore`（优点）

- `.gitignore` 覆盖全面：缓存、虚拟环境、构建产物、IDE 配置、日志、TensorBoard、数据集/输出目录等。
- 对本项目尤其重要：忽略 `output/`、`runs/`、`dataset/`、`couplet/` 这类大文件目录（但当前仓库里仍然存在 `couplet/`，说明实际是否跟踪取决于 git 历史状态）。

### 6.3 依赖管理（不足）

- 缺少 `requirements.txt` / `pyproject.toml`：
  - 用户需要“猜”依赖并手动安装。
  - Docker 也无法可靠复现。

> 从代码和 README 推断依赖至少包括：`torch`、`flask`（可选）、`nltk`、`tqdm`、`tensorboard`。

---

## 7. Docker 与部署（`docker/Dockerfile`）

### 7.1 现状

- 基于 `pytorch/pytorch:latest`（未锁定版本）。
- Dockerfile 内部直接 `git clone https://github.com/WiseDoge/CoupletAI.git`，而不是构建当前工作区代码。
- 仅安装了 `flask`，未安装 `nltk/tqdm/tensorboard` 等训练/评估依赖。
- 直接下载并解压 couplet 数据集到容器中。

### 7.2 优点

- 能快速得到一个“具备 PyTorch 环境 + 数据集”的镜像雏形。

### 7.3 不足与风险

- **不可复现**：`latest`、外部仓库 clone、外部数据下载，都会让构建结果随时间变化。
- **与本仓库脱节**：Dockerfile 构建的并非你当前修改的 CoupletAI 代码。
- 依赖不全：训练或评估很可能在容器内缺包失败。

---

## 8. 文档与开源合规

### 8.1 README

- 优点：说明了数据集来源、基本用法、效果展示、截图。
- 不足：缺少可复现环境说明、缺少模型对比表/超参数建议、缺少 API 文档、缺少常见问题排障。

### 8.2 LICENSE

- MIT License：分发与二次开发友好。

---

## 9. 主要问题清单（按优先级）

### P0（会影响运行）

1. **checkpoint 中 tokenizer 键名不一致**：`tokenzier` vs `tokenizer`（`main.py`/`clidemo.py`/`webdemo.py`）。
2. **`Tokenizer.convert_ids_to_tokens()` 逻辑错误**（`module/tokenizer.py`）：当 `ignore_pad=False` 时会返回空列表，`decode()` 实际会解出空串（这会显著影响 demo 输出的正确性）。

### P1（影响可复现/可维护）

3. 缺少依赖清单（`requirements.txt` 等）。
4. Dockerfile 不构建本地源码、版本不锁定。
5. 训练缺少随机种子、断点续训、早停等基本工程能力。

### P2（质量提升）

6. 推理策略单一（逐位 argmax），缺少对联任务常用的约束/搜索。
7. Web 输入缺少长度限制/错误处理。
8. 文档缺少性能对比与排障。

---

## 10. 改进建议（可执行）

### 10.1 先修复功能正确性（建议优先做）

- 统一 checkpoint 字段名：建议统一为 `tokenizer`（同时兼容旧字段，加载时做 fallback）。
- 修复 `Tokenizer.convert_ids_to_tokens()`：确保 `ignore_pad=False` 时返回完整 token 序列。

### 10.2 提升可复现

- 增加 `requirements.txt` 并在 README 写明 Python/PyTorch 版本建议。
- 在训练脚本加 `--seed` 并统一设置 `random/numpy/torch`。

### 10.3 提升训练工程性

- 保存 optimizer/scheduler 状态到 checkpoint，支持断点续训。
- 加 Early Stopping。
- fp16 优先使用 `torch.cuda.amp`，把 Apex 作为可选项。

### 10.4 推理与服务优化

- Web 端增加输入校验：长度上限、空输入、异常捕获。
- 提供 GPU 推理开关或自动检测。
- 可选：beam search/约束解码（对联任务往往收益明显）。

### 10.5 Docker 改造方向

- Dockerfile 改为 `COPY . /app` 构建当前源码。
- 锁定基础镜像版本与依赖版本。
- 明确区分“训练镜像”和“推理镜像”。

---

## 11. 总体评分（主观但有依据）

- **创新性**：★★★★☆（把对联生成转为序列标注，思路很巧）
- **可扩展性**：★★★★☆（模型模块化良好，便于做消融与对比）
- **工程质量**：★★☆☆☆（存在关键 bug；缺依赖清单/测试/复现手段）
- **可用性（demo）**：★★★☆☆（有 CLI/Web，但健壮性不足）

**综合：★★★☆☆（3/5）**

适合作为学习/课程实验/研究对比的项目雏形；若要“长期维护或部署”，建议先解决 P0/P1 问题并补齐可复现与健壮性能力。
