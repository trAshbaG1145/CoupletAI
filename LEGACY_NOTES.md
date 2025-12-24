# CoupletAI：过时内容与现代化建议（5~6 年老项目梳理）

> 说明：本文件聚焦“过时/不推荐/需要现代化”的点，包含：现状 → 为什么过时 → 推荐替代 → 修改成本（粗略）。
> 
> 评审基于当前仓库内容：`README.md`、`main.py`、`preprocess.py`、`clidemo.py`、`webdemo.py`、`module/*`、`docker/Dockerfile`、`templates/index.html`、`.gitignore`。

---

## 1. 混合精度训练：Apex AMP（`main.py`）

- **现状**：`--fp16` 走 NVIDIA Apex 的 `amp.initialize(...)`，并要求安装 Apex。
- **为什么过时**：PyTorch 已长期提供官方 AMP（`torch.cuda.amp` / `torch.autocast` / `GradScaler`），更易安装、维护更稳定、与 PyTorch 版本更匹配。
- **推荐替代**：
  - 用 `torch.cuda.amp.autocast` + `torch.cuda.amp.GradScaler` 实现 fp16。
  - Apex 作为可选兼容（若存在则可用 fused 优化），但不作为默认依赖。
- **修改成本**：中（主要改训练 step：forward/loss/backward/step 部分）。

---

## 2. 多卡训练：`torch.nn.DataParallel`（`main.py`）

- **现状**：检测到多卡就用 `DataParallel(model)`。
- **为什么过时**：`DataParallel` 已不再推荐作为主方案；性能、可控性与容错都不如 DDP。
- **推荐替代**：
  - 使用 `torch.distributed.run` + `torch.nn.parallel.DistributedDataParallel (DDP)`。
  - 在研究/课程环境里，也可继续单卡；但若要多卡，建议 DDP。
- **修改成本**：中到高（涉及启动方式、Sampler、rank/world_size、日志与保存）。

---

## 3. 依赖版本声明过老（`README.md`）

- **现状**：README 写 `python 3.6+`、`pytorch 1.2+`。
- **为什么过时**：Python 3.6 已停止维护多年；PyTorch 1.2 也非常老，很多生态（TensorBoard、CUDA、AMP、依赖解析）都已变化。
- **推荐替代**：
  - 将 README 调整为更现实的范围（示例：Python 3.9/3.10/3.11，PyTorch 2.x）。
  - 增加 `requirements.txt` 或 `pyproject.toml` 锁定依赖范围。
- **修改成本**：低（文档+依赖清单），但要实际验证可运行（中）。

---

## 4. 缺少依赖清单（`requirements.txt`/`pyproject.toml` 缺失）

- **现状**：仓库内没有 `requirements.txt` / `pyproject.toml`。
- **为什么算“过时实践”**：现代 Python 项目基本都会提供可复现依赖描述；否则安装全靠猜。
- **推荐替代**：
  - 最低限度：提供 `requirements.txt`（至少把 `torch/flask/nltk/tqdm/tensorboard` 写清）。
  - 更规范：用 `pyproject.toml` + `pip-tools/poetry/uv` 管理依赖。
- **修改成本**：低。

---

## 5. Dockerfile：基于 `latest` + 直接 `git clone` 外部仓库（`docker/Dockerfile`）

- **现状**：
  - `FROM pytorch/pytorch:latest`
  - `RUN git clone https://github.com/WiseDoge/CoupletAI.git`
  - 容器内再下载数据集
- **为什么过时/不推荐**：
  - `latest` 不可复现；
  - 构建时 clone 外部仓库，与你本地修改无关；
  - 运行时下载数据集，构建慢且易失败；
  - 依赖安装不完整（只 `pip install flask`）。
- **推荐替代**：
  - 锁定基础镜像版本（例如 `pytorch/pytorch:2.x.x-cudaXX-runtime`）。
  - `COPY . /app` 构建当前工作区代码。
  - 用 `requirements.txt` 一次性安装依赖。
  - 将数据集下载步骤外置（文档引导或单独 stage），或用 volume 挂载。
- **修改成本**：中。

---

## 6. Web 服务与参数解析：`sys.argv[1]` + `app.run(...)`（`webdemo.py`）

- **现状**：
  - 用 `MODEL_PATH = sys.argv[1]` 取参数
  - `app.run(host='0.0.0.0')` 直接启动开发服务器
  - API 设计为 `/<coupletup>`（把用户输入放在 URL path）
- **为什么过时/不推荐**：
  - `sys.argv` 方式不健壮、不自解释；
  - Flask 内置 server 仅适用于开发；
  - 把输入放 path 不利于编码/长度控制，也不便于扩展。
- **推荐替代**：
  - 用 `argparse`（或环境变量）读取模型路径。
  - 生产部署用 gunicorn/uwsgi（Windows 可用 waitress）或容器内跑 gunicorn。
  - API 用 `POST /predict` + JSON body，并做长度限制与错误处理。
- **修改成本**：中。

---

## 7. 前端依赖：Bootstrap 3 CDN（`templates/index.html`）

- **现状**：引用 `https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css`。
- **为什么算“过时点”**：Bootstrap 3 已非常老；同时外部 CDN 在部分网络环境可能不稳定。
- **推荐替代**：
  - 若只做 demo：直接写少量原生 CSS，或升级到 Bootstrap 5。
  - 或把静态资源 vendoring 到本地（`static/`），避免外链。
- **修改成本**：低到中。

---

## 8. 训练工程习惯：未设置随机种子/未保存优化器状态（`main.py`）

- **现状**：
  - 没有 `--seed`，也没有 `random/numpy/torch` 的全套 seed。
  - checkpoint 只保存 `model.state_dict()` + `args` + `tokenizer`，未保存 optimizer/scheduler。
- **为什么算“老项目常见欠缺”**：现代 ML 项目通常强调可复现与可中断恢复。
- **推荐替代**：
  - 增加 `--seed` 并设置 `torch.backends.cudnn.deterministic/benchmark`（按需求）。
  - checkpoint 增加 `optimizer_state_dict`、`scheduler_state_dict`、`epoch/global_step`。
- **修改成本**：中。

---

## 9. 评估指标库：NLTK BLEU（`module/metric.py`）

- **现状**：使用 NLTK `sentence_bleu`。
- **为什么可能“偏老/不够合适”**：
  - 对字符级对联任务，BLEU 对语义与对仗未必敏感；
  - NLTK 依赖可能较重，安装也常因资源/版本出问题。
- **推荐替代**（可选）：
  - 更轻量的实现或替代指标（chrF、BERTScore、任务自定义约束分数）。
  - 保留 BLEU/Rouge-L 作为 baseline，但补充更贴合对联的指标。
- **修改成本**：低到中。

---

## 10. 其他“老味道”但不一定要改

- `TensorDataset`/`torch.save(train.pkl)` 的数据持久化方式：能用但偏“脚本化”；若项目要长期维护，可考虑自定义 Dataset + mmap/arrow 等。
- `ReduceLROnPlateau` 仍可用，但现代常见的是 warmup + cosine/linear schedule（尤其 transformer）。

---

## 建议的现代化路线（最小可行）

1. **先补可复现**：加 `requirements.txt`，更新 README 的 Python/PyTorch 建议版本。
2. **替换 Apex**：`--fp16` 改成 `torch.cuda.amp`。
3. **修 Dockerfile**：锁定镜像版本、COPY 本地代码、安装完整依赖、不要在构建时 clone 外部仓库。
4. **改 Web demo**：argparse + `POST /predict` + 输入校验；生产环境用 waitress/gunicorn。

> 如果你愿意，我也可以直接在代码里把第 2/3/4 步落地（改 `main.py` AMP、补 `requirements.txt`、改 `docker/Dockerfile`、重写 `webdemo.py` 的参数与路由）。
