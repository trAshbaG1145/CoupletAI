# CoupletAI：Legacy 清单与现代化改造状态（合并版）

> 本文将 `LEGACY_NOTES.md`（过时点清单）与 `LEGACY_UPDATES.md`（对照核查与落地状态）合并为一份文档，便于维护与引用。
> 
> 更新时间：2025-12-24

---

## 总览（按条目）

- 1) AMP（Apex → PyTorch AMP）：已落地
- 2) 多卡（DataParallel → DDP）：已落地（默认不再自动启用 DataParallel）
- 3) README 版本声明：已落地
- 4) 依赖清单（requirements）：已落地
- 5) Dockerfile（latest/clone/data download）：已落地（可复现构建）
- 6) Web 服务（sys.argv/path 参数/生产部署）：部分落地（已提供 `POST /predict`，仍为 demo server）
- 7) 前端依赖（Bootstrap 3）：已落地（升级到 Bootstrap 5 CDN）
- 8) 训练工程（seed/checkpoint/resume）：已落地（可复现 + 可恢复）
- 9) 指标（NLTK BLEU）：保留（baseline），可继续优化
- 10) 其他风格项：保留

---

## 1) 混合精度训练：Apex AMP（main.py）

- 现状（原）：`--fp16` 依赖 NVIDIA Apex AMP。
- 为什么过时：PyTorch 已长期提供官方 AMP（`autocast` / `GradScaler`），更易安装、维护更稳定。
- 推荐替代：使用 `torch.cuda.amp.autocast` + `torch.cuda.amp.GradScaler`。
- 当前状态（已落地）：
  - `main.py` 已切换到 PyTorch AMP，不再依赖 Apex。
- 仍可继续：
  - 清理历史遗留但无实际作用的参数（例如 `--fp16_opt_level` 如仍存在）。

---

## 2) 多卡训练：DataParallel（main.py）

- 现状（原）：检测到多卡就自动使用 `torch.nn.DataParallel(model)`。
- 为什么过时：`DataParallel` 性能/可控性/容错性均不如 `DistributedDataParallel (DDP)`，也不再推荐作为主方案。
- 推荐替代：用 `torchrun` + `DDP`；单卡为默认。
- 当前状态（已落地）：
  - `main.py` 支持 `--ddp` 的 DDP 训练。
  - `DataParallel` 改为显式 `--dp` 才启用（默认不再自动启用）。

---

## 3) README 依赖版本声明过老（README.md）

- 现状（原）：README 写 `python 3.6+`、`pytorch 1.2+`。
- 为什么过时：Python 3.6 与 PyTorch 1.2 均已非常老，生态/安装/AMP/CUDA 等已变化。
- 推荐替代：更新为更现实的版本范围，并提供 PyTorch 官网安装指引。
- 当前状态（已落地）：
  - `README.md` 已更新为建议 `Python 3.9+`、`PyTorch 2.x`。

---

## 4) 缺少依赖清单（requirements.txt/pyproject.toml）

- 现状（原）：仓库没有 `requirements.txt` / `pyproject.toml`。
- 为什么算“过时实践”：缺少可复现依赖描述，安装全靠猜。
- 推荐替代：
  - 最低限度：提供 `requirements.txt`。
  - 更规范：`pyproject.toml` + 锁定工具（pip-tools/poetry/uv）。
- 当前状态（已落地）：
  - 已新增 `requirements.txt`（PyTorch 仍建议按平台从官网安装）。

---

## 5) Dockerfile：latest + git clone 外部仓库 + 构建/运行期下载数据（docker/Dockerfile）

- 现状（原）：
  - `FROM pytorch/pytorch:latest`
  - 构建时 `git clone` 外部仓库（与本地修改无关）
  - 在镜像构建/运行期下载数据集
  - 依赖安装不完整
- 为什么过时/不推荐：不可复现、构建慢、易失败、与当前工作区代码脱节。
- 推荐替代：锁定基础镜像版本；`COPY .` 构建当前代码；用 `requirements.txt` 安装依赖；数据用文档/volume 管理。
- 当前状态（已落地）：
  - `docker/Dockerfile` 已锁定基础镜像版本，并改为 `COPY` 当前工作区代码 + `pip install -r requirements.txt`。

---

## 6) Web 服务与参数解析（webdemo.py / templates）

- 现状（原）：
  - 用 `sys.argv[1]` 取模型路径
  - 使用 Flask 开发服务器 `app.run(...)`
  - 把用户输入放在 URL path（例如 `/<coupletup>`）
- 为什么过时/不推荐：参数不健壮；开发 server 不适合生产；path 方案不利于编码/长度控制/扩展。
- 推荐替代：
  - 用 `argparse`（或环境变量）读取参数
  - 提供 `POST /predict` + JSON body，并做输入校验
  - 生产部署：Windows 可用 waitress；Linux/容器可用 gunicorn
- 当前状态（部分落地）：
  - `webdemo.py` 已改为 `argparse`（`--model`）。
  - 已新增 `POST /predict` JSON API，并做基本输入校验与错误返回。
- 仍可继续：
  - 若要生产化部署，将 `app.run(...)` 替换为 waitress/gunicorn 的启动方式与文档示例。

---

## 7) 前端依赖：Bootstrap 3 CDN（templates/index.html）

- 现状（原）：Bootstrap 3 CDN（非常老）。
- 推荐替代：升级到 Bootstrap 5，或本地 vendoring 静态资源。
- 当前状态（已落地）：
  - `templates/index.html` 已升级到 Bootstrap 5 CDN。
- 仍可继续：
  - 如需离线/内网稳定性，将 CSS/JS vendoring 到 `static/`。

---

## 8) 训练工程：seed / checkpoint / resume（main.py）

- 现状（原）：
  - 未提供 `--seed` 与完整随机种子设置
  - checkpoint 仅保存模型权重，缺少 optimizer/scheduler 等状态，难以恢复训练
- 推荐替代：
  - 提供 `--seed` 并设置 `random/numpy/torch`（按需求选择 deterministic/benchmark）
  - checkpoint 保存 optimizer/scheduler/scaler/epoch/global_step，并提供 `--resume`
- 当前状态（已落地）：
  - `main.py` 已提供 `--seed`。
  - checkpoint 已包含 optimizer/scheduler/scaler/epoch/global_step 等训练状态。
  - `main.py` 已提供 `--resume` 并支持从 checkpoint 恢复训练。

---

## 9) 评估指标：NLTK BLEU（module/metric.py）

- 现状（原）：使用 NLTK `sentence_bleu`。
- 为什么可能“不够合适”：对字符级对联任务，BLEU 对语义/对仗未必敏感；NLTK 依赖也相对偏重。
- 推荐替代（可选）：保留 BLEU/Rouge-L baseline，同时补充 chrF/BERTScore/或任务自定义规则指标。
- 当前状态（保留）：
  - 仍使用 NLTK BLEU 作为 baseline；依赖已在 `requirements.txt` 明确。

---

## 10) 其他“老味道”但不一定要改

- `TensorDataset` / `torch.save(train.pkl)`：脚本化但可用；若长期维护可再升级到更规范的数据流水线。
- `ReduceLROnPlateau`：仍可用；如需更现代的策略可考虑 warmup + cosine/linear。
