# CoupletAI
用 PyTorch 实现的自动对对联系统，支持多种模型。一般来讲，给定一句话生成另一句话是序列生成问题，本项目根据上下联字数相等的特点将其转化为序列标注问题，即用下联去标注上联。  
## Dependencies
* Python 3.9+（建议）
* PyTorch 2.x（建议；安装方式见下方）

安装依赖（除 PyTorch 外）：

```bash
pip install -r requirements.txt
```

PyTorch 的安装与 CUDA/CPU 版本强相关，请按官网指引安装： https://pytorch.org/get-started/locally/
## Dataset
数据集包含70多万条对联数据(26MB)，下载请[点击这里](https://github.com/wb14123/couplet-dataset/releases/download/1.0/couplet.tar.gz)，或者[百度云](https://pan.baidu.com/s/1Zqnqq0VqZxv2c4jTNlZJGQ)(提取码: wude)。
## Usage
* 将下载到的数据集解压到当前目录（解压后的文件夹名称为`couplet`）
* 运行 `preprocess.py` 进行数据预处理
* 运行 `main.py [-m model type]` 进行训练
* 运行 `clidemo.py <-p model path>` 可在控制台进行AI对对联
* 运行 `webdemo.py --model <model path> [--cuda] [--max_input_len 64]` 可在Web端进行AI对对联

更完整的部署/运行说明见：`DEPLOYMENT.md`。

推理解码（可选，默认使用带约束的贪心）：

```bash
# 逐位 argmax（原始行为）
python clidemo.py -p output/xxx.bin --decode argmax

# 约束贪心：禁 PAD/UNK（默认），可选 no-copy/标点对齐等约束
python clidemo.py -p output/xxx.bin --decode constrained --topk 20

# 带约束的 beam search（适合加“全局约束”）
python clidemo.py -p output/xxx.bin --decode beam --beam_size 5 --topk 20

# 可选约束开关（默认关闭，显式开启）
python clidemo.py -p output/xxx.bin --decode constrained --no_copy --match_punct
```

多卡训练（可选，推荐 DDP）：

```bash
torchrun --standalone --nproc_per_node=2 main.py --ddp
```

DataParallel（不推荐，兼容保留）：

```bash
python main.py --dp
```

断点续训（恢复训练状态）：

```bash
python main.py --resume output/last_state.bin
```

Early Stopping（可选）：

```bash
python main.py --early_stop_patience 3 --early_stop_min_delta 0.0
```

Web API（推荐）：

* `POST /predict`，JSON：`{"coupletup": "..."}`

命令行参数的详细说明见文件内，你也可以在 `module/model.py` 中定义你自己的模型。
## Using Docker 
基于当前代码构建镜像：

```bash
docker build -f docker/Dockerfile -t coupletai .
```

## Results Show
|   #          | 对联                               |
| ------------ | ---------------------------------- |
| 上联         | 放不开眼底乾坤，何必登斯楼把酒     |
| 下联         | 吞得尽胸中云梦，方许对古人言诗     |
| AI预测的下联 | 抛难在胸中日月，不然看此水凭诗     |
| 上联         | 春暮偶登楼，上下鱼龙，应惜满湖绿水 |
| 下联         | 酒醉休说梦，关山戎马，未如一枕黄梁 |
| AI预测的下联 | 秋寒常入酒，东来风水，更喜一岸红山 |
| 上联         | 一器成名只为茗                     |
| 下联         | 悦来客满是茶香                     |
| AI预测的下联 | 三年有梦不因诗                     |
| 上联         | 春夜灯花，几处笙歌腾朗月           |
| 下联         | 良宵美景，万家箫管乐丰年             |
| AI预测的下联 | 秋天月雨，一时风雨点清风           |
| 上联         | 一曲笙歌春似海                     |
| 下联         | 千门灯火夜如年                     |
| AI预测的下联 | 三年灯色梦如山                     |

## Screenshot
### 命令行运行
![Terminal Demo](docs/terminal_demo.png)
### 网页运行
![Web Demo](docs/web_demo.png)
### Web API
![Web API](docs/webapi_demo.png)
