# SamRefer 项目接手说明

## 1. 项目目标与当前仓库状态

这个仓库服务于指代表达式语义分割任务。输入是一张图片和一句文本描述，输出是文本所指目标的分割掩码。当前仓库的实现核心是：

- 视觉侧基于 Segment Anything
- 文本侧使用 BERT 或可选的 CLIP 文本编码器（CLIP本次未使用）
- 在当前主线实现中加入了增强损失、多数据集训练和更复杂的验证逻辑

从仓库现状看，这不是一个已经完全产品化的项目，而是一个研究型实验仓库，特点包括：

- 入口脚本明确，但配置依赖较强
- `model/` 和 `model_origin/` 两套实现并存
- 部分 shell 脚本依赖作者自己的环境
- 配置文件中存在绝对路径，需要新接手者自行改写

所以，新接手时最重要的不是直接运行训练，而是先弄清楚“代码入口在哪里、依赖哪些模型和数据、哪些配置必须先改”。

## 2. 代码结构总览

### 2.1 根目录核心入口

- `train_enhanced_multi_dataset.py`
  当前主线训练入口，负责初始化 DeepSpeed、构建模型、创建训练数据集和执行多数据集训练。

- `validate.py`
  当前主线验证入口，负责创建验证数据集、加载当前主模型并调用 `validation/evaluation.py` 中的验证逻辑。

- `validate_origin.py`
  原始模型验证入口，主要用于对照旧版实现或验证 `model_origin/` 下的模型流程。

- `get_args.py`
  全局统一参数解析入口。它会先读取 `--config` 指向的 YAML 文件，再把 YAML 中的字段自动注册成命令行参数，所以它是理解“配置如何进入程序”的第一站。


### 2.2 配置目录 `configs/`

这个目录决定了项目如何加载路径、模型、精度和训练参数。

- `main_refersam_bert.yaml`
  当前主线流程最重要的配置文件之一，定义了 batch size、epoch、数据路径、SAM 权重路径、BERT 路径、输出目录等。

- `main_origin.yaml`
  原始模型流程配置，对应 `validate_origin.py` 等旧路径。

- `enhanced_loss_config.yaml`
  增强损失配置，控制是否启用 focal、IoU、boundary、自适应权重和 curriculum learning。

- `ds_config.json`
  DeepSpeed 配置文件。

- `student.yaml`
  在学校高性能计算服务平台使用的配置文件。

### 2.3 数据目录 `dataset/`

`dataset/` 负责不同数据集的封装和样本组织。当前代码里能看到的主要数据集封装包括：

- `ReferDataset.py`
- `GRefDataset.py`
- `Dataset_referit.py`
- `RefzomDataset.py`

底层数据组织相关代码还包括：

- `refer.py`
- `gref.py`
- `refer_refzom.py`

这些文件共同负责把不同来源的数据集转成训练和验证代码可消费的统一样本格式。

### 2.4 模型目录 `model/` 与 `model_origin/`

这是新接手者最需要区分的一组目录：

- `model/`
  当前主线模型实现，建议优先阅读。这里包含：
  - `enhanced_builder.py`：主模型构建入口
  - `builder.py`：较早的 builder 实现
  - `criterion.py`、`enhanced_criterion.py`：损失函数
  - `segment_anything/`：SAM 相关实现
  - `vit_adapter/`、`models/`、`tranformer_decoder.py`：模型结构组件

- `model_origin/`
  原始模型实现，适合用来对照旧结构或排查当前版本相对原版的变化。

如果你要先理解“当前仓库主要跑哪套模型”，优先看 `model/enhanced_builder.py`。

### 2.5 验证与可视化目录

- `validation/`
  负责指标计算、验证逻辑和 Ground Truth 相关辅助逻辑。主验证入口最终会调用这里的函数。

- `visual/`
  负责推理结果可视化。`visual/visualize_inference.py` 支持从数据集抽样可视化，也支持单张图片可视化。

### 2.6 脚本与其他目录

- `scripts/`
  主要是作者个人环境下的快捷脚本样例。里面的一些脚本会直接写死 conda 环境、路径和日志文件名，所以应当把它们理解为参考，而不是标准运行接口。

- `sam3_weight/`
  为了和SAM3模型进行对比放的SAM3模型的权重


如果你要从“跑起来”开始看，优先看 `get_args.py`、`configs/main_refersam_bert.yaml`、`validate.py`。如果你要从“模型如何搭建”开始看，优先看 `model/enhanced_builder.py`。

## 3. 关键执行链路

### 3.1 训练链路

当前主线训练流程可概括为：

`train_enhanced_multi_dataset.py`
-> `get_args.py`
-> `configs/main_refersam_bert.yaml`
-> `dataset/*.py`
-> `model/enhanced_builder.py`
-> `validation/evaluation.py`

更具体地说：

1. `train_enhanced_multi_dataset.py` 调用 `get_args()` 读取配置。
2. 程序根据 `deepspeed_config` 初始化 DeepSpeed。
3. 根据参数选择构建增强损失版本或原始损失版本模型。
4. 创建一个或多个数据集对象，并通过 `ConcatDataset` 组合训练集。
5. 在训练过程中，调用验证逻辑进行评估和保存模型。

### 3.2 验证链路

当前主线验证流程可概括为：

`validate.py`
-> `get_args.py`
-> `create_datasets()`
-> `model/enhanced_builder.py`
-> `validation/evaluation.py`

主要过程是：

1. 读取 YAML 配置和命令行参数。
2. 创建多个验证数据集配置。
3. 根据 `pre_train_path` 等参数决定是否加载已有 checkpoint。
4. 调用验证函数计算指标并输出结果。

### 3.3 原始模型验证链路

原始模型验证流程与当前主线类似，但走的是另一套 builder：

`validate_origin.py`
-> `get_args.py`
-> `create_datasets()`
-> `model_origin/builder.py`
-> `validation/evaluation.py`

这条链路适合在两套实现之间做对照。

### 3.4 可视化链路

可视化流程可概括为：

`visual/visualize_inference.py`
-> 读取配置与数据
-> 加载模型
-> 推理
-> 输出可视化图片

它支持两种主要模式：

- `dataset`：从数据集中抽样做可视化
- `single`：对单张图片和一句文本做可视化

## 4. 模型与权重依赖

这一节是新接手时最需要先看清楚的部分。当前仓库依赖的模型与权重可以整理如下：

| 依赖项 | 是否必需 | 作用 | 代码或配置位置 | 说明 |
| --- | --- | --- | --- | --- |
| SAM checkpoint | 必需 | 提供视觉主干权重 | `configs/main_refersam_bert.yaml`、`configs/main_origin.yaml` 的 `checkpoint`；`model/enhanced_builder.py` 中的 `sam_model_registry[args.sam_type](checkpoint=args.checkpoint)` | 需要用户本地准备 |
| BERT 权重 | 常规必需 | 文本编码器 | `ck_bert`；`model/enhanced_builder.py` 中的 `BertModel.from_pretrained(args.ck_bert)` | 当 `clip_path` 为空时使用 |
| CLIP 权重 | 可选 | 替代文本编码器 | `clip_path`；`model/enhanced_builder.py` 中的 `CLIPTextModel.from_pretrained(args.clip_path)` | 如果不为空，则会改走 CLIP 文本编码器 |
| 训练 checkpoint | 可选 | 用于继续训练或直接验证已有模型 | `pre_train_path`；`model/enhanced_builder.py` 中会根据该字段加载 checkpoint | 支持 PyTorch checkpoint，也兼容部分 DeepSpeed checkpoint 目录格式 |
| DeepSpeed 配置 | 训练和验证常用 | 控制分布式、精度和优化器相关行为 | `get_args.py` 中的 `deepspeed_config` 参数；`configs/ds_config.json` | 需要和本地环境、GPU、精度策略匹配 |
| `sam3_weight/` 目录 | 视具体流程而定 | 仓库内已有本地资源 | 仓库目录 | 目前可以看到 `sam3.pt` 和 `bpe_simple_vocab_16e6.txt`，但它不是当前主配置文件中默认引用的唯一路径 |

另外，有两个实现细节值得单独注意：

1. `model/enhanced_builder.py` 和 `model/builder.py` 都保留了 BERT 与 CLIP 两条文本编码器加载路径。
2. `model_origin/builder.py` 和 `model_origin/builder_origin.py` 也有类似加载逻辑，因此两套实现都依赖文本编码器权重，只是入口不同。

## 5. 数据集依赖

从训练与验证代码里可以看出，仓库当前至少涉及以下数据集封装：

- RefCOCO
- RefCOCO+
- RefCOCOg
- GRefCOCO
- ReferIt（本文未使用）
- Ref-ZOM

相关代码主要出现在：

- `train_enhanced_multi_dataset.py`
- `validate.py`
- `validate_origin.py`
- `dataset/` 下各数据集类

需要重点关注的路径字段包括：

- `data_root`
- `data_referit_root`

例如当前配置里已经写入了类似 `/root/autodl-tmp/...` 这样的绝对路径。新接手者必须先把它们改成本机有效路径，否则即使代码本身没有问题，也会在数据加载阶段直接报错。

如果你只是想先验证环境，而不是马上完整复现实验，建议优先选择以下方式：

1. 先准备一个最容易访问的数据集子集。
2. 先运行验证或可视化，而不是直接跑多卡训练。
3. 确认图片路径、标注路径、文本描述读取都正常之后，再进入训练。

## 6. 环境安装

推荐按下面顺序准备环境。

### 6.1 创建环境

你可以使用 conda，也可以使用其他虚拟环境工具。项目本身没有把环境管理标准化到 `pyproject.toml` 或 `environment.yml`，所以通常需要手动创建环境。

### 6.2 安装 Python 依赖

使用 `requirements.txt` 安装依赖：

```bash
pip install -r requirements.txt
```

从文件内容看，当前环境快照里包含以下关键依赖：

- `torch`
- `torchvision`
- `deepspeed`
- `mmcv`
- `mmdet`
- `transformers`
- `clip`（通过 GitHub 源安装）
- `opencv-python`
- `tensorboard`

### 6.3 单独确认大依赖之间的兼容性

这里是最容易踩坑的地方。尤其需要注意：

- PyTorch 版本是否和 CUDA 匹配
- DeepSpeed 是否和当前 PyTorch 版本兼容
- MMCV、MMDetection 是否和 PyTorch 版本匹配
- `transformers` 与本地权重格式是否兼容

要特别注意一点：`requirements.txt` 反映的是仓库曾经使用过的环境快照，不代表任意机器都能一次性无冲突安装成功。

### 6.4 准备模型权重

至少需要准备：

1. SAM 权重文件
2. BERT 权重目录，或者 CLIP 权重目录
3. 如果要继续训练或直接复现实验结果，还需要准备已有训练 checkpoint

### 6.5 准备数据集

把数据集整理到你的本地路径后，确保配置中的以下字段可以正确指向它们：

- `data_root`
- `data_referit_root`

### 6.6 修改配置文件

首次运行前，至少检查以下配置项：

- `data_root`
- `data_referit_root`
- `checkpoint`
- `ck_bert`
- `clip_path`
- `output_dir`
- `pre_train_path`
- `deepspeed_config`

## 7. 首次运行前必须修改的配置

这里给出一个非常实际的检查清单。你第一次接手时，建议先打开 `configs/main_refersam_bert.yaml` 和 `configs/main_origin.yaml`，按顺序核对：

1. `data_root` 是否指向你的 COCO/RefCOCO 相关数据目录
2. `data_referit_root` 是否指向你的 ReferIt 数据目录（该数据集可以不用）
3. `checkpoint` 是否指向你的 SAM 权重文件
4. `ck_bert` 是否指向你的本地 BERT 权重目录
5. `clip_path` 是否为空，或者是否正确指向 CLIP 文本权重
6. `output_dir` 是否是你希望保存结果的目录
7. `pre_train_path` 是否为空，或者是否指向有效 checkpoint

如果你要使用增强损失，还需要检查：

- `configs/enhanced_loss_config.yaml` 中的开关是否符合当前实验目的
- 如 `use_focal`、`use_iou`、`use_boundary`、`use_adaptive_weighting` 等是否按预期打开

## 8. 常用运行方式

建议从轻量流程开始，逐步确认环境。

### 8.1 先看参数帮助

这是最快确认入口文件存在且参数能解析的方法：

```bash
python validate.py --help
python validate_origin.py --help
python visual/visualize_inference.py --help
python train_enhanced_multi_dataset.py --help
```

### 8.2 先跑验证

```bash
python validate.py --config configs/main_refersam_bert.yaml
```

如果要验证原始模型：

```bash
python validate_origin.py --config configs/main_origin.yaml
```

### 8.3 再跑可视化

```bash
python visual/visualize_inference.py --mode dataset --dataset_name refcoco --split val
```

如果你已经准备好了单张图片和文本，也可以使用 `single` 模式。

### 8.4 最后尝试训练

```bash
python train_enhanced_multi_dataset.py --config configs/main_refersam_bert.yaml
```

如果你的环境已经正确安装 DeepSpeed，并且你确实需要分布式训练，再考虑追加 `deepspeed` 启动方式和 `--deepspeed_config`。

## 9. 新接手用户建议阅读顺序

如果你想在最短时间内建立对项目的整体理解，建议按下面顺序阅读：

1. `README.md`
2. `get_args.py`
3. `configs/main_refersam_bert.yaml`
4. `validate.py`
5. `model/enhanced_builder.py`
6. `dataset/` 下与你当前要用的数据集对应的封装文件
7. `validation/evaluation.py`
8. `visual/visualize_inference.py`

这个顺序的目的，是让你先理解“项目怎么启动”，再理解“模型怎么搭建”，最后再进入“数据如何组织”和“结果如何可视化”。

## 10. 常见问题与风险提示

### 10.1 配置文件中的绝对路径问题

这是最常见的问题来源。仓库中的配置文件已经写死了一些作者环境路径，如果不修改，程序会在模型加载或数据加载阶段直接失败。

### 10.2 `scripts/` 目录不是标准接口

`scripts/val.sh`、`scripts/visual.sh` 等脚本包含作者自己的 conda 激活方式、固定 GPU 编号和日志命名，不能直接当成通用运行标准。尤其是 `scripts/train.sh` 当前只有 `pkill` 操作，并不是训练入口。

### 10.3 DeepSpeed 与 checkpoint 格式问题

当前主线 builder 已经对部分 DeepSpeed checkpoint 做了兼容判断，但是否能直接加载，仍然依赖 checkpoint 目录结构是否符合预期。如果你遇到加载问题，需要首先确认：

- `pre_train_path` 指向的是文件还是目录
- 是否包含 `global_step*` 之类的子目录
- `ds_config.json` 是否和训练时配置一致

### 10.4 先验证单流程，再跑多卡训练

不要把“第一次接手环境”就直接等同于“马上多卡训练”。更稳妥的顺序是：

1. 确认配置文件路径无误
2. 确认模型权重能加载
3. 确认单个验证或可视化流程可运行
4. 再进入 DeepSpeed 或多卡训练
