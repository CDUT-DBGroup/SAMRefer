# SamRefer 项目说明

## 项目简介

本项目是一个围绕指代表达式语义分割任务构建的实验代码仓库，核心思路是将 Segment Anything 的视觉能力与文本编码器结合，用于根据自然语言描述生成目标区域的分割结果。

仓库中目前同时保留了两套模型实现：

- `model/`：当前主线实现，包含增强版 builder、增强损失和当前验证流程。
- `model_origin/`：原始实现，用于对照、复现或兼容旧流程。

当前仓库已经具备训练、验证、原始模型验证和推理可视化入口，但配置、权重和数据路径仍然比较依赖本地环境，因此新接手的人需要先理解目录结构和依赖关系，再修改配置后运行。

## 适合谁阅读

- 新接手本项目，想快速知道代码从哪里进入的人
- 需要梳理模型权重、数据集依赖和配置关系的人
- 想先把环境跑通，再深入看模型实现细节的人

## 项目整体结构

- `train_enhanced_multi_dataset.py`：增强版多数据集训练入口
- `validate.py`：当前主线验证入口
- `validate_origin.py`：原始模型验证入口
- `get_args.py`：统一参数解析入口，先读取 YAML 配置，再叠加命令行参数
- `configs/`：训练、验证、DeepSpeed、损失相关配置
- `dataset/`：RefCOCO、RefCOCO+、RefCOCOg、GRefCOCO、ReferIt（该数据集实际未使用）、Ref-ZOM 等数据集封装
- `model/`：当前主模型实现，包含 builder、损失函数、SAM 适配与文本融合逻辑
- `model_origin/`：原始模型实现备份
- `validation/`：验证与指标计算逻辑
- `visual/`：推理可视化工具
- `scripts/`：作者环境下的脚本样例，不建议直接视为通用启动脚本
- `sam3_weight/`：仓库内已有的本地权重/词表资源
- `论文latex版本/`：论文相关 LaTeX 资料，不属于训练主流程

## 核心依赖一览

运行本项目之前，至少需要准备以下资源：

- Python 环境和 `requirements.txt` 中的依赖
- PyTorch 与 CUDA 环境
- DeepSpeed
- SAM 权重文件，由配置项 `checkpoint` 指定
- 文本编码器权重，由 `ck_bert` 或 `clip_path` 指定
- 可选训练 checkpoint，由 `pre_train_path` 指定
- 数据集目录，由 `data_root`、`data_referit_root` 等字段指定

从代码上看，模型依赖主要通过以下路径加载：

- `model/enhanced_builder.py` 中通过 `sam_model_registry[args.sam_type](checkpoint=args.checkpoint)` 加载 SAM 权重
- `model/enhanced_builder.py` 中通过 `BertModel.from_pretrained(args.ck_bert)` 或 `CLIPTextModel.from_pretrained(args.clip_path)` 加载文本编码器
- `model/enhanced_builder.py` 中根据 `pre_train_path` 判断是否加载已有训练 checkpoint

## 安装准备

推荐按下面顺序准备环境：

1. 创建独立的 Python 或 conda 环境。
2. 安装 `requirements.txt` 中的依赖。
3. 单独确认 PyTorch、CUDA、DeepSpeed、MMCV、MMDetection 之间的版本兼容关系。
4. 准备本地 SAM 权重、BERT 或 CLIP 权重、可选训练 checkpoint。
5. 准备数据集目录。
6. 修改 `configs/` 下 YAML 文件中的绝对路径。

说明：`requirements.txt` 更像是当前仓库曾使用过的环境快照，不代表在任意机器上都能一次性无冲突安装成功，尤其需要关注 CUDA、PyTorch、DeepSpeed、MMCV 的版本匹配。

## 运行入口

建议先从验证或可视化开始检查环境，再尝试训练。

- 训练：`python train_enhanced_multi_dataset.py --config configs/main_refersam_bert.yaml`
- 验证：`python validate.py --config configs/main_refersam_bert.yaml`
- 原始模型验证：`python validate_origin.py --config configs/main_origin.yaml`
- 可视化：`python visual/visualize_inference.py --mode dataset --dataset_name refcoco --split val`

如果你使用 DeepSpeed，也可以在命令行额外传入 `--deepspeed_config`，但前提是对应配置和环境已经准备好。

## 文档导航

更完整的项目接手说明见：`docs/project-onboarding.md`

该文档会进一步解释：

- 各目录和核心文件的职责
- 训练、验证、可视化的执行链路
- 依赖哪些模型、权重和数据集
- 首次运行前必须修改哪些配置
- 常见报错和风险点是什么

## 当前仓库注意事项

- `configs/` 中包含绝对路径，首次运行前必须改成你的本地路径。
- `scripts/` 下部分脚本依赖作者个人环境，只能作为参考，不能直接视为通用启动方式。
- `scripts/train.sh` 当前不是训练脚本，只是一个清理命令。
- 建议先验证单个流程可跑通，再尝试多卡训练或 DeepSpeed 训练。

