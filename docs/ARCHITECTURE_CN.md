# LLM EGT Forecaster - 项目架构说明书

## 项目概述

本项目实现了一个基于进化博弈论(EGT)的多智能体大语言模型框架，用于新闻驱动的时间序列预测。通过竞争驱动的进化过程，智能体能够自适应地开发多样化和有效的策略来过滤新闻并预测未来趋势。

## 核心架构

```
llm_egt_forecaster/
├── configs/                    # 配置模块
│   ├── __init__.py
│   └── base_config.py         # 基础配置，支持GPU/CPU模式
├── data/                      # 数据模块
│   ├── __init__.py
│   ├── virtual_data_generator.py  # 虚拟数据生成器
│   └── data_generator.py          # 真实数据生成器
├── src/                       # 核心源代码
│   ├── __init__.py
│   ├── agent.py              # 智能体定义
│   ├── dataset.py            # 数据集和加载器
│   ├── engine/               # 训练引擎
│   │   ├── __init__.py
│   │   ├── loss.py           # 进化损失函数
│   │   └── trainer.py        # 训练器
│   ├── models/               # 模型定义
│   │   ├── __init__.py
│   │   ├── evolutionary_framework.py  # 主框架
│   │   ├── logic_generator.py        # 逻辑生成器
│   │   └── news_selector.py          # 新闻选择器
│   └── utils/                # 工具函数
│       ├── __init__.py
│       └── helpers.py
├── docs/                     # 文档
│   ├── ARCHITECTURE_CN.md    # 中文架构说明
│   └── ARCHITECTURE_EN.md    # 英文架构说明
├── train.py                  # 训练入口
├── setup.py                  # 包配置
├── requirements.txt          # 依赖列表
└── README.md                 # 项目说明
```

## 模块详细说明

### 1. 配置模块 (configs/)

**功能**: 管理项目配置，支持不同硬件环境

**关键文件**:
- `base_config.py`: 基础配置，包含GPU_LLAMA和CPU_DEBUG两种模式

**主要配置项**:
```python
MODE = 'GPU_LLAMA'  # 或 'CPU_DEBUG'
BASE_LLM_MODEL = "meta-llama/Llama-2-7b-hf"  # 或 "distilgpt2"
USE_QUANTIZATION = True  # 是否使用量化
NUM_AGENTS = 5  # 智能体数量
NUM_EPOCHS = 3  # 训练轮数
```

### 2. 数据模块 (data/)

**功能**: 数据生成和加载

**关键文件**:
- `virtual_data_generator.py`: 生成虚拟测试数据
- `data_generator.py`: 加载和转换真实数据

**数据格式**:
```json
{
  "time_series": [数值列表],
  "candidate_news": [新闻字符串列表],
  "ground_truth": [未来值列表]
}
```

### 3. 智能体模块 (src/agent.py)

**功能**: 定义单个智能体

**关键特性**:
- 继承自 `nn.Module`，支持参数自动收集
- 每个智能体有独立的LoRA权重和gate参数
- 包含适应度(fitness)和逻辑(logic)属性

**核心方法**:
```python
class Agent(nn.Module):
    def __init__(self, agent_id, base_llm_model, lora_config, device)
    def update_fitness(self, reward, beta)  # 更新适应度
    def update_logic(self, new_logic)       # 更新策略逻辑
    def forward(self, *args, **kwargs)      # 前向传播
```

### 4. 进化框架 (src/models/evolutionary_framework.py)

**功能**: 主框架，协调多智能体进化

**关键特性**:
- 支持条件量化加载模型
- 稳健的批处理数据准备
- 智能体预测聚合

**核心方法**:
```python
class EvolutionaryFramework(nn.Module):
    def __init__(self, config)              # 初始化框架
    def _prepare_batch(self, batch)         # 准备批数据
    def forward(self, batch)                # 前向传播
```

### 5. 新闻选择器 (src/models/news_selector.py)

**功能**: 基于语义相似度选择相关新闻

**关键特性**:
- 使用sentence-transformers进行语义匹配
- 支持阈值过滤和top-k选择
- 包含回退机制

### 6. 逻辑生成器 (src/models/logic_generator.py)

**功能**: 生成智能体策略逻辑

**关键特性**:
- 支持策略梯度训练
- 基于同伴信息生成新逻辑
- 返回对数概率用于PG损失

### 7. 损失函数 (src/engine/loss.py)

**功能**: 计算进化损失

**损失组件**:
- 预测损失 (L_predict)
- 策略梯度损失 (L_PG)
- 多样性损失 (L_diversity)
- 剪枝损失 (L_pruning)

### 8. 训练器 (src/engine/trainer.py)

**功能**: 管理训练过程

**关键特性**:
- 简化的优化器初始化
- 策略梯度训练循环
- 智能体适应度更新

## 数据流

```
1. 数据加载
   virtual_data_generator.py → dataset.py → DataLoader

2. 前向传播
   EvolutionaryFramework → NewsSelector → Agent → LogicGenerator

3. 损失计算
   EvolutionaryLoss → 策略梯度 + 多样性 + 剪枝

4. 反向传播
   Trainer → 优化器 → 参数更新

5. 进化更新
   Agent.fitness ← 奖励
   Agent.logic ← 新策略
```

## 扩展指南

### 添加新的智能体策略

1. 修改 `src/agent.py` 中的 `update_logic` 方法
2. 在 `src/models/logic_generator.py` 中添加新的生成逻辑
3. 更新 `src/engine/loss.py` 中的多样性计算

### 添加新的损失函数

1. 在 `src/engine/loss.py` 中添加新的损失计算方法
2. 在 `EvolutionaryLoss.forward` 中集成新损失
3. 更新配置中的权重参数

### 支持新的数据源

1. 创建新的数据加载器类
2. 实现与现有数据格式的转换
3. 更新 `src/dataset.py` 以支持新格式

### 添加新的模型架构

1. 在 `src/models/` 中创建新的模型类
2. 继承 `nn.Module` 并实现必要方法
3. 在 `EvolutionaryFramework` 中集成新模型

## 配置说明

### 环境模式

**GPU_LLAMA模式** (推荐用于生产):
- 使用Llama-2-7B模型
- 4-bit量化以节省显存
- 需要>=16GB VRAM

**CPU_DEBUG模式** (用于调试):
- 使用distilgpt2模型
- CPU推理
- 适合快速验证逻辑

### 关键参数

```python
# 模型配置
BASE_LLM_MODEL = "meta-llama/Llama-2-7b-hf"
USE_QUANTIZATION = True
LORA_RANK = 16
LORA_ALPHA = 32

# 训练配置
NUM_AGENTS = 5
NUM_EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 2e-5

# 损失权重
LAMBDA_STRATEGY = 1.0
LAMBDA_DIVERSITY = 0.5
LAMBDA_PRUNING = 0.1
```

## 贡献指南

### 代码规范

1. **导入规范**: 使用 `llm_egt_forecaster.*` 前缀
2. **类型提示**: 为函数参数和返回值添加类型提示
3. **文档字符串**: 为所有公共方法添加详细的文档字符串
4. **错误处理**: 添加适当的异常处理

### 测试要求

1. 为每个新功能添加单元测试
2. 确保在两种模式下都能正常运行
3. 验证数据格式兼容性

### 提交规范

1. 使用清晰的提交信息
2. 一个提交只包含一个功能或修复
3. 更新相关文档

## 故障排除

### 常见问题

1. **CUDA内存不足**: 切换到CPU_DEBUG模式或减少BATCH_SIZE
2. **模型加载失败**: 检查Hugging Face认证和网络连接
3. **数据格式错误**: 验证JSON文件格式和字段完整性

### 调试技巧

1. 使用 `python -m llm_egt_forecaster.data.virtual_data_generator` 测试数据生成
2. 设置 `MODE = 'CPU_DEBUG'` 进行快速调试
3. 检查 `src/dataset.py` 的 `__main__` 块进行数据加载测试

## 运行示例

### 基本训练

```bash
# 使用虚拟数据训练
python train.py

# 使用真实数据训练
python train.py --data_path data/real_dataset.json
```

### 数据生成

```bash
# 生成虚拟数据
python -m llm_egt_forecaster.data.virtual_data_generator

# 测试数据加载
python -m llm_egt_forecaster.src.dataset
```

### 模式切换

在 `configs/base_config.py` 中修改 `MODE` 变量：
- `'GPU_LLAMA'`: 使用Llama-2-7B + 4bit量化
- `'CPU_DEBUG'`: 使用distilgpt2 + CPU推理

---

*本文档与代码库同步维护。请在项目结构发生重大变化时更新此文档。*
