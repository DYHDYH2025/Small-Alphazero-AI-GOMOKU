# 五子棋AI项目集合

本项目包含多个独立的五子棋AI实现，展示了不同的算法和训练方法。所有项目均针对**9×9棋盘**进行优化。

## 项目结构

```
Gomoku/
├── alphazero-gomoku/              # AlphaZero基础版
│   ├── train.py                   # 训练脚本
│   ├── human_play.py              # 人机对弈
│   ├── policy_value_net_pytorch.py
│   └── mcts_alphaZero.py
│
├── enhanced-gomoku-ai/            # Enhanced AlphaZero（ResNet+BN）
│   ├── train.py                   # 训练脚本（默认GPU）
│   ├── human_play.py
│   ├── policy_value_net_enhanced.py
│   └── mcts_enhanced.py
│
├── minimax-gomoku/                # Minimax算法（α-β剪枝）
│   ├── human_play.py
│   └── minimax_player.py
│
├── pure-mcts-gomoku/              # 纯MCTS（无神经网络）
│   ├── human_play.py
│   └── mcts_pure.py
│
├── weight-based-gomoku/           # 基于评分表的启发式方法
│   ├── human_play.py
│   ├── batch_play.py
│   └── weight_player.py
│
├── weight-enhanced-alphazero/     # 融合评分表特征的AlphaZero
│   ├── train.py
│   ├── human_play.py
│   ├── policy_value_net_with_patterns.py
│   └── pattern_extractor.py
│
├── knowledge-distillation-gomoku/ # 知识蒸馏（教师-学生网络）
│   ├── distillation_train.py
│   ├── human_play.py
│   ├── batch_play.py
│   ├── teacher_net.py
│   └── pattern_data_generator.py
│
├── method-comparison/             # 批量对战和结果分析
│   ├── batch_play.py              # 批量对战脚本
│   ├── players.py                  # 玩家工厂
│   ├── render_gomoku.py           # 棋盘可视化
│   ├── 结果分析.ipynb             # 结果分析
│   └── batch_results_*.csv        # 对战结果数据
│
├── html/                          # HTML项目参考（评分表算法来源）
│   └── 五子棋游戏在线玩_files/
│       └── fiveChess.min.js       # JavaScript评分表算法
│
├── AlphaZero_Gomoku-master_from_github_opensource/  # 开源参考实现
│
├── README.md                      # 本文件
├── report.md                      # 工作思路
└── 完整报告.md                    # 详细学术报告
```

## 快速开始

### 基础项目（无需训练）

```bash
# Minimax算法
cd minimax-gomoku
python human_play.py

# Pure MCTS
cd pure-mcts-gomoku
python human_play.py

# Weight-Based（启发式）
cd weight-based-gomoku
python human_play.py
```

### 深度学习项目（需要训练）

```bash
# AlphaZero基础版
cd alphazero-gomoku
python train.py              # 训练
python human_play.py         # 对弈

# Enhanced AlphaZero（推荐）
cd enhanced-gomoku-ai
python train.py              # 默认使用GPU
python human_play.py

# Weight-Enhanced AlphaZero
cd weight-enhanced-alphazero
python train.py

# Knowledge Distillation
cd knowledge-distillation-gomoku
python pattern_data_generator.py  # 生成数据
python distillation_train.py      # 训练
```

### 批量对战

```bash
cd method-comparison
python batch_play.py         # 运行批量对战
jupyter notebook 结果分析.ipynb  # 分析结果
```

## 环境要求

- Python 3.6+
- PyTorch >= 1.0.0（深度学习项目）
- NumPy
- CUDA（GPU训练，推荐）

安装依赖：
```bash
cd <project_dir>
pip install -r requirements.txt
```

## 项目说明

### 核心项目

- **alphazero-gomoku**：标准AlphaZero实现
- **enhanced-gomoku-ai**：增强版（ResNet + BatchNorm + 梯度裁剪），**最强棋力**
- **minimax-gomoku**：Minimax算法，无需训练，在9×9棋盘表现强
- **pure-mcts-gomoku**：纯MCTS基准

### 特征融合项目

- **weight-based-gomoku**：基于评分表的启发式方法
- **weight-enhanced-alphazero**：融合评分表特征的AlphaZero（辅助任务）
- **knowledge-distillation-gomoku**：知识蒸馏（教师网络 + 学生网络）

### 工具项目

- **method-comparison**：批量对战、结果分析和可视化
- **html**：评分表算法参考（JavaScript实现）

## 详细报告

完整的技术报告、实验分析和对比研究请参考：**[完整报告.md](完整报告.md)**

报告包含：
- 所有方法的详细技术原理
- Enhanced AlphaZero的核心改进（ResNet、BN、梯度裁剪等）
- 两次评估的完整实验数据和分析
- 特征融合方法的技术细节
- 综合对比和能力排序

## 技术栈

- **深度学习框架**：PyTorch
- **主要算法**：AlphaZero, MCTS, Minimax
- **网络架构**：CNN, ResNet（增强版）
- **棋盘配置**：9×9，连5子获胜

## 使用建议

1. **快速体验**：从 `minimax-gomoku` 或 `pure-mcts-gomoku` 开始
2. **标准训练**：使用 `alphazero-gomoku`
3. **最强棋力**：使用 `enhanced-gomoku-ai`（需要GPU）
4. **方法对比**：使用 `method-comparison` 进行批量评估
5. **研究学习**：查看 `完整报告.md` 了解技术细节

## 许可证

本项目基于开源代码修改和增强，各子项目保持原有许可证。
