# 城市低空起降点三维选址优化系统

[![Python 3.12](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 项目简介

本项目实现了考虑覆盖与连通的城市低空起降点三维选址优化算法，基于混合整数规划模型，结合GIS技术和多阶段启发式算法，为城市低空物流网络规划提供科学决策支持。

### 核心特性

- 🏙️ **三维空间建模**：考虑建筑高度、地形高程与固定巡航高度的动态关系
- 📊 **多源数据融合**：集成建筑轮廓、高度、高程、用地性质等多维地理数据
- ⚡ **高效算法**：多阶段启发式算法（贪婪选点 + MST连通修复 + 冗余剪枝）
- 🎯 **双目标优化**：同时优化服务覆盖率与网络连通性
- 🔧 **工程优化**：空间索引、缓存机制、多进程并行计算
- 📈 **可视化分析**：2D/3D可视化、实时选点过程、收敛性分析
- 🎪 **实时可视化**：动态展示选点过程，支持MP4视频导出

## 安装指南

### 环境要求

- **Python**: 3.12 或更高版本
- **内存**: ≥16GB（处理大规模城市数据）
- **推荐配置**: 多核CPU、独立显卡（3D可视化）

### 安装依赖

```bash
# 使用pip安装
pip install -r requirements.txt

# 或者使用conda创建环境
conda create -n vertiport python=3.8
conda activate vertiport
pip install -r requirements.txt
