# LatentMind Source Module

## Directory Structure

```
src/
├── model/           # 核心模型
│   ├── __init__.py
│   ├── hrm_backbone.py   # HRM-Text backbone
│   ├── gram_layer.py     # GRAM 多轨迹层
│   └── latent_engine.py  # 分层递归潜空间引擎
├── perception/ # 原生统一感知层
│   ├── __init__.py
│   └── cnn感知层.py
├── decoder/        # 双流解码器
│   ├── __init__.py
│   ├── language_head.py
│   └── semantic_head.py
└── utils/         # 工具
    ├── __init__.py
    ├── magic_norm.py
    └── latent_utils.py
```

## Status
- 2026-06-12: 项目初始化，目录结构创建完成