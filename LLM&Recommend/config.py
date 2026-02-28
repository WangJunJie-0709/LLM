"""
第1-2周实战项目：全局配置与常量。
包含 Criteo 数据路径、模型超参数、特征交叉方式等，供各模块解耦引用。
"""

# Criteo 数据集路径与预处理相关
CRITEO_TRAIN_PATH = None  # 填写训练集路径
CRITEO_TEST_PATH = None   # 填写测试集路径
CRITEO_CACHE_DIR = None   # 预处理缓存目录

# 通用模型 / 训练
EMBEDDING_DIM = None
BATCH_SIZE = None
LEARNING_RATE = None
NUM_EPOCHS = None
DEVICE = None  # 'cuda' / 'cpu'

# 特征交叉方式枚举（与 feature_crossing 子模块对应）
FEATURE_CROSS_TYPES = ("cartesian", "attention", "pnn", "inner_product", "outer_product")

# 日志与输出
LOG_DIR = None
SAVE_MODEL_DIR = None
TSNE_OUTPUT_PATH = None
