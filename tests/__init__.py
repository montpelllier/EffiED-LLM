"""
测试模块初始化文件
"""
import sys
from pathlib import Path

# 添加src目录到Python路径
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# 添加项目根目录到Python路径
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))
