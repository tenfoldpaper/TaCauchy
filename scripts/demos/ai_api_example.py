"""
AI API 使用示例
演示如何在项目中使用 AI API
"""

import os
from pathlib import Path

# 加载环境变量
def load_api_key():
    """从 .env 文件加载 API 密钥"""
    env_file = Path(__file__).parent.parent.parent / ".env"
    
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    
    return os.getenv('AI_API_KEY')

# 使用示例
if __name__ == "__main__":
    api_key = load_api_key()
    
    if api_key:
        print(f"✓ API Key 已加载: {api_key[:10]}...")
        # 在这里添加你的 AI API 调用代码
    else:
        print("✗ 未找到 API Key，请检查 .env 文件")
