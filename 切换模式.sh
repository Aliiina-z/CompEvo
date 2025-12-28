#!/bin/bash
# 模式切换脚本

CONFIG_FILE="llm_egt_forecaster/configs/base_config.py"

echo "=== 模式切换工具 ==="
echo ""
echo "当前配置:"
grep "^MODE = " $CONFIG_FILE | head -1
echo ""

echo "请选择模式:"
echo "1) CPU_DEBUG 模式（无需认证，适合调试）"
echo "2) MPS_LLAMA 模式（需要 Hugging Face 认证，性能更好）"
echo ""
read -p "请输入选项 (1 或 2): " choice

case $choice in
    1)
        echo "切换到 CPU_DEBUG 模式..."
        # 备份原配置
        cp $CONFIG_FILE ${CONFIG_FILE}.backup
        
        # 修改为 CPU_DEBUG
        sed -i '' 's/^MODE = .*/MODE = '\''CPU_DEBUG'\''  # <-- CPU 调试模式/' $CONFIG_FILE
        
        echo "✅ 已切换到 CPU_DEBUG 模式"
        echo "📝 原配置已备份到: ${CONFIG_FILE}.backup"
        ;;
    2)
        echo "切换到 MPS_LLAMA 模式..."
        # 备份原配置
        cp $CONFIG_FILE ${CONFIG_FILE}.backup
        
        # 修改为 MPS_LLAMA（自动检测）
        sed -i '' "s/^MODE = .*/MODE = 'MPS_LLAMA' if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else 'CPU_DEBUG'  # <-- Auto-detect MPS/" $CONFIG_FILE
        
        echo "✅ 已切换到 MPS_LLAMA 模式（自动检测）"
        echo "📝 原配置已备份到: ${CONFIG_FILE}.backup"
        echo ""
        echo "⚠️  注意: 使用 MPS_LLAMA 模式需要:"
        echo "   1. Hugging Face 账号"
        echo "   2. 申请 Llama 模型访问权限"
        echo "   3. 运行: huggingface-cli login"
        ;;
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac

echo ""
echo "当前配置:"
grep "^MODE = " $CONFIG_FILE | head -1

