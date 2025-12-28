#!/bin/bash
# 恢复 MPS_LLAMA 模式脚本

CONFIG_FILE="llm_egt_forecaster/configs/base_config.py"

echo "=== 恢复 MPS_LLAMA 模式 ==="
echo ""

# 检查是否有备份
if [ -f "${CONFIG_FILE}.backup" ]; then
    echo "📦 找到备份文件，正在恢复..."
    cp ${CONFIG_FILE}.backup $CONFIG_FILE
    echo "✅ 已从备份恢复"
else
    echo "📝 未找到备份，直接修改配置..."
    # 直接修改为 MPS_LLAMA 模式
    sed -i '' "s/^MODE = .*/MODE = 'MPS_LLAMA' if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else 'CPU_DEBUG'  # <-- Auto-detect MPS, fallback to CPU_DEBUG/" $CONFIG_FILE
    echo "✅ 已切换到 MPS_LLAMA 模式"
fi

echo ""
echo "当前配置:"
grep "^MODE = " $CONFIG_FILE | head -1
echo ""
echo "⚠️  注意: 使用 MPS_LLAMA 模式需要:"
echo "   1. Hugging Face 账号"
echo "   2. 申请 Llama 模型访问权限: https://huggingface.co/meta-llama/Llama-3.1-8B"
echo "   3. 运行: huggingface-cli login"
echo ""
echo "✅ 配置已恢复，可以运行: python train.py"

