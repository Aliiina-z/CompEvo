#!/usr/bin/env python3
"""
验证checkpoint文件是否完整的脚本
用于检查训练是否正常完成，防止服务器中途挂掉导致checkpoint不完整
"""

import torch
import os
import sys

def verify_checkpoint(checkpoint_path="/mlx_devbox/users/zhangyuxuan.alina/playground/llm_egt_forecaster-master/llm_egt_forecaster/checkpoints/best_model.pt", config=None):
    """
    验证checkpoint文件的完整性
    
    Args:
        checkpoint_path: checkpoint文件路径
        config: 配置对象（可选，用于验证参数数量）
    
    Returns:
        dict: 验证结果
    """
    results = {
        'file_exists': False,
        'file_readable': False,
        'can_load': False,
        'required_keys': [],
        'missing_keys': [],
        'epoch': None,
        'val_loss': None,
        'num_agents': None,
        'framework_params_count': None,
        'optimizer_params_count': None,
        'is_complete': False,
        'warnings': []
    }
    
    print("=" * 60)
    print("Checkpoint 完整性验证")
    print("=" * 60)
    print(f"检查文件: {checkpoint_path}\n")
    
    # 1. 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        print(f"❌ 错误: 文件不存在: {checkpoint_path}")
        return results
    
    results['file_exists'] = True
    file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
    print(f"✓ 文件存在 (大小: {file_size:.2f} MB)")
    
    # 2. 尝试加载文件
    try:
        # 尝试在不同设备上加载（优先CPU，避免GPU内存问题）
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        results['file_readable'] = True
        results['can_load'] = True
        print(f"✓ 文件可以成功加载")
    except Exception as e:
        print(f"❌ 错误: 无法加载checkpoint文件: {e}")
        print(f"   文件可能损坏或不完整！")
        return results
    
    # 3. 检查必需的键
    required_keys = [
        'epoch',
        'framework_state_dict',
        'optimizer_state_dict',
        'val_loss',
        'agent_logics',
        'agent_fitness'
    ]
    
    print(f"\n检查必需字段:")
    for key in required_keys:
        if key in checkpoint:
            results['required_keys'].append(key)
            print(f"  ✓ {key}")
        else:
            results['missing_keys'].append(key)
            print(f"  ❌ {key} - 缺失！")
    
    # 4. 检查epoch
    if 'epoch' in checkpoint:
        results['epoch'] = checkpoint['epoch']
        print(f"\n训练进度:")
        print(f"  已完成 Epoch: {checkpoint['epoch'] + 1} (保存的epoch索引: {checkpoint['epoch']})")
        
        # 优先从checkpoint中的config读取NUM_EPOCHS
        total_epochs = None
        if 'config' in checkpoint:
            checkpoint_config = checkpoint['config']
            if isinstance(checkpoint_config, dict) and 'NUM_EPOCHS' in checkpoint_config:
                total_epochs = checkpoint_config['NUM_EPOCHS']
                print(f"  总 Epochs (从checkpoint): {total_epochs}")
            elif hasattr(checkpoint_config, 'NUM_EPOCHS'):
                total_epochs = checkpoint_config.NUM_EPOCHS
                print(f"  总 Epochs (从checkpoint): {total_epochs}")
        
        # 如果checkpoint中没有，才使用传入的config
        if total_epochs is None and config and hasattr(config, 'NUM_EPOCHS'):
            total_epochs = config.NUM_EPOCHS
            print(f"  总 Epochs (从当前配置): {total_epochs}")
            results['warnings'].append("⚠️  无法从checkpoint读取NUM_EPOCHS，使用了当前配置的值")
        
        if total_epochs:
            progress = (checkpoint['epoch'] + 1) / total_epochs * 100
            print(f"  进度: {progress:.1f}%")
            if checkpoint['epoch'] + 1 < total_epochs:
                results['warnings'].append(f"训练未完成：只完成了 {checkpoint['epoch'] + 1}/{total_epochs} epochs")
            elif checkpoint['epoch'] + 1 == total_epochs:
                print(f"  ✓ 训练已完成所有 {total_epochs} 个epochs")
            else:
                results['warnings'].append(f"异常：完成的epoch数 ({checkpoint['epoch'] + 1}) 超过了配置的epoch数 ({total_epochs})")
        else:
            print(f"  ⚠️  无法确定总epoch数，无法计算训练进度")
    
    # 5. 检查验证loss
    if 'val_loss' in checkpoint:
        results['val_loss'] = checkpoint['val_loss']
        print(f"  验证 Loss: {checkpoint['val_loss']:.6f}")
    
    # 6. 检查agent数量
    if 'agent_logics' in checkpoint and 'agent_fitness' in checkpoint:
        num_agents = len(checkpoint['agent_logics'])
        results['num_agents'] = num_agents
        print(f"\nAgent 信息:")
        print(f"  Agent 数量: {num_agents}")
        
        if len(checkpoint['agent_fitness']) != num_agents:
            results['warnings'].append(f"Agent logics 和 fitness 数量不匹配！")
            print(f"  ⚠️  警告: agent_logics ({len(checkpoint['agent_logics'])}) 和 agent_fitness ({len(checkpoint['agent_fitness'])}) 数量不匹配")
        
        # 显示每个agent的logic和fitness
        print(f"\n  各 Agent 的状态:")
        for i, (logic, fitness) in enumerate(zip(checkpoint['agent_logics'], checkpoint['agent_fitness'])):
            logic_preview = logic[:50] + "..." if len(logic) > 50 else logic
            print(f"    Agent {i}: Fitness={fitness:.4f}, Logic='{logic_preview}'")
    
    # 7. 检查框架参数
    if 'framework_state_dict' in checkpoint:
        framework_state = checkpoint['framework_state_dict']
        param_count = sum(len(v.flatten()) if isinstance(v, torch.Tensor) else 0 
                         for v in framework_state.values())
        results['framework_params_count'] = param_count
        print(f"\n模型参数:")
        print(f"  框架参数数量: {param_count:,}")
        print(f"  参数键数量: {len(framework_state)}")
        
        # 检查关键的参数键
        key_keywords = ['lora', 'gate', 'prediction_head']
        found_keys = []
        for key in framework_state.keys():
            for keyword in key_keywords:
                if keyword.lower() in key.lower():
                    found_keys.append(key)
                    break
        
        if found_keys:
            print(f"  关键参数键 (LoRA/Gate/PredictionHead): {len(found_keys)} 个")
            for key in found_keys[:5]:  # 只显示前5个
                print(f"    - {key}")
            if len(found_keys) > 5:
                print(f"    ... 还有 {len(found_keys) - 5} 个")
    
    # 8. 检查优化器状态
    if 'optimizer_state_dict' in checkpoint:
        optimizer_state = checkpoint['optimizer_state_dict']
        if 'state' in optimizer_state:
            results['optimizer_params_count'] = len(optimizer_state['state'])
            print(f"  优化器状态: {len(optimizer_state['state'])} 个参数组")
    
    # 9. 检查config（如果存在）
    if 'config' in checkpoint:
        print(f"\n配置信息:")
        config_data = checkpoint['config']
        if isinstance(config_data, dict):
            print(f"  配置项数量: {len(config_data)}")
            # 显示一些关键配置
            key_configs = ['NUM_AGENTS', 'NUM_EPOCHS', 'BASE_LLM_MODEL', 'LEARNING_RATE']
            for key in key_configs:
                if key in config_data:
                    print(f"    {key}: {config_data[key]}")
        else:
            print(f"  ⚠️  警告: config 不是字典格式")
            results['warnings'].append("config 格式异常")
    
    # 10. 总体评估
    print(f"\n" + "=" * 60)
    print("验证结果总结")
    print("=" * 60)
    
    if len(results['missing_keys']) == 0:
        results['is_complete'] = True
        print("✓ 所有必需字段都存在")
    else:
        print(f"❌ 缺失 {len(results['missing_keys'])} 个必需字段")
        results['is_complete'] = False
    
    if len(results['warnings']) > 0:
        print(f"\n⚠️  警告 ({len(results['warnings'])} 个):")
        for warning in results['warnings']:
            print(f"  - {warning}")
    
    if results['is_complete'] and len(results['warnings']) == 0:
        print("\n✅ Checkpoint 文件完整，训练正常完成！")
    elif results['is_complete']:
        print("\n⚠️  Checkpoint 文件基本完整，但有警告")
    else:
        print("\n❌ Checkpoint 文件不完整，可能训练被中断！")
    
    return results


if __name__ == '__main__':
    # 默认检查 checkpoints/best_model.pt
    checkpoint_path = "/mlx_devbox/users/zhangyuxuan.alina/playground/llm_egt_forecaster-master/llm_egt_forecaster/checkpoints/best_model.pt"
    
    # 如果提供了命令行参数，使用指定的路径
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    
    # 尝试导入config以获取总epoch数（可选）
    config = None
    try:
        from llm_egt_forecaster.configs import base_config
        config = base_config
    except ImportError:
        print("注意: 无法导入配置，将跳过epoch进度检查\n")
    
    results = verify_checkpoint(checkpoint_path, config)
    
    # 退出码：0=成功，1=失败
    sys.exit(0 if results['is_complete'] else 1)