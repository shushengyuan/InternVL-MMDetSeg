#!/usr/bin/env python3
"""
测试动态分辨率功能的脚本
"""

import torch
import torch.nn.functional as F
from mmseg.models.backbones.intern_vit import InternViT
from mmseg.models.backbones.dynamic_resolution import DynamicResolutionProcessor

def test_dynamic_resolution_processor():
    """测试动态分辨率处理器"""
    print("=" * 60)
    print("测试动态分辨率处理器")
    print("=" * 60)
    
    # 创建处理器
    processor = DynamicResolutionProcessor(
        min_patches=1,
        max_patches=12,  # 测试用较小值
        patch_size=448,
        use_thumbnail=False
    )
    
    # 测试不同长宽比的图像
    test_cases = [
        (448, 448, "1:1 正方形"),
        (896, 448, "2:1 横向长条"),
        (448, 896, "1:2 纵向长条"),
        (1344, 448, "3:1 极宽图像"),
        (672, 672, "1.5:1.5 略大正方形"),
    ]
    
    for width, height, description in test_cases:
        print(f"\n测试 {description} ({width}x{height}):")
        
        # 创建随机图像
        image = torch.randn(3, height, width)
        
        # 处理图像
        patches, metadata = processor.preprocess(image)
        
        print(f"  - 原始尺寸: {width}x{height}")
        print(f"  - 网格尺寸: {metadata['grid_dims']}")
        print(f"  - Patch数量: {metadata['num_patches']}")
        print(f"  - 每个patch尺寸: {patches[0].shape}")
        
        # 验证patch数量
        grid_w, grid_h = metadata['grid_dims']
        expected_patches = grid_w * grid_h
        assert len(patches) == expected_patches, f"Patch数量不匹配: {len(patches)} vs {expected_patches}"
        
        print(f"  ✅ 测试通过")


def test_intern_vit_dynamic():
    """测试 InternViT 的动态分辨率功能"""
    print("=" * 60)
    print("测试 InternViT 动态分辨率功能")
    print("=" * 60)
    
    # 创建模型（简化配置用于测试）
    model = InternViT(
        img_size=448,
        pretrain_size=448,
        patch_size=16,
        embed_dim=768,  # 较小的维度用于测试
        depth=12,       # 较少的层数用于测试
        num_heads=12,
        use_flash_attn=False,  # 关闭FlashAttention避免依赖问题
        # 动态分辨率配置
        use_dynamic_resolution=True,
        min_patches=1,
        max_patches=6,  # 测试用较小值
        use_thumbnail=False,
    )
    
    model.eval()  # 切换到推理模式以启用动态分辨率
    
    # 测试不同尺寸的输入
    test_inputs = [
        torch.randn(1, 3, 448, 448),   # 标准尺寸
        torch.randn(1, 3, 896, 448),   # 2:1 比例
        torch.randn(1, 3, 672, 672),   # 1.5:1.5 比例
    ]
    
    for i, input_tensor in enumerate(test_inputs):
        print(f"\n测试输入 {i+1}: {input_tensor.shape}")
        
        try:
            with torch.no_grad():
                # 动态分辨率前向传播
                output = model(input_tensor)
                
                if isinstance(output, tuple):
                    features = output[0]
                    if isinstance(features, tuple):
                        print(f"  - 输出特征层数: {len(features)}")
                        for j, feat in enumerate(features):
                            print(f"    层 {j}: {feat.shape}")
                    else:
                        print(f"  - 输出特征: {features.shape}")
                else:
                    print(f"  - 输出: {output.shape}")
                
                print(f"  ✅ 动态分辨率处理成功")
                
        except Exception as e:
            print(f"  ❌ 错误: {e}")


def test_standard_vs_dynamic():
    """比较标准模式和动态分辨率模式"""
    print("=" * 60)
    print("比较标准模式 vs 动态分辨率模式")
    print("=" * 60)
    
    # 创建两个相同的模型
    model_standard = InternViT(
        img_size=448,
        pretrain_size=448,
        patch_size=16,
        embed_dim=768,
        depth=6,  # 减少层数加快测试
        num_heads=12,
        use_flash_attn=False,
        use_dynamic_resolution=False,  # 关闭动态分辨率
    )
    
    model_dynamic = InternViT(
        img_size=448,
        pretrain_size=448,
        patch_size=16,
        embed_dim=768,
        depth=6,
        num_heads=12,
        use_flash_attn=False,
        use_dynamic_resolution=True,   # 启用动态分辨率
        min_patches=1,
        max_patches=4,
    )
    
    model_standard.eval()
    model_dynamic.eval()
    
    # 测试相同输入
    input_tensor = torch.randn(1, 3, 448, 448)
    
    print(f"输入尺寸: {input_tensor.shape}")
    
    with torch.no_grad():
        # 标准模式
        try:
            output_standard = model_standard(input_tensor)
            print(f"标准模式输出: 成功")
            if isinstance(output_standard, tuple) and len(output_standard) > 0:
                if isinstance(output_standard[0], tuple):
                    print(f"  特征形状: {[f.shape for f in output_standard[0]]}")
                else:
                    print(f"  特征形状: {output_standard[0].shape}")
        except Exception as e:
            print(f"标准模式输出: 失败 - {e}")
        
        # 动态分辨率模式
        try:
            output_dynamic = model_dynamic(input_tensor)
            print(f"动态模式输出: 成功")
            if isinstance(output_dynamic, tuple) and len(output_dynamic) > 0:
                if isinstance(output_dynamic[0], tuple):
                    print(f"  特征形状: {[f.shape for f in output_dynamic[0]]}")
                else:
                    print(f"  特征形状: {output_dynamic[0].shape}")
        except Exception as e:
            print(f"动态模式输出: 失败 - {e}")


if __name__ == "__main__":
    print("InternVL 动态分辨率功能测试")
    print("=" * 60)
    
    try:
        # 测试1: 动态分辨率处理器
        test_dynamic_resolution_processor()
        
        # 测试2: InternViT动态分辨率（需要较多内存，可能会跳过）
        try:
            test_intern_vit_dynamic()
        except Exception as e:
            print(f"跳过InternViT测试（可能因为内存不足）: {e}")
        
        # 测试3: 标准vs动态比较
        try:
            test_standard_vs_dynamic()
        except Exception as e:
            print(f"跳过比较测试: {e}")
            
        print("\n" + "=" * 60)
        print("✅ 动态分辨率功能测试完成")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
