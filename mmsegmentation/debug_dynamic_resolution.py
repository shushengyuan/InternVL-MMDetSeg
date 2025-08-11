#!/usr/bin/env python3
"""
调试动态分辨率功能的脚本
"""

import torch
import traceback

def test_dynamic_processor():
    """测试动态分辨率处理器"""
    print("测试动态分辨率处理器...")
    
    try:
        from mmseg.models.backbones.dynamic_resolution import DynamicResolutionProcessor
        
        processor = DynamicResolutionProcessor(
            min_patches=1,
            max_patches=6,
            patch_size=448,
            use_thumbnail=False
        )
        
        # 测试简单图像
        image = torch.randn(3, 448, 448)
        patches, metadata = processor.preprocess(image)
        
        print(f"✅ 处理器测试成功")
        print(f"  - patches数量: {len(patches)}")
        print(f"  - metadata: {metadata}")
        
        return True
        
    except Exception as e:
        print(f"❌ 处理器测试失败: {e}")
        traceback.print_exc()
        return False

def test_intern_vit_init():
    """测试 InternViT 初始化"""
    print("\n测试 InternViT 初始化...")
    
    try:
        from mmseg.models.backbones.intern_vit import InternViT
        
        model = InternViT(
            img_size=448,
            pretrain_size=448,
            patch_size=16,
            embed_dim=768,
            depth=6,  # 减少层数
            num_heads=12,
            use_flash_attn=False,
            use_dynamic_resolution=True,
            min_patches=1,
            max_patches=6,
            use_thumbnail=False,
        )
        
        print(f"✅ 模型初始化成功")
        print(f"  - out_indices: {model.out_indices}")
        print(f"  - dynamic_processor: {model.dynamic_processor is not None}")
        
        return model
        
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        traceback.print_exc()
        return None

def test_standard_forward(model):
    """测试标准前向传播"""
    print("\n测试标准前向传播...")
    
    try:
        model.eval()
        input_tensor = torch.randn(1, 3, 448, 448)
        
        with torch.no_grad():
            output = model._forward_standard(input_tensor)
            
        print(f"✅ 标准前向传播成功")
        if isinstance(output, tuple):
            print(f"  - 输出类型: tuple, 长度: {len(output)}")
            for i, feat in enumerate(output):
                if isinstance(feat, torch.Tensor):
                    print(f"    特征 {i}: {feat.shape}")
        else:
            print(f"  - 输出类型: {type(output)}")
            
        return output
        
    except Exception as e:
        print(f"❌ 标准前向传播失败: {e}")
        traceback.print_exc()
        return None

def test_dynamic_forward(model):
    """测试动态分辨率前向传播"""
    print("\n测试动态分辨率前向传播...")
    
    try:
        model.eval()
        input_tensor = torch.randn(1, 3, 448, 448)
        
        with torch.no_grad():
            output = model._forward_dynamic_resolution(input_tensor)
            
        print(f"✅ 动态前向传播成功")
        if isinstance(output, tuple):
            print(f"  - 输出类型: tuple, 长度: {len(output)}")
            for i, feat in enumerate(output):
                if isinstance(feat, torch.Tensor):
                    print(f"    特征 {i}: {feat.shape}")
        else:
            print(f"  - 输出类型: {type(output)}")
            
        return output
        
    except Exception as e:
        print(f"❌ 动态前向传播失败: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("动态分辨率调试脚本")
    print("=" * 50)
    
    # 测试1: 动态分辨率处理器
    if not test_dynamic_processor():
        print("处理器测试失败，停止后续测试")
        exit(1)
    
    # 测试2: 模型初始化
    model = test_intern_vit_init()
    if model is None:
        print("模型初始化失败，停止后续测试")
        exit(1)
    
    # 测试3: 标准前向传播
    standard_output = test_standard_forward(model)
    if standard_output is None:
        print("标准前向传播失败，停止后续测试")
        exit(1)
    
    # 测试4: 动态前向传播
    dynamic_output = test_dynamic_forward(model)
    if dynamic_output is None:
        print("动态前向传播失败")
        exit(1)
    
    print("\n" + "=" * 50)
    print("✅ 所有测试通过!")
