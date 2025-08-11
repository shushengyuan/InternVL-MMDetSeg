# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch
import torch.nn.functional as F
import math
from typing import Tuple, List


def find_closest_aspect_ratio(aspect_ratio: float, target_ratios: List[Tuple[int, int]], 
                              width: int, height: int, image_size: int) -> Tuple[int, int]:
    """
    Find the closest aspect ratio and return the corresponding grid size.
    
    Args:
        aspect_ratio: The aspect ratio of the input image (width / height)
        target_ratios: List of supported aspect ratios as (width_ratio, height_ratio)
        width: Original image width
        height: Original image height  
        image_size: Target size for each patch (e.g., 448)
        
    Returns:
        Tuple of (grid_width, grid_height) for the closest aspect ratio
    """
    best_ratio_diff = float('inf')
    best_grid = (1, 1)
    
    for grid_width, grid_height in target_ratios:
        target_aspect_ratio = grid_width / grid_height
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_grid = (grid_width, grid_height)
            
    return best_grid


def dynamic_preprocess(image: torch.Tensor, 
                      min_num: int = 1, 
                      max_num: int = 6, 
                      image_size: int = 448, 
                      use_thumbnail: bool = False) -> Tuple[List[torch.Tensor], Tuple[int, int]]:
    """
    Dynamically preprocess an image into multiple patches based on its aspect ratio.
    参考 InternVL 官方实现
    
    Args:
        image: Input image tensor of shape (C, H, W)
        min_num: Minimum number of patches
        max_num: Maximum number of patches  
        image_size: Size of each patch (e.g., 448)
        use_thumbnail: Whether to include a thumbnail of the entire image
        
    Returns:
        List of processed image patches and the grid dimensions
    """
    orig_width, orig_height = image.shape[-1], image.shape[-2]
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio (参考官方实现)
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_image = F.interpolate(
        image.unsqueeze(0), 
        size=(target_height, target_width), 
        mode='bicubic', 
        align_corners=False
    ).squeeze(0)
    
    processed_images = []
    for i in range(blocks):
        # 计算box坐标 (参考官方实现)
        left = (i % target_aspect_ratio[0]) * image_size
        top = (i // target_aspect_ratio[0]) * image_size
        right = left + image_size
        bottom = top + image_size
        
        # split the image
        split_img = resized_image[:, top:bottom, left:right]
        processed_images.append(split_img)
    
    assert len(processed_images) == blocks
    
    # 添加缩略图 (参考官方实现)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = F.interpolate(
            image.unsqueeze(0),
            size=(image_size, image_size),
            mode='bicubic',
            align_corners=False
        ).squeeze(0)
        processed_images.append(thumbnail_img)
    
    return processed_images, target_aspect_ratio


def pixel_shuffle(x: torch.Tensor, scale_factor: int = 2) -> torch.Tensor:
    """
    Pixel shuffle operation to upsample feature maps.
    
    Args:
        x: Input tensor of shape (B, C, H, W)
        scale_factor: Upsampling scale factor
        
    Returns:
        Upsampled tensor
    """
    b, c, h, w = x.shape
    # Reshape and permute for pixel shuffle
    x = x.view(b, c // (scale_factor ** 2), scale_factor, scale_factor, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(b, c // (scale_factor ** 2), h * scale_factor, w * scale_factor)
    return x


def merge_patches(patch_features: List[torch.Tensor], 
                 grid_dims: Tuple[int, int],
                 original_size: Tuple[int, int]) -> torch.Tensor:
    """
    Merge patch features back into a single feature map.
    
    Args:
        patch_features: List of patch feature tensors
        grid_dims: Grid dimensions (width, height)
        original_size: Original image size (width, height)
        
    Returns:
        Merged feature tensor
    """
    grid_width, grid_height = grid_dims
    
    if len(patch_features) != grid_width * grid_height:
        raise ValueError(f"Number of patches {len(patch_features)} doesn't match grid size {grid_width}x{grid_height}")
    
    # Assume all patches have the same feature dimensions
    if len(patch_features[0].shape) == 4:
        B, C, patch_h, patch_w = patch_features[0].shape
    else:
        # Handle 3D tensors (C, H, W) - squeezed batch dimension
        C, patch_h, patch_w = patch_features[0].shape
        B = 1
    
    # Reconstruct the grid
    feature_grid = []
    for h_idx in range(grid_height):
        row_features = []
        for w_idx in range(grid_width):
            patch_idx = h_idx * grid_width + w_idx
            row_features.append(patch_features[patch_idx])
        feature_grid.append(torch.cat(row_features, dim=-1))  # Concatenate along width
    
    # Concatenate along height
    merged_features = torch.cat(feature_grid, dim=-2)
    
    # Resize to original proportions if needed
    target_h, target_w = original_size
    if merged_features.shape[-2:] != (target_h, target_w):
        # Ensure 4D tensor for interpolation
        if len(merged_features.shape) == 3:
            merged_features = merged_features.unsqueeze(0)  # Add batch dimension
            need_squeeze = True
        else:
            need_squeeze = False
            
        merged_features = F.interpolate(
            merged_features,
            size=(target_h, target_w),
            mode='bicubic',
            align_corners=False
        )
        
        # Remove batch dimension if we added it
        if need_squeeze:
            merged_features = merged_features.squeeze(0)
    
    return merged_features


class DynamicResolutionProcessor:
    """
    A processor for handling dynamic resolution inputs in InternVL.
    """
    
    def __init__(self, 
                 min_patches: int = 1,
                 max_patches: int = 40, 
                 patch_size: int = 448,
                 use_thumbnail: bool = False):
        self.min_patches = min_patches
        self.max_patches = max_patches
        self.patch_size = patch_size
        self.use_thumbnail = use_thumbnail
    
    def preprocess(self, image: torch.Tensor) -> Tuple[List[torch.Tensor], dict]:
        """
        Preprocess an image for dynamic resolution processing.
        
        Args:
            image: Input image tensor (C, H, W)
            
        Returns:
            List of image patches and metadata
        """
        patches, target_aspect_ratio = dynamic_preprocess(
            image, 
            self.min_patches, 
            self.max_patches, 
            self.patch_size,
            self.use_thumbnail
        )
        
        grid_width, grid_height = target_aspect_ratio
        metadata = {
            'grid_dims': target_aspect_ratio,
            'original_size': (image.shape[-2], image.shape[-1]),  # (height, width) for F.interpolate
            'num_patches': len(patches),
            'num_grid_patches': grid_width * grid_height,  # 实际网格patch数量
            'use_thumbnail': self.use_thumbnail
        }
        
        return patches, metadata
    
    def merge_features(self, 
                      patch_features: List[torch.Tensor], 
                      metadata: dict) -> torch.Tensor:
        """
        Merge patch features back into a single feature map.
        
        Args:
            patch_features: List of patch feature tensors
            metadata: Metadata from preprocessing
            
        Returns:
            Merged feature tensor
        """
        if self.use_thumbnail and metadata['use_thumbnail']:
            # Handle thumbnail separately if used
            thumbnail_features = patch_features[0]
            patch_features = patch_features[1:]
            
            # 对于缩略图模式，我们可以选择使用缩略图特征或高分辨率特征
            # 这里暂时使用高分辨率特征，忽略缩略图
            # 在实际应用中，可以设计更复杂的融合策略
        
        # 如果没有patch features，返回缩略图特征（如果有的话）
        if len(patch_features) == 0:
            if self.use_thumbnail and 'thumbnail_features' in locals():
                return thumbnail_features
            else:
                raise ValueError("No patch features available for merging")
        
        # 如果只有一个patch，直接返回
        if len(patch_features) == 1:
            return patch_features[0]
        
        merged = merge_patches(
            patch_features, 
            metadata['grid_dims'], 
            metadata.get('original_size', (patch_features[0].shape[-1], patch_features[0].shape[-2]))
        )
        
        return merged
