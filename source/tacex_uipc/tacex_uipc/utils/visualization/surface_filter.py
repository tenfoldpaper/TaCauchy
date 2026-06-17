"""
Surface filtering utilities for gelpad visualization.

This module provides functions to filter vertices based on their position,
useful for isolating contact surfaces from constraint surfaces.
"""

import numpy as np
from typing import Optional, Tuple


def filter_contact_surface_vertices(
    all_positions: np.ndarray, 
    attachment_indices: Optional[list] = None,
    z_threshold_percentile: float = 30.0,
    reference_positions: Optional[np.ndarray] = None,
    use_robust_method: bool = True,
    z_tolerance: float = 0.003,
    use_relative_tolerance: bool = True,
    relative_tolerance_ratio: float = 0.5
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Filter vertices to only include the bottom contact surface (opposite to attachment surface).
    
    使用更鲁棒的方法来识别接触表面，即使表面有凹凸不平也能正确识别。
    
    IMPORTANT: This function should use INITIAL/REFERENCE positions to determine which vertices
    belong to the contact surface, not current positions. This ensures that vertices remain
    tracked even when they deform and move inward during contact.
    
    Args:
        all_positions: Current vertex positions (N, 3) - used for output
        attachment_indices: Indices of attachment points (top surface). If provided,
                          these vertices will be excluded from the result.
        z_threshold_percentile: Percentile threshold for Z coordinate to define bottom surface.
                              Default 30.0 means the bottom 30% of vertices by Z coordinate.
                              (仅在 use_robust_method=False 时使用)
        reference_positions: Reference positions (typically initial positions) used to determine
                           which vertices belong to the contact surface (N, 3).
                           If None, uses all_positions (current positions).
        use_robust_method: 是否使用鲁棒方法（基于与附着点的Z距离）。
                          True: 使用基于gelpad厚度的自适应方法（推荐）
                          False: 使用固定百分位数方法（旧方法）
        z_tolerance: Z方向绝对容差（米），仅在 use_relative_tolerance=False 时使用。
        use_relative_tolerance: 是否使用相对容差（推荐）。
                               True: 容差相对于gelpad厚度
                               False: 使用绝对容差
        relative_tolerance_ratio: 相对容差比例（0-1），默认0.5表示容差为厚度的50%。
                                 例如：厚度2.71mm时，容差=1.36mm
        
    Returns:
        Tuple of (filtered_positions, filtered_indices, mask):
            - filtered_positions: Current positions of contact surface vertices (M, 3)
            - filtered_indices: Original indices of filtered vertices (M,)
            - mask: Boolean mask indicating which vertices were selected (N,)
        Returns (None, None, None) if input is invalid.
    
    Example:
        >>> # 使用鲁棒方法（推荐）
        >>> initial_pos = gelpad.data.nodal_pos_w.cpu().numpy()
        >>> contact_pos, contact_idx, contact_mask = filter_contact_surface_vertices(
        ...     initial_pos, 
        ...     attachment_indices,
        ...     reference_positions=initial_pos,
        ...     use_robust_method=True,  # 使用鲁棒方法
        ...     use_relative_tolerance=True,  # 使用相对容差
        ...     relative_tolerance_ratio=0.5  # 容差为厚度的50%
        ... )
    """
    if all_positions is None or len(all_positions) == 0:
        return None, None, None
    
    # Use reference positions for filtering if provided, otherwise use current positions
    positions_for_filtering = reference_positions if reference_positions is not None else all_positions
    
    if use_robust_method and attachment_indices is not None and len(attachment_indices) > 0:
        # ========== 鲁棒方法：基于与附着点的Z距离 ==========
        # 这种方法对表面凹凸不平更鲁棒
        
        # 1. 计算附着点的平均Z坐标（顶部表面）
        attachment_z_coords = positions_for_filtering[attachment_indices, 2]
        attachment_z_mean = np.mean(attachment_z_coords)
        attachment_z_std = np.std(attachment_z_coords)
        
        # 2. 计算所有点的Z坐标
        all_z_coords = positions_for_filtering[:, 2]
        
        # 3. 计算gelpad的厚度（Z方向范围）
        z_min = all_z_coords.min()
        z_max = all_z_coords.max()
        gelpad_thickness = z_max - z_min
        
        # 4. 定义接触表面：距离附着点足够远的点（在Z方向上）
        # 使用自适应阈值：至少是厚度的25%，但允许一定容差
        # 降低比例以包含更多的表面点，即使表面有凹凸
        z_distance_threshold = gelpad_thickness * 0.25
        
        # 5. 计算实际容差（相对或绝对）
        if use_relative_tolerance:
            # 相对容差：基于gelpad厚度
            actual_tolerance = gelpad_thickness * relative_tolerance_ratio
        else:
            # 绝对容差
            actual_tolerance = z_tolerance
        
        # 6. 创建掩码：Z坐标明显低于附着点平均值
        # 使用容差来包含略微凹凸的表面点
        z_distance_from_attachment = attachment_z_mean - all_z_coords
        contact_surface_mask = z_distance_from_attachment >= (z_distance_threshold - actual_tolerance)
        
        # 7. 排除附着点本身
        attachment_mask = np.ones(len(all_positions), dtype=bool)
        attachment_mask[attachment_indices] = False
        
        # 8. 组合掩码
        combined_mask = contact_surface_mask & attachment_mask
        
        # 调试信息（可选，用于诊断）
        num_selected = np.sum(combined_mask)
        num_contact_surface = np.sum(contact_surface_mask)
        num_attachment = len(attachment_indices)
        selection_ratio = num_selected / len(all_positions) * 100
        print(f"[RobustFilter] Gelpad thickness: {gelpad_thickness*1000:.2f}mm")
        print(f"[RobustFilter] Attachment Z: {attachment_z_mean:.6f}m ± {attachment_z_std*1000:.2f}mm")
        print(f"[RobustFilter] Z threshold: {z_distance_threshold*1000:.2f}mm")
        if use_relative_tolerance:
            print(f"[RobustFilter] Tolerance: {actual_tolerance*1000:.2f}mm (relative: {relative_tolerance_ratio*100:.0f}% of thickness)")
        else:
            print(f"[RobustFilter] Tolerance: {actual_tolerance*1000:.2f}mm (absolute)")
        print(f"[RobustFilter] Effective threshold: {(z_distance_threshold - actual_tolerance)*1000:.2f}mm")
        print(f"[RobustFilter] Total vertices: {len(all_positions)}")
        print(f"[RobustFilter] Attachment points: {num_attachment} ({num_attachment/len(all_positions)*100:.1f}%)")
        print(f"[RobustFilter] Contact surface (before excluding attachment): {num_contact_surface} ({num_contact_surface/len(all_positions)*100:.1f}%)")
        print(f"[RobustFilter] Selected (after excluding attachment): {num_selected} ({selection_ratio:.1f}%)")
        
        # 显示Z坐标分布
        print(f"[RobustFilter] Z distance range: [{z_distance_from_attachment.min()*1000:.2f}, {z_distance_from_attachment.max()*1000:.2f}]mm")
        print(f"[RobustFilter] Z distance mean: {z_distance_from_attachment.mean()*1000:.2f}mm")
        
        # 显示有多少点满足阈值条件
        num_above_threshold = np.sum(z_distance_from_attachment >= (z_distance_threshold - actual_tolerance))
        print(f"[RobustFilter] Points above threshold: {num_above_threshold} ({num_above_threshold/len(all_positions)*100:.1f}%)")
        
    else:
        # ========== 原始方法：基于Z坐标百分位数 ==========
        # 这种方法在表面平整时效果好，但对凹凸表面不够鲁棒
        
        z_coords = positions_for_filtering[:, 2]
        z_threshold = np.percentile(z_coords, z_threshold_percentile)
        
        # Create mask for bottom surface vertices based on REFERENCE positions
        bottom_surface_mask = z_coords <= z_threshold
        
        # If attachment indices are provided, exclude them
        if attachment_indices is not None and len(attachment_indices) > 0:
            attachment_mask = np.ones(len(all_positions), dtype=bool)
            attachment_mask[attachment_indices] = False
            combined_mask = bottom_surface_mask & attachment_mask
        else:
            combined_mask = bottom_surface_mask
    
    # Get filtered CURRENT positions using the mask determined from reference positions
    filtered_positions = all_positions[combined_mask]
    filtered_indices = np.where(combined_mask)[0]
    
    return filtered_positions, filtered_indices, combined_mask


def get_surface_statistics(
    all_positions: np.ndarray,
    filtered_positions: np.ndarray,
    attachment_indices: Optional[list] = None
) -> dict:
    """
    Get statistics about surface filtering.
    
    Args:
        all_positions: All vertex positions
        filtered_positions: Filtered contact surface positions
        attachment_indices: Attachment point indices
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_vertices': len(all_positions),
        'contact_vertices': len(filtered_positions),
        'contact_percentage': len(filtered_positions) / len(all_positions) * 100,
        'z_range_all': (all_positions[:, 2].min(), all_positions[:, 2].max()),
        'z_range_contact': (filtered_positions[:, 2].min(), filtered_positions[:, 2].max()),
    }
    
    if attachment_indices is not None:
        stats['attachment_vertices'] = len(attachment_indices)
    
    return stats
