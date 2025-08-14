#!/usr/bin/env python3
"""
Example script demonstrating incremental optimization with DUSt3R

This example shows how to:
1. Start with a base scene optimization
2. Add new images incrementally
3. Optimize only the new image parameters while keeping existing ones frozen
"""

import torch
import numpy as np
from dust3r.cloud_opt.optimizer import PointCloudOptimizer
from dust3r.cloud_opt.incremental_optimizer import IncrementalPCOptimizer

def create_dummy_data(n_images, img_shape=(256, 256)):
    """
    Create dummy data for demonstration purposes
    
    Args:
        n_images: Number of images
        img_shape: Image shape (height, width)
    
    Returns:
        Dictionary containing dummy view and prediction data
    """
    H, W = img_shape
    
    # Create dummy images
    imgs = [torch.randn(3, H, W) for _ in range(n_images)]
    
    # Create dummy predictions
    n_pixels = H * W
    pts3d = [torch.randn(n_pixels, 3) for _ in range(n_images)]
    pts3d_in_other = [torch.randn(n_pixels, 3) for _ in range(n_images)]
    conf = [torch.rand(n_pixels) for _ in range(n_images)]
    
    return {
        'imgs': imgs,
        'pts3d': pts3d,
        'pts3d_in_other': pts3d_in_other,
        'conf': conf
    }

def create_view_data(img_indices, imgs):
    """
    Create view data dictionary
    
    Args:
        img_indices: List of image indices
        imgs: List of image tensors
    
    Returns:
        View data dictionary
    """
    return {
        'idx': img_indices,
        'img': [imgs[i] for i in img_indices]
    }

def create_prediction_data(pts3d, pts3d_in_other, conf):
    """
    Create prediction data dictionary
    
    Args:
        pts3d: List of 3D points
        pts3d_in_other: List of 3D points in other view
        conf: List of confidence values
    
    Returns:
        Prediction data dictionary
    """
    return {
        'pts3d': pts3d,
        'pts3d_in_other_view': pts3d_in_other,
        'conf': conf
    }

def main():
    """
    Main function demonstrating incremental optimization
    """
    print("DUSt3R Incremental Optimization Example")
    print("=" * 50)
    
    # Step 1: Create initial scene with 3 images
    print("\n1. Creating initial scene with 3 images...")
    
    # Create dummy data for 3 images
    base_data = create_dummy_data(3)
    
    # Create view data for initial scene
    view1 = create_view_data([0, 1], [base_data['imgs'][0], base_data['imgs'][1]])
    view2 = create_view_data([1, 2], [base_data['imgs'][1], base_data['imgs'][2]])
    
    # Create prediction data
    pred1 = create_prediction_data(
        [base_data['pts3d'][0], base_data['pts3d'][1]],
        [base_data['pts3d_in_other'][0], base_data['pts3d_in_other'][1]],
        [base_data['conf'][0], base_data['conf'][1]]
    )
    pred2 = create_prediction_data(
        [base_data['pts3d'][1], base_data['pts3d'][2]],
        [base_data['pts3d_in_other'][1], base_data['pts3d_in_other'][2]],
        [base_data['conf'][1], base_data['conf'][2]]
    )
    
    # Initialize base optimizer
    base_optimizer = PointCloudOptimizer(
        view1, view2, pred1, pred2,
        dist='l1',
        conf='log',
        min_conf_thr=3,
        base_scale=0.5,
        verbose=True
    )
    
    # Optimize base scene
    print("Optimizing base scene...")
    base_loss = base_optimizer.compute_global_alignment(
        init='msp',
        lr=0.01,
        niter=50,
        schedule='cosine'
    )
    print(f"Base scene optimization completed. Final loss: {base_loss:.6f}")
    
    # Step 2: Add a new image incrementally
    print("\n2. Adding new image incrementally...")
    
    # Create dummy data for new image
    new_data = create_dummy_data(1)
    
    # Create view data for new image
    new_view = create_view_data([3], [new_data['imgs'][0]])
    
    # Create predictions from new image to existing images
    new_pred1 = create_prediction_data(
        [new_data['pts3d'][0], new_data['pts3d'][0]],  # New image to existing images 0, 1
        [new_data['pts3d_in_other'][0], new_data['pts3d_in_other'][0]],
        [new_data['conf'][0], new_data['conf'][0]]
    )
    
    # Create predictions from existing images to new image
    new_pred2 = create_prediction_data(
        [base_data['pts3d'][0], base_data['pts3d'][1]],  # Existing images 0, 1 to new image
        [base_data['pts3d_in_other'][0], base_data['pts3d_in_other'][1]],
        [base_data['conf'][0], base_data['conf'][1]]
    )
    
    # Create incremental optimizer
    incremental_optimizer = IncrementalPCOptimizer(
        base_optimizer,
        new_view,
        new_pred1,
        new_pred2,
        dist='l1',
        conf='log',
        min_conf_thr=3,
        base_scale=0.5,
        verbose=True
    )
    
    # Get optimization summary
    summary = incremental_optimizer.get_optimization_summary()
    print(f"Optimization summary: {summary}")
    
    # Optimize incrementally (freeze existing parameters)
    print("Optimizing incrementally (freezing existing parameters)...")
    incremental_loss = incremental_optimizer.optimize_incremental(
        lr=0.01,
        niter=50,
        schedule='cosine',
        freeze_existing=True
    )
    print(f"Incremental optimization completed. Final loss: {incremental_loss:.6f}")
    
    # Step 3: Add another new image
    print("\n3. Adding another new image...")
    
    # Create dummy data for second new image
    new_data2 = create_dummy_data(1)
    
    # Create view data for second new image
    new_view2 = create_view_data([4], [new_data2['imgs'][0]])
    
    # Create predictions from second new image to all existing images
    new_pred1_2 = create_prediction_data(
        [new_data2['pts3d'][0], new_data2['pts3d'][0], new_data2['pts3d'][0], new_data2['pts3d'][0]],
        [new_data2['pts3d_in_other'][0], new_data2['pts3d_in_other'][0], new_data2['pts3d_in_other'][0], new_data2['pts3d_in_other'][0]],
        [new_data2['conf'][0], new_data2['conf'][0], new_data2['conf'][0], new_data2['conf'][0]]
    )
    
    # Create predictions from all existing images to second new image
    new_pred2_2 = create_prediction_data(
        [base_data['pts3d'][0], base_data['pts3d'][1], base_data['pts3d'][2], new_data['pts3d'][0]],
        [base_data['pts3d_in_other'][0], base_data['pts3d_in_other'][1], base_data['pts3d_in_other'][2], new_data['pts3d_in_other'][0]],
        [base_data['conf'][0], base_data['conf'][1], base_data['conf'][2], new_data['conf'][0]]
    )
    
    # Add second new image
    incremental_optimizer2 = incremental_optimizer.add_new_image(
        new_view2,
        new_pred1_2,
        new_pred2_2
    )
    
    # Get updated optimization summary
    summary2 = incremental_optimizer2.get_optimization_summary()
    print(f"Updated optimization summary: {summary2}")
    
    # Optimize with second new image
    print("Optimizing with second new image...")
    final_loss = incremental_optimizer2.optimize_incremental(
        lr=0.01,
        niter=50,
        schedule='cosine',
        freeze_existing=True
    )
    print(f"Final optimization completed. Final loss: {final_loss:.6f}")
    
    print("\nIncremental optimization example completed successfully!")
    print(f"Loss progression: {base_loss:.6f} -> {incremental_loss:.6f} -> {final_loss:.6f}")

if __name__ == "__main__":
    main()
