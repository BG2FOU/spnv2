'''
Visualize heatmap predictions from trained model

Load trained model, run inference on test images, and save heatmap visualizations
'''

import os
import sys
import argparse
import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure project root on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Initialize paths (adds core/ and others to sys.path)
import _init_paths  # noqa: F401

from config import cfg, update_config
from nets import build_spnv2
from dataset import get_dataloader
from utils.checkpoints import load_checkpoint
from utils.visualize import showheatmap, imshowheatmap

def visualize_heatmap_single(image, heatmap, keyptId, output_path, title=''):
    """Visualize single keypoint heatmap and save to file"""
    
    # Convert tensors to numpy
    if torch.is_tensor(heatmap):
        heatmap = heatmap.cpu().numpy()
    if torch.is_tensor(image):
        image = image.mul(255).clamp(0,255).permute(1,2,0).byte().cpu().numpy()
    
    # Get dimensions
    num_keypts, height, width = heatmap.shape
    
    # Resize image to match heatmap
    image_resized = cv2.resize(image, (width, height))
    if len(image_resized.shape) == 3 and image_resized.shape[2] == 3:
        image_resized = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
    image_resized = np.expand_dims(image_resized, axis=-1)
    
    # Get heatmap for specific keypoint
    heatmap_single = heatmap[keyptId, :, :]
    heatmap_single = (heatmap_single * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_single, cv2.COLORMAP_HOT)
    
    # Blend with image
    image_fused = (heatmap_color * 0.7 + image_resized * 0.3) / 255
    
    # Save
    plt.figure(figsize=(10, 8))
    plt.imshow(image_fused)
    plt.title(f'{title} - Keypoint {keyptId}', fontsize=12)
    plt.axis('off')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

def visualize_all_keypoints(image, heatmap, output_path, title=''):
    """Visualize all keypoints in a grid"""
    
    # Convert tensors to numpy
    if torch.is_tensor(heatmap):
        heatmap = heatmap.cpu().numpy()
    if torch.is_tensor(image):
        image = image.mul(255).clamp(0,255).permute(1,2,0).byte().cpu().numpy()
    
    num_keypts, height, width = heatmap.shape
    
    # Create grid
    ncols = 4
    nrows = (num_keypts + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3))
    if num_keypts == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Resize image
    image_resized = cv2.resize(image, (width, height))
    if len(image_resized.shape) == 3 and image_resized.shape[2] == 3:
        image_resized = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
    
    for i in range(num_keypts):
        ax = axes[i]
        heatmap_single = heatmap[i, :, :]
        heatmap_single = (heatmap_single * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_single, cv2.COLORMAP_HOT)
        
        # Blend
        image_blend = (heatmap_color * 0.7 + np.expand_dims(image_resized, -1) * 0.3) / 255
        
        ax.imshow(image_blend)
        ax.set_title(f'Keypoint {i}', fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(num_keypts, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize heatmap predictions')
    
    parser.add_argument('--cfg', required=True, type=str,
                       help='Experiment config file')
    parser.add_argument('--checkpoint', required=True, type=str,
                       help='Path to checkpoint file (e.g., model_best.pth.tar)')
    parser.add_argument('--output-dir', type=str, default='heatmap_vis',
                       help='Output directory for visualization images')
    parser.add_argument('--num-images', type=int, default=10,
                       help='Number of images to visualize')
    parser.add_argument('--keypoint-id', type=int, default=None,
                       help='Specific keypoint to visualize (None for all)')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                       help='Modify config options using command-line')
    
    args = parser.parse_args()
    
    # Update config
    update_config(cfg, args)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Config: {args.cfg}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of images: {args.num_images}\n")
    
    # Build model
    print("Building model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_spnv2(cfg)
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'best_state_dict' in checkpoint:
        state_dict = checkpoint['best_state_dict']
    else:
        state_dict = checkpoint
    
    # Handle DataParallel/DistributedDataParallel
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully!\n")
    
    # Get validation dataloader
    print("Loading validation dataset...")
    val_loader = get_dataloader(cfg, split='val', distributed=False, load_labels=True)
    
    print(f"Processing {args.num_images} images...\n")
    
    # Visualize heatmaps
    with torch.no_grad():
        for idx, (images, targets) in enumerate(val_loader):
            if idx >= args.num_images:
                break
            
            # Get filename from target
            filename = targets.get('imgpath', [f'image_{idx:04d}'])[0]
            if isinstance(filename, str):
                filename = os.path.splitext(os.path.basename(filename))[0]
            else:
                filename = f'image_{idx:04d}'
            
            print(f"[{idx+1}/{args.num_images}] Processing {filename}...")
            
            # Move to device
            images = images.to(device)
            
            # Forward pass
            outputs = model(images, is_train=False, gpu=device)
            
            # Get heatmap (first output if multiple heads)
            heatmap = None
            for i, head_name in enumerate(cfg.TEST.HEAD):
                if head_name == 'heatmap':
                    heatmap = outputs[i]
                    break
            
            if heatmap is None:
                print(f"  Warning: No heatmap head found in model outputs")
                continue
            
            # Get image and heatmap
            image = images[0].cpu()
            heatmap = heatmap[0].cpu()
            
            # Visualize specific keypoint or all
            if args.keypoint_id is not None:
                output_path = os.path.join(args.output_dir, f'{filename}_kpt{args.keypoint_id}.png')
                visualize_heatmap_single(image, heatmap, args.keypoint_id, output_path, filename)
                print(f"  Saved: {output_path}")
            else:
                # Save all keypoints in grid
                output_path = os.path.join(args.output_dir, f'{filename}_all_keypoints.png')
                visualize_all_keypoints(image, heatmap, output_path, filename)
                print(f"  Saved: {output_path}")
    
    print(f"\n✅ Heatmap visualization complete!")
    print(f"   Output directory: {args.output_dir}")

if __name__ == '__main__':
    main()
