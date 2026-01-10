'''
Visualize training metrics and pose estimation results

This script:
1. Parses training logs to plot train/val loss curves
2. Plots pose error (rotation, translation) iteration curves
3. Generates scatter plots for pose error distribution
'''

import os
import json
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def find_log_file(log_dir):
    """Find a log file in given directory (train_rank0.log preferred)."""
    candidates = ['train_rank0.log', 'train.log', 'training.log']
    for name in candidates:
        p = os.path.join(log_dir, name)
        if os.path.exists(p):
            return p
    # Fallback: search recursively for the first match
    for name in candidates:
        matches = list(Path(log_dir).glob(f'**/{name}'))
        if matches:
            return str(matches[0])
    return None

def parse_log_file(log_file):
    """Parse training log file (timestamp format) to extract losses/metrics."""
    train_logs = {'epoch': [], 'hmap': [], 'cls': [], 'box': [], 'pose': [], 'seg': []}
    val_logs   = {'epoch': [], 'heat_eR': [], 'heat_eT': [], 'effi_eR': [], 'effi_eT': [], 'segm_iou': [], 'final_pose': []}

    epoch = 0
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if 'Current epoch learning rate' in line:
                epoch += 1
                continue

            if '* Time:' in line:
                # Training losses
                if any(k in line for k in ['hmap', 'cls', 'box', 'pose', 'seg']):
                    pattern = r'(\w+)\s+([\d.e\-]+)'
                    for name, val in re.findall(pattern, line):
                        if name in ['hmap', 'cls', 'box', 'pose', 'seg']:
                            train_logs[name].append((epoch, float(val)))
                    # Track epoch for plotting
                    train_logs['epoch'].append(epoch)

                # Validation metrics
                if any(k in line for k in ['heat_eR', 'heat_eT', 'effi_eR', 'effi_eT', 'final_pose']):
                    pattern = r'(\w+)\s+([\d.e\-]+)'
                    for name, val in re.findall(pattern, line):
                        if name in val_logs:
                            val_logs[name].append((epoch, float(val)))
                    val_logs['epoch'].append(epoch)

    return train_logs, val_logs

def parse_json_metrics(result_dir):
    """Parse JSON metric files if available"""
    metrics = {'train': {}, 'val': {}}
    
    train_file = os.path.join(result_dir, 'train_metrics.json')
    val_file = os.path.join(result_dir, 'val_metrics.json')
    
    if os.path.exists(train_file):
        with open(train_file, 'r') as f:
            metrics['train'] = json.load(f)
    
    if os.path.exists(val_file):
        with open(val_file, 'r') as f:
            metrics['val'] = json.load(f)
    
    return metrics

def plot_loss_curves(log_dir, output_dir=None):
    """Plot training and validation loss curves"""
    
    log_file = find_log_file(log_dir)
    
    if not log_file or not os.path.exists(log_file):
        print(f"Log file not found in {log_dir}")
        print("Searched for train_rank0.log / train.log / training.log")
        return
    
    train_logs, val_logs = parse_log_file(log_file)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Metrics', fontsize=16)
    
    # Plot individual loss components
    losses = ['hmap', 'cls', 'box', 'pose', 'seg']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    ax_idx = 0
    for loss_name, color in zip(losses, colors):
        if loss_name in train_logs and train_logs[loss_name]:
            ax = axes.flat[ax_idx]
            ax.plot(train_logs['epoch'], train_logs[loss_name], 
                   marker='o', label=f'Train {loss_name}', color=color, linewidth=2)
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel(f'{loss_name.upper()} Loss', fontsize=11)
            ax.set_title(f'{loss_name.upper()} Loss Curve', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax_idx += 1
    
    # Hide unused subplots
    for i in range(ax_idx, 4):
        axes.flat[i].set_visible(False)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir or log_dir, 'loss_curves.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Loss curves saved to {output_file}")
    plt.close()

def plot_pose_error_curves(log_dir, output_dir=None):
    """Plot pose error (rotation and translation) iteration curves"""
    
    log_file = find_log_file(log_dir)
    
    if not log_file or not os.path.exists(log_file):
        print(f"Log file not found in {log_dir}")
        print("Searched for train_rank0.log / train.log / training.log")
        return
    
    train_logs, val_logs = parse_log_file(log_file)
    
    if not val_logs['epoch'] or not (val_logs['heat_eR'] or val_logs['effi_eR']):
        print("No validation pose error data found in log file")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Pose Error Metrics During Training', fontsize=16)
    
    # Rotation error
    if val_logs['heat_eR']:
        axes[0, 0].plot(val_logs['epoch'], val_logs['heat_eR'], 
                       marker='o', label='Heatmap eR', linewidth=2, color='#1f77b4')
    if val_logs['effi_eR']:
        axes[0, 0].plot(val_logs['epoch'], val_logs['effi_eR'], 
                       marker='s', label='EfficientPose eR', linewidth=2, color='#ff7f0e')
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Rotation Error (deg)', fontsize=11)
    axes[0, 0].set_title('Rotation Error', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Translation error
    if val_logs['heat_eT']:
        axes[0, 1].plot(val_logs['epoch'], val_logs['heat_eT'], 
                       marker='o', label='Heatmap eT', linewidth=2, color='#1f77b4')
    if val_logs['effi_eT']:
        axes[0, 1].plot(val_logs['epoch'], val_logs['effi_eT'], 
                       marker='s', label='EfficientPose eT', linewidth=2, color='#ff7f0e')
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Translation Error (m)', fontsize=11)
    axes[0, 1].set_title('Translation Error', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Segmentation IoU
    if val_logs['segm_iou']:
        axes[1, 0].plot(val_logs['epoch'], val_logs['segm_iou'], 
                       marker='o', label='Segmentation IoU', linewidth=2, color='#2ca02c')
        axes[1, 0].set_xlabel('Epoch', fontsize=11)
        axes[1, 0].set_ylabel('IoU Score', fontsize=11)
        axes[1, 0].set_title('Segmentation IoU', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
    
    axes[1, 1].set_visible(False)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir or log_dir, 'pose_error_curves.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Pose error curves saved to {output_file}")
    plt.close()

def plot_pose_error_scatter(metrics_file, output_dir=None):
    """Plot scatter plot of pose errors"""
    
    if not os.path.exists(metrics_file):
        print(f"Metrics file not found: {metrics_file}")
        return
    
    # Try to load metrics from different formats
    data = None
    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)
    except:
        try:
            data = pd.read_csv(metrics_file).to_dict(orient='list')
        except:
            print(f"Cannot parse metrics file: {metrics_file}")
            return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Pose Error Distribution', fontsize=16)
    
    # Rotation error scatter
    if 'heat_eR' in data or 'rotation_errors' in data:
        eR = data.get('heat_eR') or data.get('rotation_errors')
        axes[0].scatter(range(len(eR)), eR, alpha=0.6, s=30)
        axes[0].axhline(y=np.mean(eR), color='r', linestyle='--', label=f'Mean: {np.mean(eR):.2f}°')
        axes[0].set_xlabel('Sample Index', fontsize=11)
        axes[0].set_ylabel('Rotation Error (deg)', fontsize=11)
        axes[0].set_title('Rotation Error Distribution', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
    
    # Translation error scatter
    if 'heat_eT' in data or 'translation_errors' in data:
        eT = data.get('heat_eT') or data.get('translation_errors')
        axes[1].scatter(range(len(eT)), eT, alpha=0.6, s=30, color='orange')
        axes[1].axhline(y=np.mean(eT), color='r', linestyle='--', label=f'Mean: {np.mean(eT):.4f}m')
        axes[1].set_xlabel('Sample Index', fontsize=11)
        axes[1].set_ylabel('Translation Error (m)', fontsize=11)
        axes[1].set_title('Translation Error Distribution', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir or os.path.dirname(metrics_file), 'pose_error_scatter.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Pose error scatter plot saved to {output_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize training metrics and pose errors')
    
    parser.add_argument('--log-dir', required=True, type=str,
                        help='Directory containing training logs')
    parser.add_argument('--metrics-file', type=str, default=None,
                        help='Path to metrics JSON/CSV file for scatter plots')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for figures (default: same as log_dir)')
    
    args = parser.parse_args()
    
    output_dir = args.output_dir or args.log_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Log directory: {args.log_dir}")
    print(f"Output directory: {output_dir}")
    
    # Plot loss curves
    print("\nPlotting loss curves...")
    plot_loss_curves(args.log_dir, output_dir)
    
    # Plot pose error curves
    print("Plotting pose error curves...")
    plot_pose_error_curves(args.log_dir, output_dir)
    
    # Plot scatter plots
    if args.metrics_file:
        print("Plotting pose error scatter...")
        plot_pose_error_scatter(args.metrics_file, output_dir)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
