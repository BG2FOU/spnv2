'''
Advanced training metrics visualization for SPNv2

Parses training logs and generates visualization plots:
1. Loss curves (hmap, cls, box, pose, seg)
2. Validation metrics curves (heat_eR, effi_eT, final_pose)
3. Error distribution histograms

All figures are saved to disk (no display for headless servers)
'''

import os
import re
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless servers
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from collections import defaultdict

class LogParser:
    """Parse SPNv2 training logs with format:
    timestamp * Time: XXX ms hmap Y cls Z box W pose V seg U
    timestamp * Time: XXX ms heat_eR Y deg effi_eT Z m final_pose W
    """
    
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.data = defaultdict(list)
    
    def parse_text_log(self, log_file):
        """Parse SPNv2 training log format"""
        
        logs = {
            'train': defaultdict(list),
            'val': defaultdict(list)
        }
        
        epoch = 0
        
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Detect epoch changes (appears once per epoch)
                if 'Current epoch learning rate' in line:
                    epoch += 1
                    continue
                
                # Training losses
                if '* Time:' in line and any(x in line for x in ['hmap', 'cls', 'box', 'pose', 'seg']):
                    pattern = r'(\w+)\s+([\d.e\-]+)'
                    matches = re.findall(pattern, line)
                    for name, val in matches:
                        if name in ['hmap', 'cls', 'box', 'pose', 'seg']:
                            logs['train'][name].append((epoch, float(val)))
                
                # Validation metrics
                if '* Time:' in line and any(x in line for x in ['heat_eR', 'heat_eT', 'effi_eR', 'effi_eT', 'final_pose']):
                    pattern = r'(\w+)\s+([\d.e\-]+)'
                    matches = re.findall(pattern, line)
                    for name, val in matches:
                        if name in ['heat_eR', 'heat_eT', 'effi_eR', 'effi_eT', 'final_pose']:
                            logs['val'][name].append((epoch, float(val)))
        
        return logs

class MetricsVisualizer:
    """Visualize training metrics - all figures saved to disk"""
    
    def __init__(self, logs, output_dir='./figures'):
        self.logs = logs
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.colors = {
            'hmap': '#1f77b4', 'cls': '#ff7f0e', 'box': '#2ca02c', 
            'pose': '#d62728', 'seg': '#9467bd',
            'heat_eR': '#1f77b4', 'effi_eT': '#ff7f0e', 'final_pose': '#2ca02c'
        }
    
    def plot_train_loss(self):
        """Plot training loss curves"""
        
        train_logs = self.logs.get('train', {})
        if not train_logs:
            print("No training logs found")
            return
        
        loss_keys = ['hmap', 'cls', 'box', 'pose', 'seg']
        loss_keys = [k for k in loss_keys if k in train_logs and train_logs[k]]
        
        if not loss_keys:
            return
        
        ncols = min(3, len(loss_keys))
        nrows = (len(loss_keys) + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        if len(loss_keys) == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, loss_key in enumerate(loss_keys):
            ax = axes[idx]
            values = train_logs[loss_key]
            
            if values and len(values) > 0:
                epochs, losses = zip(*values)
                ax.plot(epochs, losses, marker='o', linewidth=2, markersize=5, 
                       color=self.colors.get(loss_key, '#1f77b4'))
                ax.set_xlabel('Epoch', fontsize=11)
                ax.set_ylabel('Loss', fontsize=11)
                ax.set_title(f'{loss_key.upper()} Loss', fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Hide unused subplots
        for idx in range(len(loss_keys), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, 'train_loss.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
    
    def plot_val_metrics(self):
        """Plot validation metrics"""
        
        val_logs = self.logs.get('val', {})
        if not val_logs:
            print("No validation logs found")
            return
        
        # Count available metrics
        available_metrics = []
        if any(k in val_logs and val_logs[k] for k in ['heat_eR', 'effi_eR']):
            available_metrics.append('rotation')
        if any(k in val_logs and val_logs[k] for k in ['heat_eT', 'effi_eT']):
            available_metrics.append('translation')
        if 'final_pose' in val_logs and val_logs['final_pose']:
            available_metrics.append('final_pose')
        
        if not available_metrics:
            print("No validation metrics to plot")
            return
        
        # Create flexible layout: 1 row with n columns + 1 for summary
        ncols = len(available_metrics) + 1
        fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 5))
        fig.suptitle('Validation Metrics', fontsize=14, fontweight='bold')
        
        if ncols == 1:
            axes = [axes]
        
        col_idx = 0
        
        # Rotation error
        if 'rotation' in available_metrics:
            ax = axes[col_idx]
            for key in ['heat_eR', 'effi_eR']:
                if key in val_logs and val_logs[key]:
                    epochs, vals = zip(*val_logs[key])
                    ax.plot(epochs, vals, marker='o', label=key, linewidth=2, 
                           color=self.colors.get(key, '#1f77b4'))
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Rotation Error (deg)', fontsize=11)
            ax.set_title('Rotation Error', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            col_idx += 1
        
        # Translation error
        if 'translation' in available_metrics:
            ax = axes[col_idx]
            for key in ['heat_eT', 'effi_eT']:
                if key in val_logs and val_logs[key]:
                    epochs, vals = zip(*val_logs[key])
                    ax.plot(epochs, vals, marker='s', label=key, linewidth=2,
                           color=self.colors.get(key, '#ff7f0e'))
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Translation Error (m)', fontsize=11)
            ax.set_title('Translation Error', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            col_idx += 1
        
        # Final pose
        if 'final_pose' in available_metrics:
            ax = axes[col_idx]
            epochs, vals = zip(*val_logs['final_pose'])
            ax.plot(epochs, vals, marker='^', linewidth=2, markersize=6,
                   color=self.colors.get('final_pose', '#2ca02c'))
            ax.fill_between(epochs, vals, alpha=0.3, color=self.colors.get('final_pose', '#2ca02c'))
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Final Pose Score', fontsize=11)
            ax.set_title('Final Pose Score', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            col_idx += 1
        
        # Summary stats
        ax = axes[col_idx]
        ax.axis('off')
        
        summary_text = "Best Metrics:\n\n"
        all_metrics = {}
        
        for key, values in val_logs.items():
            if not values or len(values) == 0:
                continue
            # Skip if not tuple format (safety check)
            if not isinstance(values[0], tuple):
                continue
            vals = [v for _, v in values]
            # All validation metrics should minimize (errors and scores)
            is_minimize = any(x in key.lower() for x in ['er', 'et', 'error', 'pose', 'loss'])
            best_val = min(vals) if is_minimize else max(vals)
            best_epoch = values[np.argmin(vals) if is_minimize else np.argmax(vals)][0]
            all_metrics[key] = (best_val, best_epoch)
            summary_text += f"{key}: {best_val:.4f} (Epoch {best_epoch})\n"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, 'val_metrics.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
    
    def plot_all_metrics_combined(self):
        """Plot train loss and val metrics side by side"""
        
        train_logs = self.logs.get('train', {})
        val_logs = self.logs.get('val', {})
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Training Progress Summary', fontsize=14, fontweight='bold')
        
        # Row 1: Training losses
        train_loss_keys = ['hmap', 'cls', 'box', 'pose', 'seg']
        train_loss_keys = [k for k in train_loss_keys if k in train_logs and train_logs[k]]
        
        for idx, loss_key in enumerate(train_loss_keys[:3]):
            ax = axes[0, idx]
            values = train_logs[loss_key]
            epochs, losses = zip(*values)
            ax.semilogy(epochs, losses, marker='o', linewidth=2, markersize=4,
                       color=self.colors.get(loss_key, '#1f77b4'))
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Loss (log)', fontsize=10)
            ax.set_title(f'Train {loss_key.upper()}', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Hide unused training subplots
        for idx in range(len(train_loss_keys[:3]), 3):
            axes[0, idx].set_visible(False)
        
        # Row 2: Validation metrics
        val_metric_keys = ['heat_eR', 'effi_eT', 'final_pose']
        val_metric_keys = [k for k in val_metric_keys if k in val_logs and val_logs[k]]
        
        for idx, metric_key in enumerate(val_metric_keys[:3]):
            ax = axes[1, idx]
            epochs, vals = zip(*val_logs[metric_key])
            ax.plot(epochs, vals, marker='o', linewidth=2, markersize=4,
                   color=self.colors.get(metric_key, '#2ca02c'))
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Metric Value', fontsize=10)
            ax.set_title(f'Val {metric_key}', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Hide unused validation subplots
        for idx in range(len(val_metric_keys), 3):
            axes[1, idx].set_visible(False)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, 'training_summary.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
    
    def plot_performance_heatmap(self):
        """Plot training metrics heatmap over epochs"""
        
        # Collect all metrics (train + val)
        all_metrics = {}
        train_logs = self.logs.get('train', {})
        val_logs = self.logs.get('val', {})
        
        # Combine metrics
        for key, values in train_logs.items():
            if values:
                all_metrics[f'train_{key}'] = values
        for key, values in val_logs.items():
            if values:
                all_metrics[f'val_{key}'] = values
        
        if not all_metrics:
            print("No metrics for heatmap")
            return
        
        # Get all unique epochs
        all_epochs = sorted(set(
            epoch for values in all_metrics.values() for epoch, _ in values
        ))
        
        if len(all_epochs) < 2:
            print("Not enough epochs for heatmap")
            return
        
        # Build data matrix (normalize each metric to 0-1 for better visualization)
        metric_names = list(all_metrics.keys())
        data = np.zeros((len(metric_names), len(all_epochs)))
        
        for i, metric_name in enumerate(metric_names):
            epoch_dict = {e: v for e, v in all_metrics[metric_name]}
            vals = [epoch_dict.get(e, np.nan) for e in all_epochs]
            
            # Normalize to 0-1 (for losses: lower is better, so invert)
            vals_array = np.array(vals)
            valid_mask = ~np.isnan(vals_array)
            if valid_mask.any():
                vmin, vmax = np.nanmin(vals_array), np.nanmax(vals_array)
                if vmax > vmin:
                    normalized = (vals_array - vmin) / (vmax - vmin)
                    # For losses and errors (smaller is better), invert normalization
                    if any(x in metric_name.lower() for x in ['loss', 'error', 'er', 'et']):
                        normalized = 1.0 - normalized
                    vals_array = normalized
            
            data[i, :] = vals_array
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(max(12, len(all_epochs) * 0.4), max(8, len(metric_names) * 0.5)))
        
        # Use masked array to handle NaN values
        masked_data = np.ma.masked_invalid(data)
        im = ax.imshow(masked_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks
        ax.set_yticks(np.arange(len(metric_names)))
        ax.set_yticklabels(metric_names, fontsize=10)
        
        # Show every Nth epoch to avoid crowding
        epoch_stride = max(1, len(all_epochs) // 20)
        tick_indices = list(range(0, len(all_epochs), epoch_stride))
        if tick_indices[-1] != len(all_epochs) - 1:
            tick_indices.append(len(all_epochs) - 1)
        
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([all_epochs[i] for i in tick_indices], rotation=45, ha='right', fontsize=9)
        ax.set_xlabel('Epoch', fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Performance (higher=better)', rotation=270, labelpad=20, fontsize=11)
        
        ax.set_title('Training Metrics Heatmap', fontsize=14, fontweight='bold', pad=15)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, 'performance_heatmap.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
    
    def plot_radar_chart(self):
        """Plot radar chart for final training performance"""
        
        train_logs = self.logs.get('train', {})
        val_logs = self.logs.get('val', {})
        
        # Collect final epoch metrics (best or last)
        categories = []
        values = []
        
        # For training losses: use minimum (best) value
        for key in ['hmap', 'cls', 'box', 'pose', 'seg']:
            if key in train_logs and train_logs[key]:
                epochs, losses = zip(*train_logs[key])
                min_loss = min(losses)
                max_loss = max(losses)
                # Normalize and invert (lower loss = better = higher score)
                if max_loss > min_loss:
                    score = 1.0 - (min_loss - min_loss) / (max_loss - min_loss + 1e-8)
                    score = max(0.3, score)  # Give at least 30% for having the metric
                else:
                    score = 0.5
                categories.append(f'{key} Loss')
                values.append(score)
        
        # For validation metrics: use best value
        val_metrics_config = [
            ('heat_eR', 'Rot Error (Heat)', True),   # lower is better
            ('effi_eR', 'Rot Error (Effi)', True),   # lower is better
            ('heat_eT', 'Trans Error (Heat)', True), # lower is better
            ('effi_eT', 'Trans Error (Effi)', True), # lower is better
            ('final_pose', 'Final Pose', True),      # lower is better
        ]
        
        for key, label, lower_is_better in val_metrics_config:
            if key in val_logs and val_logs[key]:
                epochs, vals = zip(*val_logs[key])
                if lower_is_better:
                    best_val = min(vals)
                    worst_val = max(vals)
                    # Normalize and invert
                    if worst_val > best_val:
                        score = 1.0 - (best_val - best_val) / (worst_val - best_val + 1e-8)
                    else:
                        score = 0.5
                else:
                    best_val = max(vals)
                    worst_val = min(vals)
                    if best_val > worst_val:
                        score = (best_val - worst_val) / (best_val - worst_val + 1e-8)
                    else:
                        score = 0.5
                
                categories.append(label)
                values.append(max(0.1, min(1.0, score)))
        
        if len(categories) < 3:
            print("Not enough metrics for radar chart")
            return
        
        num_vars = len(categories)
        
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        ax.plot(angles, values, 'o-', linewidth=2.5, label='Training Performance', color='#2c3e50')
        ax.fill(angles, values, alpha=0.25, color='#3498db')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        ax.grid(True)
        ax.set_title('Training Performance Radar Chart', fontsize=16, fontweight='bold', pad=30)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, 'radar_chart.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
    
    def visualize_all(self):
        """Generate all visualizations"""
        print(f"\n📊 Generating visualizations in {self.output_dir}\n")
        self.plot_train_loss()
        self.plot_val_metrics()
        self.plot_all_metrics_combined()
        self.plot_performance_heatmap()
        self.plot_radar_chart()
        print("\n✅ All visualizations complete!")

def main():
    parser = argparse.ArgumentParser(description='Visualize SPNv2 training metrics')
    parser.add_argument('--log-dir', required=True, type=str,
                       help='Training log directory (contains train_rank0.log). For multi-part training, use comma-separated paths: dir1,dir2')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for figures (default: first log_dir/figures)')
    
    args = parser.parse_args()
    
    # Support multiple log directories (comma-separated)
    log_dirs = [d.strip() for d in args.log_dir.split(',')]
    output_dir = args.output_dir or os.path.join(log_dirs[0], 'figures')
    
    # Find log files for all directories
    log_files = []
    for log_dir in log_dirs:
        log_file = None
        for pattern in ['train_rank0.log', 'train.log', 'training.log']:
            candidate = os.path.join(log_dir, pattern)
            if os.path.exists(candidate):
                log_file = candidate
                break
        
        if not log_file:
            log_dir_path = Path(log_dir)
            if log_dir_path.is_dir():
                for pattern in ['**/train_rank0.log', '**/train.log', '**/training.log']:
                    matches = list(log_dir_path.glob(pattern))
                    if matches:
                        log_file = str(matches[0])
                        break
        
        if not log_file or not os.path.exists(log_file):
            print(f"❌ Log file not found in {log_dir}")
            print("   Searched for: train_rank0.log, train.log, training.log")
            continue
        
        log_files.append(log_file)
    
    if not log_files:
        print(f"❌ No valid log files found")
        return
    
    print(f"📂 Log directories: {', '.join(log_dirs)}")
    print(f"📄 Log files: {', '.join(log_files)}")
    print(f"📁 Output directory: {output_dir}\n")
    
    # Parse and merge logs from all files (with epoch offset to avoid overlap)
    print("Parsing training logs...")
    merged_logs = {'train': defaultdict(list), 'val': defaultdict(list)}
    current_max_epoch = -1

    for log_file in log_files:
        parser_obj = LogParser(os.path.dirname(log_file))
        logs = parser_obj.parse_text_log(log_file)

        # Determine offset so epochs from later logs continue
        # Find max epoch in current merged state
        offset = current_max_epoch + 1 if current_max_epoch >= 0 else 0

        # Shift epochs in this log and merge
        for key, values in logs['train'].items():
            shifted = []
            for e, v in values:
                shifted.append((e + offset, v))
            merged_logs['train'][key].extend(shifted)
            if shifted:
                current_max_epoch = max(current_max_epoch, max(e for e, _ in shifted))

        for key, values in logs['val'].items():
            shifted = []
            for e, v in values:
                shifted.append((e + offset, v))
            merged_logs['val'][key].extend(shifted)
            if shifted:
                current_max_epoch = max(current_max_epoch, max(e for e, _ in shifted))

    # Sort by epoch to ensure correct order (preserve duplicates order)
    for key in merged_logs['train']:
        if merged_logs['train'][key] and isinstance(merged_logs['train'][key][0], tuple):
            merged_logs['train'][key] = sorted(merged_logs['train'][key], key=lambda x: x[0])

    for key in merged_logs['val']:
        if merged_logs['val'][key] and isinstance(merged_logs['val'][key][0], tuple):
            merged_logs['val'][key] = sorted(merged_logs['val'][key], key=lambda x: x[0])
    
    # Print summary
    print(f"  Train data points:")
    for key, vals in merged_logs['train'].items():
        if vals:
            print(f"    - {key}: {len(vals)} epochs")
    print(f"  Validation data points:")
    for key, vals in merged_logs['val'].items():
        if vals:
            print(f"    - {key}: {len(vals)} epochs")
    
    # Visualize
    visualizer = MetricsVisualizer(merged_logs, output_dir)
    visualizer.visualize_all()

if __name__ == '__main__':
    main()
