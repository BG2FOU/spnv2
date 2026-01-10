'''
Visualize evaluation results for non-cooperative target recognition and segmentation.

Includes:
1. Performance tables (LaTeX/Markdown)
2. Error distributions and histograms
3. Cross-domain performance plots
4. Scatter/box/CDF/radar visualizations
5. High-quality figures for publications
'''

import os
import sys
import json
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Use broadly available Latin fonts to avoid missing glyphs on headless servers
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (10, 6)

# seaborn style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)


class EvaluationVisualizer:
    """Evaluation result visualizer"""

    @staticmethod
    def _safe_metric(stats, key, subkey, default=0.0):
        """Return numeric metric; fallback to default if missing or non-numeric"""
        val = stats.get(key, {}).get(subkey, default)
        if isinstance(val, (int, float, np.floating)):
            return float(val)
        return default

    def __init__(self, results_path, output_dir='visualization_results'):
        """Initialize visualizer and load results"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        print(f"Loading evaluation results: {results_path}")
        with open(results_path, 'rb') as f:
            self.results = pickle.load(f)

        self.stats = self.results['statistics']
        self.colors = {
            'synthetic': '#3498db',
            'lightbox': '#e74c3c',
            'sunlamp': '#f39c12',
            'prisma25': '#2ecc71',
        }

    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("\nGenerating visualizations...")
        print("  [1/8] Generating performance tables...")
        self.generate_performance_tables()
        print("  [2/8] Plotting error histograms...")
        self.plot_error_distributions()
        print("  [3/8] Plotting cross-domain comparisons...")
        self.plot_cross_domain_comparison()
        print("  [4/8] Plotting error CDFs...")
        self.plot_error_cdf()
        print("  [5/8] Plotting performance heatmap...")
        self.plot_performance_heatmap()
        print("  [6/8] Plotting boxplots...")
        self.plot_error_boxplot()
        print("  [7/8] Plotting rotation vs translation scatter...")
        self.plot_rotation_vs_translation()
        print("  [8/8] Plotting radar chart...")
        self.plot_radar_chart()
        print(f"\nAll figures saved to: {self.output_dir}")

    def generate_performance_tables(self):
        """Generate performance tables (LaTeX and Markdown)"""

        # 检查是否有有效的分割数据
        has_segmentation = any(
            self.stats['per_domain'].get(d, {}).get('segmentation_iou', {}).get('mean', 0) > 0
            for d in self.stats['per_domain']
        )
        
        latex_path = os.path.join(self.output_dir, 'performance_table.tex')
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write("% 性能对比表 (姿态相关) / Performance comparison (pose)\n")
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{姿态指标 (Pose) — tools/test.py: heat\\_eR, heat\\_eT, effi\\_eR, effi\\_eT, final\\_pose}\n")
            f.write("\\label{tab:performance}\n")
            
            # 根据是否有分割数据动态调整表格列数
            if has_segmentation:
                f.write("\\begin{tabular}{lccccc}\n")
                f.write("\\toprule\n")
                f.write("域/Domain & 样本数 & 旋转误差 Rot($^\\circ$) & 平移误差 Trans(m) & Seg IoU & SPEED \\\n")
            else:
                f.write("\\begin{tabular}{lcccc}\n")
                f.write("\\toprule\n")
                f.write("域/Domain & 样本数 & 旋转误差 Rot($^\\circ$) & 平移误差 Trans(m) & SPEED \\\n")
            
            f.write("\\midrule\n")

            for domain, stats in self.stats['per_domain'].items():
                rot_err = self._safe_metric(stats, 'rotation_error', 'mean')
                rot_std = self._safe_metric(stats, 'rotation_error', 'std')
                trans_err = self._safe_metric(stats, 'translation_error', 'mean')
                trans_std = self._safe_metric(stats, 'translation_error', 'std')
                speed = self._safe_metric(stats, 'speed_score', 'mean')
                
                if has_segmentation:
                    iou = self._safe_metric(stats, 'segmentation_iou', 'mean')
                    f.write(
                        f"{domain} & {stats['count']} & "
                        f"{rot_err:.2f}$\\pm${rot_std:.2f} & "
                        f"{trans_err:.3f}$\\pm${trans_std:.3f} & "
                        f"{iou:.3f} & {speed:.3f} \\\n"
                    )
                else:
                    f.write(
                        f"{domain} & {stats['count']} & "
                        f"{rot_err:.2f}$\\pm${rot_std:.2f} & "
                        f"{trans_err:.3f}$\\pm${trans_std:.3f} & "
                        f"{speed:.3f} \\\n"
                    )

            overall = self.stats['overall']
            rot_err = self._safe_metric(overall, 'rotation_error', 'mean')
            rot_std = self._safe_metric(overall, 'rotation_error', 'std')
            trans_err = self._safe_metric(overall, 'translation_error', 'mean')
            trans_std = self._safe_metric(overall, 'translation_error', 'std')
            speed = self._safe_metric(overall, 'speed_score', 'mean')

            f.write("\\midrule\n")
            if has_segmentation:
                iou = self._safe_metric(overall, 'segmentation_iou', 'mean')
                f.write(
                    f"\\textbf{{Overall}} & {overall['count']} & "
                    f"\\textbf{{{rot_err:.2f}}}$\\pm${rot_std:.2f} & "
                    f"\\textbf{{{trans_err:.3f}}}$\\pm${trans_std:.3f} & "
                    f"\\textbf{{{iou:.3f}}} & \\textbf{{{speed:.3f}}} \\\n"
                )
            else:
                f.write(
                    f"\\textbf{{Overall}} & {overall['count']} & "
                    f"\\textbf{{{rot_err:.2f}}}$\\pm${rot_std:.2f} & "
                    f"\\textbf{{{trans_err:.3f}}}$\\pm${trans_std:.3f} & "
                    f"\\textbf{{{speed:.3f}}} \\\n"
                )
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

            # 光学特征与分割指标表（仅在有分割数据时生成）
            if has_segmentation:
                f.write("\n% 光学特征与分割指标 / Optical & Segmentation metrics\n")
                f.write("\\begin{table}[htbp]\n")
                f.write("\\centering\n")
                f.write("\\caption{光学特征 (heatmap) 与分割 (segmentation) — heat\\_eR, heat\\_eT, segm\\_iou 等}\n")
                f.write("\\label{tab:optical}\n")
                f.write("\\begin{tabular}{lccccccc}\n")
                f.write("\\toprule\n")
                f.write("域/Domain & 样本数 & 热图旋转 Rot($^\\circ$) & 热图平移 Trans(m) & 关键点召回 KptRec & Seg IoU & Seg F1 & Pixel Acc \\\n")
                f.write("\\midrule\n")

                for domain, stats in self.stats['per_domain'].items():
                    h_rot = self._safe_metric(stats, 'heat_rotation_error', 'mean')
                    h_rot_std = self._safe_metric(stats, 'heat_rotation_error', 'std')
                    h_trans = self._safe_metric(stats, 'heat_translation_error', 'mean')
                    h_trans_std = self._safe_metric(stats, 'heat_translation_error', 'std')
                    kpt_rate = self._safe_metric(stats, 'heat_kpt_rate', 'mean')
                    seg_iou = self._safe_metric(stats, 'segmentation_iou', 'mean')
                    seg_f1 = self._safe_metric(stats, 'segmentation_f1', 'mean')
                    seg_pix = self._safe_metric(stats, 'segmentation_pixel_accuracy', 'mean')
                    f.write(
                        f"{domain} & {stats['count']} & "
                        f"{h_rot:.2f}$\\pm${h_rot_std:.2f} & "
                        f"{h_trans:.3f}$\\pm${h_trans_std:.3f} & "
                        f"{kpt_rate:.3f} & {seg_iou:.3f} & {seg_f1:.3f} & {seg_pix:.3f} \\\n"
                    )

                h_rot = self._safe_metric(overall, 'heat_rotation_error', 'mean')
                h_rot_std = self._safe_metric(overall, 'heat_rotation_error', 'std')
                h_trans = self._safe_metric(overall, 'heat_translation_error', 'mean')
                h_trans_std = self._safe_metric(overall, 'heat_translation_error', 'std')
                kpt_rate = self._safe_metric(overall, 'heat_kpt_rate', 'mean')
                seg_iou = self._safe_metric(overall, 'segmentation_iou', 'mean')
                seg_f1 = self._safe_metric(overall, 'segmentation_f1', 'mean')
                seg_pix = self._safe_metric(overall, 'segmentation_pixel_accuracy', 'mean')

                f.write("\\midrule\n")
                f.write(
                    f"\\textbf{{Overall}} & {overall['count']} & "
                    f"\\textbf{{{h_rot:.2f}}}$\\pm${h_rot_std:.2f} & "
                    f"\\textbf{{{h_trans:.3f}}}$\\pm${h_trans_std:.3f} & "
                    f"\\textbf{{{kpt_rate:.3f}}} & \\textbf{{{seg_iou:.3f}}} & \\textbf{{{seg_f1:.3f}}} & \\textbf{{{seg_pix:.3f}}} \\\n"
                )
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n")

            h_rot = self._safe_metric(overall, 'heat_rotation_error', 'mean')
            h_rot_std = self._safe_metric(overall, 'heat_rotation_error', 'std')
            h_trans = self._safe_metric(overall, 'heat_translation_error', 'mean')
            h_trans_std = self._safe_metric(overall, 'heat_translation_error', 'std')
            kpt_rate = self._safe_metric(overall, 'heat_kpt_rate', 'mean')
            seg_iou = self._safe_metric(overall, 'segmentation_iou', 'mean')
            seg_f1 = self._safe_metric(overall, 'segmentation_f1', 'mean')
            seg_pix = self._safe_metric(overall, 'segmentation_pixel_accuracy', 'mean')

            f.write("\\midrule\n")
            f.write(
                f"\\textbf{{Overall}} & {overall['count']} & "
                f"\\textbf{{{h_rot:.2f}}}$\\pm${h_rot_std:.2f} & "
                f"\\textbf{{{h_trans:.3f}}}$\\pm${h_trans_std:.3f} & "
                f"\\textbf{{{kpt_rate:.3f}}} & \\textbf{{{seg_iou:.3f}}} & \\textbf{{{seg_f1:.3f}}} & \\textbf{{{seg_pix:.3f}}} \\\n"
            )
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        markdown_path = os.path.join(self.output_dir, 'performance_table.md')
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write("# 非合作靶场性能表 / Performance Tables\n\n")
            f.write("## 姿态指标 (Pose Metrics)\n\n")
            
            # 根据是否有分割数据动态调整表格
            if has_segmentation:
                f.write("| 域/Domain | 样本数 | 旋转误差 Rot (°) | 平移误差 Trans (m) | 分割 IoU | SPEED |\n")
                f.write("|:------|:------:|:-----------:|:-----------:|:-------:|:-----:|\n")

                for domain, stats in self.stats['per_domain'].items():
                    rot_err = self._safe_metric(stats, 'rotation_error', 'mean')
                    rot_std = self._safe_metric(stats, 'rotation_error', 'std')
                    trans_err = self._safe_metric(stats, 'translation_error', 'mean')
                    trans_std = self._safe_metric(stats, 'translation_error', 'std')
                    iou = self._safe_metric(stats, 'segmentation_iou', 'mean')
                    speed = self._safe_metric(stats, 'speed_score', 'mean')
                    f.write(
                        f"| {domain} | {stats['count']} | {rot_err:.2f} ± {rot_std:.2f} | "
                        f"{trans_err:.3f} ± {trans_std:.3f} | {iou:.3f} | {speed:.3f} |\n"
                    )
            else:
                f.write("| 域/Domain | 样本数 | 旋转误差 Rot (°) | 平移误差 Trans (m) | SPEED |\n")
                f.write("|:------|:------:|:-----------:|:-----------:|:-----:|\n")

                for domain, stats in self.stats['per_domain'].items():
                    rot_err = self._safe_metric(stats, 'rotation_error', 'mean')
                    rot_std = self._safe_metric(stats, 'rotation_error', 'std')
                    trans_err = self._safe_metric(stats, 'translation_error', 'mean')
                    trans_std = self._safe_metric(stats, 'translation_error', 'std')
                    speed = self._safe_metric(stats, 'speed_score', 'mean')
                    f.write(
                        f"| {domain} | {stats['count']} | {rot_err:.2f} ± {rot_std:.2f} | "
                        f"{trans_err:.3f} ± {trans_std:.3f} | {speed:.3f} |\n"
                    )

            rot_err = self._safe_metric(overall, 'rotation_error', 'mean')
            rot_std = self._safe_metric(overall, 'rotation_error', 'std')
            trans_err = self._safe_metric(overall, 'translation_error', 'mean')
            trans_std = self._safe_metric(overall, 'translation_error', 'std')
            speed = self._safe_metric(overall, 'speed_score', 'mean')
            
            if has_segmentation:
                iou = self._safe_metric(overall, 'segmentation_iou', 'mean')
                f.write(
                    f"| **Overall** | {overall['count']} | {rot_err:.2f} ± {rot_std:.2f} | "
                    f"{trans_err:.3f} ± {trans_std:.3f} | {iou:.3f} | {speed:.3f} |\n"
                )
            else:
                f.write(
                    f"| **Overall** | {overall['count']} | {rot_err:.2f} ± {rot_std:.2f} | "
                    f"{trans_err:.3f} ± {trans_std:.3f} | {speed:.3f} |\n"
                )

            f.write("\n## 成功率 / Success Rates\n\n")
            pose_success = self.stats['overall'].get('success_rate_pose', {}).get('mean', float('nan'))
            heat_success = self.stats['overall'].get('heat_kpt_rate', {}).get('mean', float('nan'))
            f.write(f"- 姿态 1cm/1°: {pose_success*100:.2f}%\n")
            f.write(f"- 热图关键点召回 (heat_kpt_rate): {heat_success*100:.2f}%\n")
            
            if has_segmentation:
                f.write("\n### 光学与分割指标 / Optical & Segmentation Metrics\n\n")
                f.write("| 域/Domain | 样本数 | 热图旋转 Rot (°) | 热图平移 Trans (m) | 关键点召回 heat_kpt_rate | Seg IoU | Seg F1 | Pixel Acc |\n")
                f.write("|:------|:------:|:-----------:|:-----------:|:-----------:|:-------:|:-------:|:----------:|\n")

                for domain, stats in self.stats['per_domain'].items():
                    h_rot = self._safe_metric(stats, 'heat_rotation_error', 'mean')
                    h_rot_std = self._safe_metric(stats, 'heat_rotation_error', 'std')
                    h_trans = self._safe_metric(stats, 'heat_translation_error', 'mean')
                    h_trans_std = self._safe_metric(stats, 'heat_translation_error', 'std')
                    kpt_rate = self._safe_metric(stats, 'heat_kpt_rate', 'mean')
                    seg_iou = self._safe_metric(stats, 'segmentation_iou', 'mean')
                    seg_f1 = self._safe_metric(stats, 'segmentation_f1', 'mean')
                    seg_pix = self._safe_metric(stats, 'segmentation_pixel_accuracy', 'mean')
                    f.write(
                        f"| {domain} | {stats['count']} | {h_rot:.2f} ± {h_rot_std:.2f} | "
                        f"{h_trans:.3f} ± {h_trans_std:.3f} | {kpt_rate:.3f} | {seg_iou:.3f} | {seg_f1:.3f} | {seg_pix:.3f} |\n"
                    )

                h_rot = self._safe_metric(overall, 'heat_rotation_error', 'mean')
                h_rot_std = self._safe_metric(overall, 'heat_rotation_error', 'std')
                h_trans = self._safe_metric(overall, 'heat_translation_error', 'mean')
                h_trans_std = self._safe_metric(overall, 'heat_translation_error', 'std')
                kpt_rate = self._safe_metric(overall, 'heat_kpt_rate', 'mean')
                seg_iou = self._safe_metric(overall, 'segmentation_iou', 'mean')
                seg_f1 = self._safe_metric(overall, 'segmentation_f1', 'mean')
                seg_pix = self._safe_metric(overall, 'segmentation_pixel_accuracy', 'mean')
                f.write(
                    f"| **Overall** | {overall['count']} | {h_rot:.2f} ± {h_rot_std:.2f} | "
                    f"{h_trans:.3f} ± {h_trans_std:.3f} | {kpt_rate:.3f} | {seg_iou:.3f} | {seg_f1:.3f} | {seg_pix:.3f} |\n"
                )
            else:
                f.write("- 分割 IoU (segmentation_iou): 无数据\n")
                f.write("\n### 热图指标 / Heatmap Metrics\n\n")
                f.write("| 域/Domain | 样本数 | 热图旋转 Rot (°) | 热图平移 Trans (m) | 关键点召回 heat_kpt_rate |\n")
                f.write("|:------|:------:|:-----------:|:-----------:|:-----------:|\n")

                for domain, stats in self.stats['per_domain'].items():
                    h_rot = self._safe_metric(stats, 'heat_rotation_error', 'mean')
                    h_rot_std = self._safe_metric(stats, 'heat_rotation_error', 'std')
                    h_trans = self._safe_metric(stats, 'heat_translation_error', 'mean')
                    h_trans_std = self._safe_metric(stats, 'heat_translation_error', 'std')
                    kpt_rate = self._safe_metric(stats, 'heat_kpt_rate', 'mean')
                    f.write(
                        f"| {domain} | {stats['count']} | {h_rot:.2f} ± {h_rot_std:.2f} | "
                        f"{h_trans:.3f} ± {h_trans_std:.3f} | {kpt_rate:.3f} |\n"
                    )

                h_rot = self._safe_metric(overall, 'heat_rotation_error', 'mean')
                h_rot_std = self._safe_metric(overall, 'heat_rotation_error', 'std')
                h_trans = self._safe_metric(overall, 'heat_translation_error', 'mean')
                h_trans_std = self._safe_metric(overall, 'heat_translation_error', 'std')
                kpt_rate = self._safe_metric(overall, 'heat_kpt_rate', 'mean')
                f.write(
                    f"| **Overall** | {overall['count']} | {h_rot:.2f} ± {h_rot_std:.2f} | "
                    f"{h_trans:.3f} ± {h_trans_std:.3f} | {kpt_rate:.3f} |\n"
                )

        print(f"    LaTeX table: {latex_path}")
        print(f"    Markdown table: {markdown_path}")
    
    def plot_error_distributions(self):
        """Plot error distribution histograms"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 收集各域的误差数据
        domains = list(self.stats['per_domain'].keys())
        
        # 旋转误差分布
        ax = axes[0, 0]
        for domain in domains:
            rot_errors = [
                sample.get('efficientpose', {}).get('rotation_error', np.nan)
                for sample in self.results['per_sample']
                if sample['domain'] == domain
            ]
            rot_errors = [e for e in rot_errors if not np.isnan(e)]
            
            if rot_errors:
                color = self.colors.get(domain, '#95a5a6')
                ax.hist(rot_errors, bins=50, alpha=0.6, label=domain, 
                       color=color, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Rotation Error (°)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('(a) Rotation Error Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 平移误差分布
        ax = axes[0, 1]
        for domain in domains:
            trans_errors = [
                sample.get('efficientpose', {}).get('translation_error', np.nan)
                for sample in self.results['per_sample']
                if sample['domain'] == domain
            ]
            trans_errors = [e for e in trans_errors if not np.isnan(e)]
            
            if trans_errors:
                color = self.colors.get(domain, '#95a5a6')
                ax.hist(trans_errors, bins=50, alpha=0.6, label=domain,
                       color=color, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Translation Error (m)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('(b) Translation Error Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # SPEED得分分布
        ax = axes[1, 0]
        for domain in domains:
            speed_scores = [
                sample.get('efficientpose', {}).get('speed_score', np.nan)
                for sample in self.results['per_sample']
                if sample['domain'] == domain
            ]
            speed_scores = [s for s in speed_scores if not np.isnan(s)]
            
            if speed_scores:
                color = self.colors.get(domain, '#95a5a6')
                ax.hist(speed_scores, bins=50, alpha=0.6, label=domain,
                       color=color, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('SPEED Score', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('(c) SPEED Score Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 分割IoU分布
        ax = axes[1, 1]
        for domain in domains:
            ious = [
                sample.get('segmentation', {}).get('iou', np.nan)
                for sample in self.results['per_sample']
                if sample['domain'] == domain
            ]
            ious = [iou for iou in ious if not np.isnan(iou)]
            
            if ious:
                color = self.colors.get(domain, '#95a5a6')
                ax.hist(ious, bins=50, alpha=0.6, label=domain,
                       color=color, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Segmentation IoU', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('(d) Segmentation IoU Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'error_distributions.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"    Error histograms: {output_path}")
    
    def plot_cross_domain_comparison(self):
        """Plot cross-domain comparison bar charts"""
        
        domains = list(self.stats['per_domain'].keys())
        x_pos = np.arange(len(domains))
        
        # 检测是否有分割数据
        has_segmentation = any(
            self._safe_metric(self.stats['per_domain'][d], 'segmentation_iou', 'mean') > 0
            for d in domains
        )
        
        # 根据是否有分割数据，调整子图数量
        n_rows = 2 if has_segmentation else 2
        n_cols = 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12))
        
        colors_list = [self.colors.get(d, '#95a5a6') for d in domains]
        
        # 旋转误差对比
        ax = axes[0, 0]
        rot_means = [self._safe_metric(self.stats['per_domain'][d], 'rotation_error', 'mean') 
                 for d in domains]
        rot_stds = [self._safe_metric(self.stats['per_domain'][d], 'rotation_error', 'std') 
                for d in domains]
        
        bars = ax.bar(x_pos, rot_means, yerr=rot_stds, capsize=5, 
                     color=colors_list, edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.set_ylabel('Rotation Error (°)', fontsize=12)
        ax.set_title('(a) Rotation Error Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(domains, rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, mean, std in zip(bars, rot_means, rot_stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std,
                   f'{mean:.2f}', ha='center', va='bottom', fontsize=10)
        
        # 平移误差对比
        ax = axes[0, 1]
        trans_means = [self._safe_metric(self.stats['per_domain'][d], 'translation_error', 'mean') 
                   for d in domains]
        trans_stds = [self._safe_metric(self.stats['per_domain'][d], 'translation_error', 'std') 
                  for d in domains]
        
        bars = ax.bar(x_pos, trans_means, yerr=trans_stds, capsize=5,
                     color=colors_list, edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.set_ylabel('Translation Error (m)', fontsize=12)
        ax.set_title('(b) Translation Error Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(domains, rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, mean, std in zip(bars, trans_means, trans_stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std,
                   f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
        
        if has_segmentation:
            # 分割IoU对比
            ax = axes[1, 0]
            iou_means = [self._safe_metric(self.stats['per_domain'][d], 'segmentation_iou', 'mean') 
                     for d in domains]
            iou_stds = [self._safe_metric(self.stats['per_domain'][d], 'segmentation_iou', 'std') 
                    for d in domains]
            
            bars = ax.bar(x_pos, iou_means, yerr=iou_stds, capsize=5,
                         color=colors_list, edgecolor='black', linewidth=1.5, alpha=0.8)
            ax.set_ylabel('Segmentation IoU', fontsize=12)
            ax.set_title('(c) Segmentation Performance', fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(domains, rotation=15, ha='right')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, mean in zip(bars, iou_means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
            
            # 成功率对比
            ax = axes[1, 1]
        else:
            # 没有分割数据时，成功率对比移到第二行第一列
            ax = axes[1, 0]
        
        detection_rates = [self.stats['per_domain'][d].get('success_rate', {}).get('detection', 0) 
                          for d in domains]
        acc_5deg = [self.stats['per_domain'][d].get('success_rate', {}).get('pose_accuracy_5deg', 0) 
                    for d in domains]
        
        width = 0.35
        bars1 = ax.bar(x_pos - width/2, detection_rates, width, label='Detection Success',
                      color='#3498db', edgecolor='black', linewidth=1.5, alpha=0.8)
        bars2 = ax.bar(x_pos + width/2, acc_5deg, width, label='<5° Accuracy',
                      color='#e74c3c', edgecolor='black', linewidth=1.5, alpha=0.8)
        
        ax.set_ylabel('Success Rate', fontsize=12)
        ax.set_title('(d) Detection Success & Accuracy' if has_segmentation else '(c) Detection Success & Accuracy', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(domains, rotation=15, ha='right')
        ax.set_ylim([0, 1.1])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加百分比标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1%}', ha='center', va='bottom', fontsize=9)
        
        # 如果没有分割数据，隐藏右下角的空子图
        if not has_segmentation:
            axes[1, 1].set_visible(False)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'cross_domain_comparison.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"    Cross-domain bars: {output_path}")
    
    def plot_error_cdf(self):
        """Plot error cumulative distribution functions (CDFs)"""
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        domains = list(self.stats['per_domain'].keys())
        
        # 旋转误差CDF
        ax = axes[0]
        for domain in domains:
            rot_errors = [
                sample.get('efficientpose', {}).get('rotation_error', np.nan)
                for sample in self.results['per_sample']
                if sample['domain'] == domain
            ]
            rot_errors = sorted([e for e in rot_errors if not np.isnan(e)])
            
            if rot_errors:
                cdf = np.arange(1, len(rot_errors) + 1) / len(rot_errors)
                color = self.colors.get(domain, '#95a5a6')
                ax.plot(rot_errors, cdf, linewidth=2.5, label=domain, color=color)
        
        # Reference lines
        ax.axvline(x=5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='5° threshold')
        ax.axvline(x=10, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='10° threshold')
        
        ax.set_xlabel('Rotation Error (°)', fontsize=12)
        ax.set_ylabel('Cumulative Probability', fontsize=12)
        ax.set_title('(a) Rotation Error CDF', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 30])
        
        # 平移误差CDF
        ax = axes[1]
        for domain in domains:
            trans_errors = [
                sample.get('efficientpose', {}).get('translation_error', np.nan)
                for sample in self.results['per_sample']
                if sample['domain'] == domain
            ]
            trans_errors = sorted([e for e in trans_errors if not np.isnan(e)])
            
            if trans_errors:
                cdf = np.arange(1, len(trans_errors) + 1) / len(trans_errors)
                color = self.colors.get(domain, '#95a5a6')
                ax.plot(trans_errors, cdf, linewidth=2.5, label=domain, color=color)
        
        # Reference lines
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='0.5m threshold')
        ax.axvline(x=1.0, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='1.0m threshold')
        
        ax.set_xlabel('Translation Error (m)', fontsize=12)
        ax.set_ylabel('Cumulative Probability', fontsize=12)
        ax.set_title('(b) Translation Error CDF', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'error_cdf.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"    Error CDFs: {output_path}")
    
    def plot_performance_heatmap(self):
        """Plot performance heatmap"""
        
        domains = list(self.stats['per_domain'].keys())
        
        # 检查是否有有效的分割数据
        has_segmentation = False
        for stats in self.stats['per_domain'].values():
            if stats.get('segmentation_iou', {}).get('mean', 0) > 0:
                has_segmentation = True
                break
        
        # 构建指标列表（仅包含有数据的指标）
        metrics = ['Rotation Error', 'Translation Error', 'SPEED Score', 'Detection Success']
        if has_segmentation:
            metrics.insert(2, 'Segmentation IoU')
        
        # 构建数据矩阵（归一化到0-1，值越大越好）
        data = []
        for domain in domains:
            stats = self.stats['per_domain'][domain]
            row = []
            
            # 旋转误差（越小越好，反转归一化）
            rot_err = self._safe_metric(stats, 'rotation_error', 'mean')
            row.append(1.0 - min(rot_err / 20.0, 1.0))
            
            # 平移误差（越小越好，反转归一化）
            trans_err = self._safe_metric(stats, 'translation_error', 'mean')
            row.append(1.0 - min(trans_err / 2.0, 1.0))
            
            # 分割IoU（若有数据）
            if has_segmentation:
                iou = self._safe_metric(stats, 'segmentation_iou', 'mean')
                row.append(iou)
            
            # SPEED得分（越小越好，反转归一化）
            speed = self._safe_metric(stats, 'speed_score', 'mean')
            row.append(1.0 - min(speed / 0.5, 1.0))
            
            # 检测成功率（越大越好）
            detection = stats.get('success_rate', {}).get('detection', 0)
            row.append(detection)
            
            data.append(row)
        
        data = np.array(data)
        
        # 绘制热图
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # 设置坐标轴
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(domains)))
        ax.set_xticklabels(metrics, rotation=15, ha='right')
        ax.set_yticklabels(domains)
        
        # 添加数值标签
        for i in range(len(domains)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{data[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=11)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Performance Score', rotation=270, labelpad=20, fontsize=12)
        
        ax.set_title('Performance Heatmap Across Domains', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'performance_heatmap.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"    Performance heatmap: {output_path}")
    
    def plot_error_boxplot(self):
        """Plot error boxplots"""
        
        domains = list(self.stats['per_domain'].keys())
        
        # 检测是否有分割数据
        has_segmentation = any(
            self._safe_metric(self.stats['per_domain'][d], 'segmentation_iou', 'mean') > 0
            for d in domains
        )
        
        # 根据是否有分割数据，调整列数
        n_cols = 3 if has_segmentation else 2
        fig, axes = plt.subplots(1, n_cols, figsize=(18 if has_segmentation else 12, 6))
        
        # 如果只有2列，需要确保axes是可迭代的
        if n_cols == 2 and not isinstance(axes, np.ndarray):
            axes = np.array([axes[0], axes[1]])
        
        # 旋转误差箱线图
        ax = axes[0]
        rot_data = []
        for domain in domains:
            rot_errors = [
                sample.get('efficientpose', {}).get('rotation_error', np.nan)
                for sample in self.results['per_sample']
                if sample['domain'] == domain
            ]
            rot_data.append([e for e in rot_errors if not np.isnan(e)])
        
        bp = ax.boxplot(rot_data, labels=domains, patch_artist=True,
                       medianprops=dict(color='red', linewidth=2),
                       boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=1.5),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5))
        
        # 设置颜色
        for patch, domain in zip(bp['boxes'], domains):
            patch.set_facecolor(self.colors.get(domain, '#95a5a6'))
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Rotation Error (°)', fontsize=12)
        ax.set_title('(a) Rotation Error Boxplot', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=15)
        
        # 平移误差箱线图
        ax = axes[1]
        trans_data = []
        for domain in domains:
            trans_errors = [
                sample.get('efficientpose', {}).get('translation_error', np.nan)
                for sample in self.results['per_sample']
                if sample['domain'] == domain
            ]
            trans_data.append([e for e in trans_errors if not np.isnan(e)])
        
        bp = ax.boxplot(trans_data, labels=domains, patch_artist=True,
                       medianprops=dict(color='red', linewidth=2),
                       boxprops=dict(facecolor='lightgreen', edgecolor='black', linewidth=1.5),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5))
        
        for patch, domain in zip(bp['boxes'], domains):
            patch.set_facecolor(self.colors.get(domain, '#95a5a6'))
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Translation Error (m)', fontsize=12)
        ax.set_title('(b) Translation Error Boxplot', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=15)
        
        # 分割IoU箱线图（仅当有数据时）
        if has_segmentation:
            ax = axes[2]
            iou_data = []
            for domain in domains:
                ious = [
                    sample.get('segmentation', {}).get('iou', np.nan)
                    for sample in self.results['per_sample']
                    if sample['domain'] == domain
                ]
                iou_data.append([iou for iou in ious if not np.isnan(iou)])
            
            bp = ax.boxplot(iou_data, labels=domains, patch_artist=True,
                           medianprops=dict(color='red', linewidth=2),
                           boxprops=dict(facecolor='lightyellow', edgecolor='black', linewidth=1.5),
                           whiskerprops=dict(linewidth=1.5),
                           capprops=dict(linewidth=1.5))
            
            for patch, domain in zip(bp['boxes'], domains):
                patch.set_facecolor(self.colors.get(domain, '#95a5a6'))
                patch.set_alpha(0.7)
            
            ax.set_ylabel('Segmentation IoU', fontsize=12)
            ax.set_title('(c) Segmentation IoU Boxplot', fontsize=14, fontweight='bold')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=15)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'error_boxplot.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"    Error boxplots: {output_path}")
    
    def plot_rotation_vs_translation(self):
        """Plot rotation vs translation error scatter"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        domains = list(self.stats['per_domain'].keys())
        
        for domain in domains:
            rot_errors = []
            trans_errors = []
            
            for sample in self.results['per_sample']:
                if sample['domain'] == domain:
                    rot_err = sample.get('efficientpose', {}).get('rotation_error', np.nan)
                    trans_err = sample.get('efficientpose', {}).get('translation_error', np.nan)
                    
                    if not np.isnan(rot_err) and not np.isnan(trans_err):
                        rot_errors.append(rot_err)
                        trans_errors.append(trans_err)
            
            if rot_errors and trans_errors:
                color = self.colors.get(domain, '#95a5a6')
                ax.scatter(rot_errors, trans_errors, alpha=0.6, s=50,
                          label=domain, color=color, edgecolors='black', linewidth=0.5)
        
        # Reference lines
        ax.axvline(x=5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='5° threshold')
        ax.axhline(y=0.5, color='blue', linestyle='--', linewidth=2, alpha=0.5, label='0.5m threshold')
        
        ax.set_xlabel('Rotation Error (°)', fontsize=12)
        ax.set_ylabel('Translation Error (m)', fontsize=12)
        ax.set_title('Rotation vs Translation Error', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'rotation_vs_translation.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"    Scatter plot: {output_path}")
    
    def plot_radar_chart(self):
        """Plot radar chart for overall performance"""
        
        domains = list(self.stats['per_domain'].keys())
        
        # 检测是否有分割数据
        has_segmentation = any(
            self._safe_metric(self.stats['per_domain'][d], 'segmentation_iou', 'mean') > 0
            for d in domains
        )
        
        # 根据是否有分割数据，调整雷达图指标
        if has_segmentation:
            categories = ['Rot Accuracy', 'Trans Accuracy', 'Seg Accuracy', 'Detection Success', 'SPEED']
        else:
            categories = ['Rot Accuracy', 'Trans Accuracy', 'Detection Success', 'SPEED']
        
        num_vars = len(categories)
        
        # 计算角度
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for domain in domains:
            stats = self.stats['per_domain'][domain]
            
            # 构建雷达图数据（归一化到0-1，值越大越好）
            values = []
            
            # 旋转精度（反转归一化）
            rot_err = self._safe_metric(stats, 'rotation_error', 'mean')
            values.append(1.0 - min(rot_err / 20.0, 1.0))
            
            # 平移精度（反转归一化）
            trans_err = self._safe_metric(stats, 'translation_error', 'mean')
            values.append(1.0 - min(trans_err / 2.0, 1.0))
            
            # 分割精度（仅当有数据时）
            if has_segmentation:
                iou = self._safe_metric(stats, 'segmentation_iou', 'mean')
                values.append(iou)
            
            # 检测成功率
            detection = stats.get('success_rate', {}).get('detection', 0)
            values.append(detection)
            
            # SPEED性能（反转归一化）
            speed = self._safe_metric(stats, 'speed_score', 'mean')
            values.append(1.0 - min(speed / 0.5, 1.0))
            
            values += values[:1]
            
            color = self.colors.get(domain, '#95a5a6')
            ax.plot(angles, values, 'o-', linewidth=2.5, label=domain, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        ax.grid(True)
        ax.set_title('Radar: Overall Performance per Domain', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'radar_chart.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"    Radar chart: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize evaluation tables and plots for the task'
    )
    
    parser.add_argument('--results', required=True, type=str,
                       help='Path to evaluation results (pickle)')
    parser.add_argument('--output-dir', type=str,
                       default='visualization_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Check results file
    if not os.path.exists(args.results):
        print(f"Error: results file not found {args.results}")
        print("Run generate_evaluation_results.py first to produce results")
        return
    
    # Create visualizer
    visualizer = EvaluationVisualizer(args.results, args.output_dir)
    
    # Generate all visualizations
    visualizer.generate_all_visualizations()
    
    print("\nVisualization done!")


if __name__ == '__main__':
    main()
