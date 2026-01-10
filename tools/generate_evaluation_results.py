'''
生成空间非合作目标识别与分割任务的综合评估结果

功能：
1. 详细性能指标计算与保存
2. 不同场景/域的对比分析
3. 典型成功与失败案例提取
4. 部件级别识别精度统计
5. 生成可用于论文的结果汇总
'''

import os
import sys
import json
import pickle
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
import copy
import torch
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import _init_paths

from config import cfg, update_config
from nets import build_spnv2
from dataset import get_dataloader
from utils.utils import load_camera_intrinsics, load_tango_3d_keypoints
from utils.metrics import *
from utils.postprocess import solve_pose_from_heatmaps, rot_6d_to_matrix


class ComprehensiveEvaluator:
    """空间目标识别任务的综合评估器"""
    
    def __init__(self, cfg, model, device):
        self.cfg = cfg
        self.model = model
        self.device = device
        
        # 加载相机参数和关键点
        self.camera = load_camera_intrinsics(cfg.DATASET.CAMERA)
        self.keypts_true_3D = load_tango_3d_keypoints(cfg.DATASET.KEYPOINTS)
        
        # 结果存储
        self.results = {
            'per_sample': [],  # 每个样本的详细结果
            'per_domain': defaultdict(list),  # 按域分类的结果
            'best_cases': [],  # 最佳案例
            'worst_cases': [],  # 最差案例
            'statistics': {},  # 统计信息
        }
    
    def evaluate_dataset(self, data_loader, output_dir, max_samples=None, compute_and_save=True):
        """对整个数据集进行评估; 当 compute_and_save=False 时仅累计结果"""
        
        self.model.eval()
        print(f"\n开始评估，共 {len(data_loader)} 个批次...")
        
        with torch.no_grad():
            for idx, (images, targets) in enumerate(data_loader):
                if max_samples and idx >= max_samples:
                    break
                
                # 将数据移至GPU
                images = images.to(self.device, non_blocking=True)
                
                # 推理
                result = self._evaluate_single_sample(images, targets)
                
                # 保存结果
                self.results['per_sample'].append(result)
                self.results['per_domain'][result['domain']].append(result)
                
                # 进度显示
                if (idx + 1) % 50 == 0:
                    print(f"已处理: {idx + 1}/{len(data_loader)}")
        
        if compute_and_save:
            self._compute_statistics()
            self._identify_representative_cases()
            self._save_results(output_dir)
        
        return self.results

    def finalize(self, output_dir):
        """在多域累计完成后统一计算统计并保存"""
        self._compute_statistics()
        self._identify_representative_cases()
        self._save_results(output_dir)
        return self.results
    
    def _evaluate_single_sample(self, images, targets):
        """评估单个样本"""
        
        # 基本信息
        img_name_raw = targets.get('image_name', ['unknown'])
        if isinstance(img_name_raw, (list, tuple)):
            img_name = img_name_raw[0]
        else:
            img_name = img_name_raw

        sample_result = {
            'image_name': img_name,
            'domain': targets['domain'][0],
            'q_gt': targets['quaternion'][0].cpu().numpy(),
            'R_gt': targets['rotationmatrix'][0].cpu().numpy(),
            't_gt': targets['translation'][0].cpu().numpy(),
        }
        
        if 'boundingbox' in targets:
            sample_result['bbox_gt'] = targets['boundingbox'][0].cpu().numpy()
        if 'mask' in targets:
            sample_result['mask_gt'] = targets['mask'][0].cpu().numpy()
        
        # 前向推理
        outputs = self.model(images, is_train=False, gpu=self.device)
        
        # 评估各个任务头
        for i, head_name in enumerate(self.cfg.TEST.HEAD):
            head_results = {}
            
            # 跳过 segmentation 头部评估，如果没有 mask_gt 数据
            if head_name == 'segmentation' and 'mask_gt' not in sample_result:
                continue
            
            if head_name == 'heatmap':
                head_results = self._evaluate_heatmap_head(
                    outputs[i].squeeze(0).cpu(),
                    sample_result['q_gt'],
                    sample_result['t_gt'],
                    sample_result['domain']
                )
            
            elif head_name == 'efficientpose':
                head_results = self._evaluate_efficientpose_head(
                    outputs[i],
                    sample_result.get('bbox_gt'),
                    sample_result['R_gt'],
                    sample_result['t_gt'],
                    sample_result['domain']
                )
            
            elif head_name == 'segmentation':
                head_results = self._evaluate_segmentation_head(
                    outputs[i],
                    sample_result.get('mask_gt')
                )
            
            sample_result[head_name] = head_results
        
        # 计算最终综合得分
        sample_result['final_score'] = self._compute_final_score(sample_result)
        
        return sample_result
    
    def _evaluate_heatmap_head(self, heatmap, q_gt, t_gt, domain):
        """评估热图头部性能"""
        
        keypts_pr, q_pr, t_pr, reject = solve_pose_from_heatmaps(
            heatmap,
            self.cfg.DATASET.IMAGE_SIZE,
            self.cfg.TEST.HEATMAP_THRESHOLD,
            self.camera,
            self.keypts_true_3D
        )
        
        result = {
            'rejected': reject,
            'keypoints_predicted': keypts_pr,
            'q_predicted': q_pr,
            't_predicted': t_pr,
        }
        
        if not reject:
            # 姿态误差
            result['rotation_error'] = error_orientation(q_pr, q_gt, 'quaternion')
            result['translation_error'] = error_translation(t_pr, t_gt)
            
            # SPEED评分
            speed_t, speed_q, speed = speed_score(
                t_pr, q_pr, t_gt, q_gt,
                representation='quaternion',
                applyThreshold=domain in ['lightbox', 'sunlamp'],
                theta_q=self.cfg.TEST.SPEED_THRESHOLD_Q,
                theta_t=self.cfg.TEST.SPEED_THRESHOLD_T
            )
            result['speed_t'] = speed_t
            result['speed_q'] = speed_q
            result['speed_score'] = speed
            
            # 关键点检测准确率
            result['keypoint_detection_rate'] = self._compute_keypoint_detection_rate(
                keypts_pr, heatmap
            )
        
        return result
    
    def _evaluate_efficientpose_head(self, outputs, bbox_gt, R_gt, t_gt, domain):
        """评估EfficientPose头部性能"""
        
        classification, bbox_prediction, rotation_raw, translation = outputs
        _, cls_argmax = torch.max(classification, dim=1)
        
        # 预测结果
        bbox_pr = bbox_prediction[0, cls_argmax].squeeze().cpu().numpy()
        R_pr = rot_6d_to_matrix(rotation_raw[0, cls_argmax, :]).squeeze().cpu().numpy()
        t_pr = translation[0, cls_argmax].squeeze().cpu().numpy()
        
        result = {
            'bbox_predicted': bbox_pr,
            'R_predicted': R_pr,
            't_predicted': t_pr,
            'classification_confidence': torch.max(classification).item(),
        }
        
        # 边界框IoU
        if bbox_gt is not None:
            result['bbox_iou'] = float(np.squeeze(bbox_iou(bbox_pr, bbox_gt, x1y1x2y2=True)))
        
        # 姿态误差
        result['rotation_error'] = error_orientation(R_pr, R_gt, 'rotationmatrix')
        result['translation_error'] = error_translation(t_pr, t_gt)
        
        # SPEED评分
        speed_t, speed_q, speed = speed_score(
            t_pr, R_pr, t_gt, R_gt,
            representation='rotationmatrix',
            applyThreshold=domain in ['lightbox', 'sunlamp'],
            theta_q=self.cfg.TEST.SPEED_THRESHOLD_Q,
            theta_t=self.cfg.TEST.SPEED_THRESHOLD_T
        )
        result['speed_t'] = speed_t
        result['speed_q'] = speed_q
        result['speed_score'] = speed
        
        return result
    
    def _evaluate_segmentation_head(self, mask_output, mask_gt):
        """评估分割头部性能"""
        
        mask_pr = mask_output.sigmoid().cpu().numpy()
        mask_binary = (mask_pr > 0.5).astype(np.float32)
        
        result = {
            'mask_predicted': mask_binary[0],
        }
        
        if mask_gt is not None:
            # IoU
            result['iou'] = segment_iou(mask_binary[0], mask_gt)
            
            # 精确率和召回率
            intersection = np.sum(mask_binary[0] * mask_gt)
            result['precision'] = intersection / (np.sum(mask_binary[0]) + 1e-16)
            result['recall'] = intersection / (np.sum(mask_gt) + 1e-16)
            result['f1_score'] = 2 * result['precision'] * result['recall'] / \
                                (result['precision'] + result['recall'] + 1e-16)
            
            # 像素准确率
            result['pixel_accuracy'] = np.sum(
                (mask_binary[0] > 0.5) == (mask_gt > 0.5)
            ) / mask_gt.size
        
        return result
    
    def _compute_keypoint_detection_rate(self, keypts_pr, heatmap):
        """计算关键点检测成功率"""
        if keypts_pr is None:
            return 0.0
        
        num_keypts = heatmap.shape[0]
        detected = 0
        
        for i in range(num_keypts):
            if heatmap[i].max() > self.cfg.TEST.HEATMAP_THRESHOLD:
                detected += 1
        
        return detected / num_keypts
    
    def _compute_final_score(self, sample_result):
        """计算综合得分（用于排序）"""
        
        score = 0.0
        weights = {'rotation': 0.4, 'translation': 0.4, 'segmentation': 0.2}
        
        # 旋转误差贡献
        if 'efficientpose' in sample_result and 'rotation_error' in sample_result['efficientpose']:
            rot_err = sample_result['efficientpose']['rotation_error']
            score += weights['rotation'] * (1.0 / (1.0 + rot_err / 10.0))
        
        # 平移误差贡献
        if 'efficientpose' in sample_result and 'translation_error' in sample_result['efficientpose']:
            trans_err = sample_result['efficientpose']['translation_error']
            score += weights['translation'] * (1.0 / (1.0 + trans_err))
        
        # 分割IoU贡献
        if 'segmentation' in sample_result and 'iou' in sample_result['segmentation']:
            iou = sample_result['segmentation']['iou']
            score += weights['segmentation'] * iou
        
        return score
    
    def _compute_statistics(self):
        """计算汇总统计信息"""
        
        stats = {
            'total_samples': len(self.results['per_sample']),
            'per_domain': {},
            'overall': {},
        }
        
        # 整体统计
        stats['overall'] = self._compute_domain_statistics(
            self.results['per_sample']
        )
        
        # 按域统计
        for domain, samples in self.results['per_domain'].items():
            stats['per_domain'][domain] = self._compute_domain_statistics(samples)
        
        self.results['statistics'] = stats
    
    def _compute_domain_statistics(self, samples):
        """计算特定域的统计信息"""
        
        def _stats(arr, include_range=False):
            if not arr:
                return {}
            result = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'median': float(np.median(arr)),
            }
            if include_range:
                result['min'] = float(np.min(arr))
                result['max'] = float(np.max(arr))
            return result

        stats = {
            'count': len(samples),
            # pose metrics (heatmap + efficientpose aggregated)
            'rotation_error': {},
            'translation_error': {},
            'speed_score': {},
            # heatmap-specific
            'heat_rotation_error': {},
            'heat_translation_error': {},
            'heat_speed_score': {},
            'heat_kpt_rate': {},
            # segmentation-specific
            'segmentation_iou': {},
            'segmentation_precision': {},
            'segmentation_recall': {},
            'segmentation_f1': {},
            'segmentation_pixel_accuracy': {},
            # success rates
            'success_rate': {},
        }

        # 收集数据
        rot_errors, trans_errors, speed_scores = [], [], []
        heat_rot, heat_trans, heat_speed, heat_kpt = [], [], [], []
        seg_iou, seg_prec, seg_rec, seg_f1, seg_pix = [], [], [], [], []
        rejected_count = 0

        for sample in samples:
            # Heatmap头部
            if 'heatmap' in sample:
                if sample['heatmap'].get('rejected', False):
                    rejected_count += 1
                else:
                    if 'rotation_error' in sample['heatmap']:
                        heat_rot.append(sample['heatmap']['rotation_error'])
                        rot_errors.append(sample['heatmap']['rotation_error'])
                    if 'translation_error' in sample['heatmap']:
                        heat_trans.append(sample['heatmap']['translation_error'])
                        trans_errors.append(sample['heatmap']['translation_error'])
                    if 'speed_score' in sample['heatmap']:
                        heat_speed.append(sample['heatmap']['speed_score'])
                        speed_scores.append(sample['heatmap']['speed_score'])
                    if 'keypoint_detection_rate' in sample['heatmap']:
                        heat_kpt.append(sample['heatmap']['keypoint_detection_rate'])

            # EfficientPose头部
            if 'efficientpose' in sample:
                if 'rotation_error' in sample['efficientpose']:
                    rot_errors.append(sample['efficientpose']['rotation_error'])
                if 'translation_error' in sample['efficientpose']:
                    trans_errors.append(sample['efficientpose']['translation_error'])
                if 'speed_score' in sample['efficientpose']:
                    speed_scores.append(sample['efficientpose']['speed_score'])

            # 分割头部
            if 'segmentation' in sample:
                seg = sample['segmentation']
                if 'iou' in seg:
                    seg_iou.append(seg['iou'])
                if 'precision' in seg:
                    seg_prec.append(seg['precision'])
                if 'recall' in seg:
                    seg_rec.append(seg['recall'])
                if 'f1_score' in seg:
                    seg_f1.append(seg['f1_score'])
                if 'pixel_accuracy' in seg:
                    seg_pix.append(seg['pixel_accuracy'])

        # 计算统计量
        stats['rotation_error'] = _stats(rot_errors, include_range=True)
        stats['translation_error'] = _stats(trans_errors, include_range=True)
        stats['speed_score'] = _stats(speed_scores)

        stats['heat_rotation_error'] = _stats(heat_rot, include_range=True)
        stats['heat_translation_error'] = _stats(heat_trans, include_range=True)
        stats['heat_speed_score'] = _stats(heat_speed)
        stats['heat_kpt_rate'] = _stats(heat_kpt)

        stats['segmentation_iou'] = _stats(seg_iou)
        stats['segmentation_precision'] = _stats(seg_prec)
        stats['segmentation_recall'] = _stats(seg_rec)
        stats['segmentation_f1'] = _stats(seg_f1)
        stats['segmentation_pixel_accuracy'] = _stats(seg_pix)

        stats['success_rate'] = {
            'detection': 1 - (rejected_count / len(samples)) if len(samples) > 0 else 0,
            'pose_accuracy_5deg': sum(1 for e in rot_errors if e < 5.0) / len(rot_errors) if rot_errors else 0,
            'pose_accuracy_10deg': sum(1 for e in rot_errors if e < 10.0) / len(rot_errors) if rot_errors else 0,
            'heat_kpt_mean': float(np.mean(heat_kpt)) if heat_kpt else 0.0
        }

        return stats
    
    def _identify_representative_cases(self, top_k=10):
        """识别代表性案例（最佳和最差）"""
        
        # 按综合得分排序
        sorted_samples = sorted(
            self.results['per_sample'],
            key=lambda x: x['final_score'],
            reverse=True
        )
        
        # 最佳案例
        self.results['best_cases'] = sorted_samples[:top_k]
        
        # 最差案例
        self.results['worst_cases'] = sorted_samples[-top_k:]
    
    def _save_results(self, output_dir):
        """保存评估结果"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存完整结果（pickle格式，便于后续处理）
        with open(os.path.join(output_dir, 'full_results.pkl'), 'wb') as f:
            pickle.dump(self.results, f)
        
        # 保存统计信息（JSON格式，便于阅读）
        with open(os.path.join(output_dir, 'statistics.json'), 'w') as f:
            json.dump(self.results['statistics'], f, indent=4, default=str)
        
        # 保存文本报告
        self._generate_text_report(output_dir)
        
        print(f"\n评估结果已保存到: {output_dir}")
    
    def _generate_text_report(self, output_dir):
        """生成文本格式的评估报告"""
        
        report_path = os.path.join(output_dir, 'evaluation_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("空间非合作目标结构与表面部件光学特征识别与分割任务评估报告\n")
            f.write("=" * 80 + "\n\n")
            
            stats = self.results['statistics']
            
            # 整体性能
            f.write("1. 整体性能指标\n")
            f.write("-" * 80 + "\n")
            overall = stats['overall']
            f.write(f"   总样本数: {overall['count']}\n\n")
            
            if overall['rotation_error'].get('mean'):
                f.write(f"   旋转误差:\n")
                f.write(f"     - 平均值: {overall['rotation_error']['mean']:.3f}°\n")
                f.write(f"     - 标准差: {overall['rotation_error']['std']:.3f}°\n")
                f.write(f"     - 中位数: {overall['rotation_error']['median']:.3f}°\n")
                f.write(f"     - 范围: [{overall['rotation_error']['min']:.3f}°, "
                       f"{overall['rotation_error']['max']:.3f}°]\n\n")
            
            if overall['translation_error'].get('mean'):
                f.write(f"   平移误差:\n")
                f.write(f"     - 平均值: {overall['translation_error']['mean']:.4f} m\n")
                f.write(f"     - 标准差: {overall['translation_error']['std']:.4f} m\n")
                f.write(f"     - 中位数: {overall['translation_error']['median']:.4f} m\n\n")
            
            if overall['segmentation_iou'].get('mean'):
                f.write(f"   分割IoU:\n")
                f.write(f"     - 平均值: {overall['segmentation_iou']['mean']:.4f}\n")
                f.write(f"     - 标准差: {overall['segmentation_iou']['std']:.4f}\n\n")
            
            # 按域分类的性能
            f.write("\n2. 各测试域性能对比\n")
            f.write("-" * 80 + "\n")
            
            for domain, domain_stats in stats['per_domain'].items():
                f.write(f"\n   [{domain}] (样本数: {domain_stats['count']})\n")
                
                if domain_stats['rotation_error'].get('mean'):
                    f.write(f"     - 旋转误差: {domain_stats['rotation_error']['mean']:.3f}° "
                           f"± {domain_stats['rotation_error']['std']:.3f}°\n")
                
                if domain_stats['translation_error'].get('mean'):
                    f.write(f"     - 平移误差: {domain_stats['translation_error']['mean']:.4f} m "
                           f"± {domain_stats['translation_error']['std']:.4f} m\n")
                
                if domain_stats['speed_score'].get('mean'):
                    f.write(f"     - SPEED得分: {domain_stats['speed_score']['mean']:.4f}\n")
                
                if domain_stats['segmentation_iou'].get('mean'):
                    f.write(f"     - 分割IoU: {domain_stats['segmentation_iou']['mean']:.4f}\n")
                
                if 'success_rate' in domain_stats:
                    f.write(f"     - 检测成功率: {domain_stats['success_rate']['detection']:.2%}\n")
                    f.write(f"     - 5°精度达标率: "
                           f"{domain_stats['success_rate']['pose_accuracy_5deg']:.2%}\n")
            
            # 最佳和最差案例
            f.write("\n\n3. 代表性案例\n")
            f.write("-" * 80 + "\n")
            
            f.write("\n   最佳案例 (Top 10):\n")
            for i, case in enumerate(self.results['best_cases'][:10], 1):
                f.write(f"     {i}. {case['image_name']} (域: {case['domain']}, "
                       f"得分: {case['final_score']:.4f})\n")
            
            f.write("\n   最差案例 (Bottom 10):\n")
            for i, case in enumerate(self.results['worst_cases'][:10], 1):
                f.write(f"     {i}. {case['image_name']} (域: {case['domain']}, "
                       f"得分: {case['final_score']:.4f})\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"文本报告已生成: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='生成空间目标识别任务的综合评估结果'
    )
    
    parser.add_argument('--cfg', required=True, type=str,
                       help='实验配置文件路径')
    parser.add_argument('--model-path', required=True, type=str,
                       help='模型检查点路径')
    parser.add_argument('--output', '--output-dir', dest='output_dir', type=str,
                       default='evaluate_seg',
                       help='输出目录')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='最大评估样本数（用于快速测试）')
    parser.add_argument('--domains', type=str, nargs='+', default=None,
                       help='测试域列表；用空格或逗号分隔，如 "lightbox synthetic sunlamp" 或 lightbox,synthetic,sunlamp；缺省使用cfg中的默认域')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                       help='修改配置选项')
    
    args = parser.parse_args()
    
    # 更新配置
    update_config(cfg, args)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_count = torch.cuda.device_count()
    print(f"使用设备: {device}")
    if gpu_count > 1:
        print(f"检测到 {gpu_count} 个GPU，启用 DataParallel 多GPU加速")
    elif gpu_count == 1:
        print(f"使用单GPU加速")
    
    # 构建模型
    print("构建模型...")
    model = build_spnv2(cfg)
    model = model.to(device)
    
    # 注：评估逻辑按单样本处理（batch=1），DataParallel 收益微弱且与自定义forward参数不兼容
    # 故不使用多GPU并行，而是充分利用单GPU的计算能力
    if gpu_count > 1:
        print(f"注：虽有 {gpu_count} 个GPU可用，但评估为单样本逐个处理（batch=1），")
        print("    多GPU并行收益微弱；采用单GPU推理以获得最佳稳定性。")
    
    # 加载检查点
    print(f"加载检查点: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # 处理state_dict（兼容旧版DataParallel保存的模型）
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # 如果保存的模型有module前缀（旧DataParallel），移除它
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=True)
    print("模型加载完成")
    
    # 域列表
    domain_list = []
    if args.domains:
        # 支持空格或逗号分隔
        raw_list = []
        for token in args.domains:
            raw_list.extend(token.split(','))
        domain_list = [d.strip() for d in raw_list if d.strip()]
    else:
        # 从默认TEST CSV推断当前域名
        base_csv = cfg.TEST.TEST_CSV
        domain_list = [base_csv.split('/')[0]]

    base_csv = cfg.TEST.TEST_CSV
    default_domain = base_csv.split('/')[0]

    # 创建评估器
    evaluator = ComprehensiveEvaluator(cfg, model, device)

    for domain in domain_list:
        cfg_domain = cfg.clone() if hasattr(cfg, 'clone') else copy.deepcopy(cfg)
        cfg_domain.defrost()
        # 替换域名部分
        cfg_domain.TEST.TEST_CSV = base_csv.replace(default_domain, domain, 1)
        cfg_domain.freeze()

        print(f"准备数据加载器... 域: {domain}, CSV: {cfg_domain.TEST.TEST_CSV}")
        # 评估逻辑按单样本展开，保持测试 batch_size=1
        if cfg_domain.TEST.IMAGES_PER_GPU != 1:
            cfg_domain.defrost()
            cfg_domain.TEST.IMAGES_PER_GPU = 1
            cfg_domain.freeze()
        test_loader = get_dataloader(cfg_domain, split='test', load_labels=True)

        evaluator.evaluate_dataset(
            test_loader,
            args.output_dir,
            max_samples=args.max_samples,
            compute_and_save=False
        )

    # 汇总统计并保存
    results = evaluator.finalize(args.output_dir)
    
    print("\n评估完成！")
    print(f"结果保存在: {args.output_dir}")


if __name__ == '__main__':
    main()
