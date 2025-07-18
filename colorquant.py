
import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
import warnings

warnings.filterwarnings('ignore')

# Set font for matplotlib - 支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial', 'Segoe UI']
matplotlib.rcParams['axes.unicode_minus'] = False
# 确保中文字符正确显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial', 'Segoe UI']
plt.rcParams['axes.unicode_minus'] = False

# Import scientific computing libraries
try:
    from skimage.color import rgb2lab, deltaE_ciede2000, lab2rgb
    from scipy.stats import gaussian_kde
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    from scipy.spatial.distance import pdist, squareform
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    ADVANCED_MODE = True
except ImportError as e:
    print(f"⚠️ Some advanced features unavailable, missing dependency: {e}")
    print("Recommend installing: pip install scikit-learn plotly")
    ADVANCED_MODE = False

# UI library import
try:
    import PySimpleGUI as sg

    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("⚠️ PySimpleGUI unavailable, command line mode only")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('color_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AdvancedColorAnalyzer:
    """Advanced Color Analyzer"""

    def __init__(self, config_file: str = "color_analysis_config.json"):
        self.config_file = config_file
        self.config = self.load_config()

        # Colormap options
        self.colormap_list = [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'Blues', 'Greens', 'Reds', 'Oranges', 'Purples', 'gray',
            'coolwarm', 'bwr', 'Spectral', 'seismic', 'RdYlBu', 'PiYG',
            'terrain', 'ocean', 'cubehelix', 'tab10', 'Set3'
        ]

        # Supported image formats
        self.image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp')

        # Analysis result cache
        self.analysis_cache = {}

        # Default colormap configuration
        self.default_colormaps = {
            'delta_e': 'viridis',
            'bhattacharyya': 'magma',
            'euclidean_lab': 'plasma',
            'euclidean_rgb': 'inferno',
            'diversity_distance': 'cividis'
        }

    def load_config(self) -> Dict[str, Any]:
        """Load configuration file"""
        default_config = {
            "mask_threshold": 20,
            "kde_bandwidth": "scott",
            "cluster_methods": ["kmeans", "dbscan"],
            "n_clusters_range": [2, 3, 4, 5, 6],
            "output_formats": ["xlsx", "csv", "json"],
            "heatmap_size": (10, 8),
            "heatmap_dpi": 300,
            "enable_interactive_plots": True,
            "color_space_analysis": ["RGB", "LAB", "HSV"],
            "statistical_tests": True,
            "clustering_analysis": True,
            "advanced_metrics": True,
            "matrix_colormaps": {
                "delta_e": "viridis",
                "bhattacharyya": "magma",
                "euclidean_lab": "plasma",
                "euclidean_rgb": "inferno",
                "diversity_distance": "cividis"
            }
        }

        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config file, using defaults: {e}")

        return default_config

    def save_config(self):
        """Save configuration file"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config file: {e}")

    def imread_unicode(self, path: str) -> Optional[np.ndarray]:
        """Safely read images with Unicode paths"""
        try:
            with open(path, 'rb') as f:
                arr = bytearray(f.read())
            img = cv2.imdecode(np.asarray(arr, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                logger.warning(f"Cannot decode image: {path}")
                return None
            return img
        except Exception as e:
            logger.error(f"Failed to read image: {path} - {e}")
            return None

    def read_and_mask(self, path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
        """Read image and generate mask, return additional statistics"""
        img = self.imread_unicode(path)
        if img is None:
            return None, None, {}

        # Convert color space
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # Generate mask
        mask = gray > self.config['mask_threshold']
        masked_img = img_rgb.copy()
        masked_img[~mask] = [0, 0, 0]

        # Calculate image statistics
        stats = {
            'image_size': img_rgb.shape[:2],
            'total_pixels': img_rgb.shape[0] * img_rgb.shape[1],
            'valid_pixels': np.sum(mask),
            'background_ratio': 1 - np.sum(mask) / (img_rgb.shape[0] * img_rgb.shape[1]),
            'brightness_mean': np.mean(gray[mask]) if np.sum(mask) > 0 else 0,
            'brightness_std': np.std(gray[mask]) if np.sum(mask) > 0 else 0
        }

        return masked_img, mask, stats

    def extract_color_features(self, img: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Extract multiple color space features"""
        features = {}

        if np.sum(mask) == 0:
            # Default values for blank images
            return {
                'lab_mean': [50, 0, 0], 'lab_std': [1, 1, 1],
                'rgb_mean': [128, 128, 128], 'rgb_std': [1, 1, 1],
                'hsv_mean': [0, 0, 128], 'hsv_std': [1, 1, 1],
                'lab_pixels': np.array([[50, 0, 0]]),
                'dominant_colors': [(128, 128, 128)],
                'color_diversity': 0
            }

        # LAB color space
        try:
            lab = rgb2lab(img)
            L, a, b = lab[:, :, 0][mask], lab[:, :, 1][mask], lab[:, :, 2][mask]
            features['lab_mean'] = [L.mean(), a.mean(), b.mean()]
            features['lab_std'] = [L.std(), a.std(), b.std()]
            features['lab_pixels'] = np.vstack((L, a, b)).T
        except Exception as e:
            logger.warning(f"LAB color space conversion failed: {e}")
            features['lab_mean'] = [50, 0, 0]
            features['lab_std'] = [1, 1, 1]
            features['lab_pixels'] = np.array([[50, 0, 0]])

        # RGB color space
        rgb_pixels = img[mask]
        features['rgb_mean'] = rgb_pixels.mean(axis=0).tolist()
        features['rgb_std'] = rgb_pixels.std(axis=0).tolist()

        # HSV color space
        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            hsv_pixels = hsv[mask]
            features['hsv_mean'] = hsv_pixels.mean(axis=0).tolist()
            features['hsv_std'] = hsv_pixels.std(axis=0).tolist()
        except Exception as e:
            logger.warning(f"HSV color space conversion failed: {e}")
            features['hsv_mean'] = [0, 0, 128]
            features['hsv_std'] = [1, 1, 1]

        # Dominant color extraction
        if ADVANCED_MODE:
            features['dominant_colors'] = self.extract_dominant_colors(rgb_pixels)
            features['color_diversity'] = self.calculate_color_diversity(rgb_pixels)
        else:
            features['dominant_colors'] = [tuple(features['rgb_mean'])]
            features['color_diversity'] = np.std(rgb_pixels) if len(rgb_pixels) > 0 else 0

        return features

    def calculate_color_temperature(self, rgb_mean: List[float]) -> float:
        """计算色温 (开尔文)"""
        try:
            # 简化的色温计算公式
            r, g, b = rgb_mean

            # 归一化
            total = r + g + b
            if total == 0:
                return 6500  # 默认色温

            r_norm = r / total
            g_norm = g / total

            # McCamy公式近似
            n = (r_norm - 0.332) / (g_norm - 0.186)
            cct = 449 * n ** 3 + 3525 * n ** 2 + 6823.3 * n + 5520.33

            return max(1000, min(25000, cct))  # 限制在合理范围
        except:
            return 6500

    def calculate_color_harmony(self, dominant_colors: List[Tuple[int, int, int]]) -> Dict[str, float]:
        """计算色彩和谐度"""
        if len(dominant_colors) < 2:
            return {'harmony_score': 1.0, 'contrast_ratio': 0.0}

        # 转换为HSV分析色相关系
        hsv_colors = []
        for r, g, b in dominant_colors:
            hsv = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2HSV)[0, 0]
            hsv_colors.append(hsv)

        # 计算色相差异
        hue_diffs = []
        for i in range(len(hsv_colors)):
            for j in range(i + 1, len(hsv_colors)):
                diff = abs(hsv_colors[i][0] - hsv_colors[j][0])
                hue_diffs.append(min(diff, 180 - diff))  # 处理色环

        # 和谐度评分 (基于色彩理论)
        harmony_types = [0, 30, 60, 90, 120, 150, 180]  # 理想色相差
        harmony_score = 0
        for diff in hue_diffs:
            min_deviation = min(abs(diff - ht) for ht in harmony_types)
            harmony_score += 1 / (1 + min_deviation / 30)

        harmony_score /= len(hue_diffs)

        # 对比度计算
        contrast_ratio = np.std(hue_diffs) / 180 if hue_diffs else 0

        return {
            'harmony_score': round(harmony_score, 3),
            'contrast_ratio': round(contrast_ratio, 3)
        }

    def extract_dominant_colors(self, pixels: np.ndarray, n_colors: int = 5) -> List[Tuple[int, int, int]]:
        """提取主色彩"""
        if len(pixels) == 0:
            return [(128, 128, 128)]

        try:
            # 降采样以提高速度
            if len(pixels) > 10000:
                indices = np.random.choice(len(pixels), 10000, replace=False)
                pixels = pixels[indices]

            kmeans = KMeans(n_clusters=min(n_colors, len(pixels)), random_state=42, n_init=10)
            kmeans.fit(pixels)
            colors = kmeans.cluster_centers_.astype(int)
            return [tuple(color) for color in colors]
        except Exception as e:
            logger.warning(f"主色彩提取失败: {e}")
            return [tuple(pixels.mean(axis=0).astype(int))]

    def calculate_color_diversity(self, pixels: np.ndarray) -> float:
        """计算颜色多样性指数"""
        try:
            if len(pixels) < 2:
                return 0

            # 使用颜色直方图计算香农熵
            hist_r = np.histogram(pixels[:, 0], bins=32, range=(0, 256))[0]
            hist_g = np.histogram(pixels[:, 1], bins=32, range=(0, 256))[0]
            hist_b = np.histogram(pixels[:, 2], bins=32, range=(0, 256))[0]

            combined_hist = hist_r + hist_g + hist_b
            combined_hist = combined_hist[combined_hist > 0]

            if len(combined_hist) == 0:
                return 0

            probabilities = combined_hist / combined_hist.sum()
            entropy = -np.sum(probabilities * np.log2(probabilities))
            return entropy
        except Exception as e:
            logger.warning(f"颜色多样性计算失败: {e}")
            return 0

    def compute_distance_matrices(self, features_list: List[Dict[str, Any]],
                                  progress_callback=None) -> Dict[str, np.ndarray]:
        """计算多种距离矩阵"""
        n = len(features_list)
        matrices = {}

        # 提取LAB均值
        lab_means = np.array([f['lab_mean'] for f in features_list])

        # ΔE2000色差矩阵
        if progress_callback:
            progress_callback("计算ΔE2000色差矩阵...")

        delta_e = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                try:
                    delta_e[i, j] = deltaE_ciede2000([[lab_means[i]]], [[lab_means[j]]])[0, 0]
                except:
                    delta_e[i, j] = 0
        matrices['delta_e'] = delta_e

        # Bhattacharyya距离矩阵
        if progress_callback:
            progress_callback("计算Bhattacharyya距离矩阵...")

        bh_distance = np.zeros((n, n))
        kdes = []

        # 预拟合KDE
        for features in features_list:
            try:
                lab_pixels = features['lab_pixels']
                if len(lab_pixels) > 1:
                    kde = gaussian_kde(lab_pixels.T, bw_method=self.config['kde_bandwidth'])
                    kdes.append(kde)
                else:
                    kdes.append(None)
            except:
                kdes.append(None)

        # 计算Bhattacharyya距离
        for i in range(n):
            for j in range(n):
                if i == j:
                    bh_distance[i, j] = 0
                elif kdes[i] is None or kdes[j] is None:
                    bh_distance[i, j] = 1.0
                else:
                    try:
                        d1 = features_list[i]['lab_pixels']
                        d2 = features_list[j]['lab_pixels']
                        mi = np.minimum(d1.min(0), d2.min(0)) - 5
                        ma = np.maximum(d1.max(0), d2.max(0)) + 5
                        axes = [np.linspace(mi[k], ma[k], 8) for k in range(3)]
                        grid = np.array(np.meshgrid(*axes)).reshape(3, -1)
                        p, q = kdes[i](grid), kdes[j](grid)
                        p, q = np.clip(p, 1e-10, None), np.clip(q, 1e-10, None)
                        bc = np.sum(np.sqrt(p * q)) / len(p)
                        bh_distance[i, j] = -np.log(max(bc, 1e-10))
                    except Exception as e:
                        logger.warning(f"Bhattacharyya距离计算失败: {e}")
                        bh_distance[i, j] = 1.0

        matrices['bhattacharyya'] = bh_distance

        # 欧几里得距离矩阵（LAB空间）
        if progress_callback:
            progress_callback("计算欧几里得距离矩阵...")

        euclidean_dist = squareform(pdist(lab_means, metric='euclidean'))
        matrices['euclidean_lab'] = euclidean_dist

        # RGB空间欧几里得距离
        rgb_means = np.array([f['rgb_mean'] for f in features_list])
        euclidean_rgb = squareform(pdist(rgb_means, metric='euclidean'))
        matrices['euclidean_rgb'] = euclidean_rgb

        # 颜色多样性距离
        diversity_scores = np.array([f['color_diversity'] for f in features_list])
        diversity_dist = np.abs(diversity_scores[:, np.newaxis] - diversity_scores[np.newaxis, :])
        matrices['diversity_distance'] = diversity_dist

        return matrices

    def create_all_heatmaps(self, distance_matrices: Dict[str, np.ndarray],
                            filenames: List[str], output_folder: str,
                            custom_colormaps: Dict[str, str] = None) -> Dict[str, str]:
        """创建所有距离矩阵的热力图，支持自定义色系"""
        viz_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 矩阵中文名称映射
        matrix_titles = {
            'delta_e': 'ΔE2000 色差矩阵',
            'bhattacharyya': 'Bhattacharyya 距离矩阵',
            'euclidean_lab': 'LAB欧几里得距离矩阵',
            'euclidean_rgb': 'RGB欧几里得距离矩阵',
            'diversity_distance': '颜色多样性距离矩阵'
        }

        # 使用自定义色系或配置中的色系或默认色系
        if custom_colormaps:
            colormaps = custom_colormaps
        else:
            colormaps = self.config.get('matrix_colormaps', self.default_colormaps)

        for matrix_name, matrix in distance_matrices.items():
            title = matrix_titles.get(matrix_name, matrix_name)
            cmap = colormaps.get(matrix_name, 'viridis')
            filename = f'{matrix_name}_heatmap_{timestamp}.png'
            filepath = os.path.join(output_folder, filename)

            try:
                # 创建图形，设置透明背景
                fig, ax = plt.subplots(figsize=self.config['heatmap_size'])
                fig.patch.set_facecolor('none')  # 图形背景透明
                ax.set_facecolor('none')  # 坐标轴背景透明

                # 确保中文字体正确设置
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial', 'Segoe UI']
                plt.rcParams['axes.unicode_minus'] = False

                # 对于ΔE矩阵，屏蔽对角线0值
                mask = matrix == 0 if matrix_name == 'delta_e' else None

                # 创建热力图 - 增强美观性
                sns.heatmap(matrix,
                            annot=True,
                            fmt='.2f',
                            cmap=cmap,
                            cbar_kws={
                                'label': title,
                                'shrink': 0.8,  # 缩小颜色条
                                'aspect': 20,  # 颜色条宽高比
                                'pad': 0.02  # 颜色条间距
                            },
                            square=True,
                            linewidths=1.0,  # 增加线宽
                            linecolor='white',  # 白色分割线更清晰
                            xticklabels=[os.path.basename(f) for f in filenames],
                            yticklabels=[os.path.basename(f) for f in filenames],
                            mask=mask,
                            annot_kws={
                                'size': 10,  # 数字字体大小
                                'weight': 'bold',  # 数字加粗
                                'color': 'white'  # 数字颜色为白色
                            })

                # 美化标题和标签 - 明确指定字体以支持中文
                plt.title(f'{title} (色系: {cmap})',
                          fontsize=16,
                          fontweight='bold',
                          pad=25,
                          color='black',
                          fontfamily=['SimHei', 'Microsoft YaHei', 'DejaVu Sans'])

                # 美化坐标轴标签 - 明确指定字体以支持中文
                plt.xticks(rotation=45, ha='right', fontsize=11, fontweight='bold',
                          fontfamily=['SimHei', 'Microsoft YaHei', 'DejaVu Sans'])
                plt.yticks(rotation=0, fontsize=11, fontweight='bold',
                          fontfamily=['SimHei', 'Microsoft YaHei', 'DejaVu Sans'])

                # 调整布局
                plt.tight_layout()

                # 保存为透明背景PNG
                plt.savefig(filepath,
                            dpi=self.config['heatmap_dpi'],
                            bbox_inches='tight',
                            facecolor='none',  # 保存时背景透明
                            edgecolor='none',  # 边框透明
                            transparent=True,  # 启用透明度
                            pad_inches=0.2)  # 边距
                plt.close()

                viz_files[matrix_name] = filename
                logger.info(f"✅ 热力图已保存: {filename}")

            except Exception as e:
                logger.error(f"热力图创建失败 {matrix_name}: {e}")

        return viz_files

    def create_3d_color_space_plot(self, features_list: List[Dict[str, Any]],
                                   filenames: List[str], output_folder: str) -> str:
        """创建3D色彩空间可视化 - 解决标注和图例问题"""
        # 确保中文字体正确设置
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial', 'Segoe UI']
        plt.rcParams['axes.unicode_minus'] = False

        fig = plt.figure(figsize=(20, 16))  # 增大画布尺寸

        # 创建2x2的子图布局

        # LAB空间3D散点图 - 修复标注问题
        ax1 = fig.add_subplot(221, projection='3d')
        lab_data = np.array([f['lab_mean'] for f in features_list])

        # 创建散点图
        scatter = ax1.scatter(lab_data[:, 1], lab_data[:, 2], lab_data[:, 0],
                              c=lab_data[:, 0], cmap='viridis', s=300, alpha=0.8, edgecolors='black', linewidth=2)

        # 清晰的坐标轴标签 - 明确指定字体以支持中文
        ax1.set_xlabel('a* (绿-红轴)', fontsize=14, fontweight='bold', labelpad=10,
                      fontfamily=['SimHei', 'Microsoft YaHei', 'DejaVu Sans'])
        ax1.set_ylabel('b* (蓝-黄轴)', fontsize=14, fontweight='bold', labelpad=10,
                      fontfamily=['SimHei', 'Microsoft YaHei', 'DejaVu Sans'])
        ax1.set_zlabel('L* (亮度轴)', fontsize=14, fontweight='bold', labelpad=10,
                      fontfamily=['SimHei', 'Microsoft YaHei', 'DejaVu Sans'])
        ax1.set_title('LAB色彩空间3D分布', fontsize=16, fontweight='bold', pad=20,
                     fontfamily=['SimHei', 'Microsoft YaHei', 'DejaVu Sans'])

        # 改进的图片标签 - 避免重叠
        for i, filename in enumerate(filenames):
            # 使用实际的LAB坐标，添加小偏移避免重叠
            ax1.text(lab_data[i, 1],
                     lab_data[i, 2],
                     lab_data[i, 0] + 3,  # 只在Z轴上偏移
                     os.path.basename(filename)[:8],
                     fontsize=9,
                     fontweight='bold',
                     ha='center',
                     va='bottom',
                     bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7, edgecolor='black'))

        # 添加外部颜色条，避免遮挡
        cbar = plt.colorbar(scatter, ax=ax1, shrink=0.6, aspect=20, pad=0.1)
        cbar.set_label('亮度 (L*)', fontsize=12, fontweight='bold')

        # RGB空间3D散点图 - 同样修复
        ax2 = fig.add_subplot(222, projection='3d')
        rgb_data = np.array([f['rgb_mean'] for f in features_list])

        # 使用实际RGB颜色作为散点颜色
        rgb_colors = rgb_data / 255.0  # 归一化到0-1
        scatter2 = ax2.scatter(rgb_data[:, 0], rgb_data[:, 1], rgb_data[:, 2],
                               c=rgb_colors, s=300, alpha=0.8, edgecolors='black', linewidth=2)

        ax2.set_xlabel('红色通道 (R)', fontsize=14, fontweight='bold', labelpad=10)
        ax2.set_ylabel('绿色通道 (G)', fontsize=14, fontweight='bold', labelpad=10)
        ax2.set_zlabel('蓝色通道 (B)', fontsize=14, fontweight='bold', labelpad=10)
        ax2.set_title('RGB色彩空间3D分布', fontsize=16, fontweight='bold', pad=20)

        # RGB标签
        for i, filename in enumerate(filenames):
            # 计算动态偏移量，基于RGB值的范围
            rgb_range = rgb_data.max() - rgb_data.min()
            z_offset = max(30, rgb_range * 0.1)  # 动态计算偏移量

            ax2.text(rgb_data[i, 0],
                     rgb_data[i, 1],
                     rgb_data[i, 2] + z_offset,  # 使用动态偏移
                     os.path.basename(filename)[:8],
                     fontsize=9,
                     fontweight='bold',
                     ha='center',
                     va='bottom',
                     bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgreen', alpha=0.7, edgecolor='black'))

            # 添加连接线，清楚显示对应关系
            ax2.plot([rgb_data[i, 0], rgb_data[i, 0]],
                     [rgb_data[i, 1], rgb_data[i, 1]],
                     [rgb_data[i, 2], rgb_data[i, 2] + z_offset],
                     color='gray', linewidth=1, linestyle='--', alpha=0.6)

        # 色彩多样性雷达图 - 改进布局
        ax3 = fig.add_subplot(223, projection='polar')
        diversity_scores = [f['color_diversity'] for f in features_list]
        angles = np.linspace(0, 2 * np.pi, len(filenames), endpoint=False)

        # 闭合雷达图
        angles_closed = np.concatenate((angles, [angles[0]]))
        diversity_closed = diversity_scores + [diversity_scores[0]]

        ax3.plot(angles_closed, diversity_closed, 'o-', linewidth=3, markersize=10, color='darkblue')
        ax3.fill(angles_closed, diversity_closed, alpha=0.3, color='lightblue')
        ax3.set_title('色彩多样性雷达图', fontsize=16, fontweight='bold', pad=30)

        # 设置角度标签
        ax3.set_thetagrids(np.degrees(angles), [os.path.basename(f)[:8] + '...' if len(os.path.basename(f)) > 8
                                                else os.path.basename(f) for f in filenames])

        # 颜色温度分布柱状图 - 改进美观性
        ax4 = fig.add_subplot(224)
        color_temps = [self.calculate_color_temperature(f['rgb_mean']) for f in features_list]

        # 根据色温创建颜色映射
        normalized_temps = np.array(color_temps) / max(color_temps)
        colors = plt.cm.coolwarm(normalized_temps)

        bars = ax4.bar(range(len(filenames)), color_temps, color=colors,
                       edgecolor='black', linewidth=1.5, alpha=0.8)

        ax4.set_xlabel('图片', fontsize=14, fontweight='bold')
        ax4.set_ylabel('色温 (K)', fontsize=14, fontweight='bold')
        ax4.set_title('色温分布柱状图', fontsize=16, fontweight='bold')
        ax4.set_xticks(range(len(filenames)))
        ax4.set_xticklabels([os.path.basename(f)[:8] + '...' if len(os.path.basename(f)) > 8
                             else os.path.basename(f) for f in filenames],
                            rotation=45, ha='right')

        # 添加数值标签
        for i, (bar, temp) in enumerate(zip(bars, color_temps)):
            ax4.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + max(color_temps) * 0.01,
                     f'{int(temp)}K', ha='center', va='bottom', fontweight='bold')

        # 调整子图间距
        plt.tight_layout(pad=3.0)

        filename = f'3d_color_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        filepath = os.path.join(output_folder, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        return filename

    def create_dominant_colors_chart(self, features_list: List[Dict[str, Any]],
                                     filenames: List[str], output_folder: str) -> str:
        """创建主色彩分布图"""
        # 确保中文字体正确设置
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial', 'Segoe UI']
        plt.rcParams['axes.unicode_minus'] = False

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # 左图：每张图片的主色彩数量
        dominant_counts = [len(f['dominant_colors']) for f in features_list]
        bars1 = ax1.bar(range(len(filenames)), dominant_counts,
                        color='lightcoral', alpha=0.8, edgecolor='black', linewidth=1.5)

        ax1.set_xlabel('图片', fontsize=14, fontweight='bold')
        ax1.set_ylabel('主色彩数量', fontsize=14, fontweight='bold')
        ax1.set_title('每张图片的主色彩数量分布', fontsize=16, fontweight='bold')
        ax1.set_xticks(range(len(filenames)))
        ax1.set_xticklabels([os.path.basename(f) for f in filenames], rotation=45, ha='right')

        # 添加数值标签
        for bar, count in zip(bars1, dominant_counts):
            ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.1,
                     str(count), ha='center', va='bottom', fontweight='bold')

        # 右图：主色彩颜色展示
        ax2.set_title('各图片主色彩展示', fontsize=16, fontweight='bold')

        y_pos = 0
        for i, (features, filename) in enumerate(zip(features_list, filenames)):
            colors = features['dominant_colors'][:5]  # 最多显示5个主色

            # 绘制颜色条
            for j, color in enumerate(colors):
                rect = plt.Rectangle((j, y_pos), 1, 0.8,
                                     facecolor=np.array(color) / 255.0,
                                     edgecolor='black', linewidth=1)
                ax2.add_patch(rect)

                # 添加RGB值标签
                ax2.text(j + 0.5, y_pos + 0.4,
                         f'RGB\n{color[0]},{color[1]},{color[2]}',
                         ha='center', va='center', fontsize=8, fontweight='bold')

            # 添加文件名
            ax2.text(-0.5, y_pos + 0.4, os.path.basename(filename),
                     ha='right', va='center', fontweight='bold')

            y_pos += 1

        ax2.set_xlim(-2, 5)
        ax2.set_ylim(-0.5, len(filenames))
        ax2.set_xlabel('主色彩排序', fontsize=14, fontweight='bold')
        ax2.set_ylabel('图片', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(5))
        ax2.set_xticklabels(['第1色', '第2色', '第3色', '第4色', '第5色'])

        plt.tight_layout()

        filename = f'dominant_colors_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        filepath = os.path.join(output_folder, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        return filename

    def create_color_harmony_chart(self, features_list: List[Dict[str, Any]],
                                   filenames: List[str], output_folder: str) -> str:
        """创建色彩和谐度对比图"""
        # 确保中文字体正确设置
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial', 'Segoe UI']
        plt.rcParams['axes.unicode_minus'] = False

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 计算和谐度数据
        harmony_data = []
        color_temps = []

        for features in features_list:
            harmony = self.calculate_color_harmony(features['dominant_colors'])
            harmony_data.append(harmony)
            color_temps.append(self.calculate_color_temperature(features['rgb_mean']))

        harmony_scores = [h['harmony_score'] for h in harmony_data]
        contrast_ratios = [h['contrast_ratio'] for h in harmony_data]

        # 1. 和谐度评分柱状图
        bars1 = ax1.bar(range(len(filenames)), harmony_scores,
                        color='mediumseagreen', alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.set_xlabel('图片', fontsize=12, fontweight='bold')
        ax1.set_ylabel('和谐度评分', fontsize=12, fontweight='bold')
        ax1.set_title('色彩和谐度评分', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(filenames)))
        ax1.set_xticklabels([os.path.basename(f)[:8] + '...' if len(os.path.basename(f)) > 8
                             else os.path.basename(f) for f in filenames], rotation=45, ha='right')

        # 添加评分标签
        for bar, score in zip(bars1, harmony_scores):
            ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                     f'{score:.2f}', ha='center', va='bottom', fontweight='bold')

        # 2. 对比度比例柱状图
        bars2 = ax2.bar(range(len(filenames)), contrast_ratios,
                        color='orange', alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('图片', fontsize=12, fontweight='bold')
        ax2.set_ylabel('对比度比例', fontsize=12, fontweight='bold')
        ax2.set_title('色彩对比度比例', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(filenames)))
        ax2.set_xticklabels([os.path.basename(f)[:8] + '...' if len(os.path.basename(f)) > 8
                             else os.path.basename(f) for f in filenames], rotation=45, ha='right')

        for bar, ratio in zip(bars2, contrast_ratios):
            ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                     f'{ratio:.2f}', ha='center', va='bottom', fontweight='bold')

        # 3. 和谐度vs色温散点图
        scatter = ax3.scatter(color_temps, harmony_scores, s=200, alpha=0.7,
                              c=contrast_ratios, cmap='viridis', edgecolors='black', linewidth=2)
        ax3.set_xlabel('色温 (K)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('和谐度评分', fontsize=12, fontweight='bold')
        ax3.set_title('色温与和谐度关系', fontsize=14, fontweight='bold')

        # 添加点标签
        for i, filename in enumerate(filenames):
            ax3.annotate(os.path.basename(filename)[:8],
                         (color_temps[i], harmony_scores[i]),
                         xytext=(5, 5), textcoords='offset points', fontsize=9)

        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('对比度比例', fontsize=10, fontweight='bold')

        # 4. 综合评价雷达图
        ax4 = plt.subplot(224, projection='polar')

        # 标准化数据到0-1范围
        norm_harmony = np.array(harmony_scores) / max(harmony_scores) if max(harmony_scores) > 0 else np.zeros_like(
            harmony_scores)
        norm_contrast = np.array(contrast_ratios) / max(contrast_ratios) if max(contrast_ratios) > 0 else np.zeros_like(
            contrast_ratios)
        norm_diversity = np.array([f['color_diversity'] for f in features_list])
        norm_diversity = norm_diversity / max(norm_diversity) if max(norm_diversity) > 0 else np.zeros_like(
            norm_diversity)

        # 计算平均值
        avg_data = [np.mean(norm_harmony), np.mean(norm_contrast), np.mean(norm_diversity)]
        labels = ['和谐度', '对比度', '多样性']

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        angles_closed = np.concatenate((angles, [angles[0]]))
        avg_data_closed = avg_data + [avg_data[0]]

        ax4.plot(angles_closed, avg_data_closed, 'o-', linewidth=3, markersize=10, color='red')
        ax4.fill(angles_closed, avg_data_closed, alpha=0.25, color='red')
        ax4.set_thetagrids(np.degrees(angles), labels)
        ax4.set_title('综合色彩质量评价', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        filename = f'color_harmony_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        filepath = os.path.join(output_folder, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        return filename

    def create_interactive_dashboard(self, features_list: List[Dict[str, Any]],
                                     distance_matrices: Dict[str, np.ndarray],
                                     filenames: List[str], output_folder: str) -> str:
        """创建交互式分析仪表盘"""
        if not ADVANCED_MODE:
            return ""

        try:
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go

            # 创建子图布局
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=['LAB色彩空间3D分布', '主色彩数量分布', 'ΔE2000色差热力图',
                                '色彩多样性vs色温', '颜色温度分布', 'RGB色彩空间3D分布'],
                specs=[[{"type": "scatter3d"}, {"type": "bar"}],
                       [{"type": "heatmap"}, {"type": "scatter"}],
                       [{"type": "bar"}, {"type": "scatter3d"}]]
            )

            # 1. LAB 3D散点图
            lab_data = np.array([f['lab_mean'] for f in features_list])
            fig.add_trace(
                go.Scatter3d(
                    x=lab_data[:, 1], y=lab_data[:, 2], z=lab_data[:, 0],
                    mode='markers+text',
                    text=[os.path.basename(f) for f in filenames],
                    textposition="top center",
                    marker=dict(size=12, color=lab_data[:, 0], colorscale='viridis',
                                showscale=True, colorbar=dict(title="亮度 (L*)")),
                    name='LAB分布'
                ), row=1, col=1
            )

            # 2. 主色彩数量分布
            dominant_counts = [len(f['dominant_colors']) for f in features_list]
            fig.add_trace(
                go.Bar(
                    x=[os.path.basename(f) for f in filenames],
                    y=dominant_counts,
                    name='主色彩数量',
                    marker_color='lightcoral'
                ), row=1, col=2
            )

            # 3. ΔE热力图
            fig.add_trace(
                go.Heatmap(
                    z=distance_matrices['delta_e'],
                    x=[os.path.basename(f) for f in filenames],
                    y=[os.path.basename(f) for f in filenames],
                    colorscale='viridis',
                    name='ΔE2000'
                ), row=2, col=1
            )

            # 4. 色彩多样性vs色温散点图
            color_temps = [self.calculate_color_temperature(f['rgb_mean']) for f in features_list]
            diversity_scores = [f['color_diversity'] for f in features_list]

            fig.add_trace(
                go.Scatter(
                    x=color_temps,
                    y=diversity_scores,
                    mode='markers+text',
                    text=[os.path.basename(f) for f in filenames],
                    textposition="top center",
                    marker=dict(size=12, color='blue'),
                    name='多样性vs色温'
                ), row=2, col=2
            )

            # 5. 颜色温度分布
            fig.add_trace(
                go.Bar(
                    x=[os.path.basename(f) for f in filenames],
                    y=color_temps,
                    name='色温分布',
                    marker_color='orange'
                ), row=3, col=1
            )

            # 6. RGB 3D散点图
            rgb_data = np.array([f['rgb_mean'] for f in features_list])
            fig.add_trace(
                go.Scatter3d(
                    x=rgb_data[:, 0], y=rgb_data[:, 1], z=rgb_data[:, 2],
                    mode='markers+text',
                    text=[os.path.basename(f) for f in filenames],
                    textposition="top center",
                    marker=dict(size=12, color='red'),
                    name='RGB分布'
                ), row=3, col=2
            )

            # 更新布局
            fig.update_layout(
                title='颜色分析交互式仪表盘',
                showlegend=False,
                height=1200,
                font=dict(size=12)
            )

            # 更新坐标轴标签
            fig.update_layout(scene=dict(
                xaxis_title='a* (绿-红)',
                yaxis_title='b* (蓝-黄)',
                zaxis_title='L* (亮度)'
            ))

            filename = f'interactive_dashboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
            filepath = os.path.join(output_folder, filename)
            fig.write_html(filepath)

            return filename

        except Exception as e:
            logger.error(f"交互式仪表盘创建失败: {e}")
            return ""

    def run_comprehensive_analysis(self, input_folder: str, output_folder: str,
                                   progress_callback=None, custom_colormaps: Dict[str, str] = None) -> Dict[str, Any]:
        """运行综合颜色分析"""

        logger.info(f"开始综合颜色分析: {input_folder} -> {output_folder}")

        # 创建输出目录
        os.makedirs(output_folder, exist_ok=True)

        # 查找图片文件
        image_files = []
        for ext in self.image_extensions:
            # 使用glob的不区分大小写匹配
            pattern = f"*{ext}"
            image_files.extend([f for f in Path(input_folder).glob(pattern) if f.suffix.lower() == ext.lower()])

        image_files = [str(f) for f in image_files]

        if not image_files:
            raise ValueError("未找到支持的图片文件")

        if progress_callback:
            progress_callback(f"找到 {len(image_files)} 张图片")

        # 提取特征
        features_list = []
        image_stats = []

        for idx, image_path in enumerate(image_files, 1):
            if progress_callback:
                progress_callback(f"提取特征: {idx}/{len(image_files)} - {os.path.basename(image_path)}")

            img, mask, stats = self.read_and_mask(image_path)
            if img is None:
                continue

            features = self.extract_color_features(img, mask)
            features_list.append(features)
            image_stats.append(stats)

        if not features_list:
            raise ValueError("没有成功处理的图片")

        # 计算距离矩阵
        distance_matrices = self.compute_distance_matrices(features_list, progress_callback)

        # 保存基础结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 创建摘要数据
        summary_data = []
        processed_files = image_files[:len(features_list)]
        for i, (features, stats, filename) in enumerate(zip(features_list, image_stats, processed_files)):
            row = {
                'Image': os.path.basename(filename),
                # LAB色彩空间 - 均值和标准差
                'L_Mean': round(features['lab_mean'][0], 3),
                'L_Std': round(features['lab_std'][0], 3),
                'a_Mean': round(features['lab_mean'][1], 3),
                'a_Std': round(features['lab_std'][1], 3),
                'b_Mean': round(features['lab_mean'][2], 3),
                'b_Std': round(features['lab_std'][2], 3),
                # RGB色彩空间 - 均值和标准差
                'R_Mean': round(features['rgb_mean'][0], 3),
                'R_Std': round(features['rgb_std'][0], 3),
                'G_Mean': round(features['rgb_mean'][1], 3),
                'G_Std': round(features['rgb_std'][1], 3),
                'B_Mean': round(features['rgb_mean'][2], 3),
                'B_Std': round(features['rgb_std'][2], 3),
                # HSV色彩空间 - 均值和标准差
                'H_Mean': round(features['hsv_mean'][0], 3),
                'H_Std': round(features['hsv_std'][0], 3),
                'S_Mean': round(features['hsv_mean'][1], 3),
                'S_Std': round(features['hsv_std'][1], 3),
                'V_Mean': round(features['hsv_mean'][2], 3),
                'V_Std': round(features['hsv_std'][2], 3),
                # 其他统计信息
                'Color_Diversity': round(features['color_diversity'], 3),
                'Valid_Pixels': stats['valid_pixels'],
                'Total_Pixels': stats['total_pixels'],
                'Background_Ratio': round(stats['background_ratio'], 3),
                'Brightness_Mean': round(stats['brightness_mean'], 3),
                'Brightness_Std': round(stats['brightness_std'], 3),
                'Image_Width': stats['image_size'][1],
                'Image_Height': stats['image_size'][0]
            }
            summary_data.append(row)

        # 保存Excel文件
        excel_path = os.path.join(output_folder, f'颜色分析结果_{timestamp}.xlsx')
        try:
            summary_df = pd.DataFrame(summary_data)
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                summary_df.to_excel(writer, 'LAB统计', index=False)

                # 保存距离矩阵
                filenames = [os.path.basename(f) for f in processed_files]
                for name, matrix in distance_matrices.items():
                    df = pd.DataFrame(matrix, columns=filenames, index=filenames)
                    df.to_excel(writer, f'{name}矩阵')

            if progress_callback:
                progress_callback(f"✅ Excel已保存: {excel_path}")
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Excel保存失败: {e}")

        # 生成所有可视化
        viz_files = {}

        # 1. 生成热力图
        if progress_callback:
            progress_callback("生成距离矩阵热力图...")

        heatmap_files = self.create_all_heatmaps(distance_matrices, processed_files, output_folder, custom_colormaps)
        viz_files.update(heatmap_files)

        # 2. 生成3D色彩空间分布图
        if progress_callback:
            progress_callback("生成3D色彩空间分布图...")

        viz_files['3d_plot'] = self.create_3d_color_space_plot(features_list, processed_files, output_folder)

        # 3. 生成主色彩分布图
        if progress_callback:
            progress_callback("生成主色彩分布图...")

        viz_files['dominant_colors'] = self.create_dominant_colors_chart(features_list, processed_files, output_folder)

        # 4. 生成色彩和谐度分析图
        if progress_callback:
            progress_callback("生成色彩和谐度分析图...")

        viz_files['harmony_analysis'] = self.create_color_harmony_chart(features_list, processed_files, output_folder)

        # 5. 生成交互式仪表盘
        if progress_callback:
            progress_callback("生成交互式HTML仪表盘...")

        dashboard_file = self.create_interactive_dashboard(features_list, distance_matrices, processed_files,
                                                           output_folder)
        if dashboard_file:
            viz_files['dashboard'] = dashboard_file

        logger.info("颜色分析完成")

        return {
            'features': features_list,
            'distance_matrices': distance_matrices,
            'image_files': processed_files,
            'excel_path': excel_path,
            'visualization_files': viz_files
        }


class AdvancedGUI:
    """Advanced Graphical User Interface - Material Design 3 Style"""

    def __init__(self):
        if not GUI_AVAILABLE:
            raise ImportError("PySimpleGUI not available")

        self.analyzer = AdvancedColorAnalyzer()

        # Material Design 3 配色方案
        self.md3_colors = {
            'primary': '#6750A4',
            'on_primary': '#FFFFFF',
            'primary_container': '#EADDFF',
            'on_primary_container': '#21005D',
            'secondary': '#625B71',
            'on_secondary': '#FFFFFF',
            'secondary_container': '#E8DEF8',
            'on_secondary_container': '#1D192B',
            'tertiary': '#7D5260',
            'on_tertiary': '#FFFFFF',
            'tertiary_container': '#FFD8E4',
            'on_tertiary_container': '#31111D',
            'surface': '#FEF7FF',
            'on_surface': '#1C1B1F',
            'surface_variant': '#E7E0EC',
            'on_surface_variant': '#49454F',
            'outline': '#79747E',
            'outline_variant': '#CAC4D0',
            'background': '#FEF7FF',
            'on_background': '#1C1B1F',
            'error': '#B3261E',
            'on_error': '#FFFFFF',
            'error_container': '#F9DEDC',
            'on_error_container': '#410E0B'
        }

        # 设置Material Design 3主题
        self.setup_md3_theme()

    def setup_md3_theme(self):
        """设置Material Design 3主题"""
        try:
            # 创建自定义主题
            sg.LOOK_AND_FEEL_TABLE['MD3'] = {
                'BACKGROUND': self.md3_colors['background'],
                'TEXT': self.md3_colors['on_background'],
                'INPUT': self.md3_colors['surface'],
                'TEXT_INPUT': self.md3_colors['on_surface'],
                'SCROLL': self.md3_colors['outline_variant'],
                'BUTTON': (self.md3_colors['on_primary'], self.md3_colors['primary']),
                'PROGRESS': (self.md3_colors['primary'], self.md3_colors['surface_variant']),
                'BORDER': 1,
                'SLIDER_DEPTH': 0,
                'PROGRESS_DEPTH': 0
            }
            sg.theme('MD3')
        except:
            sg.theme('SystemDefault')

    def create_card_frame(self, title, content, key_suffix=""):
        """创建Material Design 3卡片样式框架"""
        return sg.Frame(
            title,
            content,
            font=('Segoe UI', 11, 'bold'),
            title_color=self.md3_colors['primary'],
            background_color=self.md3_colors['surface'],
            border_width=0,
            relief='flat',
            pad=((8, 8), (8, 8)),
            key=f'-CARD_{key_suffix}-'
        )

    def create_elevated_button(self, text, key, size=(12, 1), button_type='primary'):
        """创建Material Design 3风格的按钮"""
        if button_type == 'primary':
            return sg.Button(
                text,
                key=key,
                size=size,
                font=('Segoe UI', 10, 'bold'),
                button_color=(self.md3_colors['on_primary'], self.md3_colors['primary']),
                border_width=0,
                pad=((4, 4), (8, 8))
            )
        elif button_type == 'secondary':
            return sg.Button(
                text,
                key=key,
                size=size,
                font=('Segoe UI', 10),
                button_color=(self.md3_colors['on_secondary_container'], self.md3_colors['secondary_container']),
                border_width=1,
                pad=((4, 4), (8, 8))
            )
        else:  # outlined
            return sg.Button(
                text,
                key=key,
                size=size,
                font=('Segoe UI', 10),
                button_color=(self.md3_colors['primary'], self.md3_colors['surface']),
                border_width=1,
                pad=((4, 4), (8, 8))
            )

    def create_layout(self):
        """创建Material Design 3风格的GUI布局"""

        # 精简标题区域 - 更好的比例和平衡
        header_layout = [
            [sg.Text(
                'Color Quantification Tool',
                font=('Segoe UI', 28, 'bold'),
                text_color=self.md3_colors['primary'],
                background_color=self.md3_colors['background'],
                pad=((24, 24), (32, 8)),
                justification='center'
            )],
            [sg.Text(
                'Professional Color Quantification Platform',
                font=('Segoe UI', 13),
                text_color=self.md3_colors['on_surface_variant'],
                background_color=self.md3_colors['background'],
                pad=((24, 24), (0, 32)),
                justification='center'
            )]
        ]

        # 优化基础设置卡片 - 更好的间距和比例
        basic_content = [
            [sg.Text('Input Directory', font=('Segoe UI', 12, 'bold'),
                    text_color=self.md3_colors['on_surface'], pad=((0, 0), (12, 6)))],
            [sg.Input(
                key='-FOLDER-',
                size=(55, 1),
                font=('Segoe UI', 11),
                background_color=self.md3_colors['surface_variant'],
                text_color=self.md3_colors['on_surface_variant'],
                border_width=0,
                pad=((0, 12), (0, 12))
            ), sg.FolderBrowse(
                button_text='Browse',
                font=('Segoe UI', 10, 'bold'),
                button_color=(self.md3_colors['on_secondary_container'], self.md3_colors['secondary_container']),
                size=(8, 1)
            )],
            [sg.Text('Output Directory', font=('Segoe UI', 12, 'bold'),
                    text_color=self.md3_colors['on_surface'], pad=((0, 0), (20, 6)))],
            [sg.Input(
                key='-OUTPUT-',
                size=(55, 1),
                font=('Segoe UI', 11),
                background_color=self.md3_colors['surface_variant'],
                text_color=self.md3_colors['on_surface_variant'],
                border_width=0,
                pad=((0, 12), (0, 12))
            ), sg.FolderBrowse(
                button_text='选择',
                font=('Segoe UI', 10, 'bold'),
                button_color=(self.md3_colors['on_secondary_container'], self.md3_colors['secondary_container']),
                size=(8, 1)
            )]
        ]

        # 精简色系设置卡片 - 更紧凑的布局
        colormap_content = [
            [sg.Text('Colormap Configuration', font=('Segoe UI', 12, 'bold'),
                    text_color=self.md3_colors['on_surface'],
                    pad=((0, 0), (12, 16)))],
            [sg.Text('ΔE2000:', font=('Segoe UI', 11), size=(16, 1)),
             sg.Combo(self.analyzer.colormap_list, default_value='viridis', key='-CMAP_DE-',
                     font=('Segoe UI', 10), size=(18, 1), pad=((8, 0), (4, 4)))],
            [sg.Text('Bhattacharyya:', font=('Segoe UI', 11), size=(16, 1)),
             sg.Combo(self.analyzer.colormap_list, default_value='magma', key='-CMAP_BH-',
                     font=('Segoe UI', 10), size=(18, 1), pad=((8, 0), (4, 4)))],
            [sg.Text('LAB Euclidean:', font=('Segoe UI', 11), size=(16, 1)),
             sg.Combo(self.analyzer.colormap_list, default_value='plasma', key='-CMAP_LAB-',
                     font=('Segoe UI', 10), size=(18, 1), pad=((8, 0), (4, 4)))],
            [sg.Text('RGB Euclidean:', font=('Segoe UI', 11), size=(16, 1)),
             sg.Combo(self.analyzer.colormap_list, default_value='inferno', key='-CMAP_RGB-',
                     font=('Segoe UI', 10), size=(18, 1), pad=((8, 0), (4, 4)))],
            [sg.Text('Color Diversity:', font=('Segoe UI', 11), size=(16, 1)),
             sg.Combo(self.analyzer.colormap_list, default_value='cividis', key='-CMAP_DIV-',
                     font=('Segoe UI', 10), size=(18, 1), pad=((8, 0), (4, 4)))]
        ]

        # Simplified advanced settings card - better grouping and spacing
        advanced_content = [
            [sg.Text('Analysis Parameters', font=('Segoe UI', 12, 'bold'),
                    text_color=self.md3_colors['on_surface'],
                    pad=((0, 0), (12, 16)))],
            [sg.Text('Mask Threshold:', font=('Segoe UI', 11), size=(12, 1)),
             sg.Slider(range=(1, 100), default_value=20, orientation='h',
                      key='-THRESHOLD-', size=(32, 18), font=('Segoe UI', 9))],
            [sg.Text('Cluster Range:', font=('Segoe UI', 11), size=(12, 1)),
             sg.Input('2,3,4,5,6', key='-CLUSTERS-', size=(22, 1), font=('Segoe UI', 10))],

            [sg.Text('Analysis Options', font=('Segoe UI', 11, 'bold'), pad=((0, 0), (20, 12)))],
            [sg.Checkbox('Clustering Analysis', default=True, key='-ENABLE_CLUSTERING-', font=('Segoe UI', 10)),
             sg.Checkbox('Statistical Tests', default=True, key='-ENABLE_STATS-', font=('Segoe UI', 10))],
            [sg.Checkbox('Interactive Charts', default=True, key='-ENABLE_INTERACTIVE-', font=('Segoe UI', 10))],

            [sg.Text('Output Formats', font=('Segoe UI', 11, 'bold'), pad=((0, 0), (20, 12)))],
            [sg.Checkbox('Excel', default=True, key='-OUT_XLSX-', font=('Segoe UI', 10)),
             sg.Checkbox('CSV', default=True, key='-OUT_CSV-', font=('Segoe UI', 10)),
             sg.Checkbox('JSON', default=True, key='-OUT_JSON-', font=('Segoe UI', 10))],

            [sg.Text('Color Spaces', font=('Segoe UI', 11, 'bold'), pad=((0, 0), (20, 12)))],
            [sg.Checkbox('RGB', default=True, key='-SPACE_RGB-', font=('Segoe UI', 10)),
             sg.Checkbox('LAB', default=True, key='-SPACE_LAB-', font=('Segoe UI', 10)),
             sg.Checkbox('HSV', default=True, key='-SPACE_HSV-', font=('Segoe UI', 10))]
        ]

        # 优化标签页布局 - 更好的比例和间距
        tab_layout = [
            [sg.TabGroup([
                [sg.Tab('Basic Settings', [[self.create_card_frame('', basic_content, 'BASIC')]],
                       background_color=self.md3_colors['background'], border_width=0)],
                [sg.Tab('Colormap Config', [[self.create_card_frame('', colormap_content, 'COLORMAP')]],
                       background_color=self.md3_colors['background'], border_width=0)],
                [sg.Tab('Advanced Settings', [[self.create_card_frame('', advanced_content, 'ADVANCED')]],
                       background_color=self.md3_colors['background'], border_width=0)]
            ],
            font=('Segoe UI', 12, 'bold'),
            background_color=self.md3_colors['background'],
            selected_background_color=self.md3_colors['primary_container'],
            selected_title_color=self.md3_colors['on_primary_container'],
            title_color=self.md3_colors['on_surface_variant'],
            border_width=0,
            pad=((24, 24), (16, 24)))]
        ]

        # 优化操作按钮区域 - 更好的间距和比例
        button_layout = [
            [sg.Push(),
             self.create_elevated_button('Start Analysis', 'Start Analysis', (16, 2), 'primary'),
             sg.Text('    ', background_color=self.md3_colors['background']),
             self.create_elevated_button('Load Config', 'Load Config', (12, 2), 'secondary'),
             sg.Text('  ', background_color=self.md3_colors['background']),
             self.create_elevated_button('Save Config', 'Save Config', (12, 2), 'secondary'),
             sg.Text('  ', background_color=self.md3_colors['background']),
             self.create_elevated_button('Exit', 'Exit', (10, 2), 'outlined'),
             sg.Push()]
        ]

        # 优化进度条区域 - 更好的比例
        progress_layout = [
            [sg.Text('Analysis Progress', font=('Segoe UI', 11, 'bold'),
                    text_color=self.md3_colors['on_surface'], pad=((24, 0), (20, 8)))],
            [sg.ProgressBar(
                100, orientation='h', size=(80, 24), key='-PROGRESS-',
                bar_color=(self.md3_colors['primary'], self.md3_colors['surface_variant']),
                border_width=0,
                pad=((24, 24), (0, 20))
            )]
        ]

        # 优化输出日志区域 - 更好的比例和字体
        output_layout = [
            [sg.Text('Analysis Log', font=('Segoe UI', 11, 'bold'),
                    text_color=self.md3_colors['on_surface'], pad=((24, 0), (12, 8)))],
            [sg.Multiline(
                size=(95, 16),
                key='-OUTPUT-',
                autoscroll=True,
                disabled=True,
                font=('Consolas', 10),
                background_color=self.md3_colors['surface_variant'],
                text_color=self.md3_colors['on_surface_variant'],
                border_width=0,
                pad=((24, 24), (0, 24))
            )]
        ]

        # 主布局组合
        layout = [
            [sg.Column(header_layout, background_color=self.md3_colors['background'], pad=(0, 0))],
            [sg.Column(tab_layout, background_color=self.md3_colors['background'], pad=(0, 0))],
            [sg.Column(button_layout, background_color=self.md3_colors['background'], pad=((0, 0), (16, 16)))],
            [sg.Column(progress_layout, background_color=self.md3_colors['background'], pad=(0, 0))],
            [sg.Column(output_layout, background_color=self.md3_colors['background'], pad=(0, 0))]
        ]

        return layout

    def update_config_from_gui(self, values):
        """从GUI更新配置"""
        self.analyzer.config.update({
            'mask_threshold': int(values['-THRESHOLD-']),
            'clustering_analysis': values['-ENABLE_CLUSTERING-'],
            'statistical_tests': values['-ENABLE_STATS-'],
            'enable_interactive_plots': values['-ENABLE_INTERACTIVE-'],
            'output_formats': [
                fmt for fmt, enabled in [
                    ('xlsx', values['-OUT_XLSX-']),
                    ('csv', values['-OUT_CSV-']),
                    ('json', values['-OUT_JSON-'])
                ] if enabled
            ],
            'color_space_analysis': [
                space for space, enabled in [
                    ('RGB', values['-SPACE_RGB-']),
                    ('LAB', values['-SPACE_LAB-']),
                    ('HSV', values['-SPACE_HSV-'])
                ] if enabled
            ],
            'matrix_colormaps': {
                'delta_e': values['-CMAP_DE-'],
                'bhattacharyya': values['-CMAP_BH-'],
                'euclidean_lab': values['-CMAP_LAB-'],
                'euclidean_rgb': values['-CMAP_RGB-'],
                'diversity_distance': values['-CMAP_DIV-']
            }
        })

        # 解析聚类数范围
        try:
            clusters_str = values['-CLUSTERS-'].replace(' ', '')
            clusters = [int(x) for x in clusters_str.split(',') if x.isdigit()]
            self.analyzer.config['n_clusters_range'] = clusters
        except:
            pass

    def log_message(self, window, message):
        """在GUI中显示消息"""
        try:
            window['-OUTPUT-'].print(message)
            window.refresh()
        except:
            print(message)

    def create_md3_popup(self, title, message, popup_type='info'):
        """创建Material Design 3风格的弹窗"""
        if popup_type == 'error':
            icon = '❌'
            color = self.md3_colors['error']
        elif popup_type == 'success':
            icon = '✅'
            color = self.md3_colors['primary']
        else:
            icon = 'ℹ️'
            color = self.md3_colors['secondary']

        layout = [
            [sg.Text(f'{icon} {title}',
                    font=('Segoe UI', 14, 'bold'),
                    text_color=color,
                    background_color=self.md3_colors['surface'],
                    pad=((16, 16), (16, 8)))],
            [sg.Text(message,
                    font=('Segoe UI', 11),
                    text_color=self.md3_colors['on_surface'],
                    background_color=self.md3_colors['surface'],
                    pad=((16, 16), (8, 16)))],
            [sg.Push(),
             self.create_elevated_button('确定', '-OK-', (8, 1), 'primary'),
             sg.Push()]
        ]

        popup_window = sg.Window(
            title,
            layout,
            background_color=self.md3_colors['surface'],
            border_depth=0,
            no_titlebar=False,
            grab_anywhere=True,
            modal=True,
            finalize=True
        )

        while True:
            event, values = popup_window.read()
            if event in (sg.WIN_CLOSED, '-OK-'):
                break
        popup_window.close()

    def animate_progress(self, window, target_value, duration_ms=300):
        """Material Design 3风格的进度条动画"""
        current_value = 0
        steps = 20
        step_value = target_value / steps
        step_duration = duration_ms // steps

        for i in range(steps + 1):
            current_value = min(target_value, i * step_value)
            window['-PROGRESS-'].update(current_count=current_value)
            window.refresh()
            if i < steps:
                window.read(timeout=step_duration)

    def run(self):
        """运行Material Design 3风格的GUI"""
        layout = self.create_layout()

        # 创建主窗口
        window = sg.Window(
            'Color Quantification Tool v4.0',
            layout,
            resizable=True,
            finalize=True,
            background_color=self.md3_colors['background'],
            margins=(0, 0),
            element_padding=((0, 0), (0, 0)),
            border_depth=0,
            icon=None,
            size=(1280, 960),
            location=(None, None)
        )

        # Simplified welcome message
        welcome_msg = """Color Quantification Tool Started

✨ Core Features:
• Multi-color space analysis (RGB, LAB, HSV)
• Professional distance matrix algorithms
• 3D visualization charts
• Color harmony assessment
• Multi-format data export

📋 Workflow:
1. Select image folder
2. Set output directory
3. Configure analysis parameters
4. Start analysis

System ready, please begin your analysis work."""

        self.log_message(window, welcome_msg)

        while True:
            event, values = window.read(timeout=100)

            if event in (sg.WIN_CLOSED, 'Exit'):
                break

            elif event == 'Start Analysis':
                input_folder = values['-FOLDER-']
                output_folder = values['-OUTPUT-']

                if not input_folder or not os.path.isdir(input_folder):
                    self.create_md3_popup('Input Error', 'Please select a valid image folder!', 'error')
                    continue

                if not output_folder:
                    self.create_md3_popup('Input Error', 'Please select an output folder!', 'error')
                    continue

                # 更新配置
                self.update_config_from_gui(values)

                # 获取自定义色系设置
                custom_colormaps = {
                    'delta_e': values['-CMAP_DE-'],
                    'bhattacharyya': values['-CMAP_BH-'],
                    'euclidean_lab': values['-CMAP_LAB-'],
                    'euclidean_rgb': values['-CMAP_RGB-'],
                    'diversity_distance': values['-CMAP_DIV-']
                }

                # Start analysis
                try:
                    progress_step = 0
                    total_steps = 8

                    def progress_callback(message):
                        nonlocal progress_step
                        self.log_message(window, f"📊 {message}")
                        progress_step += 1
                        progress_value = int((progress_step / total_steps) * 100)
                        self.animate_progress(window, progress_value, 200)

                    self.log_message(window, "\n🚀 Starting comprehensive color analysis...")
                    self.log_message(window, f"📁 Input directory: {input_folder}")
                    self.log_message(window, f"💾 Output directory: {output_folder}")
                    self.log_message(window, f"🎨 Colormap configuration: {custom_colormaps}")

                    window['-PROGRESS-'].update(current_count=0)

                    results = self.analyzer.run_comprehensive_analysis(
                        input_folder, output_folder, progress_callback, custom_colormaps
                    )

                    self.animate_progress(window, 100, 300)
                    self.log_message(window, "\n✅ Analysis completed!")

                    # Display detailed result summary
                    summary = f"""
🎉 Analysis Completion Summary Report:

📊 Data Statistics:
   • Number of processed images: {len(results['image_files'])} images
   • Generated distance matrices: {len(results['distance_matrices'])} matrices
   • Generated visualization files: {len(results['visualization_files'])} files

📁 Output Files:
   • Excel report: {os.path.basename(results['excel_path'])}
   • Visualization charts: {', '.join(results['visualization_files'].keys())}

🎨 Colormap Configuration Used:
   • ΔE2000: {custom_colormaps['delta_e']}
   • Bhattacharyya: {custom_colormaps['bhattacharyya']}
   • LAB Euclidean: {custom_colormaps['euclidean_lab']}
   • RGB Euclidean: {custom_colormaps['euclidean_rgb']}
   • Color Diversity: {custom_colormaps['diversity_distance']}

📈 Generated Professional Charts:
   • 3D color space distribution plots
   • Dominant color extraction and distribution charts
   • Color harmony analysis charts
   • Interactive HTML dashboard
   • Multiple distance matrix heatmaps

🎯 All files have been saved to the output directory and are ready for use!"""

                    self.log_message(window, summary)
                    self.create_md3_popup('Analysis Complete', f'All results have been successfully saved to:\n{output_folder}\n\nPlease check the files in the output directory!', 'success')

                except Exception as e:
                    self.log_message(window, f"\n❌ Error occurred during analysis: {str(e)}")
                    self.create_md3_popup('Analysis Failed', f'Error occurred during analysis:\n{str(e)}\n\nPlease check input files and settings, then try again.', 'error')

                finally:
                    window['-PROGRESS-'].update(current_count=0)

            elif event == 'Load Config':
                config_file = sg.popup_get_file(
                    'Select Configuration File',
                    file_types=(('JSON Config Files', '*.json'),),
                    no_window=True
                )
                if config_file:
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            cfg = json.load(f)
                        self.analyzer.config.update(cfg)
                        self.log_message(window, f"📂 Configuration file loaded successfully: {os.path.basename(config_file)}")
                        self.create_md3_popup('Load Successful', f'Configuration file loaded successfully:\n{os.path.basename(config_file)}', 'success')
                    except Exception as e:
                        self.log_message(window, f"❌ Configuration loading failed: {str(e)}")
                        self.create_md3_popup('Load Failed', f'Configuration file loading failed:\n{str(e)}', 'error')

            elif event == 'Save Config':
                config_file = sg.popup_get_file(
                    'Save Configuration File',
                    save_as=True,
                    file_types=(('JSON Config Files', '*.json'),),
                    default_extension='.json',
                    no_window=True
                )
                if config_file:
                    try:
                        self.update_config_from_gui(values)
                        with open(config_file, 'w', encoding='utf-8') as f:
                            json.dump(self.analyzer.config, f, ensure_ascii=False, indent=2)
                        self.log_message(window, f"💾 Configuration file saved successfully: {os.path.basename(config_file)}")
                        self.create_md3_popup('Save Successful', f'Configuration file saved successfully:\n{os.path.basename(config_file)}', 'success')
                    except Exception as e:
                        self.log_message(window, f"❌ Configuration saving failed: {str(e)}")
                        self.create_md3_popup('Save Failed', f'Configuration file saving failed:\n{str(e)}', 'error')

        window.close()


def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser(description='Advanced Color Analysis Tool v1.0')
    parser.add_argument('--gui', action='store_true', help='Launch graphical interface')
    parser.add_argument('--input', help='Input folder path')
    parser.add_argument('--output', help='Output folder path')
    parser.add_argument('--config', help='Configuration file path')

    args = parser.parse_args()

    # GUI mode
    if args.gui or (not args.input and not args.output):
        try:
            gui = AdvancedGUI()
            gui.run()
        except ImportError:
            print("GUI unavailable, please install PySimpleGUI or use command line mode")
        except Exception as e:
            print(f"GUI startup failed: {e}")
        return

    # Command line mode
    if not args.input or not args.output:
        print("Command line mode requires input and output folders")
        print("Usage: python color_analysis.py --input image_folder --output output_folder")
        return

    try:
        analyzer = AdvancedColorAnalyzer(args.config if args.config else "color_analysis_config.json")

        def progress_callback(message):
            print(message)

        print("Starting color analysis...")
        results = analyzer.run_comprehensive_analysis(args.input, args.output, progress_callback)
        print("✅ Analysis completed!")
        print(f"Results saved to: {args.output}")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")


if __name__ == "__main__":
      main()
