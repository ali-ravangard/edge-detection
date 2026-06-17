"""
Visualization Module
Handles plotting and saving of edge detection results
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
import os


class Visualizer:
    """
    A class for visualizing edge detection results.
    """
    
    @staticmethod
    def plot_comparison(original: np.ndarray,
                       grayscale: np.ndarray,
                       canny: np.ndarray,
                       sobel: np.ndarray,
                       title: str = "Edge Detection Comparison",
                       save_path: Optional[str] = None) -> None:
        """
        Plot original, grayscale, Canny, and Sobel results side by side.
        
        Args:
            original: Original color image
            grayscale: Grayscale image
            canny: Canny edge detection result
            sobel: Sobel edge detection result
            title: Plot title
            save_path: Path to save the figure (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Original image (convert BGR to RGB for matplotlib)
        axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image', fontsize=12)
        axes[0, 0].axis('off')
        
        # Grayscale image
        axes[0, 1].imshow(grayscale, cmap='gray')
        axes[0, 1].set_title('Grayscale Image', fontsize=12)
        axes[0, 1].axis('off')
        
        # Canny edges
        axes[1, 0].imshow(canny, cmap='gray')
        axes[1, 0].set_title('Canny Edge Detection', fontsize=12)
        axes[1, 0].axis('off')
        
        # Sobel edges
        axes[1, 1].imshow(sobel, cmap='gray')
        axes[1, 1].set_title('Sobel Edge Detection', fontsize=12)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_edge_comparison(canny: np.ndarray,
                            sobel: np.ndarray,
                            title: str = "Canny vs Sobel Edge Detection",
                            save_path: Optional[str] = None) -> None:
        """
        Plot Canny and Sobel results side by side for detailed comparison.
        
        Args:
            canny: Canny edge detection result
            sobel: Sobel edge detection result
            title: Plot title
            save_path: Path to save the figure (optional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Canny edges
        axes[0].imshow(canny, cmap='gray')
        axes[0].set_title('Canny Edge Detection\n(Binary edges with hysteresis)', fontsize=11)
        axes[0].axis('off')
        
        # Sobel edges
        axes[1].imshow(sobel, cmap='gray')
        axes[1].set_title('Sobel Edge Detection\n(Gradient magnitude)', fontsize=11)
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
    
    @staticmethod
    def save_individual_results(grayscale: np.ndarray,
                               canny: np.ndarray,
                               sobel: np.ndarray,
                               output_dir: str,
                               prefix: str = "result") -> None:
        """
        Save individual result images to disk.
        
        Args:
            grayscale: Grayscale image
            canny: Canny edge detection result
            sobel: Sobel edge detection result
            output_dir: Directory to save results
            prefix: Filename prefix
        """
        os.makedirs(output_dir, exist_ok=True)
        
        cv2.imwrite(os.path.join(output_dir, f"{prefix}_grayscale.png"), grayscale)
        cv2.imwrite(os.path.join(output_dir, f"{prefix}_canny.png"), canny)
        cv2.imwrite(os.path.join(output_dir, f"{prefix}_sobel.png"), sobel)
        
        print(f"Individual results saved to: {output_dir}")
    
    @staticmethod
    def plot_statistics(stats: dict, save_path: Optional[str] = None) -> None:
        """
        Plot statistics comparison between Canny and Sobel methods.
        
        Args:
            stats: Dictionary containing statistics from EdgeDetector.get_edge_statistics()
            save_path: Path to save the figure (optional)
        """
        if 'canny' not in stats or 'sobel' not in stats:
            print("Insufficient statistics data for plotting.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Edge Detection Statistics Comparison', fontsize=16, fontweight='bold')
        
        # Edge percentage comparison
        methods = ['Canny', 'Sobel']
        edge_percentages = [stats['canny']['edge_percentage'], 
                           stats['sobel']['edge_percentage']]
        
        axes[0].bar(methods, edge_percentages, color=['#2E86AB', '#A23B72'])
        axes[0].set_ylabel('Edge Pixels (%)', fontsize=11)
        axes[0].set_title('Edge Pixel Percentage', fontsize=12)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(edge_percentages):
            axes[0].text(i, v + 0.5, f'{v:.2f}%', ha='center', fontweight='bold')
        
        # Mean intensity comparison
        mean_intensities = [stats['canny']['mean_intensity'],
                           stats['sobel']['mean_intensity']]
        
        axes[1].bar(methods, mean_intensities, color=['#2E86AB', '#A23B72'])
        axes[1].set_ylabel('Mean Intensity', fontsize=11)
        axes[1].set_title('Mean Edge Intensity', fontsize=12)
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(mean_intensities):
            axes[1].text(i, v + 1, f'{v:.2f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Statistics plot saved to: {save_path}")
        
        plt.show()
