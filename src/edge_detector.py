"""
Edge Detection Module
Implements Canny and Sobel edge detection algorithms
"""

import cv2
import numpy as np
from typing import Tuple, Dict


class EdgeDetector:
    """
    A class for performing edge detection using Canny and Sobel methods.
    """
    
    def __init__(self):
        """Initialize the EdgeDetector."""
        self.original_image = None
        self.gray_image = None
        self.canny_edges = None
        self.sobel_edges = None
        
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from the specified path.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Loaded image as numpy array
        """
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        return self.original_image
    
    def convert_to_grayscale(self) -> np.ndarray:
        """
        Convert the loaded image to grayscale.
        
        Returns:
            Grayscale image as numpy array
        """
        if self.original_image is None:
            raise ValueError("No image loaded. Call load_image() first.")
        
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        return self.gray_image
    
    def apply_canny(self, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
        """
        Apply Canny edge detection algorithm.
        
        Args:
            low_threshold: Lower threshold for edge detection
            high_threshold: Upper threshold for edge detection
            
        Returns:
            Edge-detected image using Canny method
        """
        if self.gray_image is None:
            raise ValueError("No grayscale image available. Call convert_to_grayscale() first.")
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(self.gray_image, (5, 5), 1.4)
        
        # Apply Canny edge detection
        self.canny_edges = cv2.Canny(blurred, low_threshold, high_threshold)
        return self.canny_edges
    
    def apply_sobel(self, ksize: int = 3) -> np.ndarray:
        """
        Apply Sobel edge detection algorithm.
        
        Args:
            ksize: Size of the Sobel kernel (must be odd: 1, 3, 5, 7)
            
        Returns:
            Edge-detected image using Sobel method
        """
        if self.gray_image is None:
            raise ValueError("No grayscale image available. Call convert_to_grayscale() first.")
        
        # Apply Sobel operator in X direction
        sobel_x = cv2.Sobel(self.gray_image, cv2.CV_64F, 1, 0, ksize=ksize)
        
        # Apply Sobel operator in Y direction
        sobel_y = cv2.Sobel(self.gray_image, cv2.CV_64F, 0, 1, ksize=ksize)
        
        # Compute gradient magnitude
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Normalize to 0-255 range
        self.sobel_edges = np.uint8(sobel_magnitude / sobel_magnitude.max() * 255)
        
        return self.sobel_edges
    
    def get_edge_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for detected edges.
        
        Returns:
            Dictionary containing statistics for both methods
        """
        stats = {}
        
        if self.canny_edges is not None:
            canny_edge_pixels = np.count_nonzero(self.canny_edges)
            total_pixels = self.canny_edges.size
            stats['canny'] = {
                'edge_pixels': int(canny_edge_pixels),
                'total_pixels': int(total_pixels),
                'edge_percentage': float(canny_edge_pixels / total_pixels * 100),
                'mean_intensity': float(np.mean(self.canny_edges)),
                'std_intensity': float(np.std(self.canny_edges))
            }
        
        if self.sobel_edges is not None:
            # Threshold Sobel edges for fair comparison
            sobel_binary = cv2.threshold(self.sobel_edges, 50, 255, cv2.THRESH_BINARY)[1]
            sobel_edge_pixels = np.count_nonzero(sobel_binary)
            total_pixels = self.sobel_edges.size
            stats['sobel'] = {
                'edge_pixels': int(sobel_edge_pixels),
                'total_pixels': int(total_pixels),
                'edge_percentage': float(sobel_edge_pixels / total_pixels * 100),
                'mean_intensity': float(np.mean(self.sobel_edges)),
                'std_intensity': float(np.std(self.sobel_edges))
            }
        
        return stats
    
    def process_image(self, image_path: str, 
                     canny_low: int = 50, 
                     canny_high: int = 150,
                     sobel_ksize: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete pipeline: load, convert, and apply both edge detection methods.
        
        Args:
            image_path: Path to input image
            canny_low: Canny lower threshold
            canny_high: Canny upper threshold
            sobel_ksize: Sobel kernel size
            
        Returns:
            Tuple of (grayscale_image, canny_edges, sobel_edges)
        """
        self.load_image(image_path)
        gray = self.convert_to_grayscale()
        canny = self.apply_canny(canny_low, canny_high)
        sobel = self.apply_sobel(sobel_ksize)
        
        return gray, canny, sobel
