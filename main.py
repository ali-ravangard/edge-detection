"""
Main script for edge detection project
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from edge_detector import EdgeDetector
from visualizer import Visualizer


def print_statistics(stats: dict) -> None:
    """Print edge detection statistics in a formatted way."""
    print("\n" + "="*60)
    print("EDGE DETECTION STATISTICS")
    print("="*60)
    
    if 'canny' in stats:
        print("\n📊 CANNY EDGE DETECTION:")
        print(f"  • Edge pixels: {stats['canny']['edge_pixels']:,}")
        print(f"  • Total pixels: {stats['canny']['total_pixels']:,}")
        print(f"  • Edge percentage: {stats['canny']['edge_percentage']:.2f}%")
        print(f"  • Mean intensity: {stats['canny']['mean_intensity']:.2f}")
        print(f"  • Std intensity: {stats['canny']['std_intensity']:.2f}")
    
    if 'sobel' in stats:
        print("\n📊 SOBEL EDGE DETECTION:")
        print(f"  • Edge pixels: {stats['sobel']['edge_pixels']:,}")
        print(f"  • Total pixels: {stats['sobel']['total_pixels']:,}")
        print(f"  • Edge percentage: {stats['sobel']['edge_percentage']:.2f}%")
        print(f"  • Mean intensity: {stats['sobel']['mean_intensity']:.2f}")
        print(f"  • Std intensity: {stats['sobel']['std_intensity']:.2f}")
    
    print("\n" + "="*60 + "\n")


def main():
    """Main execution function."""
    
    # Setup paths
    input_dir = Path("data/input")
    output_dir = Path("data/output")
    
    # Create directories if they don't exist
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("EDGE DETECTION PROJECT")
    print("="*60)
    
    # Check for input images
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpeg"))
    
    if not image_files:
        print("\n⚠️  No images found in data/input/")
        print("Please add some images to the data/input/ directory and run again.")
        print("\nYou can use any .jpg, .png, or .jpeg images.")
        return
    
    print(f"\n✓ Found {len(image_files)} image(s) to process\n")
    
    # Process each image
    for idx, image_path in enumerate(image_files, 1):
        print(f"\n{'─'*60}")
        print(f"Processing image {idx}/{len(image_files)}: {image_path.name}")
        print(f"{'─'*60}")
        
        try:
            # Initialize detector
            detector = EdgeDetector()
            
            # Process image
            print("  [1/5] Loading image...")
            detector.load_image(str(image_path))
            
            print("  [2/5] Converting to grayscale...")
            gray = detector.convert_to_grayscale()
            
            print("  [3/5] Applying Canny edge detection...")
            canny = detector.apply_canny(low_threshold=50, high_threshold=150)
            
            print("  [4/5] Applying Sobel edge detection...")
            sobel = detector.apply_sobel(ksize=3)
            
            print("  [5/5] Generating visualizations...")
            
            # Get statistics
            stats = detector.get_edge_statistics()
            print_statistics(stats)
            
            # Create output filename prefix
            output_prefix = image_path.stem
            
            # Save individual results
            Visualizer.save_individual_results(
                gray, canny, sobel,
                str(output_dir),
                prefix=output_prefix
            )
            
            # Plot and save comparison
            comparison_path = output_dir / f"{output_prefix}_comparison.png"
            Visualizer.plot_comparison(
                detector.original_image,
                gray,
                canny,
                sobel,
                title=f"Edge Detection: {image_path.name}",
                save_path=str(comparison_path)
            )
            
            # Plot edge comparison
            edge_comparison_path = output_dir / f"{output_prefix}_edge_comparison.png"
            Visualizer.plot_edge_comparison(
                canny,
                sobel,
                title=f"Canny vs Sobel: {image_path.name}",
                save_path=str(edge_comparison_path)
            )
            
            # Plot statistics
            stats_path = output_dir / f"{output_prefix}_statistics.png"
            Visualizer.plot_statistics(stats, save_path=str(stats_path))
            
            print(f"\n✓ Successfully processed: {image_path.name}")
            
        except Exception as e:
            print(f"\n✗ Error processing {image_path.name}: {str(e)}")
            continue
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"\n✓ All results saved to: {output_dir.absolute()}")
    print("\nGenerated files for each image:")
    print("  • *_grayscale.png - Grayscale conversion")
    print("  • *_canny.png - Canny edge detection result")
    print("  • *_sobel.png - Sobel edge detection result")
    print("  • *_comparison.png - Side-by-side comparison")
    print("  • *_edge_comparison.png - Detailed edge comparison")
    print("  • *_statistics.png - Statistical analysis")
    print()


if __name__ == "__main__":
    main()
