#!/usr/bin/env python3
"""
Disaster Analysis Main Script

This script implements the core disaster analysis functionality:
1. Generate building density heatmap from before-disaster image
2. Generate damage assessment heatmap from before/after comparison
3. Fuse both heatmaps to create final emergency priority heatmap

Usage:
    python disaster_analysis_main.py --before before.png --after after.png --output results/
    python disaster_analysis_main.py --example4  # Use example4 test data

Author: AWS Hackathon Team
Date: 2025
"""

import argparse
import sys
import os
from pathlib import Path
import cv2
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional, Any

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import core modules
from mapping import process_image_pair
from house import create_light_gray_heatmap, overlay_heatmap_on_image
from fusion import (
    create_damage_heatmap, 
    sample_density_for_patches, 
    fuse_heatmaps, 
    calculate_urgency,
    save_heatmap_image,
    create_fusion_map
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DisasterAnalyzer:
    """
    Main disaster analysis class that orchestrates the complete workflow.
    """
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize the disaster analyzer.
        
        Args:
            output_dir: Directory to save all output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "heatmaps").mkdir(exist_ok=True)
        (self.output_dir / "overlays").mkdir(exist_ok=True)
        (self.output_dir / "geojson").mkdir(exist_ok=True)
        (self.output_dir / "html_maps").mkdir(exist_ok=True)
        
        # Analysis results storage
        self.results = {}
        
        logger.info(f"DisasterAnalyzer initialized with output directory: {self.output_dir}")
    
    def analyze_building_density(self, before_image_path: str) -> Dict[str, Any]:
        """
        Step 1: Analyze building density from before-disaster image.
        
        Args:
            before_image_path: Path to before-disaster image
            
        Returns:
            Dictionary containing density analysis results
        """
        logger.info("=" * 60)
        logger.info("Step 1: Building Density Analysis")
        logger.info("=" * 60)
        
        # Load before image
        before_image = cv2.imread(before_image_path)
        if before_image is None:
            raise FileNotFoundError(f"Could not load before image: {before_image_path}")
        
        logger.info(f"Loaded before image: {before_image_path}")
        logger.info(f"Image shape: {before_image.shape}")
        
        # Generate building density heatmap
        density_heatmap = create_light_gray_heatmap(
            before_image, 
            radius=60, 
            sigma=30.0
        )
        
        # Save density heatmap
        density_heatmap_path = self.output_dir / "heatmaps" / "building_density_heatmap.png"
        save_heatmap_image(
            density_heatmap, 
            str(density_heatmap_path), 
            cv2.COLORMAP_JET
        )
        
        # Create density overlay
        density_colored = cv2.applyColorMap(
            (density_heatmap * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        density_overlay = overlay_heatmap_on_image(
            before_image, 
            density_colored, 
            alpha=0.5
        )
        density_overlay_path = self.output_dir / "overlays" / "building_density_overlay.png"
        cv2.imwrite(str(density_overlay_path), density_overlay)
        
        # Store results
        density_results = {
            'heatmap_path': str(density_heatmap_path),
            'overlay_path': str(density_overlay_path),
            'heatmap': density_heatmap,
            'original_image': before_image
        }
        
        self.results['building_density'] = density_results
        logger.info("✓ Building density analysis completed")
        
        return density_results
    
    def analyze_damage_assessment(self, before_image_path: str, after_image_path: str, 
                                 grid_size: int = 32, bbox: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Step 2: Analyze damage assessment from before/after comparison.
        
        Args:
            before_image_path: Path to before-disaster image
            after_image_path: Path to after-disaster image
            grid_size: Size of grid patches for analysis
            bbox: Optional bounding box [min_lon, min_lat, max_lon, max_lat]
            
        Returns:
            Dictionary containing damage analysis results
        """
        logger.info("=" * 60)
        logger.info("Step 2: Damage Assessment Analysis")
        logger.info("=" * 60)
        
        # Load after image for dimensions
        after_image = cv2.imread(after_image_path)
        if after_image is None:
            raise FileNotFoundError(f"Could not load after image: {after_image_path}")
        
        logger.info(f"Loaded after image: {after_image_path}")
        logger.info(f"Image shape: {after_image.shape}")
        
        # Process image pair to get damage assessment
        geojson_path = self.output_dir / "geojson" / "damage_assessment.geojson"
        
        # Use default bbox if not provided
        if bbox is None:
            bbox = [-86.8, 36.1, -86.7, 36.2]  # Default Nashville area
        
        damage_geojson, patch_results_raw = process_image_pair(
            before_path=before_image_path,
            after_path=after_image_path,
            output_geojson_path=str(geojson_path),
            grid_size=grid_size,
            bbox=bbox,
            normalize=True
        )
        
        # Combine patch results with geojson features
        patch_features = damage_geojson['features']
        for i, feature in enumerate(patch_features):
            feature.update(patch_results_raw[i])
            # Remove non-serializable objects
            if 'features' in feature:
                del feature['features']
        
        logger.info(f"Found {len(patch_features)} patches for analysis")
        
        # Create damage heatmap
        damage_heatmap = create_damage_heatmap(patch_features, after_image.shape)
        
        # Save damage heatmap
        damage_heatmap_path = self.output_dir / "heatmaps" / "damage_heatmap.png"
        save_heatmap_image(
            damage_heatmap, 
            str(damage_heatmap_path), 
            cv2.COLORMAP_HOT
        )
        
        # Create damage overlay
        damage_colored = cv2.applyColorMap(
            (damage_heatmap * 255).astype(np.uint8), 
            cv2.COLORMAP_HOT
        )
        damage_overlay = overlay_heatmap_on_image(
            after_image, 
            damage_colored, 
            alpha=0.5
        )
        damage_overlay_path = self.output_dir / "overlays" / "damage_overlay.png"
        cv2.imwrite(str(damage_overlay_path), damage_overlay)
        
        # Store results
        damage_results = {
            'geojson_path': str(geojson_path),
            'heatmap_path': str(damage_heatmap_path),
            'overlay_path': str(damage_overlay_path),
            'heatmap': damage_heatmap,
            'patch_features': patch_features,
            'original_image': after_image
        }
        
        self.results['damage_assessment'] = damage_results
        logger.info("✓ Damage assessment analysis completed")
        
        return damage_results
    
    def fuse_analysis_results(self, damage_weight: float = 0.6, density_weight: float = 0.4) -> Dict[str, Any]:
        """
        Step 3: Fuse building density and damage assessment results.
        
        Args:
            damage_weight: Weight for damage assessment (0-1)
            density_weight: Weight for building density (0-1)
            
        Returns:
            Dictionary containing fused analysis results
        """
        logger.info("=" * 60)
        logger.info("Step 3: Fusing Analysis Results")
        logger.info("=" * 60)
        
        # Get previous results
        density_results = self.results['building_density']
        damage_results = self.results['damage_assessment']
        
        density_heatmap = density_results['heatmap']
        damage_heatmap = damage_results['heatmap']
        patch_features = damage_results['patch_features']
        
        # Fuse the heatmaps
        fusion_weights = {'damage': damage_weight, 'density': density_weight}
        fused_heatmap = fuse_heatmaps(
            damage_heatmap, 
            density_heatmap, 
            fusion_weights
        )
        
        # Save fused heatmap
        fused_heatmap_path = self.output_dir / "heatmaps" / "fused_heatmap.png"
        save_heatmap_image(
            fused_heatmap, 
            str(fused_heatmap_path), 
            cv2.COLORMAP_JET
        )
        
        # Add density scores to patch features
        patches_with_density = sample_density_for_patches(density_heatmap, patch_features)
        
        # Calculate final urgency scores
        final_patches = calculate_urgency(patches_with_density, fusion_weights)
        
        # Create fused overlay
        after_image = damage_results['original_image']
        fused_colored = cv2.applyColorMap(
            (fused_heatmap * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        fused_overlay = overlay_heatmap_on_image(
            after_image, 
            fused_colored, 
            alpha=0.5
        )
        fused_overlay_path = self.output_dir / "overlays" / "fused_overlay.png"
        cv2.imwrite(str(fused_overlay_path), fused_overlay)
        
        # Store results
        fusion_results = {
            'heatmap_path': str(fused_heatmap_path),
            'overlay_path': str(fused_overlay_path),
            'heatmap': fused_heatmap,
            'patch_features': final_patches,
            'weights': fusion_weights
        }
        
        self.results['fusion'] = fusion_results
        logger.info("✓ Fusion analysis completed")
        
        return fusion_results
    
    def create_interactive_map(self, bbox: Optional[List[float]] = None) -> str:
        """
        Step 4: Create interactive HTML map with all results.
        
        Args:
            bbox: Optional bounding box for map extent
            
        Returns:
            Path to generated HTML map
        """
        logger.info("=" * 60)
        logger.info("Step 4: Creating Interactive Map")
        logger.info("=" * 60)
        
        fusion_results = self.results['fusion']
        patch_features = fusion_results['patch_features']
        damage_results = self.results['damage_assessment']
        after_image = damage_results['original_image']
        
        # Create interactive fusion map
        html_map_path = self.output_dir / "html_maps" / "disaster_analysis_map.html"
        
        # Save after image temporarily for map creation
        temp_after_path = self.output_dir / "temp_after_image.png"
        cv2.imwrite(str(temp_after_path), after_image)
        
        create_fusion_map(
            patch_features=patch_features,
            post_image_path=str(temp_after_path),
            bbox=bbox or [-86.8, 36.1, -86.7, 36.2],  # Default Nashville area
            output_html_path=str(html_map_path)
        )
        
        # Clean up temporary file
        if temp_after_path.exists():
            temp_after_path.unlink()
        
        logger.info("✓ Interactive map created")
        return str(html_map_path)
    
    def run_complete_analysis(self, before_image_path: str, after_image_path: str,
                            grid_size: int = 32, bbox: Optional[List[float]] = None,
                            damage_weight: float = 0.6, density_weight: float = 0.4) -> Dict[str, Any]:
        """
        Run the complete disaster analysis workflow.
        
        Args:
            before_image_path: Path to before-disaster image
            after_image_path: Path to after-disaster image
            grid_size: Size of grid patches for analysis
            bbox: Optional bounding box [min_lon, min_lat, max_lon, max_lat]
            damage_weight: Weight for damage assessment (0-1)
            density_weight: Weight for building density (0-1)
            
        Returns:
            Dictionary containing all analysis results
        """
        logger.info("🚀 Starting Complete Disaster Analysis")
        logger.info("=" * 80)
        
        try:
            # Step 1: Building density analysis
            density_results = self.analyze_building_density(before_image_path)
            
            # Step 2: Damage assessment analysis
            damage_results = self.analyze_damage_assessment(
                before_image_path, after_image_path, grid_size, bbox
            )
            
            # Step 3: Fuse results
            fusion_results = self.fuse_analysis_results(damage_weight, density_weight)
            
            # Step 4: Create interactive map
            html_map_path = self.create_interactive_map(bbox)
            
            # Summary
            logger.info("=" * 80)
            logger.info("🎉 Analysis Complete!")
            logger.info("=" * 80)
            logger.info(f"📁 Output directory: {self.output_dir}")
            logger.info(f"🗺️  Interactive map: {html_map_path}")
            logger.info(f"📊 Building density heatmap: {density_results['heatmap_path']}")
            logger.info(f"🔥 Damage assessment heatmap: {damage_results['heatmap_path']}")
            logger.info(f"⚡ Fused emergency heatmap: {fusion_results['heatmap_path']}")
            logger.info(f"📋 GeoJSON data: {damage_results['geojson_path']}")
            
            return self.results
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {str(e)}")
            raise


def get_example4_paths() -> Tuple[str, str]:
    """Get paths to example4 test images."""
    backend_dir = Path(__file__).parent
    before_path = backend_dir / "test_data" / "mapping examples" / "4pre.png"
    after_path = backend_dir / "test_data" / "mapping examples" / "4post.png"
    
    if not before_path.exists() or not after_path.exists():
        raise FileNotFoundError(
            f"Example4 images not found. Expected:\n"
            f"  Before: {before_path}\n"
            f"  After: {after_path}"
        )
    
    return str(before_path), str(after_path)


def main():
    """Main function to run disaster analysis."""
    parser = argparse.ArgumentParser(
        description="Disaster Analysis - Building Density + Damage Assessment + Fusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use example4 test data
  python disaster_analysis_main.py --example4
  
  # Use custom images
  python disaster_analysis_main.py --before before.png --after after.png
  
  # Use custom images with bounding box
  python disaster_analysis_main.py --before before.png --after after.png \\
    --bbox "-86.8,36.1,-86.7,36.2"
  
  # Custom weights and grid size
  python disaster_analysis_main.py --before before.png --after after.png \\
    --damage-weight 0.7 --density-weight 0.3 --grid-size 64
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--example4", 
        action="store_true",
        help="Use example4 test data (4pre.png and 4post.png)"
    )
    input_group.add_argument(
        "--before", 
        type=str,
        help="Path to before-disaster image"
    )
    
    parser.add_argument(
        "--after", 
        type=str,
        help="Path to after-disaster image (required if not using --example4)"
    )
    
    # Analysis parameters
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="outputs",
        help="Output directory for results (default: outputs)"
    )
    parser.add_argument(
        "--grid-size", 
        type=int, 
        default=32,
        help="Grid patch size for analysis (default: 32)"
    )
    parser.add_argument(
        "--bbox", 
        type=str,
        help="Bounding box as 'min_lon,min_lat,max_lon,max_lat' (optional)"
    )
    parser.add_argument(
        "--damage-weight", 
        type=float, 
        default=0.6,
        help="Weight for damage assessment (default: 0.6)"
    )
    parser.add_argument(
        "--density-weight", 
        type=float, 
        default=0.4,
        help="Weight for building density (default: 0.4)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if args.example4:
        before_path, after_path = get_example4_paths()
        logger.info("Using example4 test data")
    else:
        if not args.after:
            parser.error("--after is required when not using --example4")
        before_path = args.before
        after_path = args.after
        
        # Check if files exist
        if not Path(before_path).exists():
            parser.error(f"Before image not found: {before_path}")
        if not Path(after_path).exists():
            parser.error(f"After image not found: {after_path}")
    
    # Parse bounding box if provided
    bbox = None
    if args.bbox:
        try:
            bbox = [float(x.strip()) for x in args.bbox.split(',')]
            if len(bbox) != 4:
                raise ValueError("Bounding box must have 4 values")
        except ValueError as e:
            parser.error(f"Invalid bounding box format: {e}")
    
    # Validate weights
    if abs(args.damage_weight + args.density_weight - 1.0) > 0.01:
        logger.warning(f"Weights don't sum to 1.0: damage={args.damage_weight}, density={args.density_weight}")
    
    # Run analysis
    try:
        analyzer = DisasterAnalyzer(args.output_dir)
        results = analyzer.run_complete_analysis(
            before_image_path=before_path,
            after_image_path=after_path,
            grid_size=args.grid_size,
            bbox=bbox,
            damage_weight=args.damage_weight,
            density_weight=args.density_weight
        )
        
        logger.info("✅ Analysis completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
