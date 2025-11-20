#!/usr/bin/env python3
"""
YOLO Model Export Utility

This script exports ULTRALYTICS YOLO models to ONNX format with proper post-processing
for direct usability. The exported models include NMS (Non-Maximum Suppression) and 
other necessary post-processing steps built into the ONNX model.

Author: Assistant
Date: November 2025
"""

import os
from pathlib import Path
from typing import Dict, Optional, Union, Tuple
import logging

from ultralytics import YOLO


def setup_logging(verbose: bool = True) -> logging.Logger:
    """Set up logging configuration."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def export_yolo_models_to_onnx(
    model_weights_dict: Dict[str, str],
    output_folder: Union[str, Path],
    imgsz: Union[int, Tuple[int, int]] = 640,
    half: bool = False,
    dynamic: bool = False,
    simplify: bool = True,
    opset: int = 11,
    verbose: bool = True
) -> Dict[str, str]:
    """
    Export ULTRALYTICS YOLO models to ONNX format with proper post-processing.
    
    Args:
        model_weights_dict (Dict[str, str]): Dictionary mapping model weight paths to export names
                                           Example: {"/path/to/model.pt": "arthropod_detector"}
        output_folder (Union[str, Path]): Directory to save exported ONNX models
        imgsz (Union[int, Tuple[int, int]]): Input image size for export (default: 640)
        half (bool): Export in FP16 half precision (default: False)
        dynamic (bool): Enable dynamic input shapes (default: False)
        simplify (bool): Simplify ONNX model (default: True)
        opset (int): ONNX opset version (default: 11)
        verbose (bool): Enable verbose logging (default: True)
    
    Returns:
        Dict[str, str]: Dictionary mapping export names to exported ONNX file paths
        
    Raises:
        FileNotFoundError: If model weight file doesn't exist
        ValueError: If invalid parameters are provided
        RuntimeError: If export fails
    
    Example:
        >>> model_dict = {
        ...     "runs/train/exp/weights/best.pt": "yolo11n_arthropod",
        ...     "runs/train/exp2/weights/best.pt": "yolo11l_arthropod",
        ... }
        >>> exported = export_yolo_models_to_onnx(
        ...     model_dict, 
        ...     output_folder="exported_models",
        ...     imgsz=640
        ... )
        >>> print(exported)
        {'yolo11n_arthropod': 'exported_models/yolo11n_arthropod.onnx', ...}
    """
    
    logger = setup_logging(verbose)
    
    # Validate inputs
    if not model_weights_dict:
        raise ValueError("model_weights_dict cannot be empty")
    
    if not isinstance(output_folder, Path):
        output_folder = Path(output_folder)
    
    # Create output directory
    output_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_folder.absolute()}")
    
    # Validate image size
    if isinstance(imgsz, int):
        imgsz = (imgsz, imgsz)
    elif isinstance(imgsz, (list, tuple)) and len(imgsz) == 2:
        imgsz = tuple(imgsz)
    else:
        raise ValueError("imgsz must be int or tuple/list of 2 integers")
    
    exported_models = {}
    total_models = len(model_weights_dict)
    
    logger.info(f"Starting export of {total_models} model(s) to ONNX format")
    logger.info(f"Export parameters: imgsz={imgsz}, half={half}, dynamic={dynamic}, "
               f"simplify={simplify}, opset={opset}")
    
    for i, (weights_path, export_name) in enumerate(model_weights_dict.items(), 1):
        logger.info(f"[{i}/{total_models}] Processing: {export_name}")
        
        try:
            # Validate model weights file
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Model weights not found: {weights_path}")
            
            # Load YOLO model
            logger.info(f"  Loading model from: {weights_path}")
            model = YOLO(weights_path)
            
            # Define output path
            output_path = output_folder / f"{export_name}.onnx"
            
            # Export to ONNX with post-processing
            logger.info(f"  Exporting to ONNX: {output_path}")
            
            # Export with ULTRALYTICS built-in ONNX export
            # This includes proper YOLO head processing and NMS
            export_path = model.export(
                format='onnx',
                imgsz=imgsz,
                half=half,
                dynamic=dynamic,
                simplify=simplify,
                opset=opset,
                verbose=verbose
            )
            
            # Move to desired location with custom name
            if export_path != str(output_path):
                import shutil
                shutil.move(export_path, output_path)
                logger.info(f"  Moved to: {output_path}")
            
            # Verify exported model
            if output_path.exists():
                file_size = output_path.stat().st_size / (1024 * 1024)  # MB
                logger.info(f"  ✓ Export successful: {output_path} ({file_size:.1f} MB)")
                exported_models[export_name] = str(output_path)
            else:
                raise RuntimeError(f"Export failed: {output_path} not created")
                
        except Exception as e:
            logger.error(f"  ✗ Export failed for {export_name}: {str(e)}")
            # Continue with other models instead of failing completely
            continue
    
    # Summary
    successful_exports = len(exported_models)
    failed_exports = total_models - successful_exports
    
    logger.info(f"Export completed: {successful_exports}/{total_models} successful")
    if failed_exports > 0:
        logger.warning(f"{failed_exports} export(s) failed")
    
    return exported_models


def export_model_with_custom_postprocessing(
    weights_path: str,
    output_path: str,
    imgsz: Union[int, Tuple[int, int]] = 640,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 300,
    verbose: bool = True
) -> str:
    """
    Export a single YOLO model with custom post-processing parameters.
    
    Args:
        weights_path (str): Path to YOLO model weights
        output_path (str): Output path for ONNX model
        imgsz (Union[int, Tuple[int, int]]): Input image size
        conf_threshold (float): Confidence threshold for NMS
        iou_threshold (float): IoU threshold for NMS
        max_detections (int): Maximum number of detections
        verbose (bool): Enable verbose logging
    
    Returns:
        str: Path to exported ONNX model
    """
    
    logger = setup_logging(verbose)
    
    try:
        # Load model
        model = YOLO(weights_path)
        
        # Set NMS parameters
        model.overrides['conf'] = conf_threshold
        model.overrides['iou'] = iou_threshold
        model.overrides['max_det'] = max_detections
        
        logger.info(f"Exporting with custom NMS: conf={conf_threshold}, "
                   f"iou={iou_threshold}, max_det={max_detections}")
        
        # Export
        export_path = model.export(
            format='onnx',
            imgsz=imgsz,
            verbose=verbose
        )
        
        # Move to desired location
        if export_path != output_path:
            import shutil
            shutil.move(export_path, output_path)
        
        logger.info(f"Custom export completed: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Custom export failed: {str(e)}")
        raise


def batch_export_from_directory(
    models_directory: Union[str, Path],
    output_folder: Union[str, Path],
    name_pattern: str = "{parent_dir}_{model_name}",
    weights_filename: str = "best.pt",
    **export_kwargs
) -> Dict[str, str]:
    """
    Batch export YOLO models from a directory structure.
    
    Args:
        models_directory (Union[str, Path]): Root directory containing model folders
        output_folder (Union[str, Path]): Output directory for ONNX models
        name_pattern (str): Naming pattern for exported models (default: "{parent_dir}_{model_name}")
        weights_filename (str): Name of weights file to look for (default: "best.pt")
        **export_kwargs: Additional arguments for export_yolo_models_to_onnx
    
    Returns:
        Dict[str, str]: Dictionary of exported models
        
    Example:
        >>> # Directory structure:
        >>> # runs/
        >>> #   ├── arthropod_yolo11n/weights/best.pt
        >>> #   ├── arthropod_yolo11l/weights/best.pt
        >>> #   └── flatbug_yolo11n/weights/best.pt
        >>> 
        >>> exported = batch_export_from_directory(
        ...     models_directory="runs",
        ...     output_folder="exported_models"
        ... )
    """
    
    logger = setup_logging(export_kwargs.get('verbose', True))
    
    models_directory = Path(models_directory)
    model_weights_dict = {}
    
    # Find all weight files
    weight_files = list(models_directory.rglob(weights_filename))
    
    if not weight_files:
        logger.warning(f"No {weights_filename} files found in {models_directory}")
        return {}
    
    logger.info(f"Found {len(weight_files)} model weight files")
    
    for weight_file in weight_files:
        # Extract parent directory name for naming
        parent_dir = weight_file.parent.parent.name  # Skip 'weights' folder
        model_name = weight_file.stem  # 'best' from 'best.pt'
        
        # Format export name
        export_name = name_pattern.format(
            parent_dir=parent_dir,
            model_name=model_name,
            full_path=str(weight_file.parent.parent)
        )
        
        model_weights_dict[str(weight_file)] = export_name
        logger.info(f"  Found: {weight_file} -> {export_name}")
    
    # Export all models
    return export_yolo_models_to_onnx(model_weights_dict, output_folder, **export_kwargs)


def main():
    """
    Example usage and CLI interface.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Export YOLO models to ONNX format")
    parser.add_argument("--models-dir", type=str, help="Directory containing model folders")
    parser.add_argument("--output-dir", type=str, default="exported_models", 
                       help="Output directory for ONNX models")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--half", action="store_true", help="Export in FP16 precision")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic shapes")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.models_dir:
        # Batch export from directory
        exported = batch_export_from_directory(
            models_directory=args.models_dir,
            output_folder=args.output_dir,
            imgsz=args.imgsz,
            half=args.half,
            dynamic=args.dynamic,
            verbose=args.verbose
        )
    else:
        # Example usage
        model_weights_dict = {
            "runs/arthro_and_flatbug/train/weights/best.pt" : "yolo11l_ArthroNat+flatbug",
            "runs/arthro_and_flatbug_nano/train/weights/best.pt" : "yolo11n_ArthroNat+flatbug",
            "runs/arthro_mosaic_33/train/weights/best.pt" : "yolo11l_ArthroNat_mosaic33",
            "runs/arthro_mosaic_33_nano/train/weights/best.pt" : "yolo11n_ArthroNat_mosaic33",
        }
        
        exported = export_yolo_models_to_onnx(
            model_weights_dict=model_weights_dict,
            output_folder=args.output_dir,
            imgsz=args.imgsz,
            half=args.half,
            dynamic=args.dynamic,
            verbose=args.verbose
        )
    
    print("\n" + "="*60)
    print("EXPORT SUMMARY")
    print("="*60)
    for name, path in exported.items():
        print(f"  {name}: {path}")
    print(f"\nTotal exported: {len(exported)} models")


if __name__ == "__main__":
    main()
