#!/usr/bin/env python3
"""
YOLO Inference from Hugging Face Models

This script downloads and runs YOLO models from Hugging Face Hub in both PyTorch (.pt) 
and ONNX formats. It performs inference on images and saves results in various formats.

Author: Assistant
Date: November 2025
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Union, List, Dict, Tuple
import logging
from datetime import datetime

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ULTRALYTICS package is required. Install with: pip install ultralytics")

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    raise ImportError("huggingface_hub is required. Install with: pip install huggingface-hub")


def setup_logging(verbose: bool = True) -> logging.Logger:
    """Set up logging configuration."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def load_model_from_huggingface(
    repo_id: str,
    filename: str,
    cache_dir: str = None,
    verbose: bool = True
) -> YOLO:
    """
    Load YOLO model from Hugging Face Hub.
    
    Args:
        repo_id (str): Hugging Face repository ID (e.g., "edgaremy/arthropod-detector")
        filename (str): Model filename (e.g., "yolo11l_arthropod_0.413.pt" or "model.onnx")
        cache_dir (str): Directory to cache downloaded models
        verbose (bool): Enable verbose logging
    
    Returns:
        YOLO: Loaded YOLO model
    """
    logger = setup_logging(verbose)
    
    logger.info(f"Downloading model from Hugging Face Hub")
    logger.info(f"  Repository: {repo_id}")
    logger.info(f"  Filename: {filename}")
    
    try:
        # Download weights from Hugging Face Hub
        weights_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir
        )
        
        logger.info(f"  Downloaded to: {weights_path}")
        
        # Check file extension
        file_ext = Path(filename).suffix.lower()
        
        if file_ext == '.pt':
            logger.info("Loading PyTorch model (.pt)")
            model = YOLO(weights_path)
        elif file_ext == '.onnx':
            logger.info("Loading ONNX model (.onnx)")
            model = YOLO(weights_path, task='detect')
        else:
            raise ValueError(f"Unsupported model format: {file_ext}. Use .pt or .onnx")
        
        logger.info("Model loaded successfully!")
        
        # Print model info if available
        if hasattr(model, 'names') and model.names:
            logger.info(f"Model classes: {list(model.names.values())}")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


def get_image_files(input_path: Union[str, Path]) -> List[Path]:
    """
    Get list of image files from a path (file or directory).
    
    Args:
        input_path (Union[str, Path]): Path to image file or directory
    
    Returns:
        List[Path]: List of image file paths
    """
    input_path = Path(input_path)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    if input_path.is_file():
        if input_path.suffix.lower() in image_extensions:
            return [input_path]
        else:
            raise ValueError(f"File is not a supported image format: {input_path}")
    
    elif input_path.is_dir():
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        return sorted(image_files)
    
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")


def save_cropped_detections(
    image: np.ndarray,
    boxes: List,
    output_dir: Path,
    image_name: str
) -> int:
    """
    Save cropped detection images.
    
    Args:
        image (np.ndarray): Original image
        boxes (List): Detection boxes
        output_dir (Path): Output directory for crops
        image_name (str): Original image name (without extension)
    
    Returns:
        int: Number of crops saved
    """
    crop_count = 0
    
    for i, box in enumerate(boxes):
        # Get bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].int().cpu().numpy()
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Crop the detection
        cropped = image[y1:y2, x1:x2]
        
        if cropped.size > 0:  # Ensure crop is not empty
            # Save cropped image
            crop_filename = f"{image_name}_{i}.jpg"
            crop_path = output_dir / crop_filename
            cv2.imwrite(str(crop_path), cropped)
            crop_count += 1
    
    return crop_count


def save_yolo_labels(
    boxes: List,
    output_dir: Path,
    image_name: str,
    img_width: int,
    img_height: int
) -> None:
    """
    Save detection labels in YOLO format.
    
    Args:
        boxes (List): Detection boxes
        output_dir (Path): Output directory for labels
        image_name (str): Original image name (without extension)
        img_width (int): Image width
        img_height (int): Image height
    """
    label_file = output_dir / f"{image_name}.txt"
    
    with open(label_file, 'w') as f:
        for box in boxes:
            # Get class and confidence
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Convert to YOLO format (center_x, center_y, width, height) normalized
            center_x = ((x1 + x2) / 2) / img_width
            center_y = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            f.write(f"{cls} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")


def save_annotated_image(
    image: np.ndarray,
    boxes: List,
    model_names: Dict,
    output_dir: Path,
    image_name: str,
    image_ext: str
) -> None:
    """
    Save image with bounding boxes drawn.
    
    Args:
        image (np.ndarray): Original image
        boxes (List): Detection boxes
        model_names (Dict): Class names dictionary
        output_dir (Path): Output directory for annotated images
        image_name (str): Original image name (without extension)
        image_ext (str): Image file extension
    """
    annotated_img = image.copy()
    
    for box in boxes:
        # Get coordinates and info
        x1, y1, x2, y2 = box.xyxy[0].int().cpu().numpy()
        cls = int(box.cls.item())
        conf = float(box.conf.item())
        
        # Get class name if available
        class_name = model_names.get(cls, str(cls)) if model_names else str(cls)
        
        # Draw bounding box
        cv2.rectangle(annotated_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Draw label with confidence
        label = f"{class_name}: {conf:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        
        # Draw label background
        cv2.rectangle(
            annotated_img,
            (int(x1), int(y1) - label_size[1] - 10),
            (int(x1) + label_size[0], int(y1)),
            (0, 255, 0),
            -1
        )
        
        # Draw label text
        cv2.putText(
            annotated_img,
            label,
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )
    
    # Save annotated image
    annotated_path = output_dir / f"{image_name}{image_ext}"
    cv2.imwrite(str(annotated_path), annotated_img)


def run_inference(
    model: YOLO,
    input_path: Union[str, Path],
    results_folder: Union[str, Path] = "results",
    save_crops: bool = False,
    save_labels: bool = False,
    save_bbox_view: bool = False,
    conf_threshold: float = 0.25,
    device: str = 'cpu',
    verbose: bool = True
) -> Dict:
    """
    Run YOLO inference on images with various output options.
    
    Args:
        model (YOLO): Loaded YOLO model
        input_path (Union[str, Path]): Path to image file or folder
        results_folder (Union[str, Path]): Output folder for results
        save_crops (bool): Save cropped detection images
        save_labels (bool): Save detection labels in YOLO format
        save_bbox_view (bool): Save images with bounding boxes drawn
        conf_threshold (float): Confidence threshold for detections
        device (str): Device to run inference on ('cpu', 'cuda', '0', '1', etc.)
        verbose (bool): Enable verbose logging
    
    Returns:
        Dict: Summary of inference results
    """
    logger = setup_logging(verbose)
    
    # Setup paths
    results_folder = Path(results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)
    
    # Create output directories
    crops_dir = results_folder / "crop" if save_crops else None
    labels_dir = results_folder / "labels" if save_labels else None
    bbox_view_dir = results_folder / "view-bbox" if save_bbox_view else None
    
    if save_crops:
        crops_dir.mkdir(exist_ok=True)
    if save_labels:
        labels_dir.mkdir(exist_ok=True)
    if save_bbox_view:
        bbox_view_dir.mkdir(exist_ok=True)
    
    # Get image files
    logger.info(f"Loading images from: {input_path}")
    image_files = get_image_files(input_path)
    
    if not image_files:
        logger.warning(f"No images found in {input_path}")
        return {
            "total_images": 0,
            "images_with_detections": 0,
            "images_without_detections": 0,
            "total_detections": 0
        }
    
    logger.info(f"Found {len(image_files)} images to process")
    logger.info(f"Confidence threshold: {conf_threshold}")
    logger.info(f"Device: {device}")
    
    # Statistics
    total_images = len(image_files)
    images_with_detections = 0
    images_without_detections = 0
    total_detections = 0
    processed_images = 0
    failed_images = 0
    
    # Get model class names
    model_names = model.names if hasattr(model, 'names') else {}
    
    # Process each image
    for img_path in image_files:
        try:
            logger.info(f"Processing: {img_path.name}")
            
            # Run inference
            results = model(str(img_path), conf=conf_threshold, device=device, verbose=False)
            
            # Get the first (and only) result
            result = results[0]
            
            # Load original image
            img = cv2.imread(str(img_path))
            if img is None:
                logger.error(f"  Failed to load image: {img_path}")
                failed_images += 1
                continue
            
            img_height, img_width = img.shape[:2]
            
            # Get detections
            boxes = result.boxes
            
            image_name = img_path.stem
            image_ext = img_path.suffix
            
            if boxes is not None and len(boxes) > 0:
                num_detections = len(boxes)
                total_detections += num_detections
                images_with_detections += 1
                logger.info(f"  Found {num_detections} detections")
                
                # Save cropped detections
                if save_crops:
                    crops_saved = save_cropped_detections(img, boxes, crops_dir, image_name)
                    logger.info(f"  Saved {crops_saved} cropped images")
                
                # Save labels in YOLO format
                if save_labels:
                    save_yolo_labels(boxes, labels_dir, image_name, img_width, img_height)
                    logger.info(f"  Saved label file")
                
                # Save annotated image
                if save_bbox_view:
                    save_annotated_image(img, boxes, model_names, bbox_view_dir, image_name, image_ext)
                    logger.info(f"  Saved annotated image")
            
            else:
                images_without_detections += 1
                logger.info(f"  No detections found")
                
                # Still create empty label file if saving labels
                if save_labels:
                    label_file = labels_dir / f"{image_name}.txt"
                    label_file.touch()
            
            processed_images += 1
            
        except Exception as e:
            logger.error(f"Error processing {img_path.name}: {str(e)}")
            failed_images += 1
            continue
    
    # Create summary
    summary = {
        "total_images": total_images,
        "processed_images": processed_images,
        "failed_images": failed_images,
        "images_with_detections": images_with_detections,
        "images_without_detections": images_without_detections,
        "total_detections": total_detections,
        "average_detections_per_image": total_detections / processed_images if processed_images > 0 else 0,
        "results_folder": str(results_folder)
    }
    
    # Save summary to file
    summary_path = results_folder / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("YOLO INFERENCE SUMMARY\n")
        f.write("="*60 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input path: {input_path}\n")
        f.write(f"Results folder: {results_folder}\n")
        f.write(f"Confidence threshold: {conf_threshold}\n")
        f.write(f"Device: {device}\n")
        f.write("\n")
        f.write("STATISTICS:\n")
        f.write("-"*60 + "\n")
        f.write(f"Total images: {summary['total_images']}\n")
        f.write(f"Processed images: {summary['processed_images']}\n")
        f.write(f"Failed images: {summary['failed_images']}\n")
        f.write(f"Images with detections: {summary['images_with_detections']}\n")
        f.write(f"Images without detections: {summary['images_without_detections']}\n")
        f.write(f"Total detections (all bboxes): {summary['total_detections']}\n")
        f.write(f"Average detections per image: {summary['average_detections_per_image']:.2f}\n")
        f.write("\n")
        f.write("OUTPUT OPTIONS:\n")
        f.write("-"*60 + "\n")
        f.write(f"Save crops: {save_crops}\n")
        if save_crops:
            f.write(f"  Crops folder: {crops_dir}\n")
        f.write(f"Save labels: {save_labels}\n")
        if save_labels:
            f.write(f"  Labels folder: {labels_dir}\n")
        f.write(f"Save bbox views: {save_bbox_view}\n")
        if save_bbox_view:
            f.write(f"  Bbox views folder: {bbox_view_dir}\n")
        f.write("="*60 + "\n")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("INFERENCE COMPLETED!")
    logger.info("="*60)
    logger.info(f"Processed images: {processed_images}/{total_images}")
    logger.info(f"Images with detections: {images_with_detections}")
    logger.info(f"Images without detections: {images_without_detections}")
    logger.info(f"Total detections: {total_detections}")
    logger.info(f"Average detections per image: {summary['average_detections_per_image']:.2f}")
    logger.info(f"Results saved to: {results_folder}")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info("="*60)
    
    return summary


def main():
    """Command line interface for YOLO inference from Hugging Face."""
    parser = argparse.ArgumentParser(
        description="Run YOLO inference using arthropod detection models from Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference with YOLO11n PyTorch model (default)
  python run_yolo_model2.py image.jpg
  
  # Use YOLO11l model on a folder
  python run_yolo_model2.py images/ --model-size l
  
  # Use ONNX format with all output options
  python run_yolo_model2.py images/ --format onnx --model-size l \\
      --save-crops --save-labels --save-bbox-view
  
  # Custom confidence and GPU inference
  python run_yolo_model2.py images/ --model-size l --conf 0.5 --device cuda
  
  # Save to custom output folder
  python run_yolo_model2.py images/ --results-folder output/experiment1
        """
    )
    
    parser.add_argument("input", help="Path to image file or folder")
    
    # Model selection
    parser.add_argument("--format", choices=["pt", "onnx"], default="pt",
                       help="Model format: pt (PyTorch) or onnx (default: pt)")
    parser.add_argument("--model-size", choices=["n", "l"], default="n",
                       help="Model size: n (nano) or l (large) (default: n)")
    
    # Output options
    parser.add_argument("-r", "--results-folder", default="results",
                       help="Output folder for results (default: results)")
    parser.add_argument("--save-crops", action="store_true",
                       help="Save cropped detection images to results/crop/")
    parser.add_argument("--save-labels", action="store_true",
                       help="Save detection labels in YOLO format to results/labels/")
    parser.add_argument("--save-bbox-view", action="store_true",
                       help="Save images with bounding boxes drawn to results/view-bbox/")
    
    # Inference parameters
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Confidence threshold for detections (default: 0.25)")
    parser.add_argument("--device", default="cpu",
                       help="Device to run inference on: cpu, cuda, 0, 1, etc. (default: cpu)")
    
    # Advanced options
    parser.add_argument("--cache-dir", help="Directory to cache downloaded models")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    try:
        # Hard-coded repository
        repo_id = "edgaremy/arthropod-detector"
        
        # Construct model filename based on format and size
        model_size_name = "yolo11n" if args.model_size == "n" else "yolo11l"
        file_extension = ".pt" if args.format == "pt" else ".onnx"
        model_filename = f"{model_size_name}_ArthroNat+flatbug{file_extension}"
        
        # Load model from Hugging Face
        model = load_model_from_huggingface(
            repo_id=repo_id,
            filename=model_filename,
            cache_dir=args.cache_dir,
            verbose=args.verbose or True  # Always show model loading info
        )
        
        # Run inference
        summary = run_inference(
            model=model,
            input_path=args.input,
            results_folder=args.results_folder,
            save_crops=args.save_crops,
            save_labels=args.save_labels,
            save_bbox_view=args.save_bbox_view,
            conf_threshold=args.conf,
            device=args.device,
            verbose=args.verbose or True  # Always show inference info
        )
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
