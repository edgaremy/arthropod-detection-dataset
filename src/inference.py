from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2
import os
import argparse

def main(image_path, size, output_dir, verbose):

    # Determine model weights based on size
    if size == "l":
        model_name = "yolo11l_arthropod_0.413.pt"
        if verbose:
            print("Using large model")
    elif size == "n":
        model_name = "yolo11n_arthropod_0.437.pt"
        if verbose:
            print("Using nano model")
    else:
        if verbose:
            print("Using large model by default.")
        model_name = "yolo11l_arthropod_0.413.pt"
    
    # Download weights from Hugging Face Hub
    weights = hf_hub_download(repo_id="edgaremy/arthropod-detector", filename=model_name)

    # Load the model with Ultralytics YOLO
    model = YOLO(weights)

    # Run inference
    results = model(image_path)

    # Process results
    for result in results:
        boxes = result.boxes  # Bounding boxes
        
        # Print detection information if verbose
        if verbose:
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls]
                print(f"Detected {label} with confidence {conf:.2f}")
                
                # Get bounding box coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                print(f"Bounding box: ({x1:.2f}, {y1:.2f}) to ({x2:.2f}, {y2:.2f})")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save results with bounding boxes
    result = results[0]
    result_plotted = result.plot()

    # Extract filename from path and save to output directory
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"detection_{filename}")
    cv2.imwrite(output_path, result_plotted)
    if verbose:
        print(f"Detection result saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run arthropod detection on an image')
    parser.add_argument('image', type=str, help='Path to the image for inference')
    parser.add_argument('--size', type=str, choices=['n', 'l'], default='', 
                        help='Model size: n (nano) or l (large, default)')
    parser.add_argument('--output', type=str, default='output',
                        help='Directory to save output images (default: output)')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Print detection details (default: False)')
    
    args = parser.parse_args()
    main(args.image, args.size, args.output, args.verbose)
