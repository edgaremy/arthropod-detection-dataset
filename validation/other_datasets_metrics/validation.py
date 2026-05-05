from ultralytics import YOLO
import csv
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


def _safe_iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_w = max(0.0, min(xa2, xb2) - max(xa1, xb1))
    inter_h = max(0.0, min(ya2, yb2) - max(ya1, yb1))
    intersection = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def _yolo_label_to_xyxy(label_line, im_width, im_height):
    parts = label_line.strip().split()
    x_center = float(parts[1])
    y_center = float(parts[2])
    width = float(parts[3])
    height = float(parts[4])

    x1 = (x_center - width / 2.0) * im_width
    y1 = (y_center - height / 2.0) * im_height
    x2 = (x_center + width / 2.0) * im_width
    y2 = (y_center + height / 2.0) * im_height
    return np.array([x1, y1, x2, y2], dtype=float)

def get_metrics(label_lines, prediction_boxes, im_width, im_height, IoU_threshold=0.5):

    bbox_relative_sizes = []

    n_labels = len(label_lines)
    n_preds = len(prediction_boxes)

    if n_labels == 0:
        return 0, 0, 0, 0, np.array([]), np.array([]), 0, n_preds, 0, bbox_relative_sizes

    # Compute IoU
    combinations = np.zeros((n_preds, n_labels), dtype=float)
    label_boxes = []
    for j in range(n_labels):

        # Get the coordinates of the label bounding box
        line = label_lines[j].strip().split()
        width = float(line[3])
        height = float(line[4])

        # Compute the area of the bounding box (relative to the image size)
        bbox_relative_sizes.append(width * height)
        label_boxes.append(_yolo_label_to_xyxy(label_lines[j], im_width, im_height))

    if n_preds == 0:
        return 0, 0, 0, 0, np.array([]), np.zeros(n_labels), 0, 0, n_labels, bbox_relative_sizes

    for j in range(n_labels):
        for i in range(n_preds):
            combinations[i, j] = _safe_iou(prediction_boxes[i], label_boxes[j])

    row_ind, col_ind = linear_sum_assignment(-combinations)
    IoUs = combinations[row_ind, col_ind]
    mean_IoU = IoUs.sum() / n_labels

    # IoUs keeping the order of the label_lines, IoU = 0 if no match
    IoUs_with_zeros = np.zeros(len(label_lines))
    IoUs_with_zeros[col_ind] = IoUs

    # Compute TP, FP, FN using IoU threshold (0.5 by default)
    TP = len(IoUs[IoUs >= IoU_threshold])
    FP = n_preds - TP
    FN = n_labels - TP
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return F1, precision, recall, mean_IoU, IoUs, IoUs_with_zeros, TP, FP, FN, bbox_relative_sizes

def get_performance_metrics(model, path, split='test', confidence=0.5, IoU_threshold=0.5):
    # Load model
    model = YOLO(model) # load best.pt or last.pt of local model

    # Lists to accumulate metrics across all images
    all_F1 = []
    all_precision = []
    all_recall = []
    all_mean_IoU = []
    total_TP = 0
    total_FP = 0
    total_FN = 0
    all_bbox_sizes = []

    # Evaluate model performance on the validation set
    for img in tqdm(os.listdir(os.path.join(path, "images", split))):

        img_path = os.path.join(path, "images", split, img)
        ground_truth_label = os.path.join(path, "labels", split, img.split('.')[0] + '.txt')
        if not os.path.exists(ground_truth_label): # No label, nothing to measure
            continue

        results = model(img_path, save_txt=False, save=False, verbose=False, conf=confidence)

        with open(ground_truth_label, 'r') as file:
            lines = file.readlines()
            boxes = results[0].boxes.xyxy.cpu().numpy()

            if len(lines) == 0: # No label, nothing to measure
                continue
            # if len(boxes) == 0 and len(lines) > 0:
            #     IoU, F1, precision, recall = 0, 0, 0, 0
            else:
                # Get original img size
                im_height, im_width = results[0].orig_shape[:2]
                F1, precision, recall, mean_IoU, IoUs, IoUs_with_zeros, TP, FP, FN, bbox_sizes = get_metrics(lines, boxes, im_width, im_height, IoU_threshold)
            
                # Accumulate metrics
                all_F1.append(F1)
                all_precision.append(precision)
                all_recall.append(recall)
                all_mean_IoU.append(mean_IoU)
                total_TP += TP
                total_FP += FP
                total_FN += FN
                all_bbox_sizes.extend(bbox_sizes)
    
    # Calculate average metrics across all images
    if len(all_F1) > 0:
        avg_F1 = np.mean(all_F1)
        avg_precision = np.mean(all_precision)
        avg_recall = np.mean(all_recall)
        avg_mean_IoU = np.mean(all_mean_IoU)
        avg_bbox_size = np.mean(all_bbox_sizes) if all_bbox_sizes else 0
        
        # Calculate overall precision, recall, F1 from total TP, FP, FN
        overall_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
        overall_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
        overall_F1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        return [avg_F1, avg_precision, avg_recall, avg_mean_IoU, overall_F1, overall_precision, overall_recall, total_TP, total_FP, total_FN, avg_bbox_size]
    else:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def compare_models(models, test_datasets, confidence=0.5, IoU_threshold=0.5, output_csv='model_comparison.csv'):
    """
    Compare multiple models on multiple test datasets and write results to a CSV.
    
    Args:
        models (list): List of tuples with (model_name, model_path)
        test_datasets (list): List of tuples with (dataset_name, dataset_path)
        confidence (float): Confidence threshold for model predictions
        IoU_threshold (float): IoU threshold for evaluation
        output_csv (str): Path to output CSV file
    """
    # Write header to csv file only if file doesn't exist    
    if not os.path.exists(output_csv):
        with open(output_csv, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['model_name', 'test_dataset', 'avg_F1', 'avg_precision', 'avg_recall', 'avg_mean_IoU', 'overall_F1', 'overall_precision', 'overall_recall', 'total_TP', 'total_FP', 'total_FN', 'avg_bbox_size'])
    
    # Evaluate each model on each test dataset
    for model_name, model_path in models:
        for dataset_name, dataset_path in test_datasets:
            print(f"Evaluating {model_name} on {dataset_name}...")
            
            # Get evaluation results (now returns averaged metrics)
            avg_metrics = get_performance_metrics(model_path, dataset_path, split='test', confidence=confidence, IoU_threshold=IoU_threshold)
            
            # Write results to output CSV
            with open(output_csv, 'a', newline='') as out_file:
                writer = csv.writer(out_file)
                # Write one row per model-dataset combination with averaged metrics
                writer.writerow([model_name, dataset_name] + avg_metrics)
    
    print(f"Comparison complete. Results saved to {output_csv}")

# Example usage:
if __name__ == "__main__":
    models = [
        ("arthro_no_mosaic", "runs/arthro_nomosaic/train/weights/best.pt"),
        ("arthro_mosaic2x2", "runs/arthro/train/weights/best.pt"),
        ("arthro_mosaic3x3", "runs/arthro_mosaic_33/train/weights/best.pt"),
        ("arthro_mosaic4x4", "runs/arthro_mosaic_44/train/weights/best.pt"),
        ("arthro+flatbug", "runs/arthro_and_flatbug/train/weights/best.pt"),
        ("flatbug", "runs/flatbug/train/weights/best.pt"),
    ]
    
    test_datasets = [
        ("arthro", "dataset"),
        ("flatbug", "/media/disk2/flatbug-yolo-split"), # REPLACE with your path for the flatbug dataset
        ("SPIPOLL", "datasets(others)/SPIPOLL/"),
        ("OOD", "datasets(others)/OOD-split/"),
        ("Lepinoc", "datasets(others)/Lepinoc-split/"),
        # ("Entomo barber", "datasets(others)/Entomo_barber/"),
        # ("Entomo flower", "datasets(others)/Entomo_flower/"),
    ]
    
    compare_models(models, test_datasets, confidence=0.5, IoU_threshold=0.5, output_csv='validation/other_datasets_metrics/dataset_perf_comparison3.csv')