from ultralytics import YOLO
import csv
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

def get_metrics(label_lines, prediction_boxes, im_width, im_height, IoU_threshold=0.5):

    bbox_relative_sizes = []

    # Compute IoU
    combinations = np.zeros((len(prediction_boxes), len(label_lines)))
    for j in range(len(label_lines)):

        # Get the coordinates of the label bounding box
        line = label_lines[j].split(' ')
        label = int(line[0])
        x_center = float(line[1])
        y_center = float(line[2])
        width = float(line[3])
        height = float(line[4])

        # Compute the area of the bounding box (relative to the image size)
        bbox_relative_sizes.append(width * height)

        for i in range(len(prediction_boxes)):

            # Get the coordinates of the predicted bounding box
            pred_x1 = int(prediction_boxes[i][0])
            pred_y1 = int(prediction_boxes[i][1])
            pred_x2 = int(prediction_boxes[i][2])
            pred_y2 = int(prediction_boxes[i][3])

            # Get the coordinates of the bounding box
            x1 = int((x_center - width/2) * im_height)
            y1 = int((y_center - height/2) * im_width)
            x2 = int((x_center + width/2) * im_height)
            y2 = int((y_center + height/2) * im_width)

            # Compute the intersection over union
            intersection = max(0, min(x2, pred_x2) - max(x1, pred_x1)) * max(0, min(y2, pred_y2) - max(y1, pred_y1))
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
            union = area1 + area2 - intersection
            IoU = intersection / union

            combinations[i, j] = IoU

    row_ind, col_ind = linear_sum_assignment(-combinations)
    IoUs = combinations[row_ind, col_ind]
    mean_IoU = IoUs.sum() / len(label_lines)

    # IoUs keeping the order of the label_lines, IoU = 0 if no match
    IoUs_with_zeros = np.zeros(len(label_lines))
    IoUs_with_zeros[col_ind] = IoUs

    # Compute TP, FP, FN using IoU threshold (0.5 by default)
    TP = len(IoUs[IoUs > IoU_threshold])
    FP = len(prediction_boxes) - TP
    FN = len(label_lines) - TP
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return F1, precision, recall, mean_IoU, IoUs, IoUs_with_zeros, TP, FP, FN, bbox_relative_sizes

def get_performance_metrics(model, path, split='train', confidence=0.5, IoU_threshold=0.5):
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
                im_width, im_height = results[0].boxes.orig_shape[0], results[0].boxes.orig_shape[1]
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
        test_datasets (list): List of tuples with (dataset_name, dataset_path, split)
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
        for dataset_info in test_datasets:
            if len(dataset_info) == 3:
                dataset_name, dataset_path, split = dataset_info
            else:
                # Backward compatibility: default to 'train' if split not specified
                dataset_name, dataset_path = dataset_info
                split = 'train'
            
            print(f"Evaluating {model_name} on {dataset_name} ({split} split)...")
            
            # Get evaluation results (now returns averaged metrics)
            avg_metrics = get_performance_metrics(model_path, dataset_path, split=split, confidence=confidence, IoU_threshold=IoU_threshold)
            
            # Write results to output CSV
            with open(output_csv, 'a', newline='') as out_file:
                writer = csv.writer(out_file)
                # Write one row per model-dataset combination with averaged metrics
                writer.writerow([model_name, dataset_name] + avg_metrics)
    
    print(f"Comparison complete. Results saved to {output_csv}")

# Example usage:
if __name__ == "__main__":
    models = [
        ("arthro_mosaic2x2", "runs/arthro/train/weights/best.pt"),
        ("flatbug", "runs/flatbug/train/weights/best.pt"),
        ("arthro+flatbug", "runs/arthro_and_flatbug/train/weights/best.pt"),
        ("arthro_mosaic3x3", "runs/arthro_mosaic_33/train/weights/best.pt"),
        ("arthro_mosaic4x4", "runs/arthro_mosaic_44/train/weights/best.pt"),
        ("arthro_no_mosaic", "runs/arthro_nomosaic/train/weights/best.pt"),
        # ("arthro+flatbug_mosaic4x4", "runs/arthro_and_flatbug_mosaic44/train/weights/best.pt"),
        # ("arthro_mosaic6x6", "runs/arthro_mosaic_66/train/weights/best.pt"),
    ]
    
    # Test generalization on the 3 datasets in datasets(generalization)
    test_datasets = [
        ("same_species", "dataset", "test"),
        ("same_genus", "datasets(generalization)/same_genus", "train"),
        ("other_genus", "datasets(generalization)/other_genus", "train"),
        ("other_families", "datasets(generalization)/other_families", "train"),
    ]
    
    compare_models(models, test_datasets, confidence=0.5, IoU_threshold=0.5, output_csv='validation/generalization/model_comparison.csv')