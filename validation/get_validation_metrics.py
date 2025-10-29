from ultralytics import YOLO
import csv
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import pandas as pd

# Load the hierarchy CSV file
hierarchy_df = pd.read_csv("src/arthro_dataset_hierarchy.csv")

# Returns a list of names as [class, order, family, genus, species]
def get_hierarchy_from_taxon_id(taxon_id):
    # Convert taxon_id to int for comparison
    taxon_id = int(taxon_id)
    
    # Find the row with matching taxon_id
    matching_rows = hierarchy_df[hierarchy_df['taxon_id'] == taxon_id]
    
    if len(matching_rows) == 0:
        # Return empty hierarchy if taxon_id not found
        return ['', '', '', '', '']
    
    row = matching_rows.iloc[0]
    hierarchy = [
        row['class'] if pd.notna(row['class']) else '',
        row['order'] if pd.notna(row['order']) else '',
        row['family'] if pd.notna(row['family']) else '',
        row['genus'] if pd.notna(row['genus']) else '',
        row['specie'] if pd.notna(row['specie']) else ''
    ]
    
    return hierarchy

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

def get_performance_metrics(model, path, output_csv, split='test', confidence=0.5, IoU_threshold=0.5, include_hierarchy=True):
    # Load model
    model = YOLO(model) # load best.pt or last.pt of local model

    # Write header to csv file based on hierarchy option
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        if include_hierarchy:
            csvwriter.writerow(['class', 'order', 'family', 'genus', 'specie', 'taxon_id', 'F1', 'precision', 'recall', 'mean_IoU', 'IoUs', 'IoUs_with_zeros', 'TP', 'FP', 'FN', 'bbox_sizes'])
        else:
            csvwriter.writerow(['image_name', 'F1', 'precision', 'recall', 'mean_IoU', 'IoUs', 'IoUs_with_zeros', 'TP', 'FP', 'FN', 'bbox_sizes'])

    # Evaluate model performance on the validation set
    for img in tqdm(os.listdir(os.path.join(path, "images", split))):

        if include_hierarchy:
            taxon_id = img.split('_')[0]
            hierarchy = get_hierarchy_from_taxon_id(taxon_id)

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
            
            # print("Hierarchy: ", hierarchy)
            # print("IoU: ", mean_IoU, "F1: ", F1, "Precision: ", precision, "Recall: ", recall)

            # Converting lists to writtable strings
            IoUs_str = ','.join([str(iou) for iou in IoUs])
            IoUs_str = '[' + IoUs_str + ']'
            IoUs_with_zeros_str = ','.join([str(iou) for iou in IoUs_with_zeros])
            IoUs_with_zeros_str = '[' + IoUs_with_zeros_str + ']'

            with open(output_csv, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                if include_hierarchy:
                    csvwriter.writerow(hierarchy + [taxon_id, F1, precision, recall, mean_IoU, IoUs_str, IoUs_with_zeros_str, TP, FP, FN, bbox_sizes])
                else:
                    csvwriter.writerow([img, F1, precision, recall, mean_IoU, IoUs_str, IoUs_with_zeros_str, TP, FP, FN, bbox_sizes])
    return

# Example usage:
# model_path = "runs/detect/train50/weights/best.pt"
# conf = 0.413
# output_csv = "arthropods_dataset_scripts/benchmark/validation_conf0.413yolo11l.csv"
# # model_path = "runs/detect/train48/weights/best.pt"
# # conf = 0.372
# # output_csv = "arthropods_dataset_scripts/benchmark/validation_full_0.5_conf0.372yolo11s.csv"

# path = "/mnt/disk1/datasets/iNaturalist/Arthropods/FINAL_DATASET/"
# get_performance_metrics(model_path, path, output_csv, split='test', confidence=conf, IoU_threshold=0.5)


# model_path = "runs/detect/train45/weights/best.pt"
# conf = 0.444
# output_csv = "/home/eremy/Documents/CODE/bound_v1/arthropods_dataset_scripts/benchmark/generalization/same_genus_conf0.444yolo8n.csv"
# path = "/mnt/disk1/datasets/iNaturalist/Arthropods/generalization/2nd_french_arthro/same_genus/dataset/"
# get_performance_metrics(model_path, path, output_csv, split='train', confidence=conf, IoU_threshold=0.5)

# model_path = "runs/detect/train45/weights/best.pt"
# conf = 0.444
# output_csv = "/home/eremy/Documents/CODE/bound_v1/arthropods_dataset_scripts/benchmark/generalization/south_american_nowater_conf0.444yolo8n.csv"
# path = "/mnt/disk1/datasets/iNaturalist/Arthropods/generalization/south_american_arthro/LIMIT1/dataset(nowater)"
# get_performance_metrics(model_path, path, output_csv, split='train', confidence=conf, IoU_threshold=0.5)

# model_path = "runs/detect/train45/weights/best.pt"
# conf = 0.444
# output_csv = "/home/eremy/Documents/CODE/bound_v1/arthropods_dataset_scripts/benchmark/generalization/south_american_wateronly_conf0.444yolo8n.csv"
# path = "/mnt/disk1/datasets/iNaturalist/Arthropods/generalization/south_american_arthro/LIMIT1/dataset(wateronly)"
# get_performance_metrics(model_path, path, output_csv, split='train', confidence=conf, IoU_threshold=0.5)

## YOLO11L ##
# model_path = "runs/detect/train50/weights/best.pt"
# conf = 0.413
# output_csv = "/home/eremy/Documents/CODE/bound_v1/arthropods_dataset_scripts/benchmark/generalization/same_genus_conf0.413yolo11l.csv"
# path = "/mnt/disk1/datasets/iNaturalist/Arthropods/generalization/2nd_french_arthro/same_genus/dataset/"
# get_performance_metrics(model_path, path, output_csv, split='train', confidence=conf, IoU_threshold=0.5)
# output_csv = "/home/eremy/Documents/CODE/bound_v1/arthropods_dataset_scripts/benchmark/generalization/next_genus_conf0.413yolo11l.csv"
# path = "/mnt/disk1/datasets/iNaturalist/Arthropods/generalization/2nd_french_arthro/next_genus/dataset/"
# get_performance_metrics(model_path, path, output_csv, split='train', confidence=conf, IoU_threshold=0.5)
# output_csv = "/home/eremy/Documents/CODE/bound_v1/arthropods_dataset_scripts/benchmark/generalization/south_american_nowater_conf0.413yolo11l.csv"
# path = "/mnt/disk1/datasets/iNaturalist/Arthropods/generalization/south_american_arthro/LIMIT1/dataset(nowater)"
# get_performance_metrics(model_path, path, output_csv, split='train', confidence=conf, IoU_threshold=0.5)

## YOLO11N ##
# model_path = "runs/detect/train47/weights/best.pt"
# conf = 0.437
# output_csv = "/home/eremy/Documents/CODE/bound_v1/arthropods_dataset_scripts/benchmark/generalization/same_genus_conf0.437yolo11n.csv"
# path = "/mnt/disk1/datasets/iNaturalist/Arthropods/generalization/2nd_french_arthro/same_genus/dataset/"
# get_performance_metrics(model_path, path, output_csv, split='train', confidence=conf, IoU_threshold=0.5)
# output_csv = "/home/eremy/Documents/CODE/bound_v1/arthropods_dataset_scripts/benchmark/generalization/next_genus_conf0.437yolo11n.csv"
# path = "/mnt/disk1/datasets/iNaturalist/Arthropods/generalization/2nd_french_arthro/next_genus/dataset/"
# get_performance_metrics(model_path, path, output_csv, split='train', confidence=conf, IoU_threshold=0.5)
# output_csv = "/home/eremy/Documents/CODE/bound_v1/arthropods_dataset_scripts/benchmark/generalization/south_american_nowater_conf0.437yolo11n.csv"
# path = "/mnt/disk1/datasets/iNaturalist/Arthropods/generalization/south_american_arthro/LIMIT1/dataset(nowater)"
# get_performance_metrics(model_path, path, output_csv, split='train', confidence=conf, IoU_threshold=0.5)

## YOLO11L + flatbug ##
# model_path = "arthropods_dataset_scripts/benchmark/weights/11l_flatbug_best_conf0.409.pt"
# conf = 0.409
# output_csv = "arthropods_dataset_scripts/benchmark/validation_conf0.409yolo11l_flatbug.csv"
# path = "/mnt/disk1/datasets/iNaturalist/Arthropods/FINAL_DATASET/"
# get_performance_metrics(model_path, path, output_csv, split='test', confidence=conf, IoU_threshold=0.5)
# output_csv = "/home/eremy/Documents/CODE/bound_v1/arthropods_dataset_scripts/benchmark/generalization/same_genus_conf0.409yolo11l_flatbug.csv"
# path = "/mnt/disk1/datasets/iNaturalist/Arthropods/generalization/2nd_french_arthro/same_genus/dataset/"
# get_performance_metrics(model_path, path, output_csv, split='train', confidence=conf, IoU_threshold=0.5)
# output_csv = "/home/eremy/Documents/CODE/bound_v1/arthropods_dataset_scripts/benchmark/generalization/next_genus_conf0.409yolo11l_flatbug.csv"
# path = "/mnt/disk1/datasets/iNaturalist/Arthropods/generalization/2nd_french_arthro/next_genus/dataset/"
# get_performance_metrics(model_path, path, output_csv, split='train', confidence=conf, IoU_threshold=0.5)
# output_csv = "/home/eremy/Documents/CODE/bound_v1/arthropods_dataset_scripts/benchmark/generalization/south_american_nowater_conf0.409yolo11l_flatbug.csv"
# path = "/mnt/disk1/datasets/iNaturalist/Arthropods/generalization/south_american_arthro/LIMIT1/dataset(nowater)"
# get_performance_metrics(model_path, path, output_csv, split='train', confidence=conf, IoU_threshold=0.5)


#########################
## FLATBUG COMPARISONS ##
#########################

# TESTED ON OUR TEST DATASET

# # arthro
# model_path = "runs/arthro/train/weights/best.pt"
# output_csv = "validation/metrics/validation_arthro.csv"
# path = "dataset"
# get_performance_metrics(model_path, path, output_csv, split='test', confidence=0.5, IoU_threshold=0.5, include_hierarchy=True)


# # arthro + flatbug
# model_path = "runs/arthro_and_flatbug/train/weights/best.pt"
# output_csv = "validation/metrics/validation_arthro_and_flatbug.csv"
# path = "dataset"
# get_performance_metrics(model_path, path, output_csv, split='test', confidence=0.5, IoU_threshold=0.5, include_hierarchy=True)

# # flatbug
# model_path = "runs/flatbug/train/weights/best.pt"
# output_csv = "validation/metrics/validation_flatbug.csv"
# path = "dataset"
# get_performance_metrics(model_path, path, output_csv, split='test', confidence=0.5, IoU_threshold=0.5, include_hierarchy=True)

# # arthro mosaic 3x3
# model_path = "runs/arthro_mosaic_33/train/weights/best.pt"
# output_csv = "validation/metrics/validation_arthro_mosaic_33.csv"
# path = "dataset"
# get_performance_metrics(model_path, path, output_csv, split='test', confidence=0.5, IoU_threshold=0.5, include_hierarchy=True)

# arthro mosaic 4x4
# model_path = "runs/arthro_mosaic_44/train/weights/best.pt"
# output_csv = "validation/metrics/validation_arthro_mosaic_44.csv"
# path = "dataset"
# get_performance_metrics(model_path, path, output_csv, split='test', confidence=0.5, IoU_threshold=0.5, include_hierarchy=True)

# arthro nomosaic
# model_path = "runs/arthro_nomosaic/train/weights/best.pt"
# output_csv = "validation/metrics/validation_arthro_nomosaic.csv"
# path = "dataset"
# get_performance_metrics(model_path, path, output_csv, split='test', confidence=0.5, IoU_threshold=0.5, include_hierarchy=True)

# arthro + flatbug mosaic 4x4
# model_path = "runs/arthro_and_flatbug_mosaic44/train/weights/best.pt"
# output_csv = "validation/metrics/validation_arthro_and_flatbug_mosaic_44.csv"
# path = "dataset"
# get_performance_metrics(model_path, path, output_csv, split='test', confidence=0.5, IoU_threshold=0.5, include_hierarchy=True)

# arthro mosaic 6x6
model_path = "runs/arthro_mosaic_66/train/weights/best.pt"
output_csv = "validation/metrics/validation_arthro_mosaic_66.csv"
path = "dataset"
get_performance_metrics(model_path, path, output_csv, split='test', confidence=0.5, IoU_threshold=0.5, include_hierarchy=True)

# TESTED ON FLATBUG TEST DATASET

# # arthro
# model_path = "runs/arthro/train/weights/best.pt"
# output_csv = "validation/metrics/validation(fb)_arthro.csv"
# path =  "/media/disk2/flatbug-yolo-split"
# get_performance_metrics(model_path, path, output_csv, split='test', confidence=0.5, IoU_threshold=0.5, include_hierarchy=False)

# # arthro + flatbug
# model_path = "runs/arthro_and_flatbug/train/weights/best.pt"
# output_csv = "validation/metrics/validation(fb)_arthro_and_flatbug.csv"
# path =  "/media/disk2/flatbug-yolo-split"
# get_performance_metrics(model_path, path, output_csv, split='test', confidence=0.5, IoU_threshold=0.5, include_hierarchy=False)

# # flatbug
# model_path = "runs/flatbug/train/weights/best.pt"
# output_csv = "validation/metrics/validation(fb)_flatbug.csv"
# path =  "/media/disk2/flatbug-yolo-split"
# get_performance_metrics(model_path, path, output_csv, split='test', confidence=0.5, IoU_threshold=0.5, include_hierarchy=False)

# # arthro mosaic 3x3
# model_path = "runs/arthro_mosaic_33/train/weights/best.pt"
# output_csv = "validation/metrics/validation(fb)_arthro_mosaic_33.csv"
# path =  "/media/disk2/flatbug-yolo-split"
# get_performance_metrics(model_path, path, output_csv, split='test', confidence=0.5, IoU_threshold=0.5, include_hierarchy=False)

# arthro mosaic 4x4
# model_path = "runs/arthro_mosaic_44/train/weights/best.pt"
# output_csv = "validation/metrics/validation(fb)_arthro_mosaic_44.csv"
# path =  "/media/disk2/flatbug-yolo-split"
# get_performance_metrics(model_path, path, output_csv, split='test', confidence=0.5, IoU_threshold=0.5, include_hierarchy=False)

# arthro nomosaic
# model_path = "runs/arthro_nomosaic/train/weights/best.pt"
# output_csv = "validation/metrics/validation(fb)_arthro_nomosaic.csv"
# path =  "/media/disk2/flatbug-yolo-split"
# get_performance_metrics(model_path, path, output_csv, split='test', confidence=0.5, IoU_threshold=0.5, include_hierarchy=False)

# arthro + flatbug mosaic 4x4
# model_path = "runs/arthro_and_flatbug_mosaic44/train/weights/best.pt"
# output_csv = "validation/metrics/validation(fb)_arthro_and_flatbug_mosaic_44.csv"
# path =  "/media/disk2/flatbug-yolo-split"
# get_performance_metrics(model_path, path, output_csv, split='test', confidence=0.5, IoU_threshold=0.5, include_hierarchy=False)

# arthro mosaic 6x6
model_path = "runs/arthro_mosaic_66/train/weights/best.pt"
output_csv = "validation/metrics/validation(fb)_arthro_mosaic_66.csv"
path =  "/media/disk2/flatbug-yolo-split"
get_performance_metrics(model_path, path, output_csv, split='test', confidence=0.5, IoU_threshold=0.5, include_hierarchy=False)