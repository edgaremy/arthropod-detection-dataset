import os
import cv2
import glob

def load_yolo_labels(label_path, img_width, img_height):
    bboxes = []
    if not os.path.exists(label_path):
        return bboxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, x_center, y_center, w, h = map(float, parts)
            x1 = int((x_center - w / 2) * img_width)
            y1 = int((y_center - h / 2) * img_height)
            x2 = int((x_center + w / 2) * img_width)
            y2 = int((y_center + h / 2) * img_height)
            bboxes.append((x1, y1, x2, y2))
    return bboxes

def draw_bboxes(img, bboxes):
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    return img

def main(dataset_dir):
    img_dir = os.path.join(dataset_dir.strip(), 'images'.strip(), 'train'.strip())
    label_dir = os.path.join(dataset_dir.strip(), 'labels'.strip(), 'train'.strip())
    print(os.path.join(img_dir, '*'))
    img_paths = sorted(glob.glob(os.path.join(img_dir, '*')))
    if not img_paths:
        print("No images found in", img_dir)
        return

    idx = 0
    while True:
        img_path = img_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            idx = (idx + 1) % len(img_paths)
            continue
        h, w = img.shape[:2]
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, base + '.txt')
        bboxes = load_yolo_labels(label_path, w, h)
        img_disp = img.copy()
        img_disp = draw_bboxes(img_disp, bboxes)
        cv2.imshow('YOLO Preview', img_disp)
        key = cv2.waitKey(0)
        if key == 27:  # ESC to exit
            break
        elif key == 83 or key == ord('d'):  # right arrow or 'd'
            idx = (idx + 1) % len(img_paths)
        elif key == 81 or key == ord('a'):  # left arrow or 'a'
            idx = (idx - 1) % len(img_paths)
    cv2.destroyAllWindows()

# Example Usage
dataset_dir = "/media/disk2/flatbug-yolo"
# dataset_dir = "/media/disk2/arthropod-detection-dataset/dataset"
main(dataset_dir)