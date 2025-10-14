from typing import Any
from ultralytics.data.augment import BaseMixTransform
import random
import numpy as np
from ultralytics.utils.instance import Instances

class Mosaic36(BaseMixTransform):
    """
    Mosaic36 augmentation for image datasets with 6x6 grid support.

    This class performs mosaic augmentation by combining 36 images into a single 6x6 mosaic image.
    The augmentation is applied to a dataset with a given probability.

    Attributes:
        dataset: The dataset on which the mosaic augmentation is applied.
        imgsz (int): Image size (height and width) after mosaic pipeline of a single image.
        p (float): Probability of applying the mosaic augmentation. Must be in the range 0-1.
        border (tuple[int, int]): Border size for width and height.

    Methods:
        get_indexes: Return a list of random indexes from the dataset.
        _mix_transform: Apply mixup transformation to the input image and labels.
        _mosaic36: Create a 6x6 image mosaic.
        _update_labels: Update labels with padding.
        _cat_labels: Concatenate labels and clips mosaic border instances.

    Examples:
        >>> from training.Mosaic36 import Mosaic36
        >>> dataset = YourDataset(...)  # Your image dataset
        >>> mosaic_aug = Mosaic36(dataset, imgsz=640, p=0.5)
        >>> augmented_labels = mosaic_aug(original_labels)
    """

    def __init__(self, dataset, imgsz: int = 640, p: float = 1.0):
        """
        Initialize the Mosaic36 augmentation object.

        This class performs mosaic augmentation by combining 36 images into a single 6x6 mosaic image.
        The augmentation is applied to a dataset with a given probability.

        Args:
            dataset (Any): The dataset on which the mosaic augmentation is applied.
            imgsz (int): Image size (height and width) after mosaic pipeline of a single image.
            p (float): Probability of applying the mosaic augmentation. Must be in the range 0-1.

        Examples:
            >>> from training.Mosaic36 import Mosaic36
            >>> dataset = YourDataset(...)
            >>> mosaic_aug = Mosaic36(dataset, imgsz=640, p=0.5)
        """
        assert 0 <= p <= 1.0, f"The probability should be in range [0, 1], but got {p}."
        super().__init__(dataset=dataset, p=p)
        self.imgsz = imgsz
        self.border = (-imgsz // 2, -imgsz // 2)  # width, height
        self.buffer_enabled = self.dataset.cache != "ram"

    def get_indexes(self):
        """
        Return a list of random indexes from the dataset for mosaic augmentation.

        This method selects random image indexes either from a buffer or from the entire dataset, depending on
        the 'buffer' parameter. It is used to choose images for creating 6x6 mosaic augmentations.

        Returns:
            (list[int]): A list of 35 random image indexes (36 total images including the base image).

        Examples:
            >>> mosaic36 = Mosaic36(dataset, imgsz=640, p=1.0)
            >>> indexes = mosaic36.get_indexes()
            >>> print(len(indexes))  # Output: 35
        """
        if self.buffer_enabled:  # select images from buffer
            return random.choices(list(self.dataset.buffer), k=35)
        else:  # select any images
            return [random.randint(0, len(self.dataset) - 1) for _ in range(35)]

    def _mix_transform(self, labels: dict[str, Any]) -> dict[str, Any]:
        """
        Apply 6x6 mosaic augmentation to the input image and labels.

        This method combines 36 images into a single 6x6 mosaic image. It ensures that rectangular 
        annotations are not present and that there are other images available for mosaic augmentation.

        Args:
            labels (dict[str, Any]): A dictionary containing image data and annotations. Expected keys include:
                - 'rect_shape': Should be None as rect and mosaic are mutually exclusive.
                - 'mix_labels': A list of dictionaries containing data for other images to be used in the mosaic.

        Returns:
            (dict[str, Any]): A dictionary containing the mosaic-augmented image and updated annotations.

        Raises:
            AssertionError: If 'rect_shape' is not None or if 'mix_labels' is empty.

        Examples:
            >>> mosaic36 = Mosaic36(dataset, imgsz=640, p=1.0)
            >>> augmented_data = mosaic36._mix_transform(labels)
        """
        assert labels.get("rect_shape") is None, "rect and mosaic are mutually exclusive."
        assert len(labels.get("mix_labels", [])) >= 35, "Need at least 35 other images for 6x6 mosaic augment."
        return self._mosaic36(labels)

    def _mosaic36(self, labels: dict[str, Any]) -> dict[str, Any]:
        """
        Create a 6x6 image mosaic from thirty-six input images with same final size as 2x2 mosaic.

        This method combines thirty-six images into a single mosaic image by placing them in a 6x6 grid,
        but keeps the final output size the same as regular 2x2 mosaic to avoid memory issues.
        Each individual image will be smaller (s/3 x s/3 instead of s x s).

        Args:
            labels (dict[str, Any]): A dictionary containing image data and labels for the base image (index 0) and thirty-five
                additional images (indices 1-35) in the 'mix_labels' key.

        Returns:
            (dict[str, Any]): A dictionary containing the mosaic image and updated labels. The 'img' key contains the mosaic
                image as a numpy array, and other keys contain the combined and adjusted labels for all thirty-six images.

        Examples:
            >>> mosaic36 = Mosaic36(dataset, imgsz=640, p=1.0)
            >>> labels = {
            ...     "img": np.random.rand(480, 640, 3),
            ...     "mix_labels": [{"img": np.random.rand(480, 640, 3)} for _ in range(35)],
            ... }
            >>> result = mosaic36._mosaic36(labels)
            >>> assert result["img"].shape == (1280, 1280, 3)  # 2 * 640 = 1280 (same as regular mosaic)
        """
        mosaic_labels = []
        s = self.imgsz
        # Each cell is s/3 x s/3 to fit 6x6 grid in same space as 2x2 (s*2 x s*2)
        cell_size = s // 3
        
        for i in range(36):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            # Load image
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")

            # Calculate grid position (row, col) for 6x6 grid
            row = i // 6
            col = i % 6
            
            # Place img in img36 - 6x6 grid but same final size as 2x2 mosaic
            if i == 0:  # Initialize the base image - same size as regular 2x2 mosaic
                img36 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)
            
            # Calculate position in 6x6 grid within 2x2 space
            # Each cell is exactly cell_size x cell_size
            x1a = col * cell_size
            y1a = row * cell_size
            x2a = min(x1a + cell_size, s * 2)  # Ensure we don't exceed bounds
            y2a = min(y1a + cell_size, s * 2)
            
            # Calculate how much of the original image we can fit
            cell_w = x2a - x1a
            cell_h = y2a - y1a
            
            # Scale image to fit exactly in the cell while maintaining aspect ratio
            scale_w = cell_w / w if w > 0 else 1
            scale_h = cell_h / h if h > 0 else 1
            scale = min(scale_w, scale_h)
            
            # Calculate actual dimensions after scaling
            scaled_w = min(int(w * scale), cell_w)
            scaled_h = min(int(h * scale), cell_h)
            
            if scaled_w > 0 and scaled_h > 0:
                # Resize image using numpy indexing
                h_indices = np.linspace(0, h-1, scaled_h).astype(int)
                w_indices = np.linspace(0, w-1, scaled_w).astype(int)
                img_scaled = img[np.ix_(h_indices, w_indices)]
                
                # Place the scaled image in the cell
                img36[y1a:y1a+scaled_h, x1a:x1a+scaled_w] = img_scaled
                
                # Update labels with the scaled dimensions and position
                # The key insight: treat the scaled image as if it were the original size
                labels_patch["resized_shape"] = (scaled_h, scaled_w)
                labels_patch = self._update_labels(labels_patch, x1a, y1a)
            else:
                # Empty patch - still need to add to maintain structure
                labels_patch["resized_shape"] = (1, 1)  # Minimal size
                labels_patch = self._update_labels(labels_patch, x1a, y1a)
                
            mosaic_labels.append(labels_patch)
            
        final_labels = self._cat_labels(mosaic_labels)
        final_labels["img"] = img36
        return final_labels

    @staticmethod
    def _update_labels(labels, padw: int, padh: int) -> dict[str, Any]:
        """
        Update label coordinates with padding values.

        This method adjusts the bounding box coordinates of object instances in the labels by adding padding
        values. It also denormalizes the coordinates if they were previously normalized.

        Args:
            labels (dict[str, Any]): A dictionary containing image and instance information.
            padw (int): Padding width to be added to the x-coordinates.
            padh (int): Padding height to be added to the y-coordinates.

        Returns:
            (dict): Updated labels dictionary with adjusted instance coordinates.

        Examples:
            >>> labels = {"img": np.zeros((100, 100, 3)), "instances": Instances(...)}
            >>> padw, padh = 50, 50
            >>> updated_labels = Mosaic16._update_labels(labels, padw, padh)
        """
        # Use the original image dimensions for proper coordinate transformation
        nh, nw = labels["resized_shape"]
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(nw, nh)
        labels["instances"].add_padding(padw, padh)
        return labels

    def _cat_labels(self, mosaic_labels: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Concatenate and process labels for 4x4 mosaic augmentation.

        This method combines labels from 16 images used in mosaic augmentation, clips instances to the
        mosaic border, and removes zero-area boxes. Output size is same as 2x2 mosaic for memory efficiency.

        Args:
            mosaic_labels (list[dict[str, Any]]): A list of label dictionaries for each image in the mosaic.

        Returns:
            (dict[str, Any]): A dictionary containing concatenated and processed labels for the mosaic image, including:
                - im_file (str): File path of the first image in the mosaic.
                - ori_shape (tuple[int, int]): Original shape of the first image.
                - resized_shape (tuple[int, int]): Shape of the mosaic image (imgsz * 2, imgsz * 2).
                - cls (np.ndarray): Concatenated class labels.
                - instances (Instances): Concatenated instance annotations.
                - mosaic_border (tuple[int, int]): Mosaic border size.
                - texts (list[str], optional): Text labels if present in the original labels.

        Examples:
            >>> mosaic16 = Mosaic16(dataset, imgsz=640)
            >>> mosaic_labels = [{"cls": np.array([0, 1]), "instances": Instances(...)} for _ in range(16)]
            >>> result = mosaic16._cat_labels(mosaic_labels)
            >>> print(result.keys())
            dict_keys(['im_file', 'ori_shape', 'resized_shape', 'cls', 'instances', 'mosaic_border'])
        """
        if not mosaic_labels:
            return {}
        cls = []
        instances = []
        imgsz = self.imgsz * 2  # Same as 2x2 mosaic for memory efficiency
        for labels in mosaic_labels:
            cls.append(labels["cls"])
            instances.append(labels["instances"])
        # Final labels
        final_labels = {
            "im_file": mosaic_labels[0]["im_file"],
            "ori_shape": mosaic_labels[0]["ori_shape"],
            "resized_shape": (imgsz, imgsz),
            "cls": np.concatenate(cls, 0),
            "instances": Instances.concatenate(instances, axis=0),
            "mosaic_border": self.border,
        }
        final_labels["instances"].clip(imgsz, imgsz)
        good = final_labels["instances"].remove_zero_area_boxes()
        final_labels["cls"] = final_labels["cls"][good]
        if "texts" in mosaic_labels[0]:
            final_labels["texts"] = mosaic_labels[0]["texts"]
        return final_labels
