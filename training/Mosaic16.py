from typing import Any
from ultralytics.data.augment import BaseMixTransform
import random
import numpy as np
from ultralytics.utils.instance import Instances

class Mosaic16(BaseMixTransform):
    """
    Mosaic16 augmentation for image datasets with 4x4 grid support.

    This class performs mosaic augmentation by combining 16 images into a single 4x4 mosaic image.
    The augmentation is applied to a dataset with a given probability.

    Attributes:
        dataset: The dataset on which the mosaic augmentation is applied.
        imgsz (int): Image size (height and width) after mosaic pipeline of a single image.
        p (float): Probability of applying the mosaic augmentation. Must be in the range 0-1.
        border (tuple[int, int]): Border size for width and height.

    Methods:
        get_indexes: Return a list of random indexes from the dataset.
        _mix_transform: Apply mixup transformation to the input image and labels.
        _mosaic16: Create a 4x4 image mosaic.
        _update_labels: Update labels with padding.
        _cat_labels: Concatenate labels and clips mosaic border instances.

    Examples:
        >>> from training.Mosaic import Mosaic16
        >>> dataset = YourDataset(...)  # Your image dataset
        >>> mosaic_aug = Mosaic16(dataset, imgsz=640, p=0.5)
        >>> augmented_labels = mosaic_aug(original_labels)
    """

    def __init__(self, dataset, imgsz: int = 640, p: float = 1.0):
        """
        Initialize the Mosaic16 augmentation object.

        This class performs mosaic augmentation by combining 16 images into a single 4x4 mosaic image.
        The augmentation is applied to a dataset with a given probability.

        Args:
            dataset (Any): The dataset on which the mosaic augmentation is applied.
            imgsz (int): Image size (height and width) after mosaic pipeline of a single image.
            p (float): Probability of applying the mosaic augmentation. Must be in the range 0-1.

        Examples:
            >>> from training.Mosaic import Mosaic16
            >>> dataset = YourDataset(...)
            >>> mosaic_aug = Mosaic16(dataset, imgsz=640, p=0.5)
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
        the 'buffer' parameter. It is used to choose images for creating 4x4 mosaic augmentations.

        Returns:
            (list[int]): A list of 15 random image indexes (16 total images including the base image).

        Examples:
            >>> mosaic16 = Mosaic16(dataset, imgsz=640, p=1.0)
            >>> indexes = mosaic16.get_indexes()
            >>> print(len(indexes))  # Output: 15
        """
        if self.buffer_enabled:  # select images from buffer
            return random.choices(list(self.dataset.buffer), k=15)
        else:  # select any images
            return [random.randint(0, len(self.dataset) - 1) for _ in range(15)]

    def _mix_transform(self, labels: dict[str, Any]) -> dict[str, Any]:
        """
        Apply 4x4 mosaic augmentation to the input image and labels.

        This method combines 16 images into a single 4x4 mosaic image. It ensures that rectangular 
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
            >>> mosaic16 = Mosaic16(dataset, imgsz=640, p=1.0)
            >>> augmented_data = mosaic16._mix_transform(labels)
        """
        assert labels.get("rect_shape") is None, "rect and mosaic are mutually exclusive."
        assert len(labels.get("mix_labels", [])) >= 15, "Need at least 15 other images for 4x4 mosaic augment."
        return self._mosaic16(labels)

    def _mosaic16(self, labels: dict[str, Any]) -> dict[str, Any]:
        """
        Create a 4x4 image mosaic from sixteen input images with same final size as 2x2 mosaic.

        This method combines sixteen images into a single mosaic image by placing them in a 4x4 grid,
        but keeps the final output size the same as regular 2x2 mosaic to avoid memory issues.
        Each individual image will be smaller (s/2 x s/2 instead of s x s).

        Args:
            labels (dict[str, Any]): A dictionary containing image data and labels for the base image (index 0) and fifteen
                additional images (indices 1-15) in the 'mix_labels' key.

        Returns:
            (dict[str, Any]): A dictionary containing the mosaic image and updated labels. The 'img' key contains the mosaic
                image as a numpy array, and other keys contain the combined and adjusted labels for all sixteen images.

        Examples:
            >>> mosaic16 = Mosaic16(dataset, imgsz=640, p=1.0)
            >>> labels = {
            ...     "img": np.random.rand(480, 640, 3),
            ...     "mix_labels": [{"img": np.random.rand(480, 640, 3)} for _ in range(15)],
            ... }
            >>> result = mosaic16._mosaic16(labels)
            >>> assert result["img"].shape == (1280, 1280, 3)  # 2 * 640 = 1280 (same as regular mosaic)
        """
        mosaic_labels = []
        s = self.imgsz
        # Each cell is s/2 x s/2 to fit 4x4 grid in same space as 2x2 (s*2 x s*2)
        cell_size = s // 2
        
        for i in range(16):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            # Load image
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")

            # Calculate grid position (row, col) for 4x4 grid
            row = i // 4
            col = i % 4
            
            # Place img in img16 - 4x4 grid but same final size as 2x2 mosaic
            if i == 0:  # Initialize the base image - same size as regular 2x2 mosaic
                img16 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)
            
            # Calculate position in 4x4 grid within 2x2 space
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
                img16[y1a:y1a+scaled_h, x1a:x1a+scaled_w] = img_scaled
                
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
        final_labels["img"] = img16
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

# ORIGINAL MOSAIC IMPLEMENTATION BY ULTRALYTICS #
#################################################

# class Mosaic(BaseMixTransform):
#     """
#     Mosaic augmentation for image datasets.

#     This class performs mosaic augmentation by combining multiple (4 or 9) images into a single mosaic image.
#     The augmentation is applied to a dataset with a given probability.

#     Attributes:
#         dataset: The dataset on which the mosaic augmentation is applied.
#         imgsz (int): Image size (height and width) after mosaic pipeline of a single image.
#         p (float): Probability of applying the mosaic augmentation. Must be in the range 0-1.
#         n (int): The grid size, either 4 (for 2x2) or 9 (for 3x3).
#         border (tuple[int, int]): Border size for width and height.

#     Methods:
#         get_indexes: Return a list of random indexes from the dataset.
#         _mix_transform: Apply mixup transformation to the input image and labels.
#         _mosaic3: Create a 1x3 image mosaic.
#         _mosaic4: Create a 2x2 image mosaic.
#         _mosaic9: Create a 3x3 image mosaic.
#         _update_labels: Update labels with padding.
#         _cat_labels: Concatenate labels and clips mosaic border instances.

#     Examples:
#         >>> from ultralytics.data.augment import Mosaic
#         >>> dataset = YourDataset(...)  # Your image dataset
#         >>> mosaic_aug = Mosaic(dataset, imgsz=640, p=0.5, n=4)
#         >>> augmented_labels = mosaic_aug(original_labels)
#     """

#     def __init__(self, dataset, imgsz: int = 640, p: float = 1.0, n: int = 4):
#         """
#         Initialize the Mosaic augmentation object.

#         This class performs mosaic augmentation by combining multiple (4 or 9) images into a single mosaic image.
#         The augmentation is applied to a dataset with a given probability.

#         Args:
#             dataset (Any): The dataset on which the mosaic augmentation is applied.
#             imgsz (int): Image size (height and width) after mosaic pipeline of a single image.
#             p (float): Probability of applying the mosaic augmentation. Must be in the range 0-1.
#             n (int): The grid size, either 4 (for 2x2) or 9 (for 3x3).

#         Examples:
#             >>> from ultralytics.data.augment import Mosaic
#             >>> dataset = YourDataset(...)
#             >>> mosaic_aug = Mosaic(dataset, imgsz=640, p=0.5, n=4)
#         """
#         assert 0 <= p <= 1.0, f"The probability should be in range [0, 1], but got {p}."
#         assert n in {4, 9}, "grid must be equal to 4 or 9."
#         super().__init__(dataset=dataset, p=p)
#         self.imgsz = imgsz
#         self.border = (-imgsz // 2, -imgsz // 2)  # width, height
#         self.n = n
#         self.buffer_enabled = self.dataset.cache != "ram"

#     def get_indexes(self):
#         """
#         Return a list of random indexes from the dataset for mosaic augmentation.

#         This method selects random image indexes either from a buffer or from the entire dataset, depending on
#         the 'buffer' parameter. It is used to choose images for creating mosaic augmentations.

#         Returns:
#             (list[int]): A list of random image indexes. The length of the list is n-1, where n is the number
#                 of images used in the mosaic (either 3 or 8, depending on whether n is 4 or 9).

#         Examples:
#             >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=4)
#             >>> indexes = mosaic.get_indexes()
#             >>> print(len(indexes))  # Output: 3
#         """
#         if self.buffer_enabled:  # select images from buffer
#             return random.choices(list(self.dataset.buffer), k=self.n - 1)
#         else:  # select any images
#             return [random.randint(0, len(self.dataset) - 1) for _ in range(self.n - 1)]

#     def _mix_transform(self, labels: dict[str, Any]) -> dict[str, Any]:
#         """
#         Apply mosaic augmentation to the input image and labels.

#         This method combines multiple images (3, 4, or 9) into a single mosaic image based on the 'n' attribute.
#         It ensures that rectangular annotations are not present and that there are other images available for
#         mosaic augmentation.

#         Args:
#             labels (dict[str, Any]): A dictionary containing image data and annotations. Expected keys include:
#                 - 'rect_shape': Should be None as rect and mosaic are mutually exclusive.
#                 - 'mix_labels': A list of dictionaries containing data for other images to be used in the mosaic.

#         Returns:
#             (dict[str, Any]): A dictionary containing the mosaic-augmented image and updated annotations.

#         Raises:
#             AssertionError: If 'rect_shape' is not None or if 'mix_labels' is empty.

#         Examples:
#             >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=4)
#             >>> augmented_data = mosaic._mix_transform(labels)
#         """
#         assert labels.get("rect_shape") is None, "rect and mosaic are mutually exclusive."
#         assert len(labels.get("mix_labels", [])), "There are no other images for mosaic augment."
#         return (
#             self._mosaic3(labels) if self.n == 3 else self._mosaic4(labels) if self.n == 4 else self._mosaic9(labels)
#         )  # This code is modified for mosaic3 method.

#     def _mosaic3(self, labels: dict[str, Any]) -> dict[str, Any]:
#         """
#         Create a 1x3 image mosaic by combining three images.

#         This method arranges three images in a horizontal layout, with the main image in the center and two
#         additional images on either side. It's part of the Mosaic augmentation technique used in object detection.

#         Args:
#             labels (dict[str, Any]): A dictionary containing image and label information for the main (center) image.
#                 Must include 'img' key with the image array, and 'mix_labels' key with a list of two
#                 dictionaries containing information for the side images.

#         Returns:
#             (dict[str, Any]): A dictionary with the mosaic image and updated labels. Keys include:
#                 - 'img' (np.ndarray): The mosaic image array with shape (H, W, C).
#                 - Other keys from the input labels, updated to reflect the new image dimensions.

#         Examples:
#             >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=3)
#             >>> labels = {
#             ...     "img": np.random.rand(480, 640, 3),
#             ...     "mix_labels": [{"img": np.random.rand(480, 640, 3)} for _ in range(2)],
#             ... }
#             >>> result = mosaic._mosaic3(labels)
#             >>> print(result["img"].shape)
#             (640, 640, 3)
#         """
#         mosaic_labels = []
#         s = self.imgsz
#         for i in range(3):
#             labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
#             # Load image
#             img = labels_patch["img"]
#             h, w = labels_patch.pop("resized_shape")

#             # Place img in img3
#             if i == 0:  # center
#                 img3 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 3 tiles
#                 h0, w0 = h, w
#                 c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
#             elif i == 1:  # right
#                 c = s + w0, s, s + w0 + w, s + h
#             elif i == 2:  # left
#                 c = s - w, s + h0 - h, s, s + h0

#             padw, padh = c[:2]
#             x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coordinates

#             img3[y1:y2, x1:x2] = img[y1 - padh :, x1 - padw :]  # img3[ymin:ymax, xmin:xmax]
#             # hp, wp = h, w  # height, width previous for next iteration

#             # Labels assuming imgsz*2 mosaic size
#             labels_patch = self._update_labels(labels_patch, padw + self.border[0], padh + self.border[1])
#             mosaic_labels.append(labels_patch)
#         final_labels = self._cat_labels(mosaic_labels)

#         final_labels["img"] = img3[-self.border[0] : self.border[0], -self.border[1] : self.border[1]]
#         return final_labels

#     def _mosaic4(self, labels: dict[str, Any]) -> dict[str, Any]:
#         """
#         Create a 2x2 image mosaic from four input images.

#         This method combines four images into a single mosaic image by placing them in a 2x2 grid. It also
#         updates the corresponding labels for each image in the mosaic.

#         Args:
#             labels (dict[str, Any]): A dictionary containing image data and labels for the base image (index 0) and three
#                 additional images (indices 1-3) in the 'mix_labels' key.

#         Returns:
#             (dict[str, Any]): A dictionary containing the mosaic image and updated labels. The 'img' key contains the mosaic
#                 image as a numpy array, and other keys contain the combined and adjusted labels for all four images.

#         Examples:
#             >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=4)
#             >>> labels = {
#             ...     "img": np.random.rand(480, 640, 3),
#             ...     "mix_labels": [{"img": np.random.rand(480, 640, 3)} for _ in range(3)],
#             ... }
#             >>> result = mosaic._mosaic4(labels)
#             >>> assert result["img"].shape == (1280, 1280, 3)
#         """
#         mosaic_labels = []
#         s = self.imgsz
#         yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)  # mosaic center x, y
#         for i in range(4):
#             labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
#             # Load image
#             img = labels_patch["img"]
#             h, w = labels_patch.pop("resized_shape")

#             # Place img in img4
#             if i == 0:  # top left
#                 img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
#                 x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
#                 x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
#             elif i == 1:  # top right
#                 x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
#                 x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
#             elif i == 2:  # bottom left
#                 x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
#                 x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
#             elif i == 3:  # bottom right
#                 x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
#                 x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

#             img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
#             padw = x1a - x1b
#             padh = y1a - y1b

#             labels_patch = self._update_labels(labels_patch, padw, padh)
#             mosaic_labels.append(labels_patch)
#         final_labels = self._cat_labels(mosaic_labels)
#         final_labels["img"] = img4
#         return final_labels

#     def _mosaic9(self, labels: dict[str, Any]) -> dict[str, Any]:
#         """
#         Create a 3x3 image mosaic from the input image and eight additional images.

#         This method combines nine images into a single mosaic image. The input image is placed at the center,
#         and eight additional images from the dataset are placed around it in a 3x3 grid pattern.

#         Args:
#             labels (dict[str, Any]): A dictionary containing the input image and its associated labels. It should have
#                 the following keys:
#                 - 'img' (np.ndarray): The input image.
#                 - 'resized_shape' (tuple[int, int]): The shape of the resized image (height, width).
#                 - 'mix_labels' (list[dict]): A list of dictionaries containing information for the additional
#                   eight images, each with the same structure as the input labels.

#         Returns:
#             (dict[str, Any]): A dictionary containing the mosaic image and updated labels. It includes the following keys:
#                 - 'img' (np.ndarray): The final mosaic image.
#                 - Other keys from the input labels, updated to reflect the new mosaic arrangement.

#         Examples:
#             >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=9)
#             >>> input_labels = dataset[0]
#             >>> mosaic_result = mosaic._mosaic9(input_labels)
#             >>> mosaic_image = mosaic_result["img"]
#         """
#         mosaic_labels = []
#         s = self.imgsz
#         hp, wp = -1, -1  # height, width previous
#         for i in range(9):
#             labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
#             # Load image
#             img = labels_patch["img"]
#             h, w = labels_patch.pop("resized_shape")

#             # Place img in img9
#             if i == 0:  # center
#                 img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
#                 h0, w0 = h, w
#                 c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
#             elif i == 1:  # top
#                 c = s, s - h, s + w, s
#             elif i == 2:  # top right
#                 c = s + wp, s - h, s + wp + w, s
#             elif i == 3:  # right
#                 c = s + w0, s, s + w0 + w, s + h
#             elif i == 4:  # bottom right
#                 c = s + w0, s + hp, s + w0 + w, s + hp + h
#             elif i == 5:  # bottom
#                 c = s + w0 - w, s + h0, s + w0, s + h0 + h
#             elif i == 6:  # bottom left
#                 c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
#             elif i == 7:  # left
#                 c = s - w, s + h0 - h, s, s + h0
#             elif i == 8:  # top left
#                 c = s - w, s + h0 - hp - h, s, s + h0 - hp

#             padw, padh = c[:2]
#             x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coordinates

#             # Image
#             img9[y1:y2, x1:x2] = img[y1 - padh :, x1 - padw :]  # img9[ymin:ymax, xmin:xmax]
#             hp, wp = h, w  # height, width previous for next iteration

#             # Labels assuming imgsz*2 mosaic size
#             labels_patch = self._update_labels(labels_patch, padw + self.border[0], padh + self.border[1])
#             mosaic_labels.append(labels_patch)
#         final_labels = self._cat_labels(mosaic_labels)

#         final_labels["img"] = img9[-self.border[0] : self.border[0], -self.border[1] : self.border[1]]
#         return final_labels

#     @staticmethod
#     def _update_labels(labels, padw: int, padh: int) -> dict[str, Any]:
#         """
#         Update label coordinates with padding values.

#         This method adjusts the bounding box coordinates of object instances in the labels by adding padding
#         values. It also denormalizes the coordinates if they were previously normalized.

#         Args:
#             labels (dict[str, Any]): A dictionary containing image and instance information.
#             padw (int): Padding width to be added to the x-coordinates.
#             padh (int): Padding height to be added to the y-coordinates.

#         Returns:
#             (dict): Updated labels dictionary with adjusted instance coordinates.

#         Examples:
#             >>> labels = {"img": np.zeros((100, 100, 3)), "instances": Instances(...)}
#             >>> padw, padh = 50, 50
#             >>> updated_labels = Mosaic._update_labels(labels, padw, padh)
#         """
#         nh, nw = labels["img"].shape[:2]
#         labels["instances"].convert_bbox(format="xyxy")
#         labels["instances"].denormalize(nw, nh)
#         labels["instances"].add_padding(padw, padh)
#         return labels

#     def _cat_labels(self, mosaic_labels: list[dict[str, Any]]) -> dict[str, Any]:
#         """
#         Concatenate and process labels for mosaic augmentation.

#         This method combines labels from multiple images used in mosaic augmentation, clips instances to the
#         mosaic border, and removes zero-area boxes.

#         Args:
#             mosaic_labels (list[dict[str, Any]]): A list of label dictionaries for each image in the mosaic.

#         Returns:
#             (dict[str, Any]): A dictionary containing concatenated and processed labels for the mosaic image, including:
#                 - im_file (str): File path of the first image in the mosaic.
#                 - ori_shape (tuple[int, int]): Original shape of the first image.
#                 - resized_shape (tuple[int, int]): Shape of the mosaic image (imgsz * 2, imgsz * 2).
#                 - cls (np.ndarray): Concatenated class labels.
#                 - instances (Instances): Concatenated instance annotations.
#                 - mosaic_border (tuple[int, int]): Mosaic border size.
#                 - texts (list[str], optional): Text labels if present in the original labels.

#         Examples:
#             >>> mosaic = Mosaic(dataset, imgsz=640)
#             >>> mosaic_labels = [{"cls": np.array([0, 1]), "instances": Instances(...)} for _ in range(4)]
#             >>> result = mosaic._cat_labels(mosaic_labels)
#             >>> print(result.keys())
#             dict_keys(['im_file', 'ori_shape', 'resized_shape', 'cls', 'instances', 'mosaic_border'])
#         """
#         if not mosaic_labels:
#             return {}
#         cls = []
#         instances = []
#         imgsz = self.imgsz * 2  # mosaic imgsz
#         for labels in mosaic_labels:
#             cls.append(labels["cls"])
#             instances.append(labels["instances"])
#         # Final labels
#         final_labels = {
#             "im_file": mosaic_labels[0]["im_file"],
#             "ori_shape": mosaic_labels[0]["ori_shape"],
#             "resized_shape": (imgsz, imgsz),
#             "cls": np.concatenate(cls, 0),
#             "instances": Instances.concatenate(instances, axis=0),
#             "mosaic_border": self.border,
#         }
#         final_labels["instances"].clip(imgsz, imgsz)
#         good = final_labels["instances"].remove_zero_area_boxes()
#         final_labels["cls"] = final_labels["cls"][good]
#         if "texts" in mosaic_labels[0]:
#             final_labels["texts"] = mosaic_labels[0]["texts"]
#         return final_labels
