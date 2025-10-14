from ultralytics import YOLO
from ultralytics.data.augment import Mosaic
from ultralytics.models.yolo.detect.train import DetectionTrainer
from Mosaic36 import Mosaic36

# Configuration
mosaic_type = '6x6'  # 6x6 mosaicing with 36 images
data_yaml = "dataset/Arthropoda.yaml"
project = 'arthro_mosaic_66'

class CustomTrainer(DetectionTrainer):
    """Custom trainer class to enable 6x6 mosaicing with Mosaic36"""
    
    def build_dataset(self, img_path, mode="train", batch=None):
        """Build dataset with custom Mosaic36 configuration"""
        dataset = super().build_dataset(img_path, mode, batch)
        
        if mode == "train" and mosaic_type == '6x6':
            # Look for Mosaic transform in main transforms and sub-transforms
            mosaic_found = False
            for i, transform in enumerate(dataset.transforms.transforms):
                # Check if this is a Compose transform containing other transforms
                if hasattr(transform, 'transforms'):
                    for j, sub_transform in enumerate(transform.transforms):
                        if isinstance(sub_transform, Mosaic):
                            # Store original probability and replace with Mosaic36
                            original_p = sub_transform.p
                            new_mosaic = Mosaic36(
                                dataset=dataset,
                                imgsz=self.args.imgsz,
                                p=original_p
                            )
                            transform.transforms[j] = new_mosaic
                            mosaic_found = True
                            print(f"Replaced Mosaic with Mosaic36 (6x6, 36 images) at probability {original_p}")
                            break
                
                # Also check if the transform itself is a Mosaic (direct case)
                if isinstance(transform, Mosaic):
                    original_p = transform.p
                    new_mosaic = Mosaic36(
                        dataset=dataset,
                        imgsz=self.args.imgsz,
                        p=original_p
                    )
                    dataset.transforms.transforms[i] = new_mosaic
                    mosaic_found = True
                    print(f"Replaced Mosaic with Mosaic36 (6x6, 36 images) at probability {original_p}")
                    break
                
                if mosaic_found:
                    break
        
        return dataset

# Load model
model = YOLO('yolo11l.pt')

# Configure training parameters for 6x6 mosaicing
# Use single GPU for custom trainer to avoid DDP issues with custom transforms
train_params = {
    'data': data_yaml,
    'epochs': 100,
    'project': 'runs/' + project,
    'device': 0,
    'mosaic': 0.5,  # Enable mosaic augmentation HALF OF THE TIME
    'model': 'yolo11l.pt',  # Specify the model for the trainer
}

# Create trainer instance manually to ensure our custom class is used
trainer = CustomTrainer(overrides=train_params)
print("Training with custom 6x6 mosaicing (Mosaic36) enabled")
trainer.train()

# Optional validation
# best_model = YOLO(f'runs/{project}/train/weights/best.pt')
# best_model.val(
#     data=data_yaml,
#     project='runs/' + project,
#     device=0,  # Use device 0 (which is actually GPU 1 due to CUDA_VISIBLE_DEVICES)
#     split='test'
# )