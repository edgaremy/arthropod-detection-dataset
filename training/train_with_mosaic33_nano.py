from ultralytics import YOLO
from ultralytics.data.augment import Mosaic
from ultralytics.models.yolo.detect.train import DetectionTrainer
import yaml

# Configuration
mosaic_type = '3x3'  # Change to '2x2' for default mosaicing
data_yaml = "dataset/ArthroNat.yaml"
project = f'arthro_mosaic_{mosaic_type.replace("x", "")}_nano'

class CustomTrainer(DetectionTrainer):
    """Custom trainer class to enable 3x3 mosaicing"""
    
    def build_dataset(self, img_path, mode="train", batch=None):
        """Build dataset with custom mosaic configuration"""
        dataset = super().build_dataset(img_path, mode, batch)
        
        if mode == "train" and mosaic_type == '3x3':
            # Look for Mosaic transform in main transforms and sub-transforms
            mosaic_found = False
            for i, transform in enumerate(dataset.transforms.transforms):
                # Check if this is a Compose transform containing other transforms
                if hasattr(transform, 'transforms'):
                    for j, sub_transform in enumerate(transform.transforms):
                        if isinstance(sub_transform, Mosaic):
                            # Store original probability and replace with 3x3 version
                            original_p = sub_transform.p
                            new_mosaic = Mosaic(
                                dataset=dataset,
                                imgsz=self.args.imgsz,
                                p=original_p,
                                n=9  # Use 9 images for 3x3 mosaic
                            )
                            transform.transforms[j] = new_mosaic
                            mosaic_found = True
                            break
                
                # Also check if the transform itself is a Mosaic (direct case)
                if isinstance(transform, Mosaic):
                    original_p = transform.p
                    new_mosaic = Mosaic(
                        dataset=dataset,
                        imgsz=self.args.imgsz,
                        p=original_p,
                        n=9  # Use 9 images for 3x3 mosaic
                    )
                    dataset.transforms.transforms[i] = new_mosaic
                    mosaic_found = True
                    break
                
                if mosaic_found:
                    break
        
        return dataset

# Load model
model = YOLO('yolo11n.pt')

# Configure training parameters
if mosaic_type == '3x3':
    # Use single GPU for custom trainer to avoid DDP issues
    train_params = {
        'data': data_yaml,
        'epochs': 100,
        'project': 'runs/' + project,
        'device': 1,  # Use single GPU to avoid DDP serialization issues
        'mosaic': 1.0,  # Enable mosaic augmentation
        'model': 'yolo11n.pt',  # Specify the model for the trainer
    }
    
    # Create trainer instance manually to ensure our custom class is used
    trainer = CustomTrainer(overrides=train_params)
    print("Training with custom 3x3 mosaicing enabled")
    trainer.train()
else:
    # Default 2x2 mosaicing with multi-GPU support
    train_params = {
        'data': data_yaml,
        'epochs': 100,
        'project': 'runs/' + project,
        'device': ['1'],
        'mosaic': 1.0,  # Enable mosaic augmentation (default 2x2)
    }
    print("Training with default 2x2 mosaicing")
    model.train(**train_params)

# Optional validation
# best_model = YOLO(f'runs/{project}/train/weights/best.pt')
# best_model.val(
#     data=data_yaml,
#     project='runs/' + project,
#     device=1 if mosaic_type == '3x3' else ['0', '1'],  # Match training device config
#     split='test'
# )