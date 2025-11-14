from ultralytics import YOLO
from ultralytics.data.augment import Mosaic
from ultralytics.models.yolo.detect.train import DetectionTrainer
from Mosaic16 import Mosaic16

# Force PyTorch to use only GPU 1
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# torch.cuda.set_device(0)  # Since GPU 1 becomes device 0 after CUDA_VISIBLE_DEVICES

# Configuration
mosaic_type = '4x4'  # 4x4 mosaicing with 16 images
data_yaml = "dataset/ArthroNat.yaml"
project = 'arthro_mosaic_44'

class CustomTrainer(DetectionTrainer):
    """Custom trainer class to enable 4x4 mosaicing with Mosaic16"""
    
    def build_dataset(self, img_path, mode="train", batch=None):
        """Build dataset with custom Mosaic16 configuration"""
        dataset = super().build_dataset(img_path, mode, batch)
        
        if mode == "train" and mosaic_type == '4x4':
            # Look for Mosaic transform in main transforms and sub-transforms
            mosaic_found = False
            for i, transform in enumerate(dataset.transforms.transforms):
                # Check if this is a Compose transform containing other transforms
                if hasattr(transform, 'transforms'):
                    for j, sub_transform in enumerate(transform.transforms):
                        if isinstance(sub_transform, Mosaic):
                            # Store original probability and replace with Mosaic16
                            original_p = sub_transform.p
                            new_mosaic = Mosaic16(
                                dataset=dataset,
                                imgsz=self.args.imgsz,
                                p=original_p
                            )
                            transform.transforms[j] = new_mosaic
                            mosaic_found = True
                            print(f"Replaced Mosaic with Mosaic16 (4x4, 16 images) at probability {original_p}")
                            break
                
                # Also check if the transform itself is a Mosaic (direct case)
                if isinstance(transform, Mosaic):
                    original_p = transform.p
                    new_mosaic = Mosaic16(
                        dataset=dataset,
                        imgsz=self.args.imgsz,
                        p=original_p
                    )
                    dataset.transforms.transforms[i] = new_mosaic
                    mosaic_found = True
                    print(f"Replaced Mosaic with Mosaic16 (4x4, 16 images) at probability {original_p}")
                    break
                
                if mosaic_found:
                    break
        
        return dataset

# Load model
model = YOLO('yolo11l.pt')

# Configure training parameters for 4x4 mosaicing
# Use single GPU for custom trainer to avoid DDP issues with custom transforms
train_params = {
    'data': data_yaml,
    'epochs': 100,
    'project': 'runs/' + project,
    'device': 0,  # Use device 0 (which is actually GPU 1 due to CUDA_VISIBLE_DEVICES)
    'mosaic': 1.0,  # Enable mosaic augmentation
    'model': 'yolo11l.pt',  # Specify the model for the trainer
}

# Create trainer instance manually to ensure our custom class is used
trainer = CustomTrainer(overrides=train_params)
print("Training with custom 4x4 mosaicing (Mosaic16) enabled")
trainer.train()

# Optional validation
# best_model = YOLO(f'runs/{project}/train/weights/best.pt')
# best_model.val(
#     data=data_yaml,
#     project='runs/' + project,
#     device=0,  # Use device 0 (which is actually GPU 1 due to CUDA_VISIBLE_DEVICES)
#     split='test'
# )