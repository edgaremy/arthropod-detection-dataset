from ultralytics import YOLO

# Path to your YAML config file
data_yaml = 'dataset/Arthropoda.yaml'
project = 'arthro_nomosaic'

model = YOLO('yolo11l.pt') # Use a pre-trained model

# Train the model
model.train(
    data=data_yaml,
    epochs=100,         # Set the number of epochs as needed
    project='runs/' + project,
    device= ['0'],
    mosaic=0.0          # Disable mosaic augmentation
)

# Optional validation
# model = YOLO('runs/' + project + '/train/weights/best.pt')
# model.val(
#     data=data_yaml,
#     project='runs/' + project,
#     device= ['0', '1'],
#     split='test'
# )