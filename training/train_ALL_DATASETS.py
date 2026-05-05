from ultralytics import YOLO

# Path to your YAML config file
data_yaml = 'datasets(others)/ALL_DATASETS/all_datasets.yaml'
project = 'all_datasets'

model = YOLO('yolo11l.pt') # Use a pre-trained model

# Train the model
model.train(
    data=data_yaml,
    epochs=100,         # Set the number of epochs as needed
    project='runs/' + project,
    device=['0', '1'],
)

# Optional validation
# model = YOLO('runs/' + project + '/train/weights/best.pt')
# model.val(
#     data=data_yaml,
#     project='runs/' + project,
#     device=['0', '1'],
#     split='test'
# )
