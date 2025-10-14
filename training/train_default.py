from ultralytics import YOLO

# Path to your YAML config file
data_yaml = 'dataset/Arthropoda.yaml'
project = 'arthro'
# data_yaml = 'src/datasets.yaml'
# project = 'arthro_and_flatbug'
# data_yaml = 'src/dataset_flatbug.yaml'
# project = 'flatbug'

model = YOLO('yolo11l.pt') # Use a pre-trained model

# Train the model
model.train(
    data=data_yaml,
    epochs=100,         # Set the number of epochs as needed
    project='runs/' + project,
    device= ['0'],
)

# Optional validation
# model = YOLO('runs/' + project + '/train/weights/best.pt')
# model.val(
#     data=data_yaml,
#     project='runs/' + project,
#     device= ['0', '1'],
#     split='test'
# )