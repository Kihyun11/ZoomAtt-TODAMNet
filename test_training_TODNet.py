from ultralytics import YOLO

# If you want to train a model using a customized backbone, you should modify the model_path
# Copy and paste the directory of the YAML file of the customized backbone
backbone_path = '/backbone/path.yaml'


model = YOLO(backbone_path) 
#model = YOLO(backbone_path).load('yolov8n.pt')  # build from YAML and transfer weights

# To start the train, you should copy and paste the directory of the YAML file for the dataset into data.
results = model.train(data='data/path.yaml', batch = 4, epochs=150, imgsz=640)