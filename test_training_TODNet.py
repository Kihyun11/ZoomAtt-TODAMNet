from ultralytics import YOLO

# If you want to train a model using a customized backbone, you should modify the model_path
# Copy and paste the directory of the YAML file of the customized backbone
backbone_path = 'C:/Users/white/OneDrive/바탕 화면/conv_todam_net.yaml'


model = YOLO(backbone_path) 
#model = YOLO(backbone_path).load('yolov8n.pt')  # build from YAML and transfer weights

# To start the train, you should copy and paste the directory of the YAML file for the dataset into data.
results = model.train(data='C:/Users/white/OneDrive/바탕 화면/MoonNet/data/augmentation_v1/data.yaml', batch = 8, epochs=12, imgsz=640)