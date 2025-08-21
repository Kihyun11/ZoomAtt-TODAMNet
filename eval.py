from ultralytics import YOLO

m = YOLO("/workspace/weights/best.pt")
metrics = m.val(data="/workspace/data.yaml",
                split="test", 
                imgsz=640, conf=0.001, iou=0.7, save_json=True)
print(metrics.results_dict) 