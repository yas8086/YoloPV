from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')  # 可选：yolov8s/m/l/x

# 开始训练
results = model.train(
    data='data.yaml',
    epochs=100,
    batch=16,
    imgsz=640,
    device='0',  # GPU ID
    name='custom_train'
)