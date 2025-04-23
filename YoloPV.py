from ultralytics import YOLO

# 1. 加载预训练模型
model = YOLO('yolov8n.pt')  # 可选：yolov8s/m/l/x

# 2. 使用模型
## 训练模型
results = model.train(
    data='YoloPV_dataset.yaml',
    imgsz=1280,  # 高分辨率利于小目标
    epochs=3,
    batch=16,
    lr0=0.01,
    augment=True,
    mosaic=0.75,  # 增强小目标上下文
    mixup=0.2,    # 提升泛化性
    copy_paste=0.5, # 增强污点多样性
    overlap_mask=True,
    device='cpu'
)
# 在验证集上评估模型性能
# metrics = model.val()
# 对图像进行预测

results = model.predict(
    'yolov8/datasets/images/test',
    conf=0.25,
    iou=0.45,
    imgsz=1280,
    augment=True,  # TTA增强
    max_det=1000  # 提高最大检测数
)

# 可视化输出
for result in results:
    result.show(save=True, line_width=2)

# 将模型导出为 ONNX 格式
success = model.export(format="onnx")