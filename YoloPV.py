import torch
from ultralytics.yolo.engine.model import YOLO
from ultralytics.yolo.data import build_dataloader
from ultralytics.yolo.utils import DEFAULT_CONFIG, LOGGER


def custom_train():
    # ----------------- 配置参数 -----------------
    cfg = DEFAULT_CONFIG.copy()
    cfg.update({
        'data': 'datasets/my_dataset.yaml',
        'model': 'yolov8n.yaml',  # 使用源码中的模型配置
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'device': '0' if torch.cuda.is_available() else 'cpu',
        'workers': 4,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'augment': True,
    })

    # ----------------- 初始化模型 -----------------
    model = YOLO(cfg['model']).model  # 直接访问底层模型

    # ----------------- 数据加载 -----------------
    train_loader = build_dataloader(
        cfg,
        data=cfg['data'],
        batch_size=cfg['batch'],
        mode='train'
    )

    # ----------------- 优化器配置 -----------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['lr0'],
        weight_decay=0.0005
    )

    # ----------------- 训练循环 -----------------
    for epoch in range(cfg['epochs']):
        model.train()

        for batch_i, (imgs, targets, paths, _) in enumerate(train_loader):
            imgs = imgs.to(cfg['device'])
            targets = targets.to(cfg['device'])

            # 前向传播
            preds = model(imgs)

            # 计算损失
            loss, loss_items = model.loss(preds, targets)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 日志打印
            if batch_i % 50 == 0:
                LOGGER.info(
                    f"Epoch: {epoch}/{cfg['epochs']} | "
                    f"Loss: {loss_items.mean():.4f}"
                )


if __name__ == '__main__':
    custom_train()