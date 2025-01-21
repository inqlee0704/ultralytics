from ultralytics import YOLO

def main():

    model = YOLO("yolo11l.pt")  # load a pretrained model (recommended for training)
    model.train(
        data="/home/inkyu/proj_CZII-cryET/CZII_CryoET/czii_conf.yaml",
        project="CZII_CryoET_yolo2D",
        epochs=100,
        warmup_epochs=10,
        optimizer='AdamW',
        cos_lr=True,
        lr0=3e-4,
        lrf=0.03,
        imgsz=640,
        device="0",
        weight_decay=0.005,
        batch=64,
        scale=0,
        flipud=0.5,
        fliplr=0.5,
        degrees=45,
        shear=5,
        mixup=0.2,
        copy_paste=0.25,
        seed=8620, # (｡•◡•｡)
    )

if __name__ == "__main__":
    main() 