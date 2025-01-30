# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics import YOLO3D

def main(args):
    """Train a YOLO3D model on a custom dataset."""
    # Load a model
    model = YOLO3D('yolo11n-3d.yaml')  # build a new model from YAML
    # model = YOLO3D('yolo11n-3d.pt')  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data='path/to/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device='0'  # GPU device
    )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data.yaml', help='data.yaml path')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()
    main(args) 