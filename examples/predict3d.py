# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics import YOLO3D

def main(args):
    """Run inference with a YOLO3D model."""
    # Load a model
    model = YOLO3D(args.weights)  # load a custom model

    # Run inference
    results = model.predict(
        source=args.source,
        conf=args.conf_thres,
        iou=args.iou_thres,
        max_det=args.max_det,
        device=args.device
    )

    # Process results
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        print(f'Detected {len(boxes)} objects')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolo11n-3d.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source path')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()
    main(args) 