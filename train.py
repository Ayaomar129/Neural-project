import argparse
from ultralytics import YOLO 


def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv8 for license plate detection')
    parser.add_argument('--data', default='dataset/dataset.yaml', help='dataset YAML path')
    parser.add_argument('--weights', default='yolov8n.pt', help='base YOLO weights')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='training image size')
    parser.add_argument('--device', default='cpu', help="device id or 'cpu'")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights) 
    print(f'Training on data={args.data} weights={args.weights} epochs={args.epochs} imgsz={args.imgsz} device={args.device}')

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
    )


if __name__ == '__main__':
    main()

