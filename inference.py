import argparse
import os

import cv2
import easyocr
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description='License plate detection + OCR inference')
    parser.add_argument('--weights', default='runs/train/exp/weights/best.pt', help='trained weights path')
    parser.add_argument('--source', default='0', help='video source: camera index or path to video/image file')
    parser.add_argument('--device', default='0', help="device id or 'cpu'")
    parser.add_argument('--conf', type=float, default=0.25, help='detection confidence threshold')
    return parser.parse_args()


def draw_annotations(frame, box, text, score):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f'{text} ({score:.2f})' if text else f'plate ({score:.2f})'
    cv2.putText(frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def run_inference(args):
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source

    print(f'Loading model from {args.weights}')
    model = YOLO(args.weights)
    reader = easyocr.Reader(['en'], gpu=(args.device != 'cpu'))

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f'Unable to open source: {args.source}')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=args.conf, device=args.device, stream=True)
        for result in results:
            if not hasattr(result, 'boxes') or len(result.boxes) == 0:
                continue
            boxes = result.boxes.data.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2, score, cls = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                texts = reader.readtext(crop, detail=0, paragraph=False)
                plate_text = ' '.join(texts).strip() if texts else ''
                draw_annotations(frame, (x1, y1, x2, y2), plate_text, score)

        cv2.imshow('LPR Inference', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f'Weights not found: {args.weights}')
    run_inference(args)
