from pathlib import Path
import argparse
import cv2
from ultralytics import YOLO

CLASSES = ['fire_prop', 'black_base', 'red_cloth_strip']


def draw_result(frame, boxes):
    if boxes is None:
        return frame
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        cls_id = int(box.cls[0].cpu().item())
        conf = float(box.conf[0].cpu().item())
        x1, y1, x2, y2 = xyxy
        color = (0, 255, 0) if cls_id == 0 else (255, 0, 0) if cls_id == 1 else (0, 0, 255)
        label = f'{CLASSES[cls_id] if cls_id < len(CLASSES) else cls_id}:{conf:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='best.pt')
    parser.add_argument('--source', default='0', help='0 for camera, or path to video/image')
    parser.add_argument('--conf', type=float, default=0.45)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    model = YOLO(args.model)
    source = 0 if args.source == '0' else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open source: {source}')

    writer = None
    out_path = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        frame = draw_result(frame, result.boxes)
        if args.save:
            if writer is None:
                out_path = Path('output.avi')
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                writer = cv2.VideoWriter(str(out_path), fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            writer.write(frame)
        cv2.imshow('fire_prop_demo', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    if out_path:
        print(f'Saved to {out_path.resolve()}')


if __name__ == '__main__':
    main()
