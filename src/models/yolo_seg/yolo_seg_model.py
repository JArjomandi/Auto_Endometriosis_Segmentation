from ultralytics import YOLO


def build_yolo_seg_model(weights_path: str):
    return YOLO(weights_path)