# -*- coding: utf-8 -*-
from ultralytics import YOLO

# YOLOv8 초경량 모델 (자동 다운로드)
model = YOLO("yolov8n.pt")

# 테스트 이미지로 추론
results = model("https://ultralytics.com/images/bus.jpg")

# 결과 시각화
results[0].show()