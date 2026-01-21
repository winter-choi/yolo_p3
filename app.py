import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import os
import urllib.request

st.title("책상 잡동사니 AI 클린 가이드: 실시간 이미지 처리 객체별 강조")

# --- [추가] YOLOv3 가중치 파일 자동 다운로드 로직 ---
def download_yolo_weights():
    weights_url = "https://pjreddie.com/media/files/yolov3.weights"
    weights_file = "yolov3.weights"
    if not os.path.exists(weights_file):
        with st.spinner('AI 모델 파일(weights)을 다운로드 중입니다... 잠시만 기다려 주세요.'):
            urllib.request.urlretrieve(weights_url, weights_file)
        st.success('다운로드 완료!')

download_yolo_weights()

# YOLO 모델 로드
@st.cache_resource
def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    return net, classes, output_layers

net, classes, output_layers = load_yolo()

def detect_and_process(img):
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # 강조하고 싶은 대상 리스트
            target_classes = ["book", "keyboard", "tvmonitor", "bottle", "mouse", "laptop", "cup"]
            
            if confidence > 0.5 and classes[class_id] in target_classes:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            
            # 이미지 범위를 벗어나지 않도록 좌표 제한
            x, y = max(0, x), max(0, y)
            x_end, y_end = min(width, x + w), min(height, y + h)

            # 1. 객체 내부 영역 흑백 처리 (더러운 자국 시각화 최적화)
            roi = img[y:y_end, x:x_end]
            if roi.size > 0:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                img[y:y_end, x:x_end] = cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2BGR)

            # 2. 빨간색 테두리 및 라벨 표시 (사용자 요청 디자인 적용)
            cv2.rectangle(img, (x, y), (x_end, y_end), (0, 0, 255), 3)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return img

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    processed_img = detect_and_process(img)
    return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

webrtc_streamer(
    key="ai-clean-guide", 
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
