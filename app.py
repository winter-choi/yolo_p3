import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np

st.title("AI 객체 감지 및 영역 강조 가이드")

# YOLO 모델 로드 (최초 실행 시 1회만 수행)
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
    
    # YOLO 감지를 위한 Blob 생성
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
            # 원하는 객체들만 필터링 (책, 키보드, 모니터, 물병 등)
            target_classes = ["book", "keyboard", "tvmonitor", "bottle", "mouse", "laptop"]
            
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

    # 노이즈 제거 (NMS)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            
            # 1. 객체 내부 영역 흑백 처리 (더러운 자국 강조)
            # 영역이 이미지 범위를 벗어나지 않도록 조정
            x, y = max(0, x), max(0, y)
            roi = img[y:y+h, x:x+w]
            if roi.size > 0:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                # 흑백을 다시 BGR로 변환하여 원본에 덮어씌움
                img[y:y+h, x:x+w] = cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2BGR)

            # 2. 빨간색 테두리 및 라벨 표시
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return img

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    processed_img = detect_and_process(img)
    return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

webrtc_streamer(key="object-detection", video_frame_callback=video_frame_callback)
