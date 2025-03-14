import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 여러 라이브러리에서 중복되기에 가장 먼저 설정

import paddle
paddle.utils.run_check()
import cv2
print(cv2.__version__)
from paddleocr import PaddleOCR, draw_ocr
from matplotlib import pyplot as plt
import numpy as np

# Setup model
ocr_model = PaddleOCR(lang='korean')
img_path = os.path.join('test_image.png')

# Run OCR on the image
result = ocr_model.ocr(img_path)

# 단일 이미지의 결과는 result[0]에 모든 검출 결과(각각 [박스, [텍스트, 신뢰도]])가 담깁니다.
detections = result[0]

# # 모든 검출 결과의 텍스트를 출력합니다.
# all_texts = [detection[1][0] for detection in detections]
# for i in range(0, len(all_texts), 10):
#     print(" ".join(all_texts[i:i+10]))

# 각 검출 결과에서 박스, 텍스트, 신뢰도를 추출하고, 박스의 중심 및 왼쪽 좌표 계산
detections_info = []
box_heights = []  # 각 박스의 높이를 저장하여 임계값 산출에 사용

for box, text, score in zip([d[0] for d in detections],  
                              [d[1][0] for d in detections],
                              [float(d[1][1]) for d in detections]):
    # 각 박스의 y 좌표 리스트와 x 좌표 리스트 계산
    y_coords = [pt[1] for pt in box]
    x_coords = [pt[0] for pt in box]
    y_center = sum(y_coords) / len(y_coords)
    x_left = min(x_coords)
    height = max(y_coords) - min(y_coords)
    box_heights.append(height)
    
    detections_info.append({
        'box': box,
        'text': text,
        'score': score,
        'y_center': y_center,
        'x_left': x_left,
        'height': height
    })

# 박스 높이의 중앙값(median)을 기준으로 임계값 산출
median_height = np.median(box_heights)
# 박스 높기의 50% 정도를 임계값으로 사용 (필요시 이 비율은 조정 가능)
line_threshold = median_height * 0.5

# y_center 기준으로 오름차순 정렬 (상단 -> 하단)
detections_info.sort(key=lambda x: x['y_center'])

# y_center 차이가 자동 계산된 임계값 이하이면 같은 줄(문장)로 그룹화
lines = []
current_line = []
for detection in detections_info:
    if not current_line:
        current_line.append(detection)
    else:
        current_line_avg = sum(d['y_center'] for d in current_line) / len(current_line)
        if abs(detection['y_center'] - current_line_avg) < line_threshold:
            current_line.append(detection)
        else:
            lines.append(current_line)
            current_line = [detection]
if current_line:
    lines.append(current_line)

# 각 줄 내에서 x_left 기준으로 정렬 (왼쪽 -> 오른쪽)
sorted_lines = []
for line in lines:
    sorted_line = sorted(line, key=lambda x: x['x_left'])
    sorted_lines.append(sorted_line)

# 정렬된 줄별로 문장을 구성 (단어 사이에 공백 추가)
sentences = []
for line in sorted_lines:
    sentence = " ".join([d['text'] for d in line])
    sentences.append(sentence)

# 정렬된 문장들을 출력
for sentence in sentences:
    print(sentence)

# 시각화를 위해 원래 박스, 텍스트, 신뢰도 리스트 복원
boxes = [d['box'] for d in detections_info]
texts = [d['text'] for d in detections_info]
scores = [d['score'] for d in detections_info]

# 한글 지원 폰트 설정 (Windows의 경우 'malgun.ttf' 사용)
font_path = r"C:/Windows/Fonts/malgun.ttf"
if not os.path.exists(font_path):
    raise FileNotFoundError(f"지정한 폰트 파일이 존재하지 않습니다: {font_path}")

# 이미지 불러오기 및 색상 변환 (BGR -> RGB)
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 시각화: 원본 이미지에 OCR 결과 표시
plt.figure(figsize=(65, 65))
annotated = draw_ocr(img, boxes, texts, scores, font_path=font_path)
plt.imshow(annotated)
plt.axis("off")
plt.show()
