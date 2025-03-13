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

# result는 이미지별 검출 결과 리스트이기에 단일 이미지를 입력한 경우, result[0]가 검출 결과
detections = result[0]

# 모든 검출 결과의 텍스트를 출력합니다.
for detection in detections:
    print(detection[1][0])

# 모든 검출 결과에서 박스, 텍스트, 신뢰도를 추출합니다.
boxes = [detection[0] for detection in detections]
texts = [detection[1][0] for detection in detections]
scores = [float(detection[1][1]) for detection in detections]

font_path = r"C:/Windows/Fonts/malgun.ttf"
if not os.path.exists(font_path):
    raise FileNotFoundError(f"지정한 폰트 파일이 존재하지 않습니다: {font_path}")

# 시각화
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(65, 65))
annotated = draw_ocr(img, boxes, texts, scores, font_path=font_path)
plt.imshow(annotated)
plt.axis("off")
plt.show()
