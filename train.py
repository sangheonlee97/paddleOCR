import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 여러 라이브러리에서 중복되기에 가장 먼저 설정
import cv2
import numpy as np
import paddle
import paddle.nn as nn
import yaml
from paddleocr import PaddleOCR  


# --- 데이터셋 정의 ---
class OCRDataset(paddle.io.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        super(OCRDataset, self).__init__()
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __getitem__(self, index):
        # 이미지 불러오기 및 RGB 변환
        img = cv2.imread(self.image_paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        label = self.labels[index]
        return img, label
    
    def __len__(self):
        return len(self.image_paths)

# --- 전처리 예시 (필요에 따라 수정) ---
def simple_transform(img):
    # (128, 32) 크기로 리사이즈, 정규화 및 채널 순서 변경 (C, H, W)
    img = cv2.resize(img, (128, 32))
    img = img.astype('float32') / 255.0
    img = np.transpose(img, (2, 0, 1))
    return img

# --- 데이터 경로 및 레이블 ---
image_paths = ['1.png', '2.png', '3.png', '4.png', '5.png', '6.png', '7.png', '8.png', '9.png', '10.png', '11.png', '12.png', '13.png', '14.png']
labels = ['대량', '스스로', '추출', '통계적', '모델', '분류', '바둑', '프로그램인', '몇년', '않았지만', '부분을', '차지하고', '최근에는', '보안으로']

train_dataset = OCRDataset(image_paths, labels, transform=simple_transform)
train_loader = paddle.io.DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)

# --- 인식 모델 로드 ---
# 최신 PaddleOCR에서는 PaddleOCR 객체에서 인식 모델을 바로 노출하지 않으므로, 내부 모듈을 사용하여 인식 모델을 로드합니다.
from ppocr.architectures import build_model

# 1. 인식 모델 구성 파일 경로 (환경에 맞게 수정)
rec_config_path = "configs/rec/rec_korean_ppocrv4.yml"  # 예: PaddleOCR repo 내 설정 파일 경로
with open(rec_config_path, 'r', encoding='utf-8') as f:
    rec_config = yaml.safe_load(f)

# 2. 인식 모델 빌드
rec_model = build_model(rec_config['Architecture'])

# 3. 사전학습된 파라미터 로드
rec_model_dir = r"C:/Users/aaaa2/.paddleocr/whl/rec/korean/korean_PP-OCRv4_rec_infer"
params_path = os.path.join(rec_model_dir, "inference.pdparams")
if os.path.exists(params_path):
    param_state = paddle.load(params_path)
    rec_model.set_state_dict(param_state)
    print("사전학습된 인식 모델 파라미터 로드 완료")
else:
    print("사전학습된 파라미터 파일을 찾을 수 없습니다. 처음부터 학습됩니다.")

# --- CTC Loss 및 옵티마이저 설정 ---
ctc_loss = nn.CTCLoss(blank=0, reduction='mean')
optimizer = paddle.optimizer.Adam(parameters=rec_model.parameters(), learning_rate=1e-4)

# 문자 사전 및 텍스트 인덱스 변환 함수
# 아래 문자열은 임의로 만든 사전입니다
char_list = "인공지능은대량의데이터에서유미한정보를스로추출해내는통계적모델을일컫분류예측생성등다양하게응용가장먼저저떠오르것구글바둑프그램알파고아크년밖흐않았만우리현재활야소식접할수있중어도안면인식이장많부차지하고있다스마트폰의얼굴인식잠금으며최신결합여물되"
char_dict = {char: idx + 1 for idx, char in enumerate(char_list)}  # 0번 인덱스는 CTC blank 토큰으로 예약

def text_to_indices(text):
    # 각 문자에 대해 인덱스로 변환, 사전에 없는 문자는 0 처리
    return [char_dict.get(char, 0) for char in text]

# 학습 루프
num_epochs = 10

for epoch in range(num_epochs):
    rec_model.train()
    epoch_loss = 0.0
    for batch_id, (images, texts) in enumerate(train_loader):
        # images: [N, C, H, W] (이미 simple_transform에서 numpy array로 변환됨)
        images = paddle.to_tensor(images, dtype='float32')
        
        # Forward pass: 인식 모델의 출력 shape은 일반적으로 [batch_size, time_steps, num_classes]
        logits = rec_model(images)
        # CTC Loss에서는 logits의 shape을 [time_steps, batch_size, num_classes]로 요구함
        logits = paddle.transpose(logits, perm=[1, 0, 2])
        
        # 각 텍스트를 인덱스 시퀀스로 변환
        target_list = [text_to_indices(text) for text in texts]
        # 각 시퀀스 길이 계산
        target_lengths = paddle.to_tensor([len(t) for t in target_list], dtype='int32')
        # 모든 시퀀스를 1차원 배열로 결합
        targets = paddle.to_tensor(np.concatenate(target_list), dtype='int32')
        
        # 입력 길이: 모든 배치에서 동일하며 logits의 첫번째 차원 길이
        input_length = logits.shape[0]
        input_lengths = paddle.to_tensor([input_length] * logits.shape[1], dtype='int32')
        
        # CTC Loss 계산
        loss = ctc_loss(logits, targets, input_lengths, target_lengths)
        
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        
        epoch_loss += loss.numpy()[0]
        
    avg_loss = epoch_loss / (batch_id + 1)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# --- 모델 저장 ---
save_path = "fine_tuned_model.pdparams"
paddle.save(rec_model.state_dict(), save_path)
print(f"파인튜닝된 인식 모델이 {save_path}에 저장되었습니다.")
