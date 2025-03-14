import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import cv2
import numpy as np
import paddle
import paddle.nn as nn
import yaml
import sys

# PaddleOCR 저장소의 상위 폴더(PaddleOCR)를 PYTHONPATH에 추가합니다.
sys.path.append(r'C:/Users/aaaa2/OneDrive/바탕 화면/taehwa/paddleOCR/PaddleOCR')

# PaddleOCR 내부 모듈 사용
try:
    from ppocr.modeling.architectures import build_model
except ImportError:
    raise ImportError("ppocr 모듈을 찾을 수 없습니다. 'pip install paddleocr' 또는 PaddleOCR 저장소를 클론한 후 PYTHONPATH에 추가하세요.")

# --- 데이터셋 정의 ---
class OCRDataset(paddle.io.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        super(OCRDataset, self).__init__()
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __getitem__(self, index):
        img = cv2.imread(self.image_paths[index])
        if img is None:
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {self.image_paths[index]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        label = self.labels[index]
        return img, label
    
    def __len__(self):
        return len(self.image_paths)

# --- 전처리 함수 ---
def simple_transform(img):
    # PP-OCRv3 인식 모델은 일반적으로 3x48x320 크기를 사용합니다.
    img = cv2.resize(img, (320, 48))
    img = img.astype('float32') / 255.0
    img = np.transpose(img, (2, 0, 1))  # (C, H, W)
    return img

# --- 데이터 경로 및 라벨 ---
image_paths = ['1.png', '2.png', '3.png', '4.png', '5.png', '6.png', '7.png',
               '8.png', '9.png', '10.png', '11.png', '12.png', '13.png', '14.png']
labels = ['대량', '스스로', '추출', '통계적', '모델', '분류', '바둑',
          '프로그램인', '몇년', '않았지만', '부분을', '차지하고', '최근에는', '보안으로']

train_dataset = OCRDataset(image_paths, labels, transform=simple_transform)
train_loader = paddle.io.DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)

# --- 인식 모델 로드 ---
rec_config_path = r"C:/Users/aaaa2/OneDrive/바탕 화면/taehwa/paddleOCR/PaddleOCR/configs/rec/PP-OCRv3/multi_language/korean_PP-OCRv3_rec.yml"
with open(rec_config_path, 'r', encoding='utf-8') as f:
    rec_config = yaml.safe_load(f)

# --- 문자 사전 설정 ---
char_list = "인공지능은대량의데이터에서유미한정보를스로추출해내는통계적모델을일컫분류예측생성등다양하게응용가장먼저저떠오르것구글바둑프그램알파고아크년밖흐않았만우리현재활야소식접할수있중어도안면인식이장많부차지하고있다스마트폰의얼굴인식잠금으며최신결합여물되"
char_dict = {char: idx + 1 for idx, char in enumerate(char_list)}  # 0번: CTC blank 토큰
num_classes = len(char_list) + 1

def text_to_indices(text):
    return [char_dict.get(char, 0) for char in text]

# --- config 수정: Head 설정에 필요한 out_channels_list 값을 추가 ---
head_config = rec_config["Architecture"].get("Head", {})
if "out_channels_list" not in head_config:
    head_config["out_channels_list"] = {
        "CTCLabelDecode": num_classes,
        "SARLabelDecode": num_classes  # 필요에 따라 조정 가능
    }
rec_config["Architecture"]["Head"] = head_config

# 인식 모델 빌드 (사전훈련 파라미터 생략)
rec_model = build_model(rec_config["Architecture"])
print("PaddleOCR 인식 모델이 초기화되었습니다. (사전훈련 파라미터 생략)")

# --- CTC Loss 및 옵티마이저 설정 ---
ctc_loss = nn.CTCLoss(blank=0, reduction='mean')
optimizer = paddle.optimizer.Adam(parameters=rec_model.parameters(), learning_rate=1e-4)

# --- 학습 루프 ---
num_epochs = 10
max_text_length = rec_config["Global"].get("max_text_length", 25)
for epoch in range(num_epochs):
    rec_model.train()
    epoch_loss = 0.0
    for batch_id, (images, texts) in enumerate(train_loader):
        images = paddle.to_tensor(images, dtype='float32')
        # CTC branch용 타깃: 각 텍스트를 인덱스 리스트로 변환
        target_list = [text_to_indices(text) for text in texts]
        batch_size = images.shape[0]
        # SAR branch용 label: target_list를 최대 길이(max_text_length)로 패딩하여 텐서로 변환
        padded_targets = []
        for t in target_list:
            if len(t) < max_text_length:
                t = t + [0] * (max_text_length - len(t))
            else:
                t = t[:max_text_length]
            padded_targets.append(t)
        dummy_sar_label = paddle.to_tensor(np.array(padded_targets, dtype=np.int32))
        # SAR branch용 meta 정보: 각 이미지의 유효 비율을 1.0 (dummy)로 가정하고, 텐서로 생성
        dummy_valid_ratios = paddle.to_tensor(np.ones(batch_size, dtype='float32'))
        # 구성: targets[0] -> CTC branch용 타깃 (target_list),
        #         targets[1] -> SAR branch의 label (dummy_sar_label),
        #         targets[2] -> SAR branch의 meta (dummy_valid_ratios)
        dummy_targets = [target_list, dummy_sar_label, dummy_valid_ratios]
        
        # forward 호출 시, 타깃을 위치 인자로 전달합니다.
        outputs = rec_model(images, dummy_targets)
        print("Output keys:", outputs.keys())
        # CTC branch 출력은 딕셔너리의 "ctc" 키로 접근합니다.
        ctc_logits = outputs["ctc"]
        ctc_logits = paddle.transpose(ctc_logits, perm=[1, 0, 2])
        
        # CTC Loss 계산용 타깃 텐서들 (모두 int32로 확실히 cast)
        target_lengths = paddle.to_tensor(np.array([len(t) for t in target_list], dtype=np.int32))
        targets_concat = paddle.to_tensor(np.array(np.concatenate(target_list), dtype=np.int32))
        input_length = ctc_logits.shape[0]
        input_lengths = paddle.to_tensor(np.array([input_length] * ctc_logits.shape[1], dtype=np.int32))
        
        # 명시적으로 int32로 캐스팅 (혹시 모를 dtype 불일치 해결)
        target_lengths = paddle.cast(target_lengths, 'int32')
        targets_concat = paddle.cast(targets_concat, 'int32')
        input_lengths = paddle.cast(input_lengths, 'int32')
        
        loss = ctc_loss(ctc_logits, targets_concat, input_lengths, target_lengths)
        
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        
        epoch_loss += loss.numpy()[0]
    avg_loss = epoch_loss / (batch_id + 1)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

save_path = "paddleocr_finetuned_model.pdparams"
paddle.save(rec_model.state_dict(), save_path)
print(f"파인튜닝된 인식 모델이 {save_path}에 저장되었습니다.")
