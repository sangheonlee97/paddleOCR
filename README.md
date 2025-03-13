작업 PC : window11, gram 노트북, cpu 사용, 
가상 환경 : python 3.7
깃 클론          git clone https://github.com/PaddlePaddle/Paddle
라이브러리 설치  conda install paddlepaddle==2.5.0 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
클론한 Paddle의 요구사항 설치 cd Paddle/python, pip install -r requirements.txt, pip install opencv-python matplotlib paddleocr
각종 버전 에러 처리
pip install protobuf==3.20.2
pip install "urllib3<2.0"