## 1. 📌 개요

### 🐔 닭 사망 탐지 프로젝트 - YOLOv8 기반 영상 분석

이 프로젝트는 **딥러닝 객체 탐지 모델(YOLOv8)**을 이용해  
**닭 사육장 영상에서 사망한 닭을 자동으로 판별**하는 시스템입니다.  
카메라로 촬영된 닭의 움직임을 분석하여 일정 시간 동안 움직임이 없을 경우  
사망 가능성을 추정하고, 이를 시간 기반으로 누적하여 판단합니다.

모델은 A100 GPU 환경에서 실행되며,  
프레임 기반 추적 + 좌표 히스토리를 결합한 방식으로 정확한 판단을 수행합니다.

## 2. 🎯 프로젝트 목표

- 닭의 움직임을 분석하여 죽은 닭을 자동으로 탐지
- YOLOv8을 기반으로 객체 탐지 및 추적 수행
- 사망 판단에 시간 기반 확률 누적 로직 적용
- 사망 판단 결과를 `.csv`와 `.mp4`로 저장하여 시각화

## 3. 📁 입력 데이터

- **입력 파일**: 천장 카메라로 촬영된 닭 사육장 영상 (MP4)
- **예시 경로**: `/home/nas/data/Y/0_8_IPC1_20221105100432.mp4`
- **모델 파일**: 이미 학습된 YOLOv8 모델 (`best.pt`)

## 4. ⚙️ 동작 방식

1. 영상을 4개 영역(좌상, 우상, 좌하, 우하)으로 분할
2. 각 영역별로 YOLOv8을 이용해 **닭의 위치와 ID를 추적**
3. ID 별로 **30프레임 버퍼**에 좌표를 저장해 이동 여부 계산
4. 일정 시간 동안 움직임이 없으면 **사망 의심으로 판단**
5. 시간 기준:
   - **1분(=1800프레임)** 움직임 없음 → 사망 추정 시작 (확률 5%)
   - 이후 **2분(=3600프레임)** 추가 정지 시 → +5% 확률 누적
6. **사망 확률이 50% 이상인 경우**, 죽은 닭으로 간주하여 별도 저장

## 5. 🧠 혼동 방지 로직

- 동일 좌표 근처에서 **여러 ID가 자주 출현한 경우** → 혼동 위험 판단
- 혼동 위험 구간에서는 ID를 **이어붙이지 않음**
- 이를 통해 닭들이 겹쳤을 때 **다른 닭을 잘못 판단하는 오류 방지**
- 이는 **혼동행렬(Confusion Matrix)** 개념에서 착안됨

## 6. 📦 출력 결과

📁 폴더 및 파일 구조
<pre> chicken_proj/ └── run_20250409_153000/ ├── topleft/ │ ├── annotated_topleft.mp4 │ ├── positions_topleft.csv │ └── dead_candidates_topleft.csv ├── topright/ │ ├── annotated_topright.mp4 │ ├── positions_topright.csv │ └── dead_candidates_topright.csv ├── bottomleft/ │ ├── annotated_bottomleft.mp4 │ ├── positions_bottomleft.csv │ └── dead_candidates_bottomleft.csv └── bottomright/ ├── annotated_bottomright.mp4 ├── positions_bottomright.csv └── dead_candidates_bottomright.csv </pre>

### 🗂️ 각 파일 설명
annotated_*.mp4
→ 닭을 YOLO로 탐지하여 박싱한 결과를 영상으로 저장한 파일

positions_*.csv
→ 각 프레임에서 닭의 ID 및 위치 좌표를 저장한 추적 결과 CSV

dead_candidates_*.csv
→ 일정 시간 움직이지 않아 죽은 닭으로 판단된 ID의 확률 및 좌표 저장

## 7. 🚀 실행 방법

1. Python 3.8 이상 및 `ultralytics` 설치
2. `best.pt` 모델 파일 준비
3. 아래와 같이 Python 코드 실행

```bash
python detect_dead_chickens.py
```
✅ GPU 자원을 효율적으로 활용하고 싶다면,
CUDA_VISIBLE_DEVICES 옵션을 통해 각 영역을 병렬 실행할 수 있습니다.

## 8. 🛠️ 사용 기술

- Python 3.8
- YOLOv8 (ultralytics)
- OpenCV
- Pandas
- PyTorch (CUDA 가속)

## 9. 📈 확장 아이디어

- 사망 확률 모델을 **딥러닝 기반 LSTM 방식**으로 전환
- YOLOv8 모델을 **ONNX / TensorRT 최적화**하여 속도 향상
- 다중 카메라 기반 3D 공간 추적 (멀티 뷰)
- 사망 판단에 **온도/습도 등 외부 센서 데이터** 결합

## 10. 🙌 개발자

- 전주대학교 인공지능학과  
- 연구자: YS 
- 실험 환경: A100 GPU 서버 (멀티 GPU 분산 실행 가능)
