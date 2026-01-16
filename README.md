<div align="center">

# 🚗 Road & Lane Segmentation

**자율주행을 위한 도로 및 차선 세그멘테이션**

<img src="assets/readme_image.png" alt="Road Lane Segmentation">

<br>

# 🏅 Tech Stack 🏅

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![segmentation_models_pytorch](https://img.shields.io/badge/SMP-FF6F00?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Albumentations](https://img.shields.io/badge/Albumentations-E8710A?style=for-the-badge&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

</div>

<br>

## Team

| ![함성민](https://github.com/raretomato.png) | ![전승호](https://github.com/jeonseungho-glitch.png) | ![주호중](https://github.com/hojoooooong.png) | ![문국현](https://github.com/GH-Door.png) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
| [함성민](https://github.com/raretomato) | [전승호](https://github.com/jeonseungho-glitch) | [주호중](https://github.com/hojoooooong) | [문국현](https://github.com/GH-Door) |
| 팀장 | 팀원 | 팀원 | 팀원 |

<br>

## Project Overview

| 항목 | 내용 |
|:-----|:-----|
| **📅 Date** | 2026.01.12 ~ 2026.01.16 |
| **👥 Type** | 팀 프로젝트 |
| **🎯 Goal** | 자율주행을 위한 도로 및 차선 실시간 세그멘테이션 시스템 구축 |
| **🔧 Tech Stack** | PyTorch, segmentation_models_pytorch, Albumentations, OpenCV, Streamlit |
| **📊 Dataset** | [BDD100K](https://www.bdd100k.com/) / [TuSimple](https://github.com/TuSimple/tusimple-benchmark) |

<br>

## 📋 목차

- [프로젝트 소개](#-프로젝트-소개)
- [주요 기능](#-주요-기능)
- [모델 아키텍처](#-모델-아키텍처)
- [설치 방법](#-설치-방법)
- [사용 방법](#-사용-방법)
- [실험 결과](#-실험-결과)
- [프로젝트 구조](#-프로젝트-구조)

<br>

## 🎬 시연 영상

<div align="center">

### 📹 모델 추론 데모

<!-- 시연 영상 링크 추가 -->
<!-- https://github.com/user-attachments/assets/your-video-id -->

</div>

---

## 🎯 프로젝트 소개

자율주행 시스템의 핵심 기술인 도로 및 차선 인식을 위한 딥러닝 기반 Semantic Segmentation 시스템입니다.

### 핵심 특징
- ✅ **Semantic Segmentation**: 픽셀 단위 도로/차선 분류
- 🚀 **실시간 추론**: 경량화된 모델로 실시간 처리 가능
- 🎨 **다양한 환경 대응**: 주/야간, 날씨 변화에 강건한 인식
- 📊 **데이터 증강**: Albumentations를 활용한 강건한 학습
- 🔬 **재현 가능한 파이프라인**: 학습/평가/추론 모듈화

<br>

## 🎯 주요 기능

### 1. 세그멘테이션 클래스
- **Road (도로)**: 주행 가능 영역 검출
- **Lane Line (차선)**: 차선 영역 검출
- **Background (배경)**: 비주행 영역

### 2. 지원 기능
- 이미지 세그멘테이션
- 동영상 실시간 세그멘테이션
- 결과 시각화 및 오버레이
- Streamlit 웹 데모

<br>

## 🏗️ 모델 아키텍처

- **Base Models**: U-Net, U-Net++, DeepLabV3+
- **Backbone**: ResNet34, ResNet50, EfficientNet-B0
- **Framework**: segmentation_models_pytorch (SMP)
- **Loss Function**: DiceLoss, FocalLoss, Combined Loss

<br>

## 🛠️ 설치 방법

### 1. 저장소 클론

```bash
git clone https://github.com/your-org/Road_Lane_segmentation.git
cd Road_Lane_segmentation
```

### 2. 의존성 설치 (uv 사용)

```bash
# uv 설치 (없는 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 패키지 설치
uv sync
```

### 3. 데이터셋 다운로드

```bash
# 데이터셋 다운로드
python scripts/download_dataset.py
```

<br>

## 🚀 사용 방법

### 모델 학습

```bash
python src/training/train.py --config configs/train_config.yaml
```

### 모델 평가

```bash
python src/evaluation/evaluate.py --weights weights/best.pt
```

### 추론 실행

```bash
# 단일 이미지
python src/inference/predict.py --image path/to/image.jpg

# 동영상
python src/inference/predict.py --video path/to/video.mp4
```

### Streamlit 데모 실행

```bash
streamlit run streamlit_app/app.py
```

<br>

## 📈 실험 결과

| Model | Backbone | mIoU | Dice Score | Inference Time |
|:------|:---------|:-----|:-----------|:---------------|
| U-Net | ResNet34 | - | - | - ms |
| U-Net++ | ResNet50 | - | - | - ms |
| DeepLabV3+ | ResNet50 | - | - | - ms |

<br>

## 📁 프로젝트 구조

```
Road_Lane_segmentation/
├── assets/               # 이미지 리소스
│   └── readme_image.png
│
├── configs/              # 설정 파일
│   └── train_config.yaml
│
├── dataset/              # 데이터셋
│   ├── raw/             # 원본 데이터
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── masks/
│   │   ├── val/
│   │   └── test/
│   └── aug/             # 증강된 데이터
│
├── src/                 # 소스 코드
│   ├── data/           # 데이터 로딩/전처리
│   │   ├── dataset.py
│   │   └── transforms.py
│   ├── models/         # 모델 정의
│   │   └── model.py
│   ├── training/       # 학습 스크립트
│   │   └── train.py
│   ├── evaluation/     # 평가 스크립트
│   │   └── evaluate.py
│   └── inference/      # 추론 스크립트
│       └── predict.py
│
├── scripts/            # 유틸리티 스크립트
│   ├── download_dataset.py
│   └── augment_data.py
│
├── weights/            # 학습된 모델 가중치
│
├── outputs/            # 추론 결과
│
├── streamlit_app/      # Streamlit 데모
│   └── app.py
│
├── notebooks/          # 실험 노트북
│
├── pyproject.toml
├── main.py
└── README.md
```

<br>

## 📝 License

This project is licensed under the MIT License.

---

<div align="center">
Made with ❤️ by Likelion AI Team
</div>
