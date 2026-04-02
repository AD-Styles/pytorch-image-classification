# 🖼️ PyTorch Vision Engineering: From Scratch to Transfer Learning

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-EE4C2C.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📌 프로젝트 요약 (Project Overview)
본 프로젝트는 PyTorch를 활용하여 **텐서 조작부터 신경망 아키텍처(MLP, CNN, VGG)의 직접 구현, 그리고 최신 전이 학습(Transfer Learning) 기법**까지 이미지 분류 파이프라인의 전 과정을 담고 있습니다. 프레임워크 뒤에 숨겨진 수학적 원리와 하드웨어 최적화 메커니즘을 코드로 증명하는 데 집중했습니다.

## 🎯 핵심 목표 (Motivation)
| 역량 키워드 | 세부 내용 (핵심 역량) |
| :--- | :--- |
| **🛠️ 아키텍처 설계** | `nn.Module`을 활용한 MLP, CNN, VGG 모델 직접 설계 및 가중치 업데이트 원리 체득 |
| **🚀 실무 최적화** | Pre-trained ResNet을 활용한 전이 학습 및 데이터 증강(Augmentation) 파이프라인 구축 |
| **🧠 하드웨어 이해** | RAM/VRAM 간 데이터 병목 현상 분석 및 `num_workers`, `Batch Size` 최적화 해결 |
| **📊 성능 비교 분석** | 아키텍처별 특징 추출 능력(Feature Extraction)을 정량적으로 비교 분석하는 역량 |

## 1. 프로젝트 구조 및 데이터셋 안내
본 프로젝트는 데이터 보안 및 용량 관리를 위해 원본 데이터셋을 포함하지 않습니다.
```text
pytorch-vision-engineering
├─ notebooks/  # 실험 및 연구 기록 보관 폴더
│  ├─ 01_pytorch_fundamentals.py  # 텐서 기초 및 Autograd
│  ├─ 02_cifar10_architecture_comparison.py  # MLP/CNN/VGG 비교
│  └─ 03_cat_dog_transfer_learning.py  # ResNet 전이 학습
├─ main.py  # argparse가 적용된 최종 통합 실행 스크립트
├─ README.md  # 프로젝트 명세서 및 회고록
├─ requirements.txt  # 필수 패키지 목록
└─ .gitignore  # 대용량 데이터 및 가중치 파일 제외
* **Dataset**: CIFAR-10(자동 다운로드), Kaggle Cats vs Dogs(PetImages 폴더 필요)

## 2. 단계별 학습 로직 (Learning Pipeline)

### Stage 1. 기초 및 원리 (Fundamentals)
- **파일**: `01_pytorch_fundamentals.py`
- **내용**: 텐서 조작(Shape, View, Permute)과 `Autograd` 자동 미분 로직 구현

### Stage 2. 아키텍처 비교 분석 (From Scratch)
- **파일**: `02_cifar10_architecture_comparison.py`
- **내용**: MLP, CNN, VGG를 밑바닥부터 구현하여 공간 정보 유지 여부에 따른 성능 차이 검증

### Stage 3. 고성능 실무 모델링 (Transfer Learning)
- **파일**: `03_cat_dog_transfer_learning.py`
- **내용**: Pre-trained ResNet18 기반의 전이 학습을 통해 실무 레벨의 정확도 달성

## 💡 회고록 및 향후 과제 (Retrospective & Future Work)

동일한 CIFAR-10 데이터를 학습시킬 때, 이미지를 1D로 펴버리는 MLP와 지역적 특징을 보존하는 CNN/VGG의 성능 차이를 보며 '구조가 곧 모델의 이해력'임을 확인했습니다. 모델이 깊어지면서 필연적으로 마주한 OOOM(Out of Memory) 문제는 단순히 하드웨어의 한계가 아니라, 배치 사이즈와 `num_workers` 등 데이터 파이프라인의 엔지니어링적 설계가 얼마나 중요한지를 깨닫게 해준 소중한 경험이었으며, 오히려 하드웨어 파이프라인을 깊게 파고드는 계기가 되었습니다. 
단순히 "코드가 돌아가는 것"에 만족하지 않고, "왜 이 모델이 더 우수한가?"를 수학적/구조적으로 분석하는 데 집중했습니다. 특히 VGG 구현 시 깊은 층에서의 기울기 소실(Vanishing Gradient) 문제를 체감하며 Batch Normalization의 중요성을 깨달았습니다. 향후에는 본 비전 역량을 바탕으로 NLP 및 GANs 분야로 기술 스택을 확장하여 멀티모달 솔루션을 구축할 계획입니다.
