# 🖼️ PyTorch Deep Learning: From Scratch to Transfer Learning

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-EE4C2C.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📌 프로젝트 요약 (Project Overview)
본 프로젝트는 PyTorch를 활용하여 **텐서의 수학적 기초부터 신경망 아키텍처(MLP, CNN, VGG)의 직접 구현, 그리고 실무적인 전이 학습(Transfer Learning)**까지 딥러닝 전 과정을 관통하는 파이프라인을 구축한 포트폴리오입니다. 단순 API 호출을 넘어, 하드웨어 메모리 최적화와 모델의 계층적 특징 추출 원리를 코드로 증명했습니다.

## 🎯 핵심 역량 (Core Competencies)
| 역량 키워드 | 세부 내용 |
| :--- | :--- |
| **🛠️ 아키텍처 설계** | `nn.Module` 기반 MLP, CNN, VGG 직접 설계 및 가중치 업데이트 메커니즘 이해 |
| **🚀 실무 최적화** | Pre-trained ResNet을 활용한 전이 학습(Transfer Learning) 및 파인튜닝 역량 |
| **🧠 파이프라인 엔지니어링** | `DataLoader`의 `num_workers` 최적화 및 RAM/VRAM 간 데이터 병목 해결 |
| **📊 데이터 중심 분석** | 데이터 증강(Augmentation)을 통한 과적합 방지 및 검증 전략 수립 |

## 💡 단계별 학습 로직 (Learning Pipeline)
### Stage 1. 기초 및 원리 (Fundamentals)
- **파일**: `01_pytorch_fundamentals.py`
- **핵심**: 텐서 조작(Shape, View, Permute)과 `Autograd`를 통한 자동 미분 원리 체득

### Stage 2. 실무형 모델링 (Transfer Learning)
- **파일**: `02_cat_dog_transfer_learning.py`
- **핵심**: ImageNet 가중치가 학습된 ResNet18을 활용한 고성능 분류기 구축

### Stage 3. 아키텍처 비교 분석 (Custom Architectures)
- **파일**: `03_cifar10_architecture_comparison.py`
- **핵심**: 동일 데이터(CIFAR-10) 기준 MLP vs CNN vs VGG 성능 지표 분석
- **인사이트**: 공간 정보 유지 여부에 따른 특징 추출 능력 차이 검증

## 📂 디렉토리 구조 및 데이터셋 안내
본 프로젝트는 데이터 보안 및 용량 관리를 위해 원본 데이터셋을 포함하지 않습니다.
* **Dataset**: CIFAR-10(자동 다운로드), Kaggle Cats vs Dogs(PetImages 폴더 필요)
* **Structure**:
  - `notebooks/`: 실험 및 연구 과정이 담긴 스크립트 모음
  - `main.py`: 최종 정제된 통합 실행 모델
  - `.gitignore`: 대용량 데이터셋 및 가중치 파일(`.pth`) 제외 설정

## 💡 회고록 및 향후 과제 (Retrospective & Future Work)

동일한 CIFAR-10 데이터를 학습시킬 때, 이미지를 1D로 펴버리는 MLP와 지역적 특징을 보존하는 CNN/VGG의 성능 차이를 보며 '구조가 곧 모델의 이해력'임을 확인했습니다. 모델이 깊어지면서 필연적으로 마주한 Out of Memory 현상은 오히려 하드웨어 파이프라인을 깊게 파고드는 계기가 되었습니다. 배치 사이즈를 조절하고 `num_workers`로 CPU-GPU 간의 병목을 해결하며 실무적인 엔지니어링 감각을 키웠습니다.
단순히 "코드가 돌아가는 것"에 만족하지 않고, "왜 이 모델이 더 우수한가?"를 수학적/구조적으로 분석하는 데 집중했습니다. 특히 VGG 구현 시 깊은 층에서의 기울기 소실(Vanishing Gradient) 문제를 체감하며 Batch Normalization의 중요성을 깨달았습니다. 향후에는 본 비전 역량을 바탕으로 NLP 및 GANs 분야로 기술 스택을 확장하여 멀티모달 솔루션을 구축할 계획입니다.
