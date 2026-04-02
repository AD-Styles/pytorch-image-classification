# 🖼️ PyTorch Image Classification: From Basic Tensors to VGG Architecture

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-EE4C2C.svg)
![Torchvision](https://img.shields.io/badge/Torchvision-0.15-EE4C2C.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📌 프로젝트 요약 (Project Overview)
안녕하세요, 다가오는 6~7월 실무 합류를 목표로 딥러닝 기본기를 탄탄히 다지고 있는 개발자 AD입니다.
본 프로젝트는 단순한 API 호출을 넘어, PyTorch의 **텐서(Tensor) 기초부터 다층 퍼셉트론(MLP), 합성곱 신경망(CNN), VGG 네트워크까지 직접 밑바닥부터 구현(From Scratch)** 해보며 딥러닝 아키텍처의 발전 과정을 코드로 증명한 프로젝트입니다. 특히, 대용량 이미지 데이터를 다룰 때 발생하는 하드웨어 병목 현상을 분석하고 파이프라인을 최적화하는 데 집중했습니다.

## 🎯 핵심 목표 및 역량 (Core Competencies)
| 역량 키워드 | 세부 내용 (핵심 역량) |
| :--- | :--- |
| **🛠️ 아키텍처 설계** | `nn.Module`을 상속받아 MLP, CNN, VGG 모델을 직접 설계하고 커스텀 학습 루프 구현 |
| **🚀 파이프라인 최적화** | `Dataset`과 `DataLoader`를 활용한 데이터 전처리 및 `num_workers` 튜닝을 통한 로드 최적화 |
| **🧠 하드웨어 메모리 이해** | Disk → RAM → VRAM(GPU)으로 이어지는 데이터 흐름을 제어하여 OOM(Out of Memory) 방지 |

## 💡 핵심 개념: 신경망 아키텍처 및 메모리 파이프라인 비교
딥러닝 모델이 공간적 맥락을 어떻게 이해하는지, 그리고 데이터가 어떻게 하드웨어를 관통하는지 체계적으로 분석했습니다.

### 1. 신경망 아키텍처 성능 비교
> **검증 데이터셋**: CIFAR-10 (32x32 RGB, 10개 클래스 다중 분류)

| 아키텍처 | 이미지 처리 방식 및 한계 | 핵심 차별점 |
| :--- | :--- | :--- |
| **MLP (다층 퍼셉트론)** | 2D 이미지를 1D로 강제로 펼침(Flatten).<br>픽셀 간의 **공간적(Spatial) 정보가 완전히 손실됨.** | 파라미터가 비효율적으로 많고, 이미지의 위치 변화나 왜곡에 매우 취약함. |
| **CNN (합성곱 신경망)** | 2D 형태를 유지한 채 필터(Filter)가 이동하며 연산.<br>이미지의 **지역적 패턴(Feature Map)을 학습함.** | 공간 정보를 유지하면서도 파라미터 수를 획기적으로 줄여 컴퓨터 비전의 표준이 됨. |
| **VGG Network** | 3x3의 작은 합성곱 필터를 여러 겹(Deep) 쌓아올림.<br>넓은 **수용 영역(Receptive Field)을 확보함.** | CNN보다 훨씬 깊은 계층적 특징(Hierarchical Feature)을 추출하여 표현력이 극대화됨. |

### 2. 하드웨어 메모리 파이프라인 최적화
| 단계 | 하드웨어 | 파이프라인 역할 | 최적화 포인트 |
| :--- | :--- | :--- | :--- |
| **Storage** | **Disk** | 전체 데이터셋 보관 (`Dataset`) | I/O 지연 최소화를 위한 직렬화 포맷 활용 |
| **Preparation** | **RAM / CPU** | 배치(Batch) 준비 및 데이터 증강 | `DataLoader`의 `num_workers` 설정을 통한 병렬 로드 |
| **Workbench** | **VRAM / GPU** | 실질적인 텐서 연산 및 가중치 업데이트 | 적절한 `Batch Size` 설정을 통한 OOM(Out of Memory) 방지 |

## 📝 회고 및 향후 과제 (Retrospective & Future Work)

**1. 공간 정보의 중요성 수치화 및 OOM 트러블슈팅**
동일한 CIFAR-10 데이터를 학습시킬 때, 이미지를 1D로 펴버리는 MLP와 지역적 특징을 보존하는 CNN/VGG의 성능 차이를 보며 '구조가 곧 모델의 이해력'임을 확인했습니다. 모델이 깊어지면서 필연적으로 마주한 Out of Memory 현상은 오히려 하드웨어 파이프라인을 깊게 파고드는 계기가 되었습니다. 배치 사이즈를 조절하고 `num_workers`로 CPU-GPU 간의 병목을 해결하며 실무적인 엔지니어링 감각을 키웠습니다.

**2. 꾸준한 트렌드 팔로우와 전문성 입증**
매일 아침 9시, 하루도 빠짐없이 글로벌 AI 뉴스 요약을 스크랩하며 급변하는 기술 동향을 파악하는 루틴을 유지하고 있습니다. 현재는 인프라와 LLM에 대한 이해도를 공식적으로 입증하기 위해 **NVIDIA NCA-GENL (Generative AI LLMs)** 자격증 취득을 최우선으로 준비 중입니다.

**3. Next Step: 비전(CV)을 넘어 자연어(NLP)와 생성 모델(GANs)로**
본 프로젝트로 다진 탄탄한 컴퓨터 비전 기본기를 바탕으로, 향후에는 NLP 모델링과 GANs 아키텍처로 제 역량을 확장할 것입니다. 궁극적으로는 이러한 멀티모달 기술들을 융합하여, **과거의 역사적 사건을 텍스트 맥락에 맞게 생동감 있는 이미지로 재현해 내는 'AI 기반 맞춤형 역사 교육 콘텐츠 생성 솔루션'**을 독자적으로 구축하는 것이 저의 목표입니다.
