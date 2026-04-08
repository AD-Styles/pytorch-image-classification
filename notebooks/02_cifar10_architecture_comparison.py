# # 09. PyTorch로 MLP, CNN, VGG 구현하기
# 
# ## 📋 학습 목표
# - PyTorch로 MLP, CNN, VGG 직접 구현
# - `nn.Module` 패턴 완전 이해
# - DataLoader와 transforms 활용
# - 여러 모델 성능 비교
# 
# ## 💡 데이터셋
# CIFAR-10 (32×32×3 컬러 이미지, 10개 클래스)
# - 학습: 45,000장
# - 검증: 5,000장
# - 테스트: 10,000장

# ## 1. 환경 설정
# 
# 이 섹션에서는 실습에 필요한 라이브러리를 import하고, **연산 디바이스(CPU / CUDA / MPS)** 를 자동 선택합니다.
# 
# - `device`를 먼저 정해두면 이후에 `model.to(device)`, `inputs.to(device)` 패턴으로 일관되게 작성할 수 있습니다.
# - 시드(seed)를 고정하면(가능한 범위에서) 학습 결과가 덜 흔들려 디버깅/비교가 쉬워집니다.

# In[1]:


# ------------------------------------------------------------
# 1) 필수 라이브러리 import
# ------------------------------------------------------------
# - torch / nn: 모델 정의(nn.Module), 레이어(nn.*), 텐서 연산(torch.*)
# - F (functional): 파라미터 없는 연산(ReLU, Pooling 등)을 함수 형태로 사용
# - optim: 옵티마이저(Adam 등)
# - datasets/transforms: 데이터셋 로드 + 전처리 파이프라인
# - summary(torchinfo): 모델 구조/파라미터 수를 보기 좋게 요약
#
#############################################
# Colab 팁: torchinfo가 없으면 설치가 필요
##############################################
try:
    from torchinfo import summary
except ImportError:
    # Colab에서 종종 torchinfo가 기본 설치가 아닐 수 있습니다.
    # (노트북 셀에서만 동작하는 매직/쉘 구문)
    get_ipython().system('pip -q install torchinfo')
    from torchinfo import summary

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# ------------------------------------------------------------
# 2) 디바이스 선택 (MPS/CUDA/CPU)
# ------------------------------------------------------------
# - Apple Silicon이면 MPS를 우선 사용
# - NVIDIA GPU가 있으면 CUDA 사용
# - 그 외에는 CPU로 실행
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ MPS (Apple Silicon) 사용")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ CUDA GPU 사용: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("⚠️ CPU 모드로 실행")

# ------------------------------------------------------------
# 3) 시드 설정 (재현성)
# ------------------------------------------------------------
# 같은 코드라도 '난수'가 들어가는 부분(초기화/셔플/드롭아웃 등) 때문에 결과가 흔들릴 수 있습니다.
# 시드를 고정하면(가능한 범위에서) 결과 변동을 줄여 디버깅/비교가 쉬워집니다.
SEED = 42
torch.manual_seed(SEED)  # CPU 난수
np.random.seed(SEED)     # NumPy 난수
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)  # CUDA 난수 (NVIDIA GPU일 때)

print(f"PyTorch 버전: {torch.__version__}")
print(f"디바이스: {device}")


# In[2]:


# from torch.utils.tensorboard import SummaryWriter
# from datetime import datetime

# def get_tensorboard_writer(model_name):
#     """
#     TensorBoard SummaryWriter 생성

#     - 학습 중 loss/acc 같은 값을 기록해 두면 곡선을 쉽게 확인할 수 있습니다.
#     - 모델별로 로그 폴더를 분리해 두면 비교가 편합니다.

#     Args:
#         model_name: 모델 이름 (예: 'MLP', 'CNN', 'VGG-style')

#     Returns:
#         SummaryWriter 인스턴스
#     """
#     # 실행할 때마다 시간 스탬프를 붙여 runs/ 아래에 새 로그 폴더를 생성
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     log_dir = f'runs/{model_name}_{timestamp}'
#     writer = SummaryWriter(log_dir=log_dir)

#     print(f"📊 TensorBoard 로그: {log_dir}")
#     print(f"   실행: tensorboard --logdir=runs")

#     # (팁) 학습이 끝나면 writer.close()를 호출해 파일 핸들을 정리하는 것이 좋습니다.
#     return writer

# print("✅ TensorBoard 설정 함수 정의 완료")


# ## 2. 데이터 준비 (CIFAR-10)
# 
# 이 섹션에서는 CIFAR-10을 다운로드하고, 학습을 위해 **Train / Validation / Test**로 나눕니다.
# 
# - **Train**: 실제로 가중치를 업데이트하는 데이터
# - **Validation**: 하이퍼파라미터/조기종료(Early Stopping) 판단에 쓰는 데이터
# - **Test**: 최종 성능 보고용(학습 과정에서 되도록 건드리지 않음)

# ### 💡 딥러닝 데이터 파이프라인과 메모리 관리
# 
# 학습 시 데이터는 **[Disk → RAM → VRAM]** 의 경로를 통해 이동합니다. 효율적인 학습을 위해 이 하드웨어적 흐름을 이해하는 것은 매우 중요합니다.
# 
# | 단계 | 위치 | 관리 객체 | 주요 역할 | 특징 |
# | :--- | :--- | :--- | :--- | :--- |
# | **1. 창고 (Storage)** | **Disk** | `Dataset` | 전체 데이터셋 보관 | 가장 크지만, I/O 속도가 가장 느림 |
# | **2. 대기실 (Preparation)** | **RAM** | `DataLoader` | 배치 구성, 데이터 변환(CPU 연산) | CPU가 전처리 수행, 중간 속도 |
# | **3. 작업대 (Workbench)** | **VRAM** | `to(device)` | **실제 모델 연산** 및 업데이트 | 가장 빠르나 용량이 매우 제한적 |
# 
# 
# #### 💡 하드웨어 핵심 개념 이해
# - **GPU와 CUDA Core**: GPU는 수만 개의 작은 `코어(CUDA Core)`들로 구성되어 있습니다. 이 코어들은 행렬 연산과 같은 단순 반복 계산을 동시에 병렬로 처리합니다. 특히 딥러닝 전용 가속기인 `Tensor Core`는 부동소수점 연산 속도를 획기적으로 향상시킵니다.
# 
# - **VRAM (그래픽 메모리)**: GPU 전용 메모리입니다. 학습에 필요한 `파라미터(W, b)`, 역전파를 위한 `기울기(Gradients)`, 연산 중 생성되는 `중간 데이터(Feature Maps)`가 상주합니다.
# 
# > **RAM vs VRAM (냉장고 vs 도마)**
# > *   **RAM**: 모든 재료를 신선하게 보관하는 **냉장고**입니다.
# > *   **VRAM**: 요리사(GPU 코어)가 지금 당장 칼질을 하는 **도마**입니다. 도마가 좁으면 아무리 요리사가 빨라도 요리를 한꺼번에 많이 할 수 없습니다.
# 
# 
# #### 💡 실제 학습 시 주의사항 (Troubleshooting)
# - **Batch Size와 OOM**: 배치 사이즈는 도마(VRAM) 위에 한 번에 올리는 재료의 양입니다. 이 값이 너무 크면 VRAM 용량을 초과하여 `Out of Memory (OOM)` 에러가 발생하며 학습이 중단됩니다.  
# 
# - **Params vs Pass Size**: 모델 자체의 크기(`Params`)보다, 데이터가 층을 통과하며 만들어내는 중간 결과물(`Pass Size`)이 VRAM을 더 많이 차지하는 경우가 많습니다. 특히 CNN은 이미지의 공간 구조를 유지하기 때문에 이 크기가 급격히 커집니다.  
# 
# - **Optimizer State**: `Adam` 같은 옵티마이저는 학습 효율을 위해 파라미터 크기의 2~3배에 달하는 추가 메모리를 소모합니다. (단, 추론 시에는 사용되지 않습니다.)  
# 
# - **Memory Wall**: 최근 GPU는 연산 속도(`TFLOPS`)보다 VRAM에서 코어로 데이터를 실어나르는 대역폭(`Bandwidth`)에서 병목이 자주 발생합니다. 이를 극복하기 위해 `HBM(고대역폭 메모리)` 기술이 사용됩니다.  
# 
# - **num_workers**: CPU 일꾼의 수입니다. 일꾼이 미리 RAM에 배치를 준비해두면 GPU가 노는 시간 없이 계속 일할 수 있어 학습 효율이 극대화됩니다.
# 
# 

# ### 2.1 전처리 및 데이터 로드
# 
# 이 섹션에서는 **CIFAR-10 이미지를 학습에 바로 넣을 수 있는 텐서 형태로 변환**하고, `DataLoader`로 **배치(batch)** 단위로 꺼내오는 파이프라인을 만듭니다.
# 
# - `transforms.ToTensor()` : PIL/ndarray 이미지를 `torch.Tensor`로 바꾸고 값 범위를 `[0,1]`로 정규화
# - `transforms.Normalize(mean, std)` : 채널별 정규화(학습 안정화에 도움)
# - `random_split` : Train/Val 분할(검증 데이터는 **학습에 사용하지 않음**)
# - `DataLoader` : `batch_size`, `shuffle`, `num_workers`로 학습 효율을 조절

# In[3]:


# ------------------------------------------------------------
# 하이퍼파라미터 (데이터 관련)
# ------------------------------------------------------------
IMG_SIZE = 32
BATCH_SIZE = 64
NUM_CLASSES = 10

# ------------------------------------------------------------
# 전처리 정의 (CPU에서 수행됨)
# ------------------------------------------------------------
# CIFAR-10 원본은 (H,W,C) 이미지이며, transforms가 (C,H,W) 텐서로 바꿔줍니다.
# - ToTensor(): [0,255] -> [0,1] 범위로 변환 + 채널 우선(C,H,W)로 변경
# - Normalize(): 채널별 평균/표준편차로 정규화 (학습 안정화)
transform = transforms.Compose([
    transforms.ToTensor(),  # [0, 255] → [0, 1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # → 대략 [-1, 1]
])

# ------------------------------------------------------------
# 데이터셋 다운로드/로드
# ------------------------------------------------------------
print("데이터셋 다운로드 중...")
full_train = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform,
 )
test_dataset = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform,
 )

# ------------------------------------------------------------
# Train/Val 분할 (45K / 5K)
# ------------------------------------------------------------
# generator에 시드를 주면 split 결과가 매번 동일해져 재현성이 좋아집니다.
train_size = 45000
val_size = 5000
train_dataset, val_dataset = random_split(
    full_train, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED),
)

# ------------------------------------------------------------
# DataLoader 생성
# ------------------------------------------------------------
# - shuffle=True: train에서만 섞어줌(학습 안정/일반화에 도움)
# - num_workers: 배치 준비를 병렬 처리(환경에 따라 성능/재현성에 영향)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
 )
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
 )
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
 )

print(f"✅ 학습: {train_size}장, 검증: {val_size}장, 테스트: {len(test_dataset)}장")
print(f"배치 크기: {BATCH_SIZE}")


# In[4]:


test_dataset[0][0]


# ### 2.2 데이터 시각화
# 
# 전처리가 올바르게 적용됐는지 확인하기 위해 **배치에서 샘플 이미지를 꺼내 시각화**합니다.
# 
# - `Normalize`를 적용했다면, 화면에 보여주기 전에 **역정규화(unnormalize)** 비슷한 처리가 필요합니다.
# - CIFAR-10 텐서 shape는 보통 `(C, H, W)` 이므로 `matplotlib`에 그리려면 `(H, W, C)`로 축을 바꿉니다.

# In[5]:


# ------------------------------------------------------------
# CIFAR-10 샘플 이미지 시각화
# ------------------------------------------------------------
# 목표: 전처리/정규화가 의도대로 적용됐는지 눈으로 확인
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def imshow(img, title=None):
    """이미지(텐서)를 matplotlib으로 표시하는 헬퍼 함수"""
    # transforms.Normalize((0.5),(0.5))를 했으므로 대략 역정규화처럼 되돌려 보기
    img = img / 2 + 0.5  # [-1, 1] → [0, 1]
    # 주의: numpy 변환은 CPU 텐서에서만 가능 (.cpu() 필요할 수 있음)
    npimg = img.numpy()
    # PyTorch 텐서는 (C,H,W)라서 matplotlib용 (H,W,C)로 축 변경
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title:
        plt.title(title, fontsize=12)
    plt.axis('off')

# 배치 가져오기 (images: (B,C,H,W), labels: (B,))
dataiter = iter(train_loader)
images, labels = next(dataiter)

# 8개 샘플 시각화
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    plt.sca(ax)
    imshow(images[i], title=class_names[labels[i]])

plt.tight_layout()
plt.show()

print(f"이미지 shape: {images[0].shape} (C, H, W)")
print(f"레이블 shape: {labels.shape}")


# ### 3.1 MLP (Multi-Layer Perceptron)
# 
# > **Style: Basic**
# > - 모든 연산(Activation, Dropout, Flatten 등)을 `nn.Module`의 레이어로 정의하는 방식입니다.
# > - 모델의 구조가 `print(model)` 시 순차적으로 모두 표시되어 초보자가 흐름을 이해하기에 가장 좋습니다.
# 

# In[6]:


# ------------------------------------------------------------
# MLP 모델 정의
# ------------------------------------------------------------
# MLP는 이미지를 1D로 펼친 뒤(Flatten) Linear 레이어로만 분류합니다.
# CIFAR-10 입력: (B, 3, 32, 32) -> Flatten -> (B, 3*32*32)
class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        # 입력 특징 수: 32*32*3
        self.fc1 = nn.Linear(IMG_SIZE*IMG_SIZE*3, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        # 마지막은 클래스 개수만큼 logits 출력 (softmax는 loss에서 내부적으로 처리)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x  # logits: (B, num_classes)

mlp_model = MLP(num_classes=NUM_CLASSES).to(device)

print("MLP 모델 구조:")
# 모델 요약 정보 출력
summary(mlp_model,
        input_size=(BATCH_SIZE, 3, 32, 32),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        row_settings=["var_names", "depth"],
        verbose=0)


# ### 3.2 CNN (Convolutional Neural Network)
# 
# > **Style: Intermediate**
# > - 가중치(Weight)가 있는 레이어(Conv, Linear, BN)만 `__init__`에 정의하고, 파라미터가 없는 연산(ReLU, Pooling 등)은 `forward`에서 Functional API로 처리하는 방식입니다.
# > - 가중치 레이어 중심의 설계로 코드가 깔끔하며, PyTorch 공식 라이브러리에서 가장 흔히 사용하는 표준 스타일입니다.
# 

# In[7]:


# ------------------------------------------------------------
# CNN 모델 정의
# ------------------------------------------------------------
# 합성곱(Conv) + 풀링(Pooling)으로 공간 정보를 활용하는 기본 CNN
# 입력: (B, 3, 32, 32) -> 출력 logits: (B, num_classes)
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        # Conv는 채널 수를 바꾸며 특징을 추출
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # 두 번의 2x2 maxpool -> (32,32) -> (16,16) -> (8,8)
        # 채널 64 * 8 * 8 = 4096
        self.fc1 = nn.Linear(4096, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Block 1: Conv -> BN -> ReLU -> Pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Block 2: Conv -> BN -> ReLU -> Pool
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Flatten: (B,64,8,8) -> (B,4096)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # Dropout은 학습(train)에서만 활성화되도록 training 플래그를 전달
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.fc2(x)
        return x  # logits

cnn_model = CNN(num_classes=NUM_CLASSES).to(device)

print("CNN 모델 구조:")
summary(cnn_model, input_size=(BATCH_SIZE, 3, 32, 32),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        row_settings=["var_names", "depth"],
        verbose=0)


# ### 3.3 VGG-style 네트워크
# 
# > **Style: Expert**
# > - 반복되는 구조를 블록(Block) 단위로 모듈화하고, 이를 `nn.Sequential` 등으로 그룹화하여 관리하는 방식입니다.
# > - 대규모 모델 설계 시 재사용성과 유지보수성이 압도적으로 높으며, 실무 레벨의 템플릿으로 권장되는 스타일입니다.
# 

# #### Style A: 단일 Sequential 방식 (Single Sequential) - **[입문자 권장]**
# 
# - **핵심 개념**: 모든 레이어를 단 하나의 `nn.Sequential` 컨테이너에 순차적으로 담는 방식입니다.
# - **장점**:
#     - **가장 직관적**: 입문자가 코드를 보았을 때 데이터가 흐르는 순서를 한눈에 이해하기 가장 쉽습니다.
#     - **최소한의 코드**: 레이어 정의와 연결이 한 곳에서 이루어지므로 `forward` 코드가 매우 짧습니다.
# - **단점**:
#     - **낮은 가용성**: 모델이 커지면 한눈에 파악하기 어렵고, 특정 부분만 재사용하기가 까다롭습니다.
#     - **유연성 부족**: 일직선 구조 외에 복잡한 연결(Skip Connection 등)을 구현할 수 없습니다.
# 

# In[8]:


class VGGStyleA(nn.Module):
    def __init__(self, num_classes=10):
        super(VGGStyleA, self).__init__()

        self.all_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),

            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.all_layers(x)


# #### Style B: Sequential 모듈화 방식 (Modular Assembly)
# 
# - **핵심 개념**: 관련 있는 블록들을 `nn.Sequential`이라는 컨테이너에 담아 묶음으로 관리하고, `forward()`에서는 **컨테이너 단위로 실행**하는 방식입니다.
# - **장점**:
#     - **가독성 및 관리 효율**: '특징 추출기', '분류기' 등 모델의 역할을 논리적으로 분리하여 관리할 수 있습니다.
#     - **깔끔한 forward**: 레이어가 수백 개라도 `forward` 함수는 한두 줄로 깔끔하게 끝납니다.
# - **단점**:
#     - **구조적 경직성**: `Sequential` 내부에서는 데이터가 일직선으로만 흘러야 합니다.
# 

# In[9]:


class VGGBlock(nn.Module):
    """BatchNorm이 포함된 VGG 기본 블록"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels) # 배치 정규화 추가

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


# In[10]:


class VGGStyleB(nn.Module):
    """Style B: Sequential 모듈화 - 특징 추출기와 분류기를 분리"""
    def __init__(self, num_classes=10):
        super(VGGStyleB, self).__init__()

        # 특징 추출기 (Feature Extractor)
        self.features = nn.Sequential(
            # Block 1
            VGGBlock(3, 64),
            VGGBlock(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            VGGBlock(64, 128),
            VGGBlock(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            VGGBlock(128, 256),
            VGGBlock(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 분류기 (Classifier)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

print("✅ VGGStyleB 정의 완료")


# #### Style C: 명시적 서브 모듈화 (Explicit Connection) - **[전문가 권장]**
# 
# - **핵심 개념**: 재사용 가능한 서브 모듈(`VGGBlock`)을 먼저 정의하고, 이를 메인 모델에서 **개별 속성(attribute)**으로 명시적으로 연결하는 방식입니다.
# - **장점**:
#     - **최고의 유연성**: 각 블록에 이름을 부여하여 `forward()`에서 조건부 실행, Skip Connection, Attention 등 복잡한 연결 패턴 구현이 자유롭습니다.
#     - **디버깅 용이**: 특정 블록의 출력을 중간에 추출하거나, 일부만 freeze하는 등 세밀한 제어가 가능합니다.
#     - **실무 표준**: ResNet, EfficientNet 등 최신 아키텍처는 모두 이 방식으로 구현됩니다.
# - **단점**:
#     - **코드량 증가**: 블록이 많아지면 `__init__`과 `forward` 코드가 길어집니다.
#     - **관리 부담**: 블록 간 연결 순서를 수동으로 관리해야 하므로 실수 가능성이 있습니다.

# In[11]:


class VGGStyleC(nn.Module):
    """Style C: 명시적 서브 모듈화 - 각 블록을 개별 속성으로 정의"""
    def __init__(self, num_classes=10):
        super(VGGStyleC, self).__init__()

        # Block 1
        self.block1_conv1 = VGGBlock(3, 64)
        self.block1_conv2 = VGGBlock(64, 64)
        self.block1_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.block2_conv1 = VGGBlock(64, 128)
        self.block2_conv2 = VGGBlock(128, 128)
        self.block2_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3
        self.block3_conv1 = VGGBlock(128, 256)
        self.block3_conv2 = VGGBlock(256, 256)
        self.block3_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classifier
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.bn = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Block 1
        x = self.block1_conv1(x)
        x = self.block1_conv2(x)
        x = self.block1_pool(x)

        # Block 2
        x = self.block2_conv1(x)
        x = self.block2_conv2(x)
        x = self.block2_pool(x)

        # Block 3
        x = self.block3_conv1(x)
        x = self.block3_conv2(x)
        x = self.block3_pool(x)

        # Classifier
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

print("✅ VGGStyleC 정의 완료")


# ### 4.1 학습 함수
# 
# #### 💡 Training Loop 핵심 단계
# 
# 딥러닝 모델의 실제 학습은 아래의 순서로 반복(Loop)되며 진행됩니다.
# 
# | 단계 | 코드 | 주요 역할 |
# | :--- | :--- | :--- |
# | **1. 데이터 추출** | `for images, labels in loader:` | DataLoader에서 배치 사이즈만큼 데이터를 꺼내와 반복 작업 수행 |
# | **2. Device 설정** | `images.to(device)` | 데이터를 GPU(VRAM) 또는 CPU로 이동시켜 연산 준비 |
# | **3. Forward Pass** | `pred = model(images)` | 이미지 데이터를 모델에 입력하여 예측값(Probability) 산출 |
# | **4. 오차 계산** | `loss = criterion(pred, labels)` | 손실 함수를 통해 예측값과 정답 사이의 오차(Loss) 계산 |
# | **5. 기울기 초기화** | `optimizer.zero_grad()` | 이전 배치의 기울기(Gradient) 값을 비움 (누적 방지) |
# | **6. Backprop** | `loss.backward()` | 역전파를 통해 각 파라미터별 오차 기여도(Gradient) 계산 |
# | **7. 가중치 업데이트** | `optimizer.step()` | 계산된 기울기를 학습률(LR)만큼 파라미터에 적용하여 가중치 갱신 |
# 

# In[12]:


def train_epoch(model, loader, criterion, optimizer, device):
    """
    1 에포크(epoch) 동안 모델을 학습시키는 함수
    - loader의 모든 배치를 1번씩 보며(=1 epoch) 가중치를 업데이트합니다.
    """
    model.train()  # Dropout/BN 등을 학습 모드로 (학습 시 동작이 달라짐)
    running_loss = 0.0
    correct = 0
    total = 0

    # tqdm 진행 표시줄
    pbar = tqdm(loader, desc="Training", leave=False)

    for inputs, labels in pbar:
        # 1) 데이터를 연산 디바이스로 이동
        # inputs: (B,C,H,W), labels: (B,)
        inputs, labels = inputs.to(device), labels.to(device)

        # 2) Forward + Loss
        optimizer.zero_grad()  # 기울기 누적 방지 (배치마다 초기화)
        outputs = model(inputs)  # logits: (B, num_classes)
        loss = criterion(outputs, labels)  # CE Loss는 logits + 정답 인덱스를 받음

        # 3) Backward + Update
        loss.backward()
        optimizer.step()

        # 4) 통계 누적 (epoch 단위 평균을 내기 위해 '합'으로 모아둠)
        # loss.item()은 스칼라 텐서를 파이썬 float로 변환(로깅/누적용)
        running_loss += loss.item() * inputs.size(0)
        # 예측 클래스: outputs에서 가장 큰 값의 인덱스
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 5) 진행 상황 표시 (현재 배치 loss / 누적 acc)
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    # epoch 최종 평균
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """
    검증 데이터셋으로 성능을 평가하는 함수
    - 평가에서는 가중치 업데이트를 하지 않습니다.
    """
    model.eval()  # Dropout 비활성화, BN 고정(평가 모드)
    running_loss = 0.0
    correct = 0
    total = 0

    # 평가 시에는 gradient가 필요 없어서 no_grad로 메모리/속도 최적화
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


print("✅ 학습/평가 함수 정의 완료")


# ### 4.2 Early Stopping (조기 종료) 클래스
# 
# 학습 과정에서 검증 손실(Val Loss)이 일정 에포크 동안 개선되지 않으면 학습을 중단하는 기능입니다.
# 이를 통해 **과적합(Overfitting)을 방지**하고 불필요한 연산 시간을 줄일 수 있습니다.

# In[13]:


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, verbose=True):
        self.patience = patience    # 성능 개선이 없을 때 참아줄 에포크 수
        self.min_delta = min_delta  # 개선되었다고 판단할 최소 변화량
        self.counter = 0             # 현재 참은 횟수
        self.best_loss = None        # 역사상 최저 검증 손실
        self.early_stop = False      # 학습 종료 신호
        self.verbose = verbose

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

print("✅ EarlyStopping 클래스 정의 완료")


# ### 4.3 전체 학습 루프 (Canonical Style)
# 
# 한 에포크 학습()과 검증()을 묶어 전체 에포크를 반복하는 최종 학습 함수입니다.
# 학습률 스케줄러와 조기 종료 기능을 통합한 정석적인 구조입니다.

# In[14]:


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    early_stopping=None,
    epochs=10,
    model_name="Model",
):
    """
    전체 학습 과정을 조율하는 정석적인 학습 루프

    흐름:
    1) train_epoch()로 학습(가중치 업데이트)
    2) validate()로 검증(가중치 업데이트 X)
    3) scheduler/early stopping/checkpoint를 적용
    4) history에 기록해 학습 곡선 그리기
    """
    # 학습 과정 기록(그래프/비교용)
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    print(f"\n{'='*60}")
    print(f"{model_name} 학습 시작 (Device: {device})")
    print(f"{'='*60}")

    start_time = time.time()
    best_val_acc = 0.0  # 체크포인트 저장 기준(여기서는 val_acc)

    for epoch in range(epochs):
        # 현재 LR 확인 (optimizer에 들어있는 값)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{epochs} [LR: {current_lr:.6f}]")
        print("-" * 60)

        # 1) 학습 (Training Phase): backward/step 수행
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 2) 검증 (Validation Phase): eval + no_grad로 평가
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

# 3) 학습률 스케줄러 (Scheduler Step)
# - StepLR 같은 스케줄러: epoch마다 step()
# - ReduceLROnPlateau: 보통 val_loss 같은 지표를 보고 step(metric)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # 4) 결과 기록(그래프용)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 결과 출력
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # 5) 체크포인트 저장: 가장 좋은 모델 가중치를 파일로 남김
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_{model_name}.pth')
            print(f"✅ 최고 성능 갱신! (저장됨)")

        # 6) 조기 종료(Early Stopping): 개선이 없으면 학습을 중단
        if early_stopping is not None:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("🛑 조기 종료 조건 충족! 학습을 중단합니다.")
                break

    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"학습 완료! (소요 시간: {elapsed_time/60:.1f}분)")
    print(f"최고 정확도: {best_val_acc:.4f}")
    print(f"{'='*60}")

    return history


# ## 5. 모델 학습
# 
# 여기부터는 앞에서 만든 **DataLoader + 모델 + 학습 루프**를 조합해 실제로 학습을 돌립니다.
# 
# - 각 모델(MLP/CNN/VGG)에 대해 동일한 학습 루프를 사용하면, 구조 차이에 따른 성능 차이를 더 공정하게 비교할 수 있습니다.
# - 아래 코드 셀은 학습 시간이 걸릴 수 있으니, 필요하면 `epochs`를 줄여 빠르게 확인해도 됩니다.
# 
# ### 5.1 MLP 학습

# In[15]:


# 1. 모델, 손실함수, 옵티마이저, 스케줄러 설정
mlp_model = MLP(num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=1e-3)
# 5 에포크마다 학습률을 0.1배로 감소시키는 스케줄러
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
early_stopping = EarlyStopping(patience=3, verbose=True)

# 2. 학습 실행 (인자들을 명시적으로 전달)
history_mlp = train_model(
    mlp_model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    early_stopping=early_stopping,
    epochs=10,
    model_name="MLP"
)


# ### 5.2 BasicCNN 학습 (예시 코드)
# 
# 아래 코드는 **MLP 학습과 동일한 패턴**으로 `BasicCNN`을 학습하는 예시입니다.
# (현재 셀은 마크다운이므로 실행되지 않습니다. 실행하려면 같은 내용을 코드 셀로 옮겨 실행하세요.)
# 
# ```python
# # 1. 모델, 손실함수, 옵티마이저, 스케줄러 설정
# cnn_model = BasicCNN(num_classes=NUM_CLASSES).to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(cnn_model.parameters(), lr=1e-3)
# 
# # 학습률 스케줄러 설정 (예: 5 epoch마다 LR을 0.1배)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
# early_stopping = EarlyStopping(patience=3, verbose=True)
# 
# # 2. 학습 실행
# history_cnn = train_model(
#     cnn_model,
#     train_loader,
#     val_loader,
#     criterion,
#     optimizer,
#     scheduler,
#     device,
#     early_stopping=early_stopping,
#     epochs=10,
#     model_name="BasicCNN",
# )
# ```

# In[16]:


# 1. 모델, 손실함수, 옵티마이저, 스케줄러 설정
cnn_model = CNN(num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=1e-3)
# 학습률 스케줄러 설정
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
early_stopping = EarlyStopping(patience=3, verbose=True)

# 2. 학습 실행
history_cnn = train_model(
    cnn_model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    early_stopping=early_stopping,
    epochs=10,
    model_name="CNN"
)


# ### 5.3 VGG 학습
# 
# VGG-style 모델은 CNN보다 더 깊고 표현력이 커서 성능이 좋아질 수 있지만, **연산량/메모리 사용량도 증가**합니다.
# 
# - 실습에서는 같은 `train_model()` 함수를 그대로 재사용해 학습 흐름을 통일합니다.
# - VGG는 종종 더 작은 학습률(LR)이 안정적인 경우가 있어, 아래 코드에서 LR 관련 주석을 확인하세요.

# In[17]:


# 1. 모델, 손실함수, 옵티마이저, 스케줄러 설정
vgg_model = VGGStyleA(num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg_model.parameters(), lr=1e-3) # VGG는 더 작은 LR 권장
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 2. 학습 실행
history_vgg = train_model(
    vgg_model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs=10,
    model_name="VGGStyleA"
)


# ## 6. 결과 분석
# 
# 학습이 끝나면 “학습을 잘 했는지/과적합은 없는지/모델 간 차이는 무엇인지”를 확인해야 합니다.
# 이 섹션에서는 **곡선(learning curve)**과 **테스트 성능**을 통해 모델들을 비교합니다.
# 
# ### 6.1 학습 곡선 비교
# 
# - `val_loss`가 계속 떨어지면 일반화가 좋아지는 신호일 수 있습니다.
# - `train_acc`는 오르는데 `val_acc`가 정체/하락하면 과적합 가능성이 있습니다.

# In[18]:


# 학습 곡선 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 손실 곡선
axes[0].plot(history_mlp['val_loss'], label='MLP', marker='o', linewidth=2)
axes[0].plot(history_cnn['val_loss'], label='CNN', marker='s', linewidth=2)
axes[0].plot(history_vgg['val_loss'], label='VGG', marker='^', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Validation Loss', fontsize=12)
axes[0].set_title('Validation Loss Comparison', fontsize=14)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# 정확도 곡선
axes[1].plot(history_mlp['val_acc'], label='MLP', marker='o', linewidth=2)
axes[1].plot(history_cnn['val_acc'], label='CNN', marker='s', linewidth=2)
axes[1].plot(history_vgg['val_acc'], label='VGG', marker='^', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Validation Accuracy', fontsize=12)
axes[1].set_title('Validation Accuracy Comparison', fontsize=14)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ### 6.2 테스트 성능 평가
# 
# 검증(Validation)은 학습 중 의사결정(early stopping, 튜닝)에 쓰고, **최종 성능 보고는 Test**로 하는 것이 정석입니다.
# 
# - `model.eval()` + `torch.no_grad()` 상태로 평가하여, dropout/BN 동작을 고정하고 불필요한 gradient 계산을 막습니다.

# In[19]:


# 테스트 평가 함수
def evaluate_on_test(model, test_loader, device, model_name):
    """테스트 데이터셋 평가"""
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"{model_name:15} - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    return test_loss, test_acc

# 세 모델 평가
print("\n테스트 데이터셋 평가")
print("=" * 60)
mlp_test_loss, mlp_test_acc = evaluate_on_test(mlp_model, test_loader, device, "MLP")
cnn_test_loss, cnn_test_acc = evaluate_on_test(cnn_model, test_loader, device, "CNN")
vgg_test_loss, vgg_test_acc = evaluate_on_test(vgg_model, test_loader, device, "VGG-style")
print("=" * 60)


# ### 6.3 모델 비교 요약
# 
# 여기서는 세 모델(MLP/CNN/VGG)의 결과를 한 표로 모아 **파라미터 수(모델 크기)**와 **정확도(성능)**를 함께 비교합니다.
# 
# - 파라미터 수가 많다고 항상 좋은 것은 아니며(과적합/연산량 증가), CNN처럼 **구조적 가정(공간 정보)**을 잘 쓰는 모델이 더 효율적일 수 있습니다.

# In[20]:


import pandas as pd

# 파라미터 수 계산
mlp_params = sum(p.numel() for p in mlp_model.parameters())
cnn_params = sum(p.numel() for p in cnn_model.parameters())
vgg_params = sum(p.numel() for p in vgg_model.parameters())

# 결과 테이블
results = pd.DataFrame({
    'Model': ['MLP', 'CNN', 'VGG-style'],
    'Parameters': [f"{mlp_params:,}", f"{cnn_params:,}", f"{vgg_params:,}"],
    'Val Acc': [
        f"{max(history_mlp['val_acc']):.4f}",
        f"{max(history_cnn['val_acc']):.4f}",
        f"{max(history_vgg['val_acc']):.4f}"
    ],
    'Test Acc': [
        f"{mlp_test_acc:.4f}",
        f"{cnn_test_acc:.4f}",
        f"{vgg_test_acc:.4f}"
    ]
})

print("\n📊 모델 성능 비교")
print("=" * 60)
print(results.to_string(index=False))
print("=" * 60)
print("\n💡 관찰:")
print("  - MLP: 가장 많은 파라미터, 낮은 성능 (공간 정보 손실)")
print("  - CNN: 적은 파라미터, 개선된 성능 (지역 패턴 학습)")
print("  - VGG: 깊은 구조로 최고 성능 (계층적 특징 추출)")


# ## 7. 모델 저장 및 로드
# 
# 학습이 끝난 모델은 보통 **가중치(state_dict)**를 파일로 저장해 두고, 나중에 다시 불러와 추론/재현/배포에 사용합니다.
# 
# - 가장 흔한 방식: `torch.save(model.state_dict(), path)` / `model.load_state_dict(torch.load(path))`
# - 이 노트북에서는 학습 중 최고 성능 모델을 저장(checkpoint)하고, 저장된 가중치를 다시 로드해 테스트 성능을 재확인합니다.

# In[21]:


# 모델 저장
torch.save({
    'model_state_dict': vgg_model.state_dict(),
    'test_acc': vgg_test_acc,
}, 'best_vgg_model.pth')

print("✅ VGG 모델 저장 완료: best_vgg_model.pth")
print()

# 모델 로드 예제
print("모델 로드 예제:")
print("""
# 새 모델 인스턴스 생성
loaded_model = VGGStyle(num_classes=10).to(device)

# 체크포인트 로드
checkpoint = torch.load('best_vgg_model.pth')
loaded_model.load_state_dict(checkpoint['model_state_dict'])
print(f"테스트 정확도: {checkpoint['test_acc']:.4f}")
""")


# ### 7.1 저장된 모델 불러오기 및 최종 테스트
# 
# 학습이 끝난 후 저장된 최적의 가중치를 불러와서, 실제로 테스트 데이터셋에 대해 똑같은 성능이 나오는지 확인해봅니다.

# In[22]:


# 1. 모델 인스턴스 생성 (학습 때와 동일한 구조)
test_model = VGGStyleA(num_classes=NUM_CLASSES).to(device)

# 2. 저장된 파일 로드
# train_model 함수에서 자동으로 저장된 파일 또는 위에서 수동으로 저장한 파일 사용
model_path = 'best_VGGStyleA.pth' # 또는 'best_vgg_model.pth'

try:
    checkpoint = torch.load(model_path, map_location=device)

    # 만약 딕셔너리 형태로 저장했다면 (state_dict 추출)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        test_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 가중치(state_dict)만 바로 저장한 경우
        test_model.load_state_dict(checkpoint)

    print(f"✅ 모델 로드 성공: {model_path}")

    # 3. 최종 테스트 수행
    criterion = nn.CrossEntropyLoss()
    final_loss, final_acc = validate(test_model, test_loader, criterion, device)

    print(f"\n[불러온 모델의 최종 성능]")
    print(f"Test Loss: {final_loss:.4f}")
    print(f"Test Accuracy: {final_acc*100:.2f}%")

except FileNotFoundError:
    print(f"❌ 모델 파일을 찾을 수 없습니다. ({model_path}) 학습을 먼저 완료해주세요.")


# ## 8. 요약 및 다음 단계
# 
# ### ✅ 학습한 내용
# 1. PyTorch DataLoader와 transforms 활용
# 2. nn.Module 패턴으로 MLP, CNN, VGG 구현
# 3. 직접 작성한 학습 루프로 모델 학습
# 4. 여러 모델 성능 비교 및 분석
# 5. 모델 저장 및 로드
# 
# ### 💡 핵심 패턴 복습
# 1. **데이터**: transforms → Dataset → DataLoader
# 2. **모델**: nn.Module 클래스 상속 → forward() 정의
# 3. **학습**: optimizer.zero_grad() → loss.backward() → optimizer.step()
# 4. **평가**: model.eval() → torch.no_grad()
# 
# ### 📊 성능 차이 이유
# - **MLP**: 이미지를 1D로 펼치면서 공간 정보 손실
# - **CNN**: 합성곱으로 지역 패턴 학습, 파라미터 효율적
# - **VGG**: 더 깊은 구조로 계층적 특징 추출, 높은 표현력

# In[22]:




