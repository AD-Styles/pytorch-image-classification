#!/usr/bin/env python
# coding: utf-8

# # 08. PyTorch 기초 튜토리얼
# 
# ## 📋 학습 목표
# - PyTorch 설치 및 환경 설정 (GPU/MPS 감지)
# - 텐서 생성, 형태 변환, 차원 조작 완벽 이해
# - Aggregation 연산 및 고급 인덱싱 기법 습득
# - Autograd를 통한 자동 미분 이해
# - PyTorch vs TensorFlow 주요 차이점 학습
# - 기본 학습 루프 구조 이해
# 
# ## 💡 TensorFlow와의 차이점
# 이 노트북에서는 이미 학습한 TensorFlow/Keras 개념을 PyTorch로 어떻게 구현하는지 배웁니다.

# ## 1. 환경 설정 및 GPU 확인
# 
# PyTorch는 Apple Silicon (MPS), NVIDIA GPU (CUDA), CPU를 모두 지원합니다.

# In[ ]:


from torchinfo import summary
import torch
import platform
import numpy as np
import matplotlib.pyplot as plt

def system_config():
    """GPU 감지 및 디바이스 설정"""
    print("=" * 50)
    print("시스템 정보")
    print("=" * 50)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"PyTorch 버전: {torch.__version__}")
    print()

    # MPS (Apple Silicon) > CUDA > CPU 우선순위
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ MPS (Apple Silicon) 사용")
        print("   - M 시리즈 칩의 GPU 가속 활성화")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ CUDA GPU 사용: {torch.cuda.get_device_name(0)}")
        print(f"   - GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device("cpu")
        print("⚠️ CPU 모드로 실행")
        print("   - GPU 가속을 사용할 수 없습니다")

    print("=" * 50)
    return device

# 디바이스 설정
device = system_config()


# ## 2. 텐서 기초
# 
# ### 2.1 텐서 생성 및 조작
# 
# PyTorch의 텐서는 TensorFlow의 텐서와 유사하지만, NumPy와 더 긴밀하게 통합되어 있습니다.

# In[ ]:


# 텐서 생성
print("1️⃣ 텐서 생성 방법")
print("=" * 50)

# 리스트에서 생성
x1 = torch.tensor([1, 2, 3, 4, 5])
print(f"리스트에서 생성: {x1}")
print(f"shape: {x1.shape}, dtype: {x1.dtype}")
print()

# 특정 값으로 초기화
x2 = torch.zeros(3, 4)  # 0으로 채운 3×4 텐서
x3 = torch.ones(2, 3)   # 1로 채운 2×3 텐서
x4 = torch.rand(2, 3)   # 0~1 사이 랜덤 값

print(f"zeros(3, 4):\n{x2}")
print()
print(f"ones(2, 3):\n{x3}")
print()
print(f"rand(2, 3):\n{x4}")
print()

# arange로 순차 텐서 생성
x5 = torch.arange(10)
print(f"arange(10): {x5}")
print(f"dtype: {x5.dtype}, shape: {x5.shape}")


# ### 2.2 NumPy 변환 (Tensor ↔ Array)
# 
# 딥러닝 파이프라인에서 **모델 학습/추론은 Tensor**, **데이터 전처리·시각화·외부 라이브러리는 NumPy**로 처리하는 경우가 많아서 둘 사이 변환을 자주 합니다.
# 
# - Array → Tensor: `torch.from_numpy(x)` (대개 **메모리 공유**, 빠름)
# - Tensor → Array: `t.detach().cpu().numpy()` (대개 **메모리 공유**, CPU 텐서만 가능)
# - **공유 끊기(독립 복사본)**: `x.copy()`, `t.clone()`, `numpy().copy()` 등을 사용
# - Tensor → **Python 숫자**: `t.detach().item()` (loss 같은 **스칼라 값 로깅/출력**에 자주 사용)
# 
# #### 딥러닝에서 언제 쓰나? (실전 예시)
# - OpenCV/PIL/NumPy로 읽고 전처리한 이미지를 **모델 입력**으로 넣을 때 (Array → Tensor)
# - `Dataset`/`DataLoader`에서 CPU 전처리 후 **텐서로 변환해 배치 구성**할 때
# - 모델 출력(예측/확률/특징맵)을 `matplotlib`, `sklearn`로 **시각화·평가**할 때 (Tensor → Array)
# - 디버깅: 중간 텐서를 NumPy로 바꿔 **값/분포를 빠르게 확인**할 때
# - 학습 루프: `loss.item()`으로 **로그/프로그레스바/early stopping 조건**에 쓰기
# 
# #### 언제 메모리 공유를 유지하는 게 유리한가?
# - 큰 이미지/배치를 자주 변환할 때 **불필요한 복사를 줄여 속도·메모리를 절약**합니다. (대신, 이후 그 버퍼를 in-place로 수정하지 않는 전제가 필요)
# 
# #### 언제 공유를 끊어야 안전한가?
# - 증강/정규화 등에서 원본이 바뀌면 안 되는 경우(원본 보존, 캐시/재사용 데이터)
# - “왜 값이 같이 바뀌지?” 같은 **사이드 이펙트 버그를 원천 차단**하고 싶을 때
# 
# #### 언제 `item()`을 쓰나?
# - 스칼라 텐서(예: `loss`)를 **파이썬 float/int로 뽑아서** 출력/저장/조건문에 쓰고 싶을 때
# - 주의: GPU/MPS 텐서에서 `item()`은 값을 가져오며 동기화가 생길 수 있어 너무 자주 호출하면 느려질 수 있습니다.

# In[ ]:


import numpy as np
import torch

print("2️⃣ NumPy 변환")
print("=" * 50)

# NumPy → PyTorch
# - torch.from_numpy: NumPy의 메모리를 '그대로' 바라보는 Tensor(view) 생성 → 빠르지만 보통 메모리 공유
np_array = np.array([1, 2, 3, 4, 5])
torch_tensor = torch.from_numpy(np_array)  # ✅ (대개) 공유 O: np_array를 바꾸면 torch_tensor도 같이 변할 수 있음
print(f"NumPy → PyTorch: {torch_tensor}")
print(f"타입: {type(torch_tensor)}")
print()

# PyTorch → NumPy
# - tensor.numpy(): (CPU 텐서일 때) Tensor의 메모리를 '그대로' 바라보는 ndarray 생성 → 빠르지만 보통 메모리 공유
back_to_numpy = torch_tensor.numpy()  # ✅ (대개) 공유 O: torch_tensor를 바꾸면 back_to_numpy도 같이 변할 수 있음
print(f"PyTorch → NumPy: {back_to_numpy}")
print(f"타입: {type(back_to_numpy)}")
print()

# 💡 주의: 공유(view) 관계면 한쪽을 in-place로 수정할 때 다른 쪽도 그대로 반영됩니다
np_array[0] = 999
print(f"NumPy 수정 후 PyTorch 텐서: {torch_tensor}")
print("⚠️ 메모리를 공유하면 한쪽 수정 → 다른 쪽도 변경")
print()

# ✅ 해결: 공유를 끊고 싶다면 '복사본'을 만드세요 (원본 보존/사이드이펙트 방지)
print("✅ 해결: 메모리 공유 방지(복사본 만들기)")

# (1) NumPy → Tensor: from_numpy(공유) vs torch.tensor(복사) 차이
# - torch.tensor: 내용을 '복사'해서 새 Tensor 생성 → 안전(원본이 바뀌어도 영향 없음)
np_array2 = np.array([1, 2, 3, 4, 5])
t_shared = torch.from_numpy(np_array2)                 # 공유 O (빠름, 원본 수정 안 할 때 유리)
t_safe1 = torch.tensor(np_array2)                      # 공유 X (복사, 안전)
t_safe2 = torch.from_numpy(np_array2.copy())           # 공유 X (NumPy에서 먼저 copy → 이후 수정해도 안전)
t_safe3 = torch.from_numpy(np_array2).clone()          # 공유 X (Tensor에서 clone → 새 메모리)

np_array2[0] = 777
print(f"공유 O (from_numpy): {t_shared}")
print(f"공유 X (torch.tensor): {t_safe1}")
print(f"공유 X (copy+from_numpy): {t_safe2}")
print(f"공유 X (from_numpy+clone): {t_safe3}")
print()

# (2) Tensor → NumPy: numpy()(공유) vs numpy().copy()(복사) 차이
# - detach(): autograd 그래프에서 분리(grad 추적 끔). grad가 필요한 텐서는 보통 detach 후 NumPy로 변환
# - cpu(): GPU 텐서는 NumPy로 바로 변환 불가 → CPU로 이동
# - copy(): NumPy에서 새 메모리로 복사 → 공유 끊기
t2 = torch.arange(5)
n_shared = t2.numpy()                                  # 공유 O (CPU 텐서만 가능)
n_safe = t2.detach().cpu().numpy().copy()              # 공유 X (권장: grad/GPU/공유 이슈를 한 번에 회피)

t2[0] = 999
print(f"Tensor 수정 후 공유 NumPy: {n_shared}")
print(f"Tensor 수정 후 복사 NumPy: {n_safe}")
print()

# (3) Tensor → Python 스칼라: item()
# - item(): 스칼라 텐서(또는 원소 1개 텐서)에서 파이썬 숫자(float/int)를 꺼냄
# - 주 사용처: loss 로깅, print 포맷팅, if 조건(early stopping 등)
# - 주의: GPU/MPS 텐서에서 item()은 값을 가져오며 동기화가 생길 수 있어 너무 자주 쓰면 느려질 수 있음
t_scalar = torch.tensor(3.14)
print(f"t_scalar: {t_scalar}, item(): {t_scalar.item()} (type={type(t_scalar.item())})")

# requires_grad=True인 텐서는 로깅 목적이라면 보통 detach 후 item()
x = torch.tensor([2.0], requires_grad=True)
loss_like = (x ** 2 + 3 * x + 1).sum()  # 스칼라 텐서
print(f"loss_like (tensor): {loss_like}")
print(f"loss_like.detach().item() (logging용): {loss_like.detach().item()}")


# ### 2.3 Shape와 차원 정보
# 
# 텐서의 형태(shape)와 차원(dimension) 정보를 확인하는 방법을 학습합니다.
# 
# #### 왜 중요한가?
# 
# 딥러닝에서 **텐서의 형태는 코드의 정확성을 보장하는 첫 번째 방어선**입니다:
# - 형태가 예상과 다르면 런타임 에러 발생 → 디버깅의 출발점
# - 배치 크기, 채널 수, 이미지 크기를 명확히 추적해야 함
# - 모델 입출력 형태를 이해해야 파이프라인 구축 가능
# 
# #### 핵심 개념
# 
# **차원(dimension) vs 형태(shape)**:
# - **차원**: 텐서가 몇 차원인가? (스칼라=0D, 벡터=1D, 행렬=2D, ...)
# - **형태**: 각 차원의 크기는 얼마인가? (예: `[3, 224, 224]`)
# 
# **이미지 텐서의 규약**:
# - PyTorch: `(B, C, H, W)` - Batch, Channel, Height, Width
# - TensorFlow: `(B, H, W, C)` - 프레임워크마다 다름!
# - OpenCV: `(H, W, C)` - 배치 차원 없음
# 
# **음수 인덱싱 `-1`**:
# - 마지막 차원을 의미 (`shape[-1]` = 너비)
# - 배치 크기가 가변적일 때 유용 (항상 채널/너비는 고정)
# 
# #### 실전 활용
# 
# 1. **모델 디버깅**: `print(tensor.shape)`로 각 층의 출력 확인
# 2. **차원 검증**: `assert tensor.shape == (16, 3, 224, 224)`
# 3. **동적 배치 처리**: `batch_size = tensor.shape[0]`

# In[ ]:


print("3️⃣ Shape와 차원 정보")
print("=" * 50)

# 3차원 텐서 생성
tensor = torch.rand(3, 4, 5)
print(f"tensor shape: {tensor.shape}")        # torch.Size([3, 4, 5])
print(f"tensor.size(): {tensor.size()}")      # torch.Size([3, 4, 5])
print(f"tensor.ndim: {tensor.ndim}")          # 3
print()

# 특정 차원 크기 접근
print(f"첫 번째 차원: {tensor.shape[0]}")     # 3
print(f"두 번째 차원: {tensor.size(1)}")      # 4
print(f"마지막 차원: {tensor.size(-1)}")      # 5
print()

# 실전 예제: 이미지 텐서
image_batch = torch.rand(16, 3, 224, 224)  # 배치 크기 16, RGB 채널 3, 224×224 이미지
print(f"이미지 배치 shape: {image_batch.shape}")
print(f"배치 크기: {image_batch.shape[0]}")
print(f"채널 수: {image_batch.shape[1]}")
print(f"높이: {image_batch.shape[2]}, 너비: {image_batch.shape[3]}")


# ### 2.4 데이터 타입 변환
# 
# 텐서의 데이터 타입을 확인하고 변환하는 방법을 학습합니다.
# 
# #### 왜 타입 변환이 필요한가?
# 
# 1. **GPU 연산 호환성**: 대부분의 GPU 연산은 `float32`만 지원
# 2. **메모리 효율성**: `float32` vs `float64`는 2배 메모리 차이
# 3. **정밀도 트레이드오프**: 학습은 `float32`, 추론은 `int8`/`float16` 가능
# 4. **손실 계산**: `mean()` 등은 float 타입 필수
# 
# #### 타입 선택 가이드
# 
# | 타입 | 용도 | 메모리 | 정밀도 |
# |------|------|--------|--------|
# | `float32` | **기본 학습** (권장) | 4바이트 | 충분 |
# | `float64` | 고정밀도 계산 | 8바이트 | 매우 높음 |
# | `int64` | 인덱스, 레이블 | 8바이트 | - |
# | `int32` | 정수 연산 | 4바이트 | - |
# | `float16` | 대규모 모델 추론 | 2바이트 | 낮음 |
# 
# #### 실전 패턴
# 
# ```python
# # 패턴 1: 손실 계산 전 float 변환
# labels = labels.float()  # int → float
# loss = criterion(predictions, labels)
# 
# # 패턴 2: GPU 이동과 타입 변환 동시
# tensor = tensor.to(device=device, dtype=torch.float32)
# 
# # 패턴 3: 추론 최적화 (메모리 절약)
# model = model.half()  # float16으로 변환
# ```
# 
# #### 주의사항
# 
# ⚠️ **타입 불일치 에러**:
# ```python
# # 오류 발생
# torch.mean(torch.tensor([1, 2, 3]))  # int는 mean 불가
# 
# # 해결
# torch.mean(torch.tensor([1, 2, 3]).float())  # float 변환 필요
# ```

# In[ ]:


print("4️⃣ 데이터 타입 변환")
print("=" * 50)

# 정수 텐서 생성
tensor = torch.tensor([1, 2, 3])
print(f"원본 타입: {tensor.dtype}")  # torch.int64
print()

# 방법 1: 간편 메서드
tensor_float = tensor.float()  # torch.float32
tensor_int32 = tensor.int()    # torch.int32
print(f"float(): {tensor_float.dtype}")
print(f"int(): {tensor_int32.dtype}")
print()

# 방법 2: type() 메서드
tensor_long = tensor.type(torch.int64)
print(f"type(torch.int64): {tensor_long.dtype}")
print()

# 방법 3: to() 메서드 (GPU 이동과 함께 사용 가능)
tensor_double = tensor.to(torch.float64)
tensor_gpu = tensor.to(device=device, dtype=torch.float32)
print(f"to(torch.float64): {tensor_double.dtype}")
print(f"to(device, dtype): {tensor_gpu.dtype}, device: {tensor_gpu.device}")


# ### 2.5 GPU/CPU 간 이동
# 
# PyTorch에서는 `.to(device)` 메서드로 텐서를 GPU/CPU로 이동시킵니다.

# In[ ]:


print("5️⃣ GPU/CPU 간 텐서 이동")
print("=" * 50)

# CPU 텐서 생성
x_cpu = torch.rand(3, 3)
print(f"CPU 텐서: {x_cpu.device}")
print(x_cpu)
print()

# GPU로 이동
x_gpu = x_cpu.to(device)
print(f"GPU 텐서: {x_gpu.device}")
print(x_gpu)
print()

# ⚠️ GPU 텐서를 NumPy로 변환하려면 먼저 CPU로 이동해야 합니다
if x_gpu.device.type != 'cpu':
    x_back_to_cpu = x_gpu.cpu()
    x_numpy = x_back_to_cpu.numpy()
    print("GPU → CPU → NumPy 변환 완료")
else:
    x_numpy = x_gpu.numpy()
    print("CPU → NumPy 변환 완료")


# ### 2.6 텐서 연산 (Tensor Ops)
# 
# PyTorch 텐서는 NumPy와 비슷한 문법으로 연산할 수 있어요. (대부분 “NumPy에서 하던 것 = PyTorch에도 있다” 느낌)
# 
# - **원소별 연산**: `a + b`, `a * b`  ↔  NumPy도 동일(`a + b`, `a * b`), 브로드캐스팅도 같은 감각
# - **내적(벡터)**: `torch.dot(a, b)`  ↔  `np.dot(a, b)`
# - **행렬곱**: `A @ B` 또는 `torch.matmul(A, B)`  ↔  `A @ B` 또는 `np.matmul(A, B)`
# - **요약/통계(리덕션)**: `x.mean()`, `x.std()`, `x.max()`  ↔  `x.mean()`, `x.std()`, `x.max()` (NumPy와 거의 동일)
# - **reshape/view**: 둘 다 shape만 바꾸는 기능이며, `view`는 가능한 경우 메모리를 공유하는 “뷰”로 동작  ↔  NumPy의 `reshape`도 상황에 따라 view로 동작
# - **딥러닝에서 PyTorch만의 포인트**: 텐서는 `cuda`/`mps`에서 연산 가능 + 학습 시 `requires_grad=True`면 autograd가 연산을 기록해 `backward()`로 미분 가능
# - 참고: `xxx_()`는 in-place 연산인 경우가 많아(값이 즉시 바뀜) 디버깅/그래프 추적에서 주의

# In[ ]:


print("6️⃣ 텐서 연산")
print("=" * 50)

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# 기본 연산
print(f"덧셈: {a + b}")
print(f"곱셈: {a * b}")
print(f"내적: {torch.dot(a, b)}")
print()

# 행렬 연산
A = torch.rand(3, 4)
B = torch.rand(4, 2)
C = torch.matmul(A, B)  # 또는 A @ B
print(f"행렬 곱셈 (3×4) @ (4×2) = {C.shape}")
print(C)
print()

# 유용한 함수들
x = torch.randn(3, 4)  # 표준정규분포에서 샘플링
print(f"평균: {x.mean():.4f}")
print(f"표준편차: {x.std():.4f}")
print(f"최댓값: {x.max():.4f}")
print(f"최솟값: {x.min():.4f}")


# ### 2.7 Random 값 생성
# 
# 재현 가능한(reproducible) 랜덤 값 생성 방법을 학습합니다.
# 
# #### 왜 재현 가능성이 중요한가?
# 
# **과학적 방법론**: 동일한 실험을 반복하면 동일한 결과를 얻어야 함
# - 디버깅: 오류를 재현해야 수정 가능
# - 하이퍼파라미터 튜닝: 공정한 비교를 위해 같은 초기 조건 필요
# - 논문 재현: 다른 연구자가 결과를 검증할 수 있어야 함
# 
# #### 각 분포의 사용처
# 
# **1. `torch.rand()` - 균일 분포 (0~1)**
# - 드롭아웃: `if rand() > 0.5: drop`
# - 데이터 증강: 랜덤 회전 각도
# - 확률적 샘플링
# 
# **2. `torch.randn()` - 정규 분포 (평균=0, 분산=1)**
# - **가중치 초기화**: Xavier/He 초기화의 기반
# - 노이즈 생성: VAE, Diffusion 모델
# - 정규화된 입력 생성
# 
# **3. `torch.randint()` - 정수 난수**
# - 데이터 샘플링: 배치 인덱스 생성
# - 레이블 생성: 합성 데이터
# - 셔플링
# 
# #### 재현 가능성 체크리스트
# 
# ```python
# # ✅ 올바른 재현 가능성 설정
# torch.manual_seed(42)
# np.random.seed(42)
# random.seed(42)
# 
# # 🔧 GPU 사용 시 추가 필요
# torch.cuda.manual_seed(42)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# ```
# 
# #### 주의사항
# 
# ⚠️ **재현 불가 시나리오**:
# - DataLoader의 `num_workers > 0` (멀티프로세싱)
# - 서로 다른 GPU 환경
# - PyTorch 버전 차이

# In[ ]:


print("7️⃣ Random 값 생성")
print("=" * 50)

# 재현 가능성을 위한 시드 설정
torch.manual_seed(42)

# 균일 분포 (0~1 사이)
uniform = torch.rand(3, 4)
print(f"균일 분포:\n{uniform}")
print(f"범위: {uniform.min():.4f} ~ {uniform.max():.4f}")
print()

# 정수 난수 (0~99)
random_ints = torch.randint(low=0, high=100, size=(3, 4))
print(f"정수 난수:\n{random_ints}")
print()

# 정규 분포 (평균=0, 분산=1)
normal = torch.randn(3, 4)
print(f"정규 분포:\n{normal}")
print(f"평균: {normal.mean():.4f}, 표준편차: {normal.std():.4f}")


# ### 2.8 텐서 형태 변환: reshape vs view
# 
# `reshape()`와 `view()`의 차이점과 사용법을 학습합니다.
# 
# #### 핵심 차이점
# 
# | 특성 | `reshape()` | `view()` |
# |------|-------------|----------|
# | **동작** | 항상 성공 | contiguous 필요 |
# | **메모리** | 가끔 복사 | 복사 없음 (metadata만) |
# | **안전성** | ✅ 높음 | ⚠️ 낮음 (오류 가능) |
# | **권장** | **일반 용도** | 성능 최적화 필요 시 |
# 
# #### Contiguous Memory란?
# 
# **C-contiguous**: 메모리에서 연속적으로 배치됨
# ```
# [1, 2, 3, 4, 5, 6] → (2, 3) 형태로 해석 가능
# ```
# 
# **Non-contiguous**: 메모리가 흩어져 있음 (permute 후)
# ```
# permute 후: 메모리는 [1,4,2,5,3,6]처럼 섞임
# → view() 불가, contiguous() 필요
# ```
# 
# #### 실전 패턴
# 
# **패턴 1: 이미지 Flatten** (가장 흔함)
# ```python
# # CNN 출력 → FC 입력
# features = cnn_output.view(batch_size, -1)  # (B,C,H,W) → (B,C*H*W)
# # 또는 안전하게
# features = cnn_output.reshape(batch_size, -1)
# ```
# 
# **패턴 2: permute 후 reshape**
# ```python
# # 이미지 차원 변경 후 flatten
# x = x.permute(0, 2, 3, 1)  # (B,C,H,W) → (B,H,W,C)
# # x.view(B, -1)  # ❌ 오류!
# x = x.contiguous().view(B, -1)  # ✅ 정상
# # 또는
# x = x.reshape(B, -1)  # ✅ 항상 안전
# ```
# 
# #### 성능 고려사항
# 
# - `view()`: metadata만 변경 → 매우 빠름
# - `reshape()`: contiguous 체크 → contiguous면 view, 아니면 복사
# - `contiguous()`: 메모리 복사 발생 → 느림
# 
# **권장**: 특별한 이유 없으면 `reshape()` 사용 (안전성 > 성능)
# 
# #### -1의 의미 (자동 차원 추론)
# 
# ```python
# x = torch.arange(12)
# 
# x.view(3, 4)    # 명시적
# x.view(3, -1)   # -1 = 4 (자동 계산)
# x.view(-1, 4)   # -1 = 3 (자동 계산)
# 
# # 배치 처리에서 유용
# x.view(batch_size, -1)  # 배치 크기만 지정, 나머지 자동
# ```

# In[ ]:


print("8️⃣ reshape vs view")
print("=" * 50)

# 1차원 텐서 생성
tensor = torch.arange(12)
print(f"원본: {tensor.shape}")  # (12,)
print(tensor)
print()

# reshape: 항상 동작
reshaped = tensor.reshape(3, 4)
print(f"reshape 결과: {reshaped.shape}")  # (3, 4)
print(reshaped)
print()

# view: contiguous memory에서만 동작
viewed = tensor.view(3, 4)
print(f"view 결과: {viewed.shape}")  # (3, 4)
print(viewed)
print()

# -1로 자동 차원 추론
auto_shaped = tensor.view(3, -1)  # (3, 4)
print(f"자동 추론: {auto_shaped.shape}")
print()

# contiguous 여부 확인
print(f"연속 메모리? {tensor.is_contiguous()}")  # True
print()

# ⚠️ permute 후에는 view 사용 주의
matrix = tensor.view(3, 4).permute(1, 0)
print(f"permute 후 연속 메모리? {matrix.is_contiguous()}")  # False
# flattened = matrix.view(-1)  # 오류 발생!
flattened = matrix.contiguous().view(-1)  # 정상 동작
print(f"contiguous() 후 view: {flattened.shape}")


# ### 2.9 차원 재배열: permute, transpose, t
# 
# 텐서의 차원 순서를 변경하는 다양한 방법을 학습합니다.
# 
# #### 왜 차원 재배열이 필요한가?
# 
# **이미지 포맷 전쟁**:
# - TensorFlow/Keras: `(B, H, W, C)` - "channels-last"
# - PyTorch: `(B, C, H, W)` - "channels-first"
# - OpenCV/PIL: `(H, W, C)` - 배치 없음
# - NumPy 배열: `(H, W, C)` - 일반적
# 
# → 데이터 로드/전처리/모델 입력 시 형식 변환 필수!
# 
# #### 각 함수의 역할
# 
# **`permute(dims)`**: 임의 차원 재배열 (가장 범용적)
# ```python
# # 이미지 로드: (H, W, C) → PyTorch: (C, H, W)
# image = torch.rand(224, 224, 3)
# pytorch_image = image.permute(2, 0, 1)  # (3, 224, 224)
# 
# # 배치 이미지: (B, H, W, C) → (B, C, H, W)
# batch = torch.rand(16, 224, 224, 3)
# batch = batch.permute(0, 3, 1, 2)  # (16, 3, 224, 224)
# ```
# 
# **`transpose(dim0, dim1)`**: 두 차원만 교환 (제한적)
# ```python
# # H와 W만 교환
# tensor = torch.rand(16, 3, 224, 224)
# swapped = tensor.transpose(2, 3)  # (16, 3, 224, 224)
# ```
# 
# **`t()`**: 2D 행렬 전치만 (매우 제한적)
# ```python
# matrix = torch.rand(3, 4)
# transposed = matrix.t()  # (4, 3)
# ```
# 
# #### 성능 및 메모리 주의사항
# 
# ⚠️ **permute는 non-contiguous를 만듦**:
# ```python
# x = torch.rand(3, 224, 224)
# x_perm = x.permute(1, 2, 0)
# 
# print(x_perm.is_contiguous())  # False
# 
# # view() 사용 불가!
# # x_perm.view(-1)  # RuntimeError!
# 
# # 해결 방법
# x_perm.contiguous().view(-1)  # ✅ 작동
# x_perm.reshape(-1)             # ✅ 작동 (권장)
# ```
# 
# #### 실전 패턴
# 
# **패턴 1: 데이터 로더에서 자동 변환**
# ```python
# from torchvision import transforms
# 
# transform = transforms.Compose([
#     transforms.ToTensor(),  # 자동으로 (H,W,C) → (C,H,W)
# ])
# ```
# 
# **패턴 2: 배치 처리**
# ```python
# # NumPy 배열 → PyTorch (배치 포함)
# images_np = np.array([...])  # (B, H, W, C)
# images = torch.from_numpy(images_np).permute(0, 3, 1, 2)  # (B, C, H, W)
# ```
# 
# **패턴 3: 시각화**
# ```python
# # PyTorch → matplotlib (H, W, C 필요)
# image = torch.rand(3, 224, 224)
# plt.imshow(image.permute(1, 2, 0))  # (224, 224, 3)
# ```

# In[ ]:


print("9️⃣ 차원 재배열")
print("=" * 50)

# 이미지 텐서 차원 변경: (H, W, C) -> (C, H, W)
image = torch.rand(224, 224, 3)
image_chw = image.permute(2, 0, 1)
print(f"이미지 차원 변환: {image.shape} → {image_chw.shape}")
print()

# 배치 이미지: (B, H, W, C) -> (B, C, H, W)
batch_images = torch.rand(16, 224, 224, 3)
batch_chw = batch_images.permute(0, 3, 1, 2)
print(f"배치 이미지 변환: {batch_images.shape} → {batch_chw.shape}")
print()

# transpose: 두 차원만 교환
tensor = torch.rand(16, 3, 224, 224)
swapped = tensor.transpose(2, 3)  # H와 W 교환
print(f"transpose 결과: {tensor.shape} → {swapped.shape}")
print()

# t(): 2D 전치
matrix = torch.rand(3, 4)
transposed = matrix.t()
print(f"전치 행렬: {matrix.shape} → {transposed.shape}")
print()

# ⚠️ permute 후 contiguous 주의
print(f"permute 후 연속 메모리? {image_chw.is_contiguous()}")


# ### 2.10 차원 추가/제거: squeeze, unsqueeze
# 
# 배치 처리를 위한 차원 조작 기법을 학습합니다.
# 
# #### 왜 배치 차원이 필수인가?
# 
# **모델은 배치 입력을 가정**:
# - 신경망은 병렬 처리를 위해 배치 단위로 동작
# - 단일 샘플도 `(1, ...)`처럼 배치 차원 필요
# - 배치 정규화(Batch Normalization)는 배치 통계에 의존
# 
# #### squeeze vs unsqueeze
# 
# **`unsqueeze(dim)`: 차원 추가**
# ```python
# # 단일 이미지 → 배치 (모델 입력용)
# image = torch.rand(3, 224, 224)      # (C, H, W)
# batch = image.unsqueeze(0)            # (1, C, H, W)
# # model(image)  # ❌ 오류: 배치 차원 없음
# model(batch)    # ✅ 정상
# ```
# 
# **`squeeze(dim)`: 차원 제거**
# ```python
# # 배치 출력 → 단일 결과
# batch_output = model(batch)           # (1, num_classes)
# single_output = batch_output.squeeze(0)  # (num_classes,)
# ```
# 
# #### 주의사항
# 
# ⚠️ **squeeze() 위험성**:
# ```python
# x = torch.rand(1, 3, 1, 224, 224)
# 
# # ❌ 위험: 모든 크기 1 차원 제거
# x.squeeze()  # (3, 224, 224) - 의도와 다를 수 있음!
# 
# # ✅ 안전: 명시적으로 차원 지정
# x.squeeze(dim=0)  # (3, 1, 224, 224)
# x.squeeze(dim=2)  # (1, 3, 224, 224)
# ```
# 
# #### 실전 버그 패턴
# 
# **버그 1: 추론 시 배치 차원 누락**
# ```python
# # ❌ 잘못된 코드
# image = load_image()  # (3, 224, 224)
# output = model(image)  # RuntimeError!
# 
# # ✅ 올바른 코드
# image = load_image().unsqueeze(0)  # (1, 3, 224, 224)
# output = model(image)
# prediction = output.squeeze(0)  # (num_classes,)
# ```
# 
# **버그 2: 배치 정규화 오류**
# ```python
# # BatchNorm은 배치 차원 필요
# bn = nn.BatchNorm2d(3)
# x = torch.rand(3, 224, 224)  # (C, H, W)
# # bn(x)  # ❌ 오류: 배치 차원 없음
# 
# x = x.unsqueeze(0)  # (1, C, H, W)
# bn(x)  # ✅ 정상
# ```

# In[ ]:


print("🔟 squeeze / unsqueeze")
print("=" * 50)

# squeeze: 크기 1인 차원 제거
batch_of_one = torch.rand(1, 3, 224, 224)
single_image = batch_of_one.squeeze(dim=0)
print(f"squeeze 후: {batch_of_one.shape} → {single_image.shape}")
print()

# unsqueeze: 차원 추가
image = torch.rand(3, 224, 224)
batch = image.unsqueeze(dim=0)
print(f"unsqueeze 후: {image.shape} → {batch.shape}")
print()

# 실전 활용: 단일 이미지를 배치로 만들기
single_img = torch.rand(3, 64, 64)
img_batch = single_img.unsqueeze(0)  # 모델 입력용
print(f"모델 입력 준비: {single_img.shape} → {img_batch.shape}")
# predictions = model(img_batch)
# prediction_single = predictions.squeeze(0)  # 배치 차원 제거
print()

# 여러 차원 동시 제거
tensor_with_ones = torch.rand(1, 3, 1, 224, 224)
squeezed = tensor_with_ones.squeeze()  # 모든 크기 1 차원 제거
print(f"모두 squeeze: {tensor_with_ones.shape} → {squeezed.shape}")


# ### 2.11 Aggregation 연산 심화
# 
# `dim` 인자를 활용한 차원별 집계 연산을 마스터합니다.
# 
# #### dim 파라미터의 의미
# 
# **"특정 축을 따라 축약한다"**:
# ```python
# data = torch.arange(24).view(2, 3, 4)  # (2, 3, 4)
# 
# # dim=0: 첫 번째 차원을 축약 → (3, 4)
# data.sum(dim=0)  # 2개 요소를 더함
# 
# # dim=1: 두 번째 차원을 축약 → (2, 4)
# data.sum(dim=1)  # 3개 요소를 더함
# 
# # dim=2: 세 번째 차원을 축약 → (2, 3)
# data.sum(dim=2)  # 4개 요소를 더함
# ```
# 
# **다중 dim**: 여러 차원 동시 축약
# ```python
# data.sum(dim=(1, 2))  # → (2,)
# ```
# 
# #### keepdim=True의 중요성
# 
# **브로드캐스트 호환성 유지**:
# ```python
# x = torch.rand(16, 3, 224, 224)
# 
# # keepdim=False (기본)
# mean = x.mean(dim=1)  # (16, 224, 224) - 차원 감소
# # x - mean  # ❌ 브로드캐스트 불가!
# 
# # keepdim=True
# mean = x.mean(dim=1, keepdim=True)  # (16, 1, 224, 224)
# x - mean  # ✅ 브로드캐스트 가능
# ```
# 
# #### 실전 활용 패턴
# 
# **패턴 1: 배치 평균 (손실 함수)**
# ```python
# losses = criterion(predictions, targets, reduction='none')  # (B,)
# loss = losses.mean()  # 배치 평균
# ```
# 
# **패턴 2: 채널별 통계 (정규화)**
# ```python
# # 배치 정규화: 배치와 공간 차원에서 평균/분산
# images = torch.rand(16, 3, 224, 224)  # (B, C, H, W)
# mean = images.mean(dim=(0, 2, 3), keepdim=True)  # (1, 3, 1, 1)
# std = images.std(dim=(0, 2, 3), keepdim=True)
# normalized = (images - mean) / (std + 1e-5)
# ```
# 
# **패턴 3: argmax (분류)**
# ```python
# logits = torch.rand(4, 10)  # (B, num_classes)
# predicted_classes = logits.argmax(dim=1)  # (B,)
# # dim=1: 각 샘플에서 최대 클래스 인덱스
# ```
# 
# **패턴 4: 계층 정규화 (LayerNorm)**
# ```python
# # 마지막 차원에서 정규화
# features = torch.rand(16, 512)  # (B, D)
# mean = features.mean(dim=-1, keepdim=True)  # (B, 1)
# std = features.std(dim=-1, keepdim=True)
# normalized = (features - mean) / (std + 1e-5)
# ```
# 
# #### 흔한 실수
# 
# ❌ **차원 인덱스 오류**:
# ```python
# x = torch.rand(16, 3, 224, 224)  # (B, C, H, W)
# 
# # 잘못: 공간 평균을 구하려 했으나...
# x.mean(dim=(1, 2))  # (16, 224) - 채널과 높이를 축약!
# 
# # 올바름
# x.mean(dim=(2, 3))  # (16, 3) - 공간 차원 축약
# ```

# In[ ]:


print("1️⃣1️⃣ Aggregation 연산 심화")
print("=" * 50)

# 3차원 텐서 생성
data = torch.arange(24).view(2, 3, 4)
print(f"원본 shape: {data.shape}")  # (2, 3, 4)
print(data)
print()

# 전체 합계
total = data.sum()
print(f"전체 합: {total}")
print()

# 특정 차원 따라 합계
sum_0 = data.sum(dim=0)  # (3, 4) - 첫 번째 차원 제거
sum_1 = data.sum(dim=1)  # (2, 4) - 두 번째 차원 제거
sum_2 = data.sum(dim=2)  # (2, 3) - 세 번째 차원 제거

print(f"dim=0 합계 shape: {sum_0.shape}")
print(f"dim=1 합계 shape: {sum_1.shape}")
print(f"dim=2 합계 shape: {sum_2.shape}")
print()

# 다중 차원 합계
sum_12 = data.sum(dim=(1, 2))  # (2,) - 두 차원 동시 제거
print(f"dim=(1,2) 합계 shape: {sum_12.shape}")
print(f"결과: {sum_12}")
print()

# max는 값과 인덱스 반환
max_values, max_indices = data.max(dim=1)
print(f"최댓값 shape: {max_values.shape}, 인덱스 shape: {max_indices.shape}")
print()

# argmax: 인덱스만 반환
argmax_indices = data.argmax(dim=1)
print(f"argmax shape: {argmax_indices.shape}")
print()

# mean (주의: float 타입 필요)
mean_values = data.float().mean(dim=0)
print(f"평균 shape: {mean_values.shape}")


# In[ ]:


# 실전 예제: 분류 모델 예측
print("실전 예제: 분류 모델 예측")
print("=" * 50)

# 분류 모델 출력 (batch_size=4, num_classes=10)
torch.manual_seed(42)
logits = torch.rand(4, 10)

# 각 샘플의 최고 확률 클래스 찾기
predicted_classes = logits.argmax(dim=1)
print(f"예측 클래스: {predicted_classes}")  # tensor([7, 2, 9, 1])
print(f"각 클래스의 확률:\n{logits}")


# In[ ]:


# ✅ axis(dim) 때문에 결과가 달라지는 아주 쉬운 예제
# - NumPy의 axis == PyTorch의 dim
# - "어느 축을 줄이느냐(=합/평균을 어디 방향으로 모으느냐)"에 따라 결과의 의미/shape가 바뀝니다.
print("axis(dim) 예제: 축을 바꾸면 결과가 달라진다")
print("=" * 50)

# 이 셀은 단독 실행해도 되도록 import를 여기서 합니다.
import numpy as np
import torch

# ------------------------------------------------------------
# 1) 2D 예제 (가장 기본)
# ------------------------------------------------------------
# x.shape=(2,3) 이면 (행 2개, 열 3개)
# - axis/dim=0: 0번 축(행 축)을 "줄인다" => 각 열을 행 방향으로 모아 (3,) 결과
# - axis/dim=1: 1번 축(열 축)을 "줄인다" => 각 행을 열 방향으로 모아 (2,) 결과
#   (즉, axis/dim 값은 "남길 축"이 아니라 "줄일 축"을 고르는 값)
x_np = np.array([[1, 2, 3],
                 [4, 5, 6]])  # shape=(2,3)
x_t = torch.tensor(x_np)  # NumPy→Tensor (여기서는 단순 예제라 복사/공유는 신경 X)
print("\n[2D] x.shape:", x_np.shape)
print(x_np)

print("\nNumPy sum(axis=0)  (열별 합):", x_np.sum(axis=0))
print("NumPy sum(axis=1)  (행별 합):", x_np.sum(axis=1))
print("Torch sum(dim=0)  (열별 합):", x_t.sum(dim=0))
print("Torch sum(dim=1)  (행별 합):", x_t.sum(dim=1))

# (팁) 차원을 유지하고 싶으면
# - NumPy: keepdims=True
# - PyTorch: keepdim=True
# 예) x_t.sum(dim=1, keepdim=True) -> shape (2,1)

# ------------------------------------------------------------
# 2) 딥러닝에서 자주 보는 4D 이미지 텐서 (B, C, H, W)
# ------------------------------------------------------------
# - B: 배치, C: 채널, H/W: 높이/너비(공간축)
# - dim=(2,3) => H,W만 평균내서 "이미지 1장당 채널별 평균" -> (B,C)로 요약
#   (Global Average Pooling(GAP)과 같은 감각)
img = torch.arange(2 * 3 * 2 * 2).view(2, 3, 2, 2)  # (B=2,C=3,H=2,W=2)
print("\n[4D] img.shape:", tuple(img.shape))
gap = img.float().mean(dim=(2, 3))  # H,W 축을 줄여서 (B,C)
print("img.mean(dim=(2,3)) -> (B,C) shape:", tuple(gap.shape))
print(gap)


# ### 2.12 고급 인덱싱
# 
# Boolean indexing, Fancy indexing 등 고급 인덱싱 기법을 학습합니다.
# 
# #### 고급 인덱싱이란?
# 
# **기본 인덱싱**: 정수, 슬라이스 사용
# ```python
# x[0], x[:5], x[1:3]
# ```
# 
# **고급 인덱싱**: 조건, 배열 사용
# - **Boolean indexing**: 조건으로 필터링
# - **Fancy indexing**: 정수 배열로 선택
# - **Masking**: 특정 요소만 연산
# 
# #### Boolean Indexing의 특성
# 
# ⚠️ **PyTorch는 1D 텐서 반환**:
# ```python
# matrix = torch.arange(20).view(4, 5)
# mask = matrix > 10
# 
# filtered = matrix[mask]  # (9,) - 1D!
# # 원본 차원 유지 안 됨
# 
# # 차원 유지 필요 시: torch.where()
# result = torch.where(mask, matrix, torch.tensor(0))  # (4, 5)
# ```
# 
# **NumPy와의 차이**:
# - NumPy: 원본 shape 유지 가능
# - PyTorch: 항상 1D flatten
# 
# **이유**: 어떤 요소가 선택될지 컴파일 타임에 불명확 → 동적 shape
# 
# #### Fancy Indexing
# 
# **정수 텐서로 인덱싱**:
# ```python
# matrix = torch.arange(20).view(4, 5)
# indices = torch.tensor([0, 2, 3])
# 
# selected = matrix[indices]  # (3, 5) - 0, 2, 3번째 행 선택
# ```
# 
# #### 실전 활용
# 
# **패턴 1: 패딩 마스크 (NLP)**
# ```python
# # 문장 길이가 다를 때 패딩 무시
# sentences = torch.randint(0, 1000, (16, 50))  # (B, seq_len)
# padding_mask = sentences == 0  # 패딩 토큰
# 
# # 주의(Attention) 계산 시 패딩 마스킹
# attention = torch.rand(16, 50, 50)
# attention = attention.masked_fill(padding_mask.unsqueeze(1), -1e9)
# ```
# 
# **패턴 2: 신뢰도 필터링 (객체 탐지)**
# ```python
# boxes = torch.rand(100, 4)      # (N, 4) - bbox
# scores = torch.rand(100)        # (N,) - confidence
# 
# # 높은 신뢰도만 선택
# high_conf = scores > 0.5
# filtered_boxes = boxes[high_conf]  # (M, 4) where M < N
# filtered_scores = scores[high_conf]
# ```
# 
# **패턴 3: 클래스 불균형 처리**
# ```python
# labels = torch.randint(0, 10, (1000,))
# 
# # 특정 클래스만 선택
# rare_class = 9
# rare_samples = labels == rare_class
# rare_indices = torch.where(rare_samples)[0]
# 
# # 오버샘플링
# oversampled = torch.cat([labels, labels[rare_indices]])
# ```
# 
# #### gather / scatter (고급)
# 
# **Fancy indexing의 확장**:
# ```python
# # gather: 특정 인덱스에서 값 수집
# logits = torch.rand(4, 10)  # (B, num_classes)
# targets = torch.tensor([3, 5, 2, 7])  # (B,)
# 
# # targets 위치의 logits 추출
# selected = logits.gather(1, targets.unsqueeze(1))  # (4, 1)
# 
# # scatter: 특정 인덱스에 값 할당
# output = torch.zeros(4, 10)
# output.scatter_(1, targets.unsqueeze(1), 1.0)  # one-hot encoding
# ```

# In[ ]:


print("1️⃣2️⃣ 고급 인덱싱")
print("=" * 50)

# 2차원 텐서 생성
matrix = torch.arange(20).view(4, 5)
print("원본 행렬:")
print(matrix)
print()

# 정수 인덱싱
element = matrix[0, 0]  # 0
value = matrix[1, 2]    # 7
print(f"matrix[0,0] = {element}, matrix[1,2] = {value}")
print()

# 슬라이싱
first_row = matrix[0, :]     # 첫 번째 행
first_col = matrix[:, 0]     # 첫 번째 열
partial = matrix[0, 1:4]     # 일부 열
print(f"첫 행: {first_row}")
print(f"첫 열: {first_col}")
print(f"일부 선택: {partial}")
print()

# Fancy indexing
indices = torch.tensor([0, 2, 3])
selected_rows = matrix[indices]  # 0, 2, 3번째 행 선택
print(f"선택된 행 shape: {selected_rows.shape}")
print(f"선택된 행:\n{selected_rows}")
print()

# Boolean indexing (주의: PyTorch는 1D 반환)
mask = matrix > 10
filtered = matrix[mask]  # 1D tensor
print(f"10보다 큰 값들 (1D): {filtered}")
print()

# torch.where (차원 유지)
result = torch.where(matrix > 10, matrix, torch.tensor(0))
print(f"torch.where 결과 shape: {result.shape}")  # (4, 5) 유지
print(f"torch.where 결과:\n{result}")


# ### 2.13 행렬 연산 심화
# 
# `dot()`과 `matmul()`의 차이점과 배치 행렬곱을 학습합니다.
# 
# #### 수학적 의미
# 
# **`torch.dot(v1, v2)`**: 벡터 내적 (scalar 반환)
# ```
# v1 = [1, 2, 3]
# v2 = [4, 5, 6]
# dot = 1*4 + 2*5 + 3*6 = 32
# ```
# - **제약**: 1D 벡터만 가능
# - **출력**: 스칼라
# 
# **`torch.matmul(A, B)`**: 행렬곱 (행렬 반환)
# - **1D × 1D**: 벡터 내적 (dot과 동일)
# - **2D × 2D**: 행렬곱
# - **3D+ × 3D+**: 배치 행렬곱 (마지막 2차원만)
# 
# #### matmul vs mm vs @
# 
# | 함수 | 지원 차원 | 브로드캐스트 |
# |------|-----------|--------------|
# | `matmul()` | 1D, 2D, 3D+ | ✅ 지원 |
# | `mm()` | 2D만 | ❌ 불가 |
# | `@` 연산자 | 1D, 2D, 3D+ | ✅ 지원 |
# 
# **권장**: `@` 연산자 또는 `matmul()` 사용
# 
# #### 배치 행렬곱의 브로드캐스팅
# 
# **규칙**: 마지막 2차원은 행렬곱, 앞 차원은 브로드캐스트
# 
# ```python
# # (B, M, N) @ (N, P) → (B, M, P)
# batch_a = torch.rand(2, 3, 4)
# matrix_b = torch.rand(4, 5)
# result = batch_a @ matrix_b  # (2, 3, 5)
# 
# # (M, N) @ (B, N, P) → (B, M, P)
# matrix_a = torch.rand(3, 4)
# batch_b = torch.rand(2, 4, 5)
# result = matrix_a @ batch_b  # (2, 3, 5)
# ```
# 
# **장점**: 배치 루프 불필요 → GPU 병렬화
# 
# #### 실전 활용
# 
# **패턴 1: 선형층 (Fully Connected)**
# ```python
# # 배치 입력
# x = torch.rand(16, 512)  # (B, D_in)
# W = torch.rand(512, 256)  # (D_in, D_out)
# 
# # 행렬곱
# output = x @ W  # (16, 256) = (B, D_out)
# 
# # nn.Linear는 내부적으로 이것을 수행
# linear = nn.Linear(512, 256)
# output = linear(x)  # 동일
# ```
# 
# **패턴 2: 주의 메커니즘 (Attention)**
# ```python
# # Query, Key, Value
# Q = torch.rand(16, 10, 64)  # (B, T, D)
# K = torch.rand(16, 10, 64)
# V = torch.rand(16, 10, 64)
# 
# # Attention scores
# scores = Q @ K.transpose(-2, -1)  # (16, 10, 10) = (B, T, T)
# attn = torch.softmax(scores / 8, dim=-1)
# 
# # Weighted sum
# output = attn @ V  # (16, 10, 64) = (B, T, D)
# ```
# 
# **패턴 3: 이미지 특징 변환**
# ```python
# # CNN 출력 → FC 입력
# images = torch.rand(16, 3, 32, 32)  # (B, C, H, W)
# features = images.view(16, -1)       # (16, 3072) flatten
# 
# # 차원 축소
# W = torch.rand(3072, 128)
# transformed = features @ W  # (16, 128)
# ```
# 
# #### einsum: 고급 다중 차원 연산
# 
# **복잡한 텐서 연산을 간결하게**:
# ```python
# # 행렬곱
# A = torch.rand(3, 4)
# B = torch.rand(4, 5)
# C = torch.einsum('ik,kj->ij', A, B)  # (3, 5)
# 
# # 배치 행렬곱
# A = torch.rand(2, 3, 4)
# B = torch.rand(2, 4, 5)
# C = torch.einsum('bik,bkj->bij', A, B)  # (2, 3, 5)
# 
# # 주의 메커니즘
# Q = torch.rand(16, 10, 64)
# K = torch.rand(16, 10, 64)
# scores = torch.einsum('btd,bsd->bts', Q, K)  # (16, 10, 10)
# ```

# In[ ]:


print("1️⃣3️⃣ 행렬 연산 심화")
print("=" * 50)

# dot: 1D 벡터 내적만 가능
vector1 = torch.tensor([1.0, 2.0, 3.0])
vector2 = torch.tensor([4.0, 5.0, 6.0])
dot_product = torch.dot(vector1, vector2)  # 1*4 + 2*5 + 3*6 = 32
print(f"내적: {dot_product}")
print()

# matmul: 2D 행렬곱
matrix_a = torch.rand(3, 4)
matrix_b = torch.rand(4, 5)
result = torch.matmul(matrix_a, matrix_b)  # (3, 5)
print(f"행렬곱 결과: {result.shape}")
print()

# 배치 행렬곱: (batch, m, n) @ (batch, n, p) = (batch, m, p)
batch_a = torch.rand(2, 3, 4)  # 배치 크기 2
batch_b = torch.rand(2, 4, 5)  # 배치 크기 2
batch_result = torch.matmul(batch_a, batch_b)  # (2, 3, 5)
print(f"배치 행렬곱: {batch_result.shape}")
print()

# 실전 예제: 이미지 특징 변환
print("실전 예제: 이미지 특징 변환")
print("=" * 50)
batch_images = torch.rand(16, 3, 32, 32)  # 배치 16개
features = batch_images.view(16, -1)       # (16, 3072) Flatten
weight_matrix = torch.rand(3072, 128)      # 변환 가중치
transformed = torch.matmul(features, weight_matrix)  # (16, 128)
print(f"원본 이미지: {batch_images.shape}")
print(f"Flatten: {features.shape}")
print(f"변환 가중치: {weight_matrix.shape}")
print(f"변환된 특징: {transformed.shape}")


# ### 2.14 Autograd: 자동 미분
# 
# PyTorch의 핵심 기능인 autograd는 자동으로 그래디언트를 계산합니다.
# `requires_grad=True`로 설정하면 연산이 추적됩니다.

# In[ ]:


print("Autograd: 자동 미분")
print("=" * 50)

# 그래디언트 추적 활성화
x = torch.tensor([2.0], requires_grad=True)
print(f"x = {x}")
print(f"requires_grad: {x.requires_grad}")
print()

# 연산 수행: y = x^2 + 3x + 1
y = x**2 + 3*x + 1
print(f"y = x^2 + 3x + 1 = {y.item():.4f}")
print()

# 역전파 수행
y.backward()  # dy/dx 계산

# 그래디언트 확인
print(f"dy/dx = 2x + 3 = {x.grad.item():.4f}")
print(f"x=2일 때 이론값: 2*2 + 3 = 7")
print()

# 💡 더 복잡한 예제
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x.sum() ** 2  # (x1 + x2 + x3)^2
y.backward()

print("복잡한 예제: y = (x1 + x2 + x3)^2")
print(f"x = {x.data}")
print(f"y = {y.item():.4f}")
print(f"dy/dx = {x.grad}")
print(f"이론값: 2*(1+2+3) = 12 (모든 원소에 동일)")


# ## 3. PyTorch vs TensorFlow 비교
# 
# ### 3.1 주요 차이점 요약

# | 항목 | TensorFlow/Keras | PyTorch |
# |------|------------------|----------|
# | **철학** | High-level API (간결함) | Low-level 제어 (유연성) |
# | **모델 정의** | `Sequential` / `Functional API` | `nn.Module` 클래스 |
# | **데이터 로드** | `tf.data.Dataset` | `DataLoader` |
# | **손실 함수** | `'sparse_categorical_crossentropy'` | `nn.CrossEntropyLoss()` |
# | **최적화** | `tf.keras.optimizers.Adam()` | `torch.optim.Adam()` |
# | **학습 루프** | `model.fit()` (암묵적) | 직접 작성 (명시적) |
# | **디버깅** | 그래프 모드 (어려움) | Eager 실행 (쉬움) |
# | **산업 표준** | 프로덕션 배포 강점 | 연구/실험 강점 |

# ### 3.2 코드 비교 예제

# In[ ]:


print("PyTorch vs TensorFlow 코드 비교")
print("=" * 50)
print()
print("【TensorFlow/Keras】")
print("""
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# 모델 정의
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])

# 컴파일
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 학습 (한 줄!)
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)
""")
print()
print("=" * 50)
print()
print("【PyTorch】")
print("""
import torch.nn as nn
import torch.optim as optim

# 모델 정의
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MyModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 학습 루프 (직접 작성)
for epoch in range(10):
    for inputs, labels in train_loader:
        optimizer.zero_grad()        # 그래디언트 초기화
        outputs = model(inputs)      # Forward
        loss = criterion(outputs, labels)  # 손실 계산
        loss.backward()              # Backward
        optimizer.step()             # 가중치 업데이트
""")
print()
print("=" * 50)
print("💡 PyTorch는 학습 루프를 직접 작성하여 더 많은 제어권을 제공합니다.")


# ## 4. 간단한 선형 회귀 예제
# 
# PyTorch의 기본 패턴을 익히기 위해 간단한 선형 회귀를 구현해봅시다.
# 
# **목표:** y = 2x + 1 관계를 학습

# ### 4.1 데이터 생성

# In[ ]:


# 시드 설정 (재현 가능성)
torch.manual_seed(42)
np.random.seed(42)

# 데이터 생성: y = 2x + 1 + noise
X = np.random.randn(100, 1).astype(np.float32)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1).astype(np.float32)

# PyTorch 텐서로 변환
X_tensor = torch.from_numpy(X).to(device)
y_tensor = torch.from_numpy(y).to(device)

# 시각화
plt.figure(figsize=(8, 5))
plt.scatter(X, y, alpha=0.6, s=50, edgecolors='k')
plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Linear Regression Data (y = 2x + 1 + noise)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"데이터 shape: X={X_tensor.shape}, y={y_tensor.shape}")
print(f"디바이스: {X_tensor.device}")


# ### 4.2 모델 정의

# In[ ]:


import torch.nn as nn

class LinearRegression(nn.Module):
    """간단한 선형 회귀 모델: y = wx + b"""
    def __init__(self):
        super(LinearRegression, self).__init__()
        # 1개 입력 → 1개 출력
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        """순전파"""
        return self.linear(x)

# 모델 생성 및 GPU로 이동
model = LinearRegression().to(device)

print("모델 구조:")
# 모델 요약 정보 출력
summary(model, input_size=(1, 1),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        row_settings=["var_names", "depth"])

print("\n초기 파라미터:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.data}")


# ### 4.3 학습 루프

# In[ ]:


# 학습 루프 시작 전 장치 상태 강제 확인
model = model.to(device)
X_tensor = X_tensor.to(device)
y_tensor = y_tensor.to(device)


# In[ ]:


import torch.optim as optim

# 손실 함수와 옵티마이저
criterion = nn.MSELoss()  # Mean Squared Error
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 학습
num_epochs = 100
losses = []

print("학습 시작...")
print("=" * 50)

for epoch in range(num_epochs):
    # 1. Forward pass
    predictions = model(X_tensor)
    loss = criterion(predictions, y_tensor)

    # 2. Backward pass
    optimizer.zero_grad()  # 그래디언트 초기화 (중요!)
    loss.backward()        # 역전파
    optimizer.step()       # 파라미터 업데이트

    # 3. 손실 기록 (MPS 호환: detach().cpu() 추가)
    losses.append(loss.detach().cpu().item())

    # 4. 진행 상황 출력
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("=" * 50)
print("학습 완료!")
print()
print("최종 파라미터:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.data}")
print()
print("💡 기대값: weight ≈ 2.0, bias ≈ 1.0")


# ### 4.4 결과 시각화

# In[ ]:


# 학습 곡선
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 손실 곡선
axes[0].plot(losses, linewidth=2, color='royalblue')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss (MSE)', fontsize=12)
axes[0].set_title('Training curve', fontsize=14)
axes[0].grid(True, alpha=0.3)

# 예측 결과
model.eval()  # 평가 모드
with torch.no_grad():
    y_pred = model(X_tensor).cpu().numpy()

axes[1].scatter(X, y, alpha=0.6, s=50, label='Real data', edgecolors='k')
axes[1].plot(X, y_pred, color='red', linewidth=2, label='Predicted line')
axes[1].set_xlabel('X', fontsize=12)
axes[1].set_ylabel('y', fontsize=12)
axes[1].set_title('Prediction results', fontsize=14)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 성능 평가
final_loss = losses[-1]
print(f"최종 손실: {final_loss:.6f}")
print(f"\n✅ 모델이 y = 2x + 1 관계를 성공적으로 학습했습니다!")


# ## 5. PyTorch 학습 루프 패턴 정리
# 
# PyTorch의 기본 학습 루프는 다음 5단계로 구성됩니다:

# In[ ]:


print("PyTorch 학습 루프 표준 패턴")
print("=" * 50)
print("""
for epoch in range(num_epochs):
    # 1️⃣ Forward Pass
    predictions = model(inputs)
    loss = criterion(predictions, labels)

    # 2️⃣ 그래디언트 초기화 (중요!)
    optimizer.zero_grad()

    # 3️⃣ Backward Pass
    loss.backward()

    # 4️⃣ 파라미터 업데이트
    optimizer.step()

    # 5️⃣ 손실 기록 (선택)
    losses.append(loss.detach().cpu().item())  # MPS 호환
""")
print("=" * 50)
print()
print("💡 핵심 포인트:")
print("   - optimizer.zero_grad()를 매 반복마다 호출해야 합니다")
print("   - loss.backward()가 자동으로 그래디언트를 계산합니다")
print("   - optimizer.step()이 가중치를 업데이트합니다")
print("   - MPS 사용 시 loss 기록에 detach().cpu() 추가")


# ## 6. 요약 및 다음 단계
# 
# ### ✅ 학습한 내용
# 1. PyTorch 환경 설정 및 GPU/MPS 활용
# 2. 텐서 생성 및 기본 연산
# 3. 텐서 형태 변환 (reshape, view, permute, squeeze)
# 4. Aggregation 연산 (sum, mean, max, argmax)
# 5. 고급 인덱싱 (Boolean, Fancy indexing)
# 6. 행렬 연산 (dot, matmul, 배치 연산)
# 7. Autograd를 통한 자동 미분
# 8. PyTorch vs TensorFlow 주요 차이점
# 9. 기본 학습 루프 패턴
# 10. 간단한 선형 회귀 구현
# 
# ### 🎯 다음 노트북에서 배울 내용
# - `09_pytorch_mlp_cnn_vgg.ipynb`
#   - CIFAR-10 데이터셋 로드 및 전처리
#   - MLP, CNN, VGG 아키텍처 구현
#   - DataLoader와 transforms 활용
#   - 여러 모델 성능 비교
# 
# ### 💡 핵심 개념 복습
# - **Tensor 기본**: shape, size(), ndim, dtype
# - **형태 변환**: reshape() vs view(), contiguous memory
# - **차원 조작**: permute(), transpose(), squeeze(), unsqueeze()
# - **Aggregation**: dim 인자 활용, argmax()
# - **인덱싱**: Boolean, Fancy indexing, torch.where()
# - **행렬 연산**: dot() vs matmul(), 배치 연산
# - **nn.Module**: 모든 PyTorch 모델의 베이스 클래스
# - **Autograd**: requires_grad, backward(), optimizer.zero_grad()

# In[ ]:


print("🎉 PyTorch 기초 튜토리얼을 완료했습니다!")
print()
print("다음 단계:")
print("  → 09_pytorch_mlp_cnn_vgg.ipynb로 이동하여")
print("     실전 이미지 분류 모델을 구현해봅시다.")

