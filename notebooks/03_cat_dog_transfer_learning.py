import os

from tqdm import tqdm                 # 프로그레스 바 표시
from PIL import Image                 # 이미지 파일 읽기
import matplotlib.pyplot as plt       # 그래프 그리기
import pandas as pd                   # 데이터 정리
from pathlib import Path              # 파일 경로 다루기

import torch                          # PyTorch: 딥러닝 프레임워크
import torch.nn as nn                 # 신경망 구성 요소들
import torch.optim as optim           # 최적화 알고리즘 (학습 방법)
from torch.utils.data import DataLoader, Dataset  # 데이터 로딩 도구
from torchvision import transforms    # 이미지 변환 도구

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
print(f"사용 디바이스: {device}")

IMAGE_SIZE = (128, 128)    # 이미지 크기
BATCH_SIZE = 32            # 배치 크기 -> 한번에 처리하는 데이터 묶음의 수
NUM_EPOCHS = 1            # 에포크 수 (전체 데이터를 학습하는 횟수)
LEARNING_RATE = 1e-3       # 학습률 (0.001)

# 데이터셋 경로
DATA_DIR = Path("PetImages")

import urllib.request
import zipfile
import subprocess

# 데이터셋이 이미 있는지 확인
if not DATA_DIR.exists():
    print("\n데이터셋을 찾을 수 없습니다. 다운로드를 시작합니다...")

    # 데이터셋 URL
    dataset_url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
    zip_filename = "kagglecatsanddogs_5340.zip"

    # 다운로드
    print(f"다운로드 중: {dataset_url}")
    print("(파일 크기: 약 800MB, 시간이 걸릴 수 있습니다...)")
    urllib.request.urlretrieve(dataset_url, zip_filename)
    print(f"다운로드 완료: {zip_filename}")

    # 압축 해제 (Python zipfile 사용)
    print("\n압축 해제 중...")
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall('.')
    print("압축 해제 완료!")

    # 압축 파일 삭제 (선택사항)
    print(f"\n압축 파일 삭제 중: {zip_filename}")
    os.remove(zip_filename)
    print("삭제 완료!")
else:
    print(f"\n데이터셋이 이미 존재합니다: {DATA_DIR}")

# 손상된 이미지 제거
print("\n손상된 이미지 제거 중...")
corrupted_image = DATA_DIR / "Cat" / "11702.jpg" # 이미지가 너무 작아서 삭제
if corrupted_image.exists():
    os.remove(corrupted_image)
    print(f"  제거됨: {corrupted_image.name}")

# 데이터셋 구조 확인
print("\n데이터셋 구조 확인:")
for class_name in ['Cat', 'Dog']:
    class_dir = DATA_DIR / class_name
    if class_dir.exists():
        num_images = len(list(class_dir.glob('*.jpg')))
        print(f"  {class_name}: {num_images}개 이미지")

class CatsDogsDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        데이터셋 초기화

        Args:
            data_dir: 이미지가 들어있는 폴더
            transform: 이미지에 적용할 변환 (크기 조정, 정규화 등)
        """
        self.transform = transform
        self.images = []      # 이미지 파일 경로를 저장할 리스트
        self.labels = []      # 각 이미지의 정답 레이블 (0: 고양이, 1: 강아지)

        # Cat과 Dog 폴더에서 이미지 찾기
        for class_name in ['Cat', 'Dog']:
            class_dir = data_dir / class_name
            if not class_dir.exists():
                continue

            # 레이블 지정: Cat=0, Dog=1
            label = 0 if class_name == 'Cat' else 1

            # 폴더 안의 모든 jpg 파일 찾기
            for img_path in class_dir.glob('*.jpg'):
                self.images.append(img_path)
                self.labels.append(label)

        print(f"총 이미지 수: {len(self.images)}")
        print(f"고양이: {self.labels.count(0)}개, 강아지: {self.labels.count(1)}개")

    def __len__(self):
        """데이터셋의 전체 이미지 개수 반환"""
        return len(self.images)

    def __getitem__(self, idx):
        """
        idx번째 이미지와 레이블을 반환
        PyTorch가 학습할 때 이 함수를 호출합니다
        """
        img_path = self.images[idx]
        label = self.labels[idx]

        try:
            # 이미지 파일 읽기 (RGB 컬러로)
            image = Image.open(img_path).convert('RGB')

            # 이미지 전처리 적용
            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            # 손상된 이미지 발견 시 다음 이미지로 넘어감
            next_idx = (idx + 1) % len(self.images)
            return self.__getitem__(next_idx)

# 학습용 이미지 변환 (데이터 증강 포함)
train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),                          # (전처리)크기 조정
    transforms.RandomHorizontalFlip(),                      # (증강)랜덤 좌우 반전
    transforms.RandomRotation(10),                          # (증강)랜덤 회전 (±10도)
    transforms.ColorJitter(brightness=0.2, contrast=0.2),   # (증강)밝기/대비 조정
    transforms.ToTensor(),                                  # (전처리)Tensor로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406],        # (전처리)표준화 (ImageNet 평균)
                        std=[0.229, 0.224, 0.225])          # (전처리)표준화 (ImageNet 표준편차)
])

# 검증용 이미지 변환 (증강 없음)
val_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

full_dataset = CatsDogsDataset(DATA_DIR, transform=None)

# 학습/검증 분할 (80/20)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

torch.manual_seed(1337)  # 랜덤 시드 고정 (재현 가능하도록)
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# Transform 적용
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

# 데이터 로더: 배치 단위로 데이터를 제공
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"학습 데이터: {len(train_dataset)}개")
print(f"검증 데이터: {len(val_dataset)}개")

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # 입력: [3, 128, 128]
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # -> [32, 64, 64]
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # -> [64, 32, 32]
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # -> [128, 16, 16]
        )
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.relu_fc1 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)              # 마지막 FC 를 분류기(Classifier) 라고도 부름.

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNNClassifier()

model = model.to(device)   # 모델을 GPU 로 전송함
print(model)

print(f"\n모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

# 손실 함수: 모델의 예측이 얼마나 틀렸는지 측정
criterion = nn.BCEWithLogitsLoss()

# 옵티마이저: 손실을 줄이는 방향으로 모델을 업데이트
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_epoch(model, loader, criterion, optimizer, device):
    """한 에포크 동안 학습"""
    model.train()        # 학습 모드
    running_loss = 0.0   # 손실값 기록용 변수
    correct = 0          # 맞춘 응답 기록용 변수
    total = 0            # 전체 문제(x) 수 -> 기록용 변수

    for images, labels in tqdm(loader):   # 데이터 로더에서 배치 크기만큼 데이터 추출해라.
        # 데이터를 디바이스로 이동
        images = images.to(device)                       # x를 GPU 로 보냄.
        labels = labels.float().unsqueeze(1).to(device)  # y를 GPU 로 보냄.

        # 학습 1. 순전파: 예측값 계산
        optimizer.zero_grad()      # 기울기 초기화
        outputs = model(images)    # outputs -> 예측값

        # 학습 2. 손실 계산
        loss = criterion(outputs, labels)

        # 학습 3. 역전파: 가중치 업데이트
        loss.backward()    # 손실에대한 기울기를 구하는 과정.
        optimizer.step()   # Optimizer(경사하강 알고리즘)을 사용하여, 기울기를 바탕으로 파라미터 업뎃.

        # 4. 통계 계산
        running_loss += loss.item() * images.size(0)           # Train Loss 계산
        predicted = (torch.sigmoid(outputs) > 0.5).float()     # Train Accuracy 계산
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    """검증 데이터로 모델 평가"""
    model.eval()         # 평가 모드 (Dropout, BatchNorm 비활성화)
    running_loss = 0.0   # Validation loss 기록용 변수
    correct = 0          # 맞춘 응답 기록용 변수
    total = 0            # 전체 문제(x) 수 기록용 변수

    with torch.no_grad():  # 기울기 계산 안 함 (메모리 절약)
        for images, labels in tqdm(loader):                   # 데이터로더에서 배치 1개 추출
            images = images.to(device)                        # 이미지(x) 를 GPU 로 이동
            labels = labels.float().unsqueeze(1).to(device)   # y를 GPU 로 이동

            # 예측
            outputs = model(images)                  # 순전파 -> 예측값(prediction) 구하기
            loss = criterion(outputs, labels)        # 손실 계싼

            # 통계 계산
            running_loss += loss.item() * images.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

print("\n학습 시작...\n")
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

for epoch in range(NUM_EPOCHS):
    # 학습
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    # 검증
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # 기록
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    # 모델 저장 (나중에 다시 사용할 수 있도록)
    torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

print("\n학습 완료!")

# 학습 로그 CSV로 저장
df_history = pd.DataFrame({
    'epoch': range(1, NUM_EPOCHS + 1),
    'loss': history['train_loss'],
    'acc': history['train_acc'],
    'val_loss': history['val_loss'],
    'val_acc': history['val_acc']
})
df_history.to_csv('training_log.csv', index=False)
print("학습 로그 저장: training_log.csv")

# 학습 과정 시각화
plt.figure(figsize=(12, 4))

# 정확도 그래프
plt.subplot(1, 2, 1)
plt.plot(history['train_acc'], label='Train accuracy', marker='o')
plt.plot(history['val_acc'], label='Validation accuracy', marker='s')
plt.title('Model accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim((0, 1))
plt.legend()
plt.grid(True)

# 손실 그래프
plt.subplot(1, 2, 2)
plt.plot(history['train_loss'], label='Train loss', marker='o')
plt.plot(history['val_loss'], label='Validation loss', marker='s')
plt.title('Model loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# 최종 성능 출력
print(f"\n최종 훈련 정확도: {history['train_acc'][-1]:.4f}")
print(f"최종 검증 정확도: {history['val_acc'][-1]:.4f}")
print(f"최종 훈련 손실: {history['train_loss'][-1]:.4f}")
print(f"최종 검증 손실: {history['val_loss'][-1]:.4f}")

import numpy as np

# 시각화할 클래스 이름 정의
class_names = ['Cat', 'Dog']

# 모델을 평가 모드로 설정
model.eval()

# 검증 데이터 로더에서 배치 하나를 가져옴
# torch.no_grad() 블록 안에서 실행하여 메모리 사용량을 줄입니다.
with torch.no_grad():
    images, labels = next(iter(val_loader))
    images, labels = images.to(device), labels.to(device)

    # 모델 예측
    outputs = model(images)
    # Sigmoid 함수를 적용하여 확률로 변환하고, 0.5를 기준으로 예측 클래스 결정
    preds = (torch.sigmoid(outputs) > 0.5).squeeze()

# 이미지 정규화를 되돌리는 함수 정의
def denormalize(tensor, mean, std):
    # 텐서의 복사본을 만들어 원본이 변경되지 않도록 함
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

# 시각화를 위한 Figure 객체 생성
plt.figure(figsize=(16, 12))
plt.suptitle("Model Prediction vs. True Label", fontsize=20)

# 한 배치에서 최대 16개의 이미지만큼 반복
for i in range(min(len(images), 16)):
    ax = plt.subplot(4, 4, i + 1)

    # 이미지 텐서를 CPU로 이동시키고 정규화 역변환
    img = images[i].cpu()
    img = denormalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 텐서를 NumPy 배열로 변환하고 차원 순서 변경 (C, H, W) -> (H, W, C)
    img_np = img.numpy().transpose((1, 2, 0))
    # 픽셀 값을 0과 1 사이로 조정하여 이미지 깨짐 방지
    img_np = np.clip(img_np, 0, 1)

    # 실제 정답과 예측값 텍스트 가져오기
    true_label = class_names[labels[i].cpu().item()]
    pred_label = class_names[preds[i].cpu().item()]

    # 제목 색상 설정 (예측이 맞으면 녹색, 틀리면 빨간색)
    title_color = 'green' if true_label == pred_label else 'red'

    # 이미지 표시
    ax.imshow(img_np)
    # 제목 설정
    ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=title_color, fontsize=12)
    # 축 눈금 제거
    ax.axis('off')

# 서브플롯 간의 간격 자동 조정
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # suptitle과 겹치지 않도록 조정
# 이미지 파일로 저장
plt.savefig('prediction_visualization.png', dpi=300)
# 화면에 결과 표시
plt.show()

import urllib.request
from PIL import Image

def predict_image_from_url(url):
    """
    URL로부터 이미지를 다운로드하고, 모델로 예측한 후 결과를 시각화하는 함수
    """
    try:
        # URL에서 이미지 다운로드
        urllib.request.urlretrieve(url, "sample_image.jpg")
        image = Image.open("sample_image.jpg").convert('RGB')
    except Exception as e:
        print(f"이미지를 다운로드하거나 여는 데 실패했습니다: {e}")
        return

    # 원본 이미지 표시
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.title("Input Image")
    plt.show()

    # 이미지를 모델 입력에 맞게 전처리
    # 이전에 정의된 검증용 변환(val_transform)을 사용
    image_tensor = val_transform(image).unsqueeze(0).to(device)

    # 모델 예측
    model_tl.eval()  # 모델을 평가 모드로 설정
    with torch.no_grad():
        output = model_tl(image_tensor)

        # 확률 계산
        dog_prob = torch.sigmoid(output).item()
        cat_prob = 1 - dog_prob

        # 최종 예측 클래스 결정
        prediction_class = class_names[1 if dog_prob > 0.5 else 0]

    # 결과 출력
    print(f"🚀 모델 예측 결과: {prediction_class}")
    print(f"   - 강아지일 확률: {dog_prob:.2%}")
    print(f"   - 고양이일 확률: {cat_prob:.2%}")

# "" 사이에 이미지의 주소를 복사하여 붙여넣어 주세요
url = "https://cdn.dailyvet.co.kr/wp-content/uploads/2024/05/15231647/20240515ceva_experts4.jpg"
predict_image_from_url(url)

import torchvision.models as models
import torchvision.models as models  # Torchvision 에서 모델 다운로드 API 불러오기


# 1. ResNet-18 모델 불러오기 (ImageNet으로 사전 학습된 가중치 사용)
model_tl = models.resnet18(pretrained=True)  # ImageNet 으로 사전학습된 가중치 불러오기

# 2. 모든 파라미터를 동결 (기존에 학습된 가중치가 변하지 않도록 설정)
for param in model_tl.parameters():
    param.requires_grad = False         # 기울기 계산 불가능하게 만들기(역전파 안함)

model_tl.fc      # 변경 전

# 3. 우리의 문제에 맞게 분류기(fc layer)를 새로운 레이어로 교체
num_ftrs = model_tl.fc.in_features        # ResNet의 마지막 레이어 입력 채널 수 가져오기
model_tl.fc = nn.Linear(num_ftrs, 1)      # 2진 분류이므로 출력은 1개

model_tl.fc     # 변경 후 (2진 분류)

# 모델을 디바이스로 이동
model_tl = model_tl.to(device)

print(model_tl) # 모델 구조 확인

# 전이 학습을 위한 손실 함수와 옵티마이저 준비

# 손실 함수는 동일하게 사용
criterion_tl = nn.BCEWithLogitsLoss()

# 옵티마이저: 새로운 분류기(model_tl.fc)의 파라미터만 학습하도록 설정
# requires_grad=True인 파라미터만 업데이트됩니다.
optimizer_tl = optim.Adam(model_tl.fc.parameters(), lr=1e-4)
# ke-n : 소숫점 아래 n번째 자리에 k를 넣어라
# ex) 1e-3 : 0.001
# ex) 4e-5 : 0.00004

print("\n🚀 전이 학습 시작...\n")
history_tl = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': []
}
# 에포크는 동일하게 5번으로 설정
NUM_EPOCHS_TL = 10

# 기존에 정의한 학습 및 검증 함수(train_epoch, validate)를 그대로 사용합니다.
# 인자로 새로운 모델과 옵티마이저를 전달합니다.
for epoch in range(NUM_EPOCHS_TL):
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS_TL}]")

    # 학습
    train_loss, train_acc = train_epoch(model_tl, train_loader, criterion_tl, optimizer_tl, device)
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    # 검증
    val_loss, val_acc = validate(model_tl, val_loader, criterion_tl, device)
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # 결과 기록
    history_tl['train_loss'].append(train_loss)
    history_tl['train_acc'].append(train_acc)
    history_tl['val_loss'].append(val_loss)
    history_tl['val_acc'].append(val_acc)

print("\n학습 완료!")

# 최종 성능 출력
print(f"\n최종 훈련 정확도: {history_tl['train_acc'][-1]:.4f}")
print(f"최종 검증 정확도: {history_tl['val_acc'][-1]:.4f}")
