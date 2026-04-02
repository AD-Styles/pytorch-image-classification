import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# 1. 하드웨어 장치 설정 (VRAM 활용)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"현재 사용 중인 장치: {device}")

# 2. 하이퍼파라미터 설정
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10

# 3. 데이터 파이프라인 (Disk -> RAM 변환 및 증강)
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ==========================================
# 4. 모델 설계 (MLP, CNN, VGG 아키텍처 비교 구현)
# ==========================================

# [비교군 1] 다층 퍼셉트론 (MLP) - 공간 정보 손실 확인용
class MLPClassifier(nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(3 * 32 * 32, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1) # 2D 이미지를 1D로 강제 Flatten
        return self.fc_layer(x)

# [비교군 2] 합성곱 신경망 (CNN) - 본 학습용 메인 모델
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1) # Feature Map을 Flatten
        x = self.fc_layer(x)
        return x

# [비교군 3] VGG 스타일 네트워크 - 깊은 수용 영역 확인용
class VGGStyleClassifier(nn.Module):
    def __init__(self):
        super(VGGStyleClassifier, self).__init__()
        # 3x3 필터를 반복적으로 사용하는 VGG의 핵심 아이디어 차용
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Linear(128 * 8 * 8, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ==========================================
# 5. 학습 및 검증 루프 세팅
# ==========================================
model = CNNClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_model():
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Average Loss: {running_loss/len(train_loader):.4f}")

    print("✅ 모델 학습이 완료.")
    
    # 가중치 저장 로직
    torch.save(model.state_dict(), 'cnn_cifar10_weights.pth')
    print("💾 학습된 모델 가중치가 'cnn_cifar10_weights.pth'로 저장되었음.")

def evaluate_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"🎯 테스트 데이터셋 최종 정확도: {100 * correct / total:.2f}%")

if __name__ == '__main__':
    print("🚀 PyTorch 모델 학습 파이프라인을 시작...")
    train_model()
    evaluate_model()
