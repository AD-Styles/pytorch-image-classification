import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# 1. 하드웨어 최적화 설정
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 활성화 장치: {device}")

# 2. 데이터 파이프라인 (CIFAR-10 기준)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2) # 메모리 병목 최적화

# 3. 신경망 아키텍처 라이브러리 (README 명세 일치)
class MLP(nn.Module): #
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(3*32*32, 512), nn.ReLU(), nn.Linear(512, 10))
    def forward(self, x): return self.fc(x)

class CNN(nn.Module): #
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(64*8*8, 10)
    def forward(self, x): return self.fc(self.conv(x).view(x.size(0), -1))

# [실무 역량 어필] 전이 학습 모델 (ResNet18)
def get_transfer_model():
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, 10) # 10개 클래스로 조정
    return model.to(device)

# 4. 통합 학습 시스템
model = CNN().to(device) # 필요에 따라 MLP() 또는 get_transfer_model()로 교체 가능
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train():
    model.train()
    for epoch in range(5):
        loop = tqdm(train_loader, leave=False)
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            loop.set_description(f"Epoch [{epoch+1}/5]")
            loop.set_postfix(loss=loss.item())
    
    # 가중치 저장 (실무 필수 공정)
    torch.save(model.state_dict(), "best_model.pth")
    print("✅ 학습 완료 및 가중치 저장 성공.")

if __name__ == "__main__":
    train()
