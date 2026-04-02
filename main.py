import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# 1. 하드웨어 설정 (VRAM 가속 활성화)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 현재 사용 장치: {device}")

# 2. 데이터 파이프라인 (CIFAR-10)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), # 데이터 증강
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_loader = DataLoader(
    torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True, num_workers=2 # 병목 최적화
)

# 3. 모델 라이브러리
class CNNFromScratch(nn.Module): #
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Linear(64 * 8 * 8, 10)
    def forward(self, x): return self.classifier(self.features(x).view(x.size(0), -1))

def get_transfer_model(): #
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model.to(device)

# 4. 학습 루프
model = CNNFromScratch().to(device) # 또는 get_transfer_model()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train():
    model.train()
    for epoch in range(5):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
    
    torch.save(model.state_dict(), "final_model_weights.pth") # 가중치 저장
    print("✅ 모델 가중치 저장 완료.")

if __name__ == "__main__":
    train()
