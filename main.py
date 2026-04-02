import argparse
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

# 2. 모델 아키텍처 정의
class MLP(nn.Module): 
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(3*32*32, 512), nn.ReLU(), nn.Linear(512, 10))
    def forward(self, x): return self.fc(x)

class CNNFromScratch(nn.Module): 
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Linear(64 * 8 * 8, 10)
    def forward(self, x): return self.classifier(self.features(x).view(x.size(0), -1))

def get_transfer_model(): 
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model.to(device)

# 모델 매핑 딕셔너리
MODELS = {
    "mlp": MLP,
    "cnn": CNNFromScratch,
    "transfer": get_transfer_model
}

def train(model_name, epochs, batch_size, lr):
    # 데이터 파이프라인 (CIFAR-10 기준)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_loader = DataLoader(
        torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True, num_workers=2
    )

    # 모델 인스턴스화 및 장치 할당
    if model_name == "transfer":
        model = MODELS[model_name]()
    else:
        model = MODELS[model_name]().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"🚀 학습 시작: Model={model_name}, Epochs={epochs}, Batch={batch_size}, LR={lr}")

    model.train()
    for epoch in range(epochs):
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=f"{loss.item():.4f}")
    
    # 학습된 가중치 저장
    save_path = f"final_{model_name}_weights.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ {model_name} 학습 완료 및 가중치 저장 성공: {save_path}")

if __name__ == "__main__":
    # 터미널 인자 설정
    parser = argparse.ArgumentParser(description="PyTorch Vision Engineering Pipeline")
    parser.add_argument("--model", type=str, default="cnn", choices=["mlp", "cnn", "transfer"], help="학습할 모델 선택")
    parser.add_argument("--epochs", type=int, default=5, help="학습 에포크 수")
    parser.add_argument("--batch", type=int, default=64, help="배치 사이즈")
    parser.add_argument("--lr", type=float, default=0.001, help="학습률")

    args = parser.parse_args()
    train(model_name=args.model, epochs=args.epochs, batch_size=args.batch, lr=args.lr)
