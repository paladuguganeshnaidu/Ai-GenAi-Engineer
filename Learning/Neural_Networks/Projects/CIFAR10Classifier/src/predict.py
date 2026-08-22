from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.dataset import CLASSES
from src.model import CIFAR10CNN


PROJECT_DIR = Path(__file__).resolve().parents[1]


def predict(image_path: str) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CIFAR10CNN().to(device)
    model.load_state_dict(torch.load(PROJECT_DIR / "models" / "cifar10_cnn.pt", map_location=device))
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        label = model(image).argmax(dim=1).item()
    return CLASSES[label]


if __name__ == "__main__":
    import sys
    print(predict(sys.argv[1]))
