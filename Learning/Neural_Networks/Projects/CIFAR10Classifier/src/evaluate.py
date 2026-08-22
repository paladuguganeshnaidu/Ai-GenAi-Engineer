from pathlib import Path

import torch

from src.dataset import CLASSES, create_loaders
from src.model import CIFAR10CNN


PROJECT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = create_loaders(PROJECT_DIR / "data")
    model = CIFAR10CNN().to(device)
    model.load_state_dict(torch.load(PROJECT_DIR / "models" / "cifar10_cnn.pt", map_location=device))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    (PROJECT_DIR / "results").mkdir(exist_ok=True)
    (PROJECT_DIR / "results" / "metrics.txt").write_text(
        f"device={device}\naccuracy={accuracy:.4f}\nclasses={', '.join(CLASSES)}\n",
        encoding="utf-8",
    )
    print(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
