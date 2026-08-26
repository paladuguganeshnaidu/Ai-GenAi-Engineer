import torch
from pathlib import Path


# ==========================================
# Save model
# ==========================================

def save_model(model, path):
    """
    Save the model's learned parameters.
    """

    path = Path(path)

    # Create parent directory if it doesn't exist
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        path
    )

    print(f"Model saved to: {path}")


# ==========================================
# Load model
# ==========================================

def load_model(model, path):
    """
    Load saved parameters into a model.
    """

    model.load_state_dict(
        torch.load(
            path,
            weights_only=True
        )
    )

    print(f"Model loaded from: {path}")

    return model