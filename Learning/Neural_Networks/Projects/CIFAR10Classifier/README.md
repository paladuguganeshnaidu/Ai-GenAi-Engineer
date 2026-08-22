# CIFAR-10 Classifier

A small PyTorch CNN project for learning the complete image-classification workflow.

## Structure

- `data/`: downloaded CIFAR-10 files
- `models/`: saved model weights
- `results/`: evaluation metrics
- `src/dataset.py`: dataset and DataLoader setup
- `src/model.py`: CNN architecture
- `src/train.py`: training loop
- `src/evaluate.py`: test evaluation
- `src/predict.py`: single-image prediction

## Run

From this project directory:

```powershell
python -m pip install -r requirements.txt
python -m src.train
python -m src.evaluate
python -m src.predict path\to\image.png
```

The code automatically uses CUDA when PyTorch detects an NVIDIA GPU.
