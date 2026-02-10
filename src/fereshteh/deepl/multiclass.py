from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import matplotlib.pyplot as plt


class SimpleNN(nn.Module):
    """
    Multi-class version:
      - last layer outputs num_classes logits
      - use CrossEntropyLoss for single-label multi-class
    """

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        self.fc1 = nn.Linear(self.in_features, 3)
        self.fc2 = nn.Linear(3, 4)
        self.fc3 = nn.Linear(4, 5)
        self.fc4 = nn.Linear(5, self.num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # logits: (batch, num_classes)
        return x


@dataclass
class ClassTrainer:
    """
    Required class variables / parameters:
      X_train, Y_train, eta, epoch, loss, optimizer, loss_vector, accuracy_vector,
      model, device
    """

    X_train: torch.Tensor
    Y_train: torch.Tensor
    eta: float = 1e-3
    epoch: int = 50
    loss: Optional[nn.Module] = None
    optimizer_name: str = "adam"
    model: Optional[nn.Module] = None
    device: Optional[torch.device] = None
    batch_size: int = 1024

    loss_vector: Optional[torch.Tensor] = None
    accuracy_vector: Optional[torch.Tensor] = None

    # saved after training for evaluation()
    y_train_true: Optional[np.ndarray] = None
    y_train_pred: Optional[np.ndarray] = None
    y_test_true: Optional[np.ndarray] = None
    y_test_pred: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.model is None:
            raise ValueError("ClassTrainer requires model=SimpleNN(...)")

        self.model = self.model.to(self.device)

        # Default loss for multi-class
        if self.loss is None:
            self.loss = nn.CrossEntropyLoss()

        # Optimizer
        if self.optimizer_name.lower() == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.eta)
        elif self.optimizer_name.lower() == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.eta)
        else:
            raise ValueError("optimizer_name must be 'adam' or 'sgd'")

        # Initialize vectors (length = epochs)
        self.loss_vector = torch.zeros(self.epoch, device="cpu")
        self.accuracy_vector = torch.zeros(self.epoch, device="cpu")

    def train(self) -> None:
        self.model.train()

        X = self.X_train.to(self.device)
        y = self.Y_train.to(self.device)

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for ep in range(self.epoch):
            total_loss = 0.0
            correct = 0
            total = 0

            for xb, yb in loader:
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss_val = self.loss(logits, yb)
                loss_val.backward()
                self.optimizer.step()

                total_loss += loss_val.item() * xb.size(0)

                preds = torch.argmax(logits, dim=1)
                correct += (preds == yb).sum().item()
                total += yb.numel()

            avg_loss = total_loss / total
            acc = correct / total

            self.loss_vector[ep] = avg_loss
            self.accuracy_vector[ep] = acc

            if ep % max(1, self.epoch // 10) == 0:
                print(f"Epoch {ep:4d}/{self.epoch}  loss={avg_loss:.4f}  acc={acc:.4f}")

        # store train predictions for evaluation()
        with torch.no_grad():
            logits = self.model(X)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            self.y_train_true = y.detach().cpu().numpy()
            self.y_train_pred = preds

    def test(self, X_test: torch.Tensor, y_test: torch.Tensor) -> Dict[str, float]:
        self.model.eval()
        X = X_test.to(self.device)
        y = y_test.to(self.device)

        with torch.no_grad():
            logits = self.model(X)
            preds = torch.argmax(logits, dim=1)

        y_true = y.detach().cpu().numpy()
        y_pred = preds.detach().cpu().numpy()

        self.y_test_true = y_true
        self.y_test_pred = y_pred

        return self._metrics(y_true, y_pred)

    def predict(self, X: torch.Tensor) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X.to(self.device))
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        return preds

    def save(self, file_name: Optional[str] = None) -> Path:
        """
        Save ONNX model. If file_name is None, saves as: model.onnx
        """
        if file_name is None:
            file_name = "model.onnx"

        out = Path(file_name).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        self.model.eval()
        dummy = torch.randn(1, self.X_train.shape[1], device=self.device)

        torch.onnx.export(
            self.model,
            dummy,
            str(out),
            input_names=["input"],
            output_names=["logits"],
            opset_version=17,
        )
        print(f"Saved ONNX model to: {out}")
        return out

    def evaluation(
        self,
        class_names: Optional[list[str]] = None,
        outdir: str | Path = "reports",
        tag: str = "run",
    ) -> Dict[str, Dict[str, float]]:
        """
        Plot:
          - loss + accuracy curves
          - confusion matrices for train/test
        Also print final metrics.
        """
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        # curves
        plt.figure()
        plt.plot(self.loss_vector.numpy())
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.tight_layout()
        loss_path = outdir / f"{tag}_loss.png"
        plt.savefig(loss_path, dpi=150)
        plt.close()

        plt.figure()
        plt.plot(self.accuracy_vector.numpy())
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy")
        plt.tight_layout()
        acc_path = outdir / f"{tag}_acc.png"
        plt.savefig(acc_path, dpi=150)
        plt.close()

        results = {}

        if self.y_train_true is not None and self.y_train_pred is not None:
            train_m = self._metrics(self.y_train_true, self.y_train_pred)
            results["train"] = train_m
            self._plot_cm(self.y_train_true, self.y_train_pred, class_names, outdir / f"{tag}_cm_train.png")

        if self.y_test_true is not None and self.y_test_pred is not None:
            test_m = self._metrics(self.y_test_true, self.y_test_pred)
            results["test"] = test_m
            self._plot_cm(self.y_test_true, self.y_test_pred, class_names, outdir / f"{tag}_cm_test.png")

        return results

    def _metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }

    def _plot_cm(self, y_true, y_pred, class_names, outpath: Path):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        plt.imshow(cm)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        if class_names:
            plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
            plt.yticks(range(len(class_names)), class_names)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(outpath, dpi=150)
        plt.close()
