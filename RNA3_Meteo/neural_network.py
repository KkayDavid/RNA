"""
neural_network.py
=================
Red neuronal feedforward con PyTorch.
Soporta regresión, clasificación binaria y multiclase.
Incluye Early Stopping, activaciones configurables y checkpoint completo.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional
from config import DEFAULT_LR, DEFAULT_DROPOUT, EARLY_STOPPING_PATIENCE


# ── Mapa de activaciones ─────────────────────────────────────────
ACTIVATIONS = {
    "relu":       nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu":       nn.GELU,
    "silu":       nn.SiLU,
}


class NeuralNet(nn.Module):
    """
    Red neuronal feedforward configurable.
    Capas: [Linear → BatchNorm → Activation → Dropout] × N + Linear final.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int = 1,
        dropout: float = DEFAULT_DROPOUT,
        task: str = "regression",
        activation: str = "relu",
    ):
        super().__init__()
        self.task = task
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout
        self.activation_name = activation

        act_class = ACTIVATIONS.get(activation, nn.ReLU)
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                act_class(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, output_size))

        if task == "binary":
            layers.append(nn.Sigmoid())
        # multiclass: no Softmax (CrossEntropyLoss lo incluye)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Trainer:
    """
    Entrenador con Early Stopping, AdamW y ReduceLROnPlateau.
    Restaura automáticamente los mejores pesos cuando la validación no mejora.
    """

    def __init__(self, model: NeuralNet, lr: float = DEFAULT_LR):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Criterio según tarea
        if model.task == "regression":
            self.criterion = nn.MSELoss()
        elif model.task == "binary":
            self.criterion = nn.BCELoss()
        elif model.task == "multiclass":
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()

        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=15, factor=0.5
        )
        self.history: dict = {"train": [], "val": []}

        # Early stopping
        self._best_val_loss = float("inf")
        self._best_state = None
        self._patience_counter = 0
        self._stopped_epoch = 0

    def _make_loader(self, X, y, batch_size, shuffle=True) -> DataLoader:
        """Crea DataLoader sin mover datos al device de antemano."""
        Xt = torch.tensor(X, dtype=torch.float32)
        if self.model.task == "multiclass":
            yt = torch.tensor(y, dtype=torch.long)
        else:
            yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        use_pin = self.device.type == "cuda"
        return DataLoader(
            TensorDataset(Xt, yt),
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=use_pin,
        )

    def train(
        self,
        X_train, y_train,
        X_val=None, y_val=None,
        epochs=150, batch_size=32,
        patience: int = EARLY_STOPPING_PATIENCE,
        progress_callback=None,
    ) -> dict:
        """
        Entrena el modelo con Early Stopping.
        Restaura los mejores pesos al finalizar.
        """
        train_dl = self._make_loader(X_train, y_train, batch_size)
        val_dl = (self._make_loader(X_val, y_val, batch_size, False)
                  if X_val is not None else None)

        self._best_val_loss = float("inf")
        self._patience_counter = 0
        self._best_state = None

        for epoch in range(1, epochs + 1):
            # ── Train ────────────────────────────────────────────
            self.model.train()
            train_loss = 0.0
            n_batches = 0
            for Xb, yb in train_dl:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                train_loss += self._step(Xb, yb)
                n_batches += 1
            tl = train_loss / max(n_batches, 1)

            # ── Validation ───────────────────────────────────────
            vl = None
            if val_dl:
                self.model.eval()
                val_loss = 0.0
                vn = 0
                with torch.no_grad():
                    for Xb, yb in val_dl:
                        Xb, yb = Xb.to(self.device), yb.to(self.device)
                        val_loss += self.criterion(self.model(Xb), yb).item()
                        vn += 1
                vl = val_loss / max(vn, 1)
                self.scheduler.step(vl)

                # ── Early Stopping ───────────────────────────────
                if vl < self._best_val_loss:
                    self._best_val_loss = vl
                    self._patience_counter = 0
                    self._best_state = {
                        k: v.clone() for k, v in self.model.state_dict().items()
                    }
                else:
                    self._patience_counter += 1
                    if self._patience_counter >= patience:
                        self._stopped_epoch = epoch
                        break

            self.history["train"].append(tl)
            if vl is not None:
                self.history["val"].append(vl)

            if progress_callback:
                progress_callback(epoch, epochs, tl, vl)

        # Restaurar mejores pesos
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)

        self._stopped_epoch = epoch if self._stopped_epoch == 0 else self._stopped_epoch
        return self.history

    @property
    def stopped_epoch(self) -> int:
        return self._stopped_epoch

    def _step(self, Xb, yb) -> float:
        self.optimizer.zero_grad()
        loss = self.criterion(self.model(Xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            out = self.model(Xt).cpu().numpy()
        return out.flatten() if self.model.task != "multiclass" else out

    def save(self, path: str = "model.pt"):
        """Guarda modelo completo: pesos + arquitectura + hiperparámetros."""
        torch.save({
            "state":        self.model.state_dict(),
            "task":         self.model.task,
            "input_size":   self.model.input_size,
            "hidden_sizes": self.model.hidden_sizes,
            "output_size":  self.model.output_size,
            "dropout":      self.model.dropout_rate,
            "activation":   self.model.activation_name,
            "history":      self.history,
        }, path)

    def load(self, path: str = "model.pt"):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["state"])
        if "history" in ck:
            self.history = ck["history"]
