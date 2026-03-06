from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

class HeavyTierMLP(nn.Module):
    def __init__(self, input_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

@dataclass
class HeavyTrainConfig:
    backend: str = 'rf'
    n_estimators: int = 100
    max_depth: Optional[int] = None
    mlp_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001

def train_heavy_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: HeavyTrainConfig,
    sample_weights: Optional[np.ndarray] = None
) -> object:
    if config.backend == 'rf':
        clf = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            n_jobs=-1,
            class_weight='balanced'
        )
        clf.fit(X_train, y_train, sample_weight=sample_weights)
        return clf
    elif config.backend == 'mlp':
        # Simple training loop for MLP
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = HeavyTierMLP(input_dim=X_train.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.CrossEntropyLoss(reduction='none')

        # Convert to tensor
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.long)
        
        # Sampler only if weights are provided
        sampler = None
        if sample_weights is not None:
             # Normalize weights for sampler
             w = torch.tensor(sample_weights, dtype=torch.double)
             sampler = WeightedRandomSampler(w, len(w))
             
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=config.batch_size, 
                            shuffle=(sampler is None), sampler=sampler)

        model.train()
        for epoch in range(config.mlp_epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb).mean()
                loss.backward()
                optimizer.step()
        
        model.eval()
        return model
    else:
        raise ValueError(f"Unknown backend: {config.backend}")

def predict_heavy(
    model: object,
    X: np.ndarray
) -> Dict[str, np.ndarray]:
    if isinstance(model, RandomForestClassifier):
        probs = model.predict_proba(X)
        preds = model.predict(X)
        return {
            'probabilities': probs,
            'predictions': preds
        }
    elif isinstance(model, nn.Module):
        device = next(model.parameters()).device
        model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
        return {
            'probabilities': probs,
            'predictions': preds
        }
    else:
        raise ValueError(f"Unknown model type: {type(model)}")
