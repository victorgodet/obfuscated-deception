from abc import ABC, abstractmethod
import torch
import numpy as np
import dill as pickle
import einops
from typing import Callable
import pandas as pd
import os
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve

from src import utils

"""
Probe Training and Evaluation
"""

PROBE_REGISTRY: dict[str, type["Probe"]] = {}

def register_probe(cls: type["Probe"]) -> type["Probe"]:
    PROBE_REGISTRY[cls.file_extension] = cls
    return cls

def load_probe(path: str, **kwargs) -> type["Probe"]:
    ext = path.rsplit(".", 1)[-1] if "." in path else ""
    if ext not in PROBE_REGISTRY:
        raise ValueError(f"Unsupported probe file type: {path}. Available: {list(PROBE_REGISTRY.keys())}")
    return PROBE_REGISTRY[ext].load(path, **kwargs)

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().float().numpy()

def numpy_to_tensor(array: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.from_numpy(array).to(device=device, dtype=dtype)

class Probe(ABC):
    '''self.layers is an integer list of the layers that were used to fit the probe.

    Layers will be set by fit()
    For any prediction, a subset of these layers can be specified instead. If layers is not provided to fit, it is inferred based on test data shape 
    '''
    file_extension: str # File extension for the probe

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.layers = None # Probe can contain more layers than this in saved version, but will call based on these names and assume activations are in this order
    
    @abstractmethod
    def fit(self, acts: torch.Tensor, labels: torch.Tensor, layers: list[int] | None = None):
        '''acts dimension n_layers x n_samples x hidden_dim; labels n_samples x 1'''
        pass

    @abstractmethod
    def predict_proba(self, activations: torch.Tensor, layers: list[int] | None = None) -> torch.Tensor:
        '''acts dimesnions n_layers x n_samples x hidden_dim; output n_layers x n_samples x 1
        Returns 0 to 1 for the positive class
        '''
        pass
    
    def replace_extension(self, path: str, extension: str) -> str:
        return os.path.splitext(path)[0] + "." + extension

    @abstractmethod
    def save(self, path: str):
        pass

    @classmethod
    def load(cls, path: str) -> 'Probe':
        pass

    def predict(self, activations: torch.Tensor, threshold: float | dict[int, float] = 0.5, layers: list[int] | None = None) -> torch.Tensor:
        '''acts dimesnions n_layers x n_samples x hidden_dim; output n_layers x n_samples x 1
        Returns 1/0 binary predictions
        '''
        raw_predictions = self.predict_proba(activations, layers = layers)

        if layers is None:
            layers = self.layers

        if isinstance(threshold, float):
            threshold_t = torch.ones(raw_predictions.shape) * threshold
        else:
            threshold_t = torch.Tensor([threshold[layer] for layer in layers]).unsqueeze(0) # n_layers x 1
            threshold_t = torch.vstack([threshold_t for _ in range(raw_predictions.shape[0])]) # n_samples x n_layers
        threshold_t[threshold_t >= 1.0] = 0.5
        threshold_t[threshold_t <= 0.0] = 0.5
        threshold_t = threshold_t.to(device = raw_predictions.device)

        return (raw_predictions >= threshold_t).to(dtype = activations.dtype, device = activations.device)

    def _run_by_layer(self, y_true: torch.Tensor, y_pred: torch.Tensor, func: Callable, layers: list[int] | None = None) -> torch.Tensor:
        if layers is None:
            layers = self.layers
        predictions = {}
        for i, layer in enumerate(layers):
            predictions[layer] = func(tensor_to_numpy(y_true), tensor_to_numpy(y_pred[:, i]))
        return predictions

    def score(self, X_test: torch.Tensor, y_test: torch.Tensor, threshold: float | dict[int, float] = 0.5, layers: list[int] | None = None) -> dict[int, float]:
        y_pred = self.predict(X_test, layers = layers, threshold = threshold) # n_samples x n_layers
        return self._run_by_layer(y_test, y_pred, accuracy_score)

    def score_max(self, X_test: torch.Tensor, y_test: torch.Tensor, layers: list[int] | None = None) -> dict[int, float]:
        scores = pd.Series(self.score(X_test, y_test, layers = layers))
        return {self.layers[scores.idxmax()]: scores.max()}
    
    def roc_auc_score(self, X_test: torch.Tensor, y_test: torch.Tensor, layers: list[int] | None = None) -> dict[int, float]:
        y_pred = self.predict_proba(X_test, layers = layers)
        return self._run_by_layer(y_test, y_pred, roc_auc_score, layers = layers)

    def roc_auc_score_max(self, X_test: torch.Tensor, y_test: torch.Tensor, layers: list[int] | None = None) -> dict[int, float]:
        scores = pd.Series(self.roc_auc_score(X_test, y_test, layers = layers))
        return {self.layers[scores.idxmax()]: scores.max()}
    
    def precision_score(self, X_test: torch.Tensor, y_test: torch.Tensor, threshold: float | dict[int, float] = 0.5, layers: list[int] | None = None) -> dict[int, float]:
        y_pred = self.predict(X_test, layers = layers, threshold = threshold)
        return self._run_by_layer(y_test, y_pred, precision_score, layers = layers)
    
    def recall_score(self, X_test: torch.Tensor, y_test: torch.Tensor, threshold: float | dict[int, float] = 0.5, layers: list[int] | None = None) -> dict[int, float]:
        y_pred = self.predict(X_test, layers = layers, threshold = threshold)
        return self._run_by_layer(y_test, y_pred, recall_score, layers = layers)

    def select_threshold(self, X_test: torch.Tensor, y_test: torch.Tensor, target_fpr: float = 0.01, layers: list[int] | None = None) -> dict[int, float]:
        y_pred = self.predict_proba(X_test)
        outputs = self._run_by_layer(y_test, y_pred, roc_curve, layers = layers)
        results = {}
        for layer in outputs:
            fpr, tpr, thr = outputs[layer]
            i = int(np.argmin(np.abs(fpr - target_fpr)))
            results[layer] = float(thr[i])
        return results
    
    def evaluate(self, test_activations: torch.Tensor, test_labels: torch.Tensor, target_fpr: float = 0.05) -> dict:
        '''Returns layer by layer metrics for the probe'''

        threshold = self.select_threshold(test_activations, test_labels, target_fpr = target_fpr)

        return {
            'threshold': threshold,
            'roc_auc_score': self.roc_auc_score(test_activations, test_labels),
            'accuracy': self.score(test_activations, test_labels, threshold = threshold),
            'precision': self.precision_score(test_activations, test_labels, threshold = threshold),
            'recall': self.recall_score(test_activations, test_labels, threshold = threshold),
        }


@register_probe
class LogisticRegressionProbe(Probe):
    file_extension = "lgprobe"

    def __init__(self, clf: dict[int, LogisticRegression] | None = None, max_iter = 5000, l2_reg = 1.0, threshold: float = 0.5):
        self.clf = clf if clf is not None else {}
        self.settings = {
            'max_iter': max_iter,
            'penalty': 'l2',
            'C': 1.0 / l2_reg,
        }
        super().__init__(threshold = threshold)
        self.layers = self.clf.keys() if clf is not None else None 
    
    def fit(self, acts: torch.Tensor, labels: torch.Tensor, layers: list[int] | None = None):
        if layers is None:
            self.layers = range(acts.shape[0])
        else:
            self.layers = layers

        for i, layer in tqdm(enumerate(self.layers), desc="Fitting Logistic Regression", total=len(self.layers)):
            try:
                data_np = tensor_to_numpy(acts[i, ...])
                labels_np = tensor_to_numpy(labels)
                self.clf[layer] = LogisticRegression(**self.settings)
                self.clf[layer].fit(data_np, labels_np)
            except Exception as e:
                print(f"Error fitting logistic regression for layer {layer}", e)
                self.clf[layer] = None
    
    def predict_proba(self, activations: torch.Tensor, layers: list[int] | None = None) -> torch.Tensor:
        if layers is None:
            layers = self.layers

        predictions = []
        for i,layer in enumerate(layers):
            data_np = tensor_to_numpy(activations[i, ...])
            predictions.append(self.clf[layer].predict_proba(data_np)[:, 1]) # Get probability of the positive class only, all 1 0 values

        return torch.tensor(predictions).transpose(0, 1).to(device=activations.device, dtype=activations.dtype) # n_samples x n_layers

    def save(self, path: str):
        path = self.replace_extension(path, "lgprobe")
        utils.verify_path(path)
        with open(path, 'wb') as f:
            pickle.dump(self.clf, f)
    
    @classmethod
    def load(cls, path: str, **kwargs) -> 'LogisticRegressionProbe':
        with open(path, 'rb') as f:
            clf = pickle.load(f)
        return cls(clf, **kwargs)



@register_probe
class ApolloProbe(Probe):
    """Apollo-style deception probe using standardized dot-product scoring.

    Probe format: joblib dict with keys:
        direction:    (n_layers, hidden_dim) float32 ndarray
        scaler_mean:  (n_layers, hidden_dim) float32 ndarray
        scaler_scale: (n_layers, hidden_dim) float32 ndarray
        layers:       list[int]

    Scoring: score(x) = (1/L) * Σ_l dot( (x_l - μ_l) / σ_l, w_l )
             P(deceptive) = sigmoid(score)
    """
    file_extension = "joblib"

    def __init__(self, direction: np.ndarray, scaler_mean: np.ndarray, scaler_scale: np.ndarray, layers: list, threshold: float = 0.5):
        self.direction = direction        # (n_layers, hidden_dim)
        self.scaler_mean = scaler_mean    # (n_layers, hidden_dim)
        self.scaler_scale = scaler_scale  # (n_layers, hidden_dim)
        super().__init__(threshold=threshold)
        self.layers = list(layers)

    def fit(self, acts, labels, layers=None):
        raise NotImplementedError("ApolloProbe must be loaded from a pre-trained .joblib file.")

    def predict_proba(self, activations: torch.Tensor, layers: list | None = None) -> torch.Tensor:
        """Score activations using the Apollo formula.

        Args:
            activations: (n_layers, n_samples, hidden_dim) tensor

        Returns:
            (n_samples, 1) tensor of P(deceptive)
        """
        acts_np = tensor_to_numpy(activations)  # (n_layers, n_samples, hidden_dim)
        n_layers = acts_np.shape[0]
        scores = np.zeros(acts_np.shape[1], dtype=np.float32)
        for i in range(n_layers):
            normalized = (acts_np[i] - self.scaler_mean[i]) / self.scaler_scale[i]
            scores += normalized @ self.direction[i]
        scores /= n_layers
        probs = 1.0 / (1.0 + np.exp(-scores))  # sigmoid → (n_samples,)
        return torch.tensor(probs, dtype=activations.dtype).unsqueeze(-1)  # (n_samples, 1)

    def predict_raw(self, activations: torch.Tensor) -> torch.Tensor:
        """Return raw probe scores (before sigmoid).

        Args:
            activations: (n_layers, n_samples, hidden_dim) tensor

        Returns:
            (n_samples,) tensor of raw scores (positive = more deceptive)
        """
        acts_np = tensor_to_numpy(activations)
        n_layers = acts_np.shape[0]
        scores = np.zeros(acts_np.shape[1], dtype=np.float32)
        for i in range(n_layers):
            normalized = (acts_np[i] - self.scaler_mean[i]) / self.scaler_scale[i]
            scores += normalized @ self.direction[i]
        scores /= n_layers
        return torch.tensor(scores, dtype=activations.dtype)

    def save(self, path: str):
        import joblib
        path = self.replace_extension(path, "joblib")
        utils.verify_path(path)
        joblib.dump({
            "direction": self.direction,
            "scaler_mean": self.scaler_mean,
            "scaler_scale": self.scaler_scale,
            "layers": self.layers,
        }, path)

    @classmethod
    def load(cls, path: str, **kwargs) -> 'ApolloProbe':
        import joblib
        data = joblib.load(path)
        return cls(
            direction=np.asarray(data["direction"], dtype=np.float32),
            scaler_mean=np.asarray(data["scaler_mean"], dtype=np.float32),
            scaler_scale=np.asarray(data["scaler_scale"], dtype=np.float32),
            layers=list(data["layers"]),
            **kwargs,
        )


@register_probe
class MassMeanProbe(Probe):
    '''WARNING: This probe should only be saved if all layers are used'''
    file_extension = "mmpprobe"

    def __init__(self, direction: torch.Tensor | None = None, layers: list[int] | None = None, threshold: float = 0.5):
        self.direction = direction
        self.layers = layers
        super().__init__(threshold = threshold)
    
    def fit(self, acts: torch.Tensor, labels: torch.Tensor, layers: list[int] | None = None):
        if layers is None:
            self.layers = list(range(acts.shape[0]))
        else:
            self.layers = layers

        self.direction = acts[:, labels == 1, :].mean(dim=1) - acts[:, labels == 0, :].mean(dim=1) # shape: n_layers x hidden_dim
        self.direction = self.direction / self.direction.norm(p=2)


    def predict_proba(self, activations: torch.Tensor, layers: list[int] | None = None) -> torch.Tensor:
        if layers is None:
            layer_ind = range(len(self.layers))
        else:
            layer_ind = [self.layers.index(layer) for layer in layers]

        predictions = einops.einsum(
            activations.transpose(0, 1), self.direction[layer_ind, :], "n_samples n_layers hidden_dim, n_layers hidden_dim -> n_samples n_layers"
        )
        return torch.sigmoid(predictions)
    

    def save(self, path: str):
        path = self.replace_extension(path, "mmpprobe")
        utils.verify_path(path)
        torch.save({'direction': self.direction, 'layers': self.layers}, path)
    
    @classmethod
    def load(cls, path: str, **kwargs) -> 'MassMeanProbe':
        obj = torch.load(path, weights_only = False)
        new_cls = cls(direction = obj['direction'], layers = list(obj['layers']), **kwargs)
        new_cls.layers = list(obj['layers'])
        new_cls.direction = obj['direction']
        return new_cls





