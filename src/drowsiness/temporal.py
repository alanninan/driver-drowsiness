"""Optional small GRU; installed only with the gpu extra."""

import random

import numpy as np


def get_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install the optional training stack: uv sync --extra gpu") from exc
    return torch


def select_device(requested="auto"):
    torch = get_torch()
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("device must be auto, cuda or cpu")
    device = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; run doctor --gpu")
    return device


def make_model(features, hidden=32):
    torch = get_torch()

    class TemporalClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = torch.nn.GRU(features, hidden, batch_first=True)
            self.output = torch.nn.Linear(hidden, 3)

        def forward(self, x):
            _, state = self.gru(x)
            return self.output(state[-1])

    return TemporalClassifier()


def fit_predict(train_x, train_y, test_x, epochs=20, seed=42, device="auto"):
    torch = get_torch()
    if epochs < 1:
        raise ValueError("epochs must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = select_device(device)
    # Fit every preprocessing statistic on training subjects only.
    mean = train_x.mean(axis=(0, 1), keepdims=True)
    scale = np.maximum(train_x.std(axis=(0, 1), keepdims=True), 1e-6)
    x = torch.from_numpy(((train_x - mean) / scale).astype(np.float32))
    y = torch.from_numpy(train_y.astype(np.int64))
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=64, shuffle=True, generator=generator
    )
    model = make_model(x.shape[-1]).to(device)
    counts = np.bincount(train_y, minlength=3)
    weights = torch.tensor(
        len(train_y) / (3 * np.maximum(counts, 1)), dtype=torch.float32, device=device
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    model.train()
    for _ in range(epochs):
        for batch, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch.to(device)), target.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    predictions = []
    with torch.inference_mode():
        for offset in range(0, len(test_x), 128):
            normalized = ((test_x[offset : offset + 128] - mean) / scale).astype(np.float32)
            predictions.extend(
                model(torch.from_numpy(normalized).to(device)).argmax(dim=1).cpu().tolist()
            )
    return np.array(predictions), {
        "device": device,
        "epochs": epochs,
        "torch": torch.__version__,
        "normalization": "training subjects only",
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
    }


def gpu_diagnostic():
    torch = get_torch()
    device = select_device("auto")
    model = make_model(8).to(device)
    x = torch.randn(4, 30, 8, device=device)
    loss = model(x).square().mean()
    loss.backward()
    if device == "cuda":
        torch.cuda.synchronize()
    return {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": device,
        "forward_backward_passed": True,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "capability": list(torch.cuda.get_device_capability(0)) if device == "cuda" else None,
    }
