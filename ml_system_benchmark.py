import time
import json
import math
import platform
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =========================================================
# Defaults chosen to avoid huge RAM/swap pressure
# while still creating a meaningful ML-style benchmark
# =========================================================
DEFAULTS = {
    "seed": 42,
    "data_repeats": 300,             # much smaller than before
    "batch_size": 1024,
    "epochs": 8,
    "learning_rate": 0.0015,
    "hidden_1": 1024,
    "hidden_2": 512,
    "dropout": 0.10,
    "num_workers": 0,                # keep 0 initially for stability
    "mm_size": 512,
    "mm_repeats_per_batch": 1,
    "report_every": 25,
    "test_size": 0.2,
    "cpu_preprocess_loops": 3,
    "eval_cpu_preprocess_loops": 2,
}


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_accel_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return None


def sync_device(device: torch.device | None):
    if device is None:
        return
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def current_memory_mb(device: torch.device | None):
    if device is None:
        return None
    if device.type == "cuda":
        return torch.cuda.memory_allocated() / (1024 ** 2)
    if device.type == "mps":
        try:
            return torch.mps.current_allocated_memory() / (1024 ** 2)
        except Exception:
            return None
    return None


def build_base_dataset(data_repeats: int, seed: int):
    """
    Keeps the global dataset relatively compact.
    Heavy work is moved to per-batch transforms instead.
    """
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0
    y = digits.target.astype(np.int64)

    X = np.tile(X, (data_repeats, 1)).astype(np.float32)
    y = np.tile(y, data_repeats).astype(np.int64)

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=DEFAULTS["test_size"],
        random_state=seed,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def cpu_feature_stress(x_cpu: torch.Tensor, loops: int) -> torch.Tensor:
    """
    Compute-heavy but memory-safer preprocessing.
    Keeps shape fixed instead of repeatedly concatenating tensors.
    This is much better for CPU benchmarking than unbounded tensor growth.
    """
    out = x_cpu

    for _ in range(loops):
        a = torch.sin(out)
        b = torch.cos(out)
        c = torch.tanh(out)
        d = torch.square(out)
        e = torch.sqrt(torch.abs(out) + 1e-6)

        out = (
            0.30 * out
            + 0.20 * a
            + 0.15 * b
            + 0.15 * c
            + 0.10 * d
            + 0.10 * e
        )

        # small dense mixing to simulate feature interactions
        half = max(1, out.shape[1] // 2)
        left = out[:, :half]
        right = out[:, -half:]
        mixed = left * right
        mixed_mean = mixed.mean(dim=1, keepdim=True)
        out = out + mixed_mean

    return out


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_1: int, hidden_2: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_2, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def evaluate(model, loader, compute_device, mode, eval_cpu_preprocess_loops, loss_fn):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for xb, yb in loader:
        if mode == "cpu":
            xb_proc = cpu_feature_stress(xb, eval_cpu_preprocess_loops)
            logits = model(xb_proc)
            loss = loss_fn(logits, yb)

        elif mode == "accel":
            xb = xb.to(compute_device)
            yb = yb.to(compute_device)
            logits = model(xb)
            loss = loss_fn(logits, yb)

        elif mode == "hybrid":
            xb_proc = cpu_feature_stress(xb, eval_cpu_preprocess_loops)
            xb_proc = xb_proc.to(compute_device)
            yb = yb.to(compute_device)
            logits = model(xb_proc)
            loss = loss_fn(logits, yb)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        total_loss += loss.item() * xb.size(0)
        total_correct += (logits.argmax(dim=1) == yb).sum().item()
        total += xb.size(0)

    sync_device(compute_device if mode in {"accel", "hybrid"} else None)
    return total_loss / total, total_correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cpu", "accel", "hybrid"], required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--data-repeats", type=int, default=DEFAULTS["data_repeats"])
    parser.add_argument("--report-every", type=int, default=DEFAULTS["report_every"])
    parser.add_argument("--mm-size", type=int, default=DEFAULTS["mm_size"])
    parser.add_argument("--mm-repeats", type=int, default=DEFAULTS["mm_repeats_per_batch"])
    parser.add_argument("--cpu-preprocess-loops", type=int, default=DEFAULTS["cpu_preprocess_loops"])
    parser.add_argument("--eval-cpu-preprocess-loops", type=int, default=DEFAULTS["eval_cpu_preprocess_loops"])
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    set_seed(DEFAULTS["seed"])

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    accel_device = pick_accel_device()
    system_name = platform.system().lower()

    if args.mode in {"accel", "hybrid"} and accel_device is None:
        raise RuntimeError("No accelerator found. CUDA/MPS unavailable.")

    print("=" * 80)
    print("ML SYSTEM BENCHMARK V2")
    print("=" * 80)
    print(f"System:       {platform.platform()}")
    print(f"Mode:         {args.mode}")
    print(f"CPU threads:  {torch.get_num_threads()}")
    print(f"Accelerator:  {accel_device}")
    print()

    # -----------------------------------------------------
    # Data build phase
    # -----------------------------------------------------
    data_start = time.perf_counter()
    X_train, X_test, y_train, y_test = build_base_dataset(
        data_repeats=args.data_repeats,
        seed=DEFAULTS["seed"],
    )
    data_build_time = time.perf_counter() - data_start

    # Input dim stays fixed now
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=DEFAULTS["num_workers"],
        pin_memory=(system_name == "windows"),
        drop_last=False,
    )

    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=DEFAULTS["num_workers"],
        pin_memory=(system_name == "windows"),
        drop_last=False,
    )

    compute_device = None if args.mode == "cpu" else accel_device

    model = TabularMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_1=DEFAULTS["hidden_1"],
        hidden_2=DEFAULTS["hidden_2"],
        dropout=DEFAULTS["dropout"],
    )

    if compute_device is not None:
        model = model.to(compute_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULTS["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()

    # Extra math-kernel stress
    if args.mode == "cpu":
        A = torch.randn((args.mm_size, args.mm_size))
        B = torch.randn((args.mm_size, args.mm_size))
    else:
        A = torch.randn((args.mm_size, args.mm_size), device=compute_device)
        B = torch.randn((args.mm_size, args.mm_size), device=compute_device)

    sync_device(compute_device)
    overall_start = time.perf_counter()
    total_steps = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        running_loss = 0.0
        seen = 0

        for batch_idx, (xb, yb) in enumerate(train_loader, start=1):
            if args.mode == "cpu":
                xb_proc = cpu_feature_stress(xb, args.cpu_preprocess_loops)
                yb_proc = yb

                optimizer.zero_grad(set_to_none=True)
                logits = model(xb_proc)
                loss = loss_fn(logits, yb_proc)
                loss.backward()
                optimizer.step()

                for _ in range(args.mm_repeats):
                    C = torch.mm(A, B)
                    A = torch.relu(C)
                    B = torch.relu(torch.mm(B, A))

                compare_target = yb_proc

            elif args.mode == "accel":
                xb = xb.to(compute_device)
                yb = yb.to(compute_device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()

                for _ in range(args.mm_repeats):
                    C = torch.mm(A, B)
                    A = torch.relu(C)
                    B = torch.relu(torch.mm(B, A))

                compare_target = yb

            elif args.mode == "hybrid":
                xb_proc = cpu_feature_stress(xb, args.cpu_preprocess_loops)
                xb_proc = xb_proc.to(compute_device)
                yb = yb.to(compute_device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(xb_proc)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()

                for _ in range(args.mm_repeats):
                    C = torch.mm(A, B)
                    A = torch.relu(C)
                    B = torch.relu(torch.mm(B, A))

                compare_target = yb

            else:
                raise ValueError(f"Unknown mode: {args.mode}")

            running_loss += loss.item() * xb.size(0)
            seen += xb.size(0)
            total_steps += 1

            if batch_idx % args.report_every == 0 or batch_idx == len(train_loader):
                sync_device(compute_device)
                elapsed = time.perf_counter() - overall_start
                steps_per_sec = total_steps / elapsed if elapsed > 0 else float("nan")
                mem = current_memory_mb(compute_device)
                mem_str = f"{mem:.1f} MB" if mem is not None else "N/A"

                with torch.no_grad():
                    batch_acc = (logits.argmax(dim=1) == compare_target).float().mean().item()

                print(
                    f"Epoch {epoch:02d}/{args.epochs} | "
                    f"Batch {batch_idx:04d}/{len(train_loader)} | "
                    f"Loss {loss.item():.4f} | "
                    f"Batch Acc {batch_acc:.4f} | "
                    f"Steps/s {steps_per_sec:.2f} | "
                    f"Mem {mem_str}"
                )

        sync_device(compute_device)
        epoch_time = time.perf_counter() - epoch_start

        test_loss, test_acc = evaluate(
            model=model,
            loader=test_loader,
            compute_device=compute_device,
            mode=args.mode,
            eval_cpu_preprocess_loops=args.eval_cpu_preprocess_loops,
            loss_fn=loss_fn,
        )

        train_loss = running_loss / seen

        print(
            f"[Epoch Summary] Epoch {epoch:02d} | "
            f"Train Loss {train_loss:.4f} | "
            f"Test Loss {test_loss:.4f} | "
            f"Test Acc {test_acc:.4f} | "
            f"Epoch Time {epoch_time:.2f}s"
        )
        print("-" * 80)

    sync_device(compute_device)
    total_time = time.perf_counter() - overall_start

    final_test_loss, final_test_acc = evaluate(
        model=model,
        loader=test_loader,
        compute_device=compute_device,
        mode=args.mode,
        eval_cpu_preprocess_loops=args.eval_cpu_preprocess_loops,
        loss_fn=loss_fn,
    )

    results = {
        "machine": platform.node(),
        "system": platform.platform(),
        "mode": args.mode,
        "accel_device": str(compute_device),
        "cpu_threads": torch.get_num_threads(),
        "data_build_time_sec": data_build_time,
        "data_repeats": args.data_repeats,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "mm_size": args.mm_size,
        "mm_repeats": args.mm_repeats,
        "cpu_preprocess_loops": args.cpu_preprocess_loops,
        "eval_cpu_preprocess_loops": args.eval_cpu_preprocess_loops,
        "total_steps": total_steps,
        "total_time_sec": total_time,
        "steps_per_sec": total_steps / total_time,
        "final_test_loss": final_test_loss,
        "final_test_acc": final_test_acc,
    }

    output = args.output or f"benchmark_{args.mode}_{system_name}.json"
    Path(output).write_text(json.dumps(results, indent=2))

    print("\nDONE")
    print(json.dumps(results, indent=2))
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()