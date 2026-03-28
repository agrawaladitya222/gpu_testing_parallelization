import time
import torch
from sklearn.datasets import load_digits

RUN_SECONDS = 180
REPORT_EVERY = 10

# Increase if GPU usage is low, decrease if you hit memory errors
MM_SIZE = 2048
MM_REPEATS = 6

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available.")

    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))

    # Real dataset from scikit-learn
    digits = load_digits()
    X = torch.tensor(digits.data, dtype=torch.float32, device=device) / 16.0
    y = torch.tensor(digits.target, dtype=torch.long, device=device)

    input_dim = X.shape[1]
    num_classes = int(y.max().item()) + 1

    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, num_classes),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Pre-allocate large GPU matrices
    A = torch.randn((MM_SIZE, MM_SIZE), device=device)
    B = torch.randn((MM_SIZE, MM_SIZE), device=device)

    start = time.time()
    next_report = start + REPORT_EVERY
    steps = 0

    print(f"Running for about {RUN_SECONDS} seconds...")
    print("Watch Task Manager > Performance > GPU or run nvidia-smi -l 1\n")

    while time.time() - start < RUN_SECONDS:
        optimizer.zero_grad(set_to_none=True)

        # Training work on GPU
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        # Extra GPU-heavy matrix math
        for _ in range(MM_REPEATS):
            C = torch.mm(A, B)
            A = torch.relu(C)
            B = torch.relu(torch.mm(B, A))

        torch.cuda.synchronize()
        steps += 1

        if time.time() >= next_report:
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                acc = (preds == y).float().mean().item()

            elapsed = time.time() - start
            mem_used = torch.cuda.memory_allocated() / (1024 ** 2)

            print(
                f"Elapsed: {elapsed:6.1f}s | "
                f"Steps: {steps:3d} | "
                f"Loss: {loss.item():.4f} | "
                f"Acc: {acc:.4f} | "
                f"GPU mem: {mem_used:.1f} MB"
            )
            next_report += REPORT_EVERY

    with torch.no_grad():
        final_logits = model(X)
        final_preds = final_logits.argmax(dim=1)
        final_acc = (final_preds == y).float().mean().item()

    print("\nDone.")
    print(f"Final accuracy: {final_acc:.4f}")
    print(f"Total runtime: {time.time() - start:.1f}s")

if __name__ == "__main__":
    main()


