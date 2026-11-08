"""
Sample training script demonstrating multi-file import and output persistence.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from config import BATCH_SIZE, LEARNING_RATE, MODEL_NAME
from utils import compute_epoch_metric, get_environment_info


def main() -> None:
    """
    Main training execution loop.
    """
    parser = argparse.ArgumentParser(description="Sample training script")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to run")
    args = parser.parse_args()

    print(f"=== Starting Training for {MODEL_NAME} ===", flush=True)
    print(f"Hyperparameters: lr={LEARNING_RATE}, batch_size={BATCH_SIZE}, epochs={args.epochs}", flush=True)

    env_info = get_environment_info()
    print(f"Environment: {env_info}", flush=True)

    out_dir = Path(os.environ.get("KGRUN_OUT", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(1, args.epochs + 1):
        loss = compute_epoch_metric(epoch, 1.5)
        accuracy = compute_epoch_metric(epoch, 0.6) + 0.3
        history.append({"epoch": epoch, "loss": loss, "accuracy": round(accuracy, 4)})
        print(f"Epoch {epoch}/{args.epochs} - Loss: {loss:.4f} - Acc: {accuracy:.4f}", flush=True)
        time.sleep(1)

    result = {
        "model_name": MODEL_NAME,
        "epochs": args.epochs,
        "history": history,
        "final_loss": history[-1]["loss"],
        "final_accuracy": history[-1]["accuracy"],
        "finished_at": time.time(),
    }

    result_path = out_dir / "sample_model_results.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Successfully saved training results to {result_path}", flush=True)
    print("=== Training Complete ===", flush=True)


if __name__ == "__main__":
    main()

