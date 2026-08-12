# =============================================================================
# inference.py -- Single-sample inference with FedSentinel global model
# =============================================================================

import os, sys, argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (DATASET, MODEL_ARCH, NUM_CLASSES, CHECKPOINT_BEST,
                    RESULTS_DIR, DEVICE, DATASET_CONFIG)
from model import build_model
from utils import load_checkpoint

# CIFAR-10 class names for visualization
CIFAR10_CLASSES  = ["airplane","automobile","bird","cat","deer",
                    "dog","frog","horse","ship","truck"]
CIFAR100_CLASSES = [f"class_{i}" for i in range(100)]  # placeholder


def get_transform(dataset: str):
    """Return test-time transform for a dataset."""
    if dataset in ("cifar10", "cifar100"):
        mean = (0.4914, 0.4822, 0.4465) if dataset == "cifar10" else (0.5071, 0.4867, 0.4408)
        std  = (0.2470, 0.2435, 0.2616) if dataset == "cifar10" else (0.2675, 0.2565, 0.2761)
        return T.Compose([T.Resize((32, 32)), T.ToTensor(), T.Normalize(mean, std)])
    elif dataset == "femnist":
        return T.Compose([T.Grayscale(), T.Resize((28, 28)), T.ToTensor(),
                          T.Normalize((0.5,), (0.5,))])
    return T.Compose([T.ToTensor()])


def load_image(image_path: str, dataset: str) -> torch.Tensor:
    """Load and preprocess a single image."""
    img = Image.open(image_path).convert("RGB")
    tf  = get_transform(dataset)
    return tf(img).unsqueeze(0)   # (1, C, H, W)


@torch.no_grad()
def predict(model, x: torch.Tensor, class_names=None):
    """Run forward pass and return top-5 predictions."""
    model.eval()
    x     = x.to(DEVICE)
    logits = model(x)
    probs  = F.softmax(logits, dim=-1)[0]
    top5   = probs.topk(min(5, len(probs)))

    results = []
    for i, (prob, idx) in enumerate(zip(top5.values.tolist(), top5.indices.tolist())):
        name = class_names[idx] if class_names and idx < len(class_names) else f"class_{idx}"
        results.append({"rank": i + 1, "class_id": idx, "class_name": name, "confidence": prob})
    return results, probs.cpu().numpy()


def visualize_prediction(image_path: str, predictions: list, probs: np.ndarray,
                          dataset: str, save_path: str) -> None:
    """Visualize image alongside prediction bar chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    img = Image.open(image_path).convert("RGB")
    ax1.imshow(img)
    top = predictions[0]
    ax1.set_title(f"Predicted: {top['class_name']}\nConfidence: {top['confidence']:.2%}")
    ax1.axis("off")

    names  = [p["class_name"] for p in predictions]
    confs  = [p["confidence"] for p in predictions]
    colors = ["tab:red" if i == 0 else "tab:blue" for i in range(len(names))]
    ax2.barh(names[::-1], confs[::-1], color=colors[::-1])
    ax2.set_xlabel("Confidence")
    ax2.set_title(f"Top-{len(predictions)} Predictions")
    ax2.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[inference] Visualization saved: {save_path}")


def run_inference(image_path: str, checkpoint: str = CHECKPOINT_BEST) -> dict:
    """Full inference pipeline."""
    print(f"[inference] Loading model from: {checkpoint}")
    model = build_model(MODEL_ARCH, NUM_CLASSES)
    load_checkpoint(checkpoint, model)
    model.to(DEVICE).eval()

    # Load image
    x = load_image(image_path, DATASET)
    print(f"[inference] Input shape: {x.shape}")

    # Class names
    class_names = CIFAR10_CLASSES if DATASET == "cifar10" else None

    # Predict
    predictions, probs = predict(model, x, class_names)

    print("\n[inference] Top-5 Predictions:")
    for p in predictions:
        print(f"  #{p['rank']:1d}  {p['class_name']:<20s}  {p['confidence']:>6.2%}")

    # Visualize
    save_path = os.path.join(RESULTS_DIR, "inference_output.png")
    visualize_prediction(image_path, predictions, probs, DATASET, save_path)

    return {"top_class": predictions[0]["class_name"],
            "confidence": predictions[0]["confidence"],
            "all_predictions": predictions}


def main():
    parser = argparse.ArgumentParser(description="FedSentinel Inference")
    parser.add_argument("--image",      type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_BEST,
                        help="Model checkpoint path")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)

    result = run_inference(args.image, args.checkpoint)
    print(f"\n[RESULT] {result['top_class']} ({result['confidence']:.2%})")


if __name__ == "__main__":
    main()
