import os
import torch

from fereshteh.deepl.two_layer_binary_classification import binary_classification
from fereshteh.animation import animate_weight_heatmap  # for small-ish matrices


def main():
    # HW02 suggested parameters
    dt = 0.04
    epochs = 5000
    eta = 0.01
    d = 200
    n = 40000

    # Run training + collect weight histories (you must have modified binary_classification)
    out = binary_classification(d=d, n=n, epochs=epochs, eta=eta)

    # Expect you returned: ..., W1_hist, W2_hist, W3_hist, W4_hist at the end
    # Based on my previous return order:
    W1_hist = out[5]
    W2_hist = out[6]
    W3_hist = out[7]
    W4_hist = out[8]

    # Ensure media folder exists (Manim also creates media automatically, but this is fine)
    os.makedirs("media", exist_ok=True)

    print("Animating W1...")
    animate_weight_heatmap(
        W1_hist,
        dt=dt,
        file_name="W1_animation",
        title_str="W1 Weight Evolution",
    )

    print("Animating W2...")
    animate_weight_heatmap(
        W2_hist,
        dt=dt,
        file_name="W2_animation",
        title_str="W2 Weight Evolution",
    )

    print("Animating W3...")
    animate_weight_heatmap(
        W3_hist,
        dt=dt,
        file_name="W3_animation",
        title_str="W3 Weight Evolution",
    )

    print("Animating W4...")
    animate_weight_heatmap(
        W4_hist,
        dt=dt,
        file_name="W4_animation",
        title_str="W4 Weight Evolution",
    )

    print("\nDone. Your mp4 files should be in:")
    print("  media/videos/1080p30/")


if __name__ == "__main__":
    main()

