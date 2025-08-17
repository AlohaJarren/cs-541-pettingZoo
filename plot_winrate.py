import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="plot winrate")
    parser.add_argument("--file", type=str, default="models/B_winrate_vs_A.txt")
    parser.add_argument("--steps-per-burst", type=int, default=5000)
    parser.add_argument("--out", type=str, default="models/B_winrate_vs_A.png")
    args = parser.parse_args()

    y = np.loadtxt(args.file)
    if y.ndim == 0:
        y = np.array([float(y)])
    x = np.arange(1, len(y) + 1) * args.steps_per_burst

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel("training steps")
    plt.ylabel("win rate (B vs A)")
    plt.title("adaptation curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()