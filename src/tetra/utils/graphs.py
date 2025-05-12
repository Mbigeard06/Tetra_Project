import pandas as pd
import matplotlib.pyplot as plt


def compare_results(res_paths, fn, names, metric=None, ylabel="Metric"):
    plt.figure(figsize=(10, 6))

    for path, label in zip(res_paths, names):
        df = pd.read_csv(path)
        if metric is not None:
            y = metric_fn(df,metric)
        else:
            y = fn(df)  # Applique la fonction passée
        plt.plot(df.index[:len(y)], y, label=label)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title("Comparaison personnalisée")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def total_loss(df):
    return 7.5 * df["val/box_loss"] + 0.5 * df["val/cls_loss"] + 1.5 * df["val/dfl_loss"]

def metric_fn(df, metric_name):
    return df[metric_name]