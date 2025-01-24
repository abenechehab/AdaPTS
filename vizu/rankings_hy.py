import pandas as pd
import tyro
from dataclasses import dataclass

import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
df = pd.read_csv("results/data.csv")
df["adapter"] = df["adapter"].fillna("no_adapter")

# Read hyperopt data
df_hyperopt = pd.read_csv("results/hyperopt.csv")


@dataclass
class Args:
    model: str = "AutonLab/MOMENT-1-large"
    metric: str = "mse"
    dataset: str = "ETTh1"
    forecasting_horizon: int = 96


# Parse command line arguments
args = tyro.cli(Args)

filtered_df = df.loc[
    (df["foundational_model"] == args.model)
    & (~df["is_fine_tuned"].isin(["True"]))
    & (df["metric"] == args.metric)
    & (df["dataset"] == args.dataset)
    & (df["forecasting_horizon"] == args.forecasting_horizon)
]

filtered_df_hyperopt = df_hyperopt.loc[
    (df_hyperopt["model"] == args.model)
    & (df_hyperopt["dataset"] == f"{args.dataset}_pred={args.forecasting_horizon}")
    & (df_hyperopt["forecasting_horizon"] == args.forecasting_horizon)
]

# Create the plot
plt.figure(figsize=(10, 6))

# Get all unique adapters except 'no_adapter'
adapters = filtered_df_hyperopt["adapter"].unique()

# Define a color palette for the adapters
# Option 1: Use a different seaborn palette
colors = sns.color_palette("deep", n_colors=len(adapters))

# Option 2: Use tableau colors
# colors = sns.color_palette("tab10", n_colors=len(adapters))

# Option 3: Use bright colors
# colors = sns.color_palette("bright", n_colors=len(adapters))

# Option 4: Use pastel colors
# colors = sns.color_palette("pastel", n_colors=len(adapters))

for adapter, color in zip(adapters, colors):
    adapter_data = filtered_df_hyperopt[filtered_df_hyperopt["adapter"] == adapter]

    # Group by n_components and find best validation MSE for each
    best_val = adapter_data.loc[adapter_data.groupby("n_components")["mse"].idxmin()]
    # Group by n_components and find best test MSE for each
    best_test = adapter_data.loc[
        adapter_data.groupby("n_components")["test_mse"].idxmin()
    ]

    # Plot validation line (solid)
    sns.lineplot(
        data=best_val,
        x="n_components",
        y="test_mse",
        label=f"{adapter} (val)",
        marker="o",
        color=color,
    )
    # Plot test line (dashed)
    sns.lineplot(
        data=best_test,
        x="n_components",
        y="test_mse",
        label=f"{adapter} (test)",
        marker="s",
        color=color,
        linestyle="--",
    )

# Plot the baseline
baseline = filtered_df[filtered_df["adapter"] == "no_adapter"]["value"].mean()
baseline_std = filtered_df[filtered_df["adapter"] == "no_adapter"]["value"].std()
plt.axhline(y=baseline, color="black", linestyle="--", label="baseline")
plt.fill_between(
    [filtered_df["n_components"].min(), filtered_df["n_components"].max()],
    baseline - baseline_std,
    baseline + baseline_std,
    color="black",
    alpha=0.1,
)

if "pca" in filtered_df["adapter"].unique():
    pca_data = filtered_df.loc[filtered_df["adapter"] == "pca"]
    sns.lineplot(
        data=pca_data,
        x="n_components",
        y="value",
        label="PCA",
        marker="+",
        color="red",
        linestyle=":",
    )

plt.title(
    f"metric '{args.metric.upper()}' on {args.dataset}_pred={args.forecasting_horizon} "
    f"({args.model.split('/')[-1]})."
)

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    f"figures/hyperopt/{args.dataset}_pred={args.forecasting_horizon}_"
    f"{args.model.split('/')[-1]}_{args.metric}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
