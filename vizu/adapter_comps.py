import pandas as pd
import tyro
from dataclasses import dataclass

import matplotlib.pyplot as plt

# Read the data
df = pd.read_csv("results/data.csv")


@dataclass
class Args:
    model: str = "AutonLab/MOMENT-1-large"
    metric: str = "mse"
    dataset: str = "ETTh1"
    adapter: str = "pca"
    forecasting_horizon: int = 96


# Parse command line arguments
args = tyro.cli(Args)

# Get baseline values (no adapter)
filtered_df = df[
    (df["adapter"].isna())
    & (df["foundational_model"] == args.model)
    & (~df["is_fine_tuned"])
    & (df["metric"] == args.metric)
    & (df["dataset"] == args.dataset)
    & (df["forecasting_horizon"] == args.forecasting_horizon)
]["value"]
no_adapter_base = filtered_df.iloc[0] if not filtered_df.empty else float("nan")

filtered_df = df[
    (df["adapter"].isna())
    & (df["foundational_model"] == args.model)
    & (df["is_fine_tuned"])
    & (df["metric"] == args.metric)
    & (df["dataset"] == args.dataset)
    & (df["forecasting_horizon"] == args.forecasting_horizon)
]["value"]
no_adapter_ft = filtered_df.iloc[0] if not filtered_df.empty else float("nan")

# Get PCA data
pca_base = df[
    (df["adapter"] == args.adapter)
    & (df["foundational_model"] == args.model)
    & (~df["is_fine_tuned"])
    & (df["metric"] == args.metric)
    & (df["dataset"] == args.dataset)
    & (df["forecasting_horizon"] == args.forecasting_horizon)
].sort_values("n_components")

pca_ft = df[
    (df["adapter"] == args.adapter)
    & (df["foundational_model"] == args.model)
    & (df["is_fine_tuned"])
    & (df["metric"] == args.metric)
    & (df["dataset"] == args.dataset)
    & (df["forecasting_horizon"] == args.forecasting_horizon)
].sort_values("n_components")

# Create the plot
plt.figure(figsize=(10, 6))

# Plot PCA curves
plt.plot(
    pca_base["n_components"],
    pca_base["value"],
    "o-",
    color="lightblue",
    label=args.adapter,
)
plt.plot(
    pca_ft["n_components"],
    pca_ft["value"],
    "o-",
    color="darkblue",
    label=f"{args.adapter} + Fine-tuning",
)

# Plot baseline horizontal lines
plt.axhline(y=no_adapter_base, color="lightblue", linestyle="--", label="No adapter")
plt.axhline(
    y=no_adapter_ft, color="darkblue", linestyle="--", label="No adapter + Fine-tuning"
)
# Add confidence intervals for PCA curves
if len(pca_base.groupby("seed")) > 1:
    mean_base = pca_base.groupby("n_components")["value"].mean()
    std_base = pca_base.groupby("n_components")["value"].std()
    plt.fill_between(
        mean_base.index,
        mean_base - std_base,
        mean_base + std_base,
        alpha=0.2,
        color="lightblue",
    )

if len(pca_ft.groupby("seed")) > 1:
    mean_ft = pca_ft.groupby("n_components")["value"].mean()
    std_ft = pca_ft.groupby("n_components")["value"].std()
    plt.fill_between(
        mean_ft.index, mean_ft - std_ft, mean_ft + std_ft, alpha=0.2, color="darkblue"
    )

plt.xlabel("Number of Components")
plt.ylabel(f"{args.metric.upper()}")
plt.title(
    f"{args.adapter} Adapter Performance on "
    f"{args.dataset}_pred={args.forecasting_horizon} ({args.model.split('/')[-1]})"
)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    f"figures/viz_ada/{args.dataset}_pred={args.forecasting_horizon}_"
    f"{args.model.split('/')[-1]}_{args.adapter}_{args.metric}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
