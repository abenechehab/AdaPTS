import pandas as pd
import tyro
from dataclasses import dataclass

import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
df = pd.read_csv("results/data.csv")
df["adapter"] = df["adapter"].fillna("no_adapter")


@dataclass
class Args:
    model: str = "AutonLab/MOMENT-1-large"
    metric: str = "mse"
    dataset: str = "ETTh1"
    forecasting_horizon: int = 96
    is_fine_tuned: bool = False


# Parse command line arguments
args = tyro.cli(Args)

if args.is_fine_tuned:
    filtered_df = df.loc[
        (df["foundational_model"] == args.model)
        & (df["is_fine_tuned"].isin(["True"]))
        & (df["metric"] == args.metric)
        & (df["dataset"] == args.dataset)
        & (df["forecasting_horizon"] == args.forecasting_horizon)
    ]
else:
    filtered_df = df.loc[
        (df["foundational_model"] == args.model)
        & (~df["is_fine_tuned"].isin(["True"]))
        & (df["metric"] == args.metric)
        & (df["dataset"] == args.dataset)
        & (df["forecasting_horizon"] == args.forecasting_horizon)
    ]

# Create the plot
plt.figure(figsize=(10, 6))

sns.lineplot(
    data=filtered_df.loc[filtered_df["adapter"] != "no_adapter"],
    x="n_components",
    y="value",
    hue="adapter",
    markers=True,
    dashes=False,
)

baseline = filtered_df[filtered_df["adapter"] == "no_adapter"]["value"].iloc[0]
plt.axhline(y=baseline, color="black", linestyle="--", label="no_adapter")

plt.title(
    f"metric '{args.metric.upper()}' on {args.dataset}_pred={args.forecasting_horizon} "
    f"({args.model.split('/')[-1]}). fine_tuning={args.is_fine_tuned}"
)

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    f"figures/viz_comparison/{args.dataset}_pred={args.forecasting_horizon}_"
    f"{args.model.split('/')[-1]}_ft={args.is_fine_tuned}_{args.metric}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
