import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def scatter_plot_speed_vs_accuracy(df, x_col, y_col, hue_col, size_col, title):
    """
    Creates a scatter (bubble) plot with customizable x and y axes.

    Parameters:
    - df: DataFrame containing the data
    - x_col: Column name for the x-axis (e.g., throughput, latency)
    - y_col: Column name for the y-axis (e.g., accuracy, latency)
    - hue_col: Column name for the hue (color grouping, e.g., model name)
    - size_col: Column name for the size (bubble size, e.g., data fraction)
    - title: Custom plot title
    """
    plt.figure(figsize=(7, 5))
    scatter = sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        size=size_col,
        sizes=(20, 400),
        alpha=0.7,
        palette="turbo"
    )
    x_label_cleaned = x_col.replace("eval_", "").replace("_", " ").title()
    y_label_cleaned = y_col.replace("eval_", "").replace("_", " ").title()
    scatter.set_title(title)
    scatter.set_xlabel(x_label_cleaned)
    scatter.set_ylabel(y_label_cleaned)
    handles, labels = scatter.get_legend_handles_labels()
    model_names = df[hue_col].astype(str).unique()
    fraction_values = sorted(df[size_col].astype(str).unique())
    hue_handles = [h for h, l in zip(handles, labels) if l in model_names]
    hue_labels = [l for l in labels if l in model_names]
    size_handles = [h for h, l in zip(handles, labels) if l in fraction_values]
    size_labels = [l for l in labels if l in fraction_values]
    hue_legend = plt.legend(hue_handles, hue_labels, title="Model Name", loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.gca().add_artist(hue_legend)
    size_legend = plt.legend(size_handles, size_labels, title="Fraction Raw", loc="lower left", bbox_to_anchor=(1.05, 0))
    plt.tight_layout()
    plt.show()


def heatmap_label_f1_scores(
    df,
    fraction_col="fraction_label",
    model_label_col="model_label",
    f1_score_prefix="eval_",
    f1_score_suffix="_f1-score",
    avg_identifier="avg",
    title="F1 Scores for Each Label Across Models",
    note=None,
    figsize=(10, 7),
    cmap="cividis"
):
    """
    Plots a heatmap of F1 scores for each label across models, for a given fraction of data.
    Parameters:
    - df: DataFrame containing the results
    - fraction_col: Column indicating data fraction (default: 'fraction_label')
    - model_label_col: Column for model names (default: 'model_label')
    - f1_score_prefix: Prefix for F1-score columns (default: 'eval_')
    - f1_score_suffix: Suffix for F1-score columns (default: '_f1-score')
    - avg_identifier: Substring to exclude average columns (default: 'avg')
    - title: Plot title
    - note: Optional note to add below the plot
    - figsize: Figure size
    - cmap: Colormap for heatmap
    """
    # Filter for 100% data (or the max fraction if not present)
    if df[fraction_col].dtype == float or df[fraction_col].dtype == int:
        max_fraction = df[fraction_col].max()
        df_100 = df[df[fraction_col] == max_fraction]
    else:
        df_100 = df[df[fraction_col] == "100.0%"]
        if df_100.empty:
            # fallback: use max fraction
            max_fraction = df[fraction_col].max()
            df_100 = df[df[fraction_col] == max_fraction]
    # Identify label F1 columns (not averages)
    label_f1_columns = [
        col for col in df_100.columns
        if col.startswith(f1_score_prefix) and col.endswith(f1_score_suffix) and avg_identifier not in col
    ]
    # Prepare DataFrame for heatmap
    df_f1_100 = df_100[[model_label_col] + label_f1_columns]
    df_f1_melted_100 = df_f1_100.melt(id_vars=[model_label_col], var_name="Label", value_name="F1 Score")
    df_f1_melted_100["Label"] = (
        df_f1_melted_100["Label"]
        .str.replace(f1_score_prefix, "")
        .str.replace(f1_score_suffix, "")
        .str.replace("_", " ")
        .str.title()
    )
    df_f1_pivot_100 = df_f1_melted_100.pivot(index=model_label_col, columns="Label", values="F1 Score")
    # Remove 'Incident Summary' if present
    df_f1_pivot_100 = df_f1_pivot_100.drop(columns="Incident Summary", errors="ignore")
    # Sort event types (columns) by their average F1 score across models (descending order)
    event_order = df_f1_pivot_100.mean().sort_values(ascending=False).index
    df_f1_pivot_100 = df_f1_pivot_100[event_order]
    # Sort models (rows) by their Micro F1 score (descending order) if available
    if "eval_micro avg_f1-score" in df_100.columns:
        df_model_avg_f1 = df_100.set_index(model_label_col)["eval_micro avg_f1-score"].sort_values(ascending=False)
        model_order = df_model_avg_f1.index
        df_f1_pivot_100 = df_f1_pivot_100.loc[model_order]
    # Plot
    plt.figure(figsize=figsize)
    ax = sns.heatmap(df_f1_pivot_100, annot=True, fmt=".2f", cmap=cmap, linewidths=0.5, linecolor="gray", cbar_kws={'label': 'F1 Score'})
    ax.set_title(title, pad=20)
    if note:
        plt.figtext(0.5, -0.08, note, ha="center", fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.xaxis.tick_bottom()
    ax.xaxis.set_label_position('bottom')
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show() 