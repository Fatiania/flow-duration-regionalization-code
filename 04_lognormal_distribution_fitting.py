"""Script 04: Fit log-normal distribution to normalized non-zero flow data.

This script reads station Excel files, extracts a flow column from each sheet,
keeps positive flow values, normalizes by the station/sheet mean, transforms the
normalized values to log space, fits a normal distribution in log space, and
creates Q-Q plots and summary metrics.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from utils import as_list, ensure_dir, load_config, resolve_path, safe_name

CM_TO_INCH = 1 / 2.54


def calculate_metrics(observed: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """Calculate common fit metrics between observed and fitted log quantiles."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    differences = predicted - observed
    pbias = np.sum(differences) / np.sum(observed) * 100 if np.sum(observed) != 0 else np.nan
    rmse = np.sqrt(np.mean(differences ** 2))
    mae = np.mean(np.abs(differences))
    nonzero_mask = observed != 0
    mape = np.mean(np.abs(differences[nonzero_mask] / observed[nonzero_mask])) * 100 if np.any(nonzero_mask) else np.nan
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    nse = 1 - np.sum(differences ** 2) / denominator if denominator != 0 else np.nan
    r_squared = 1 - np.sum(differences ** 2) / denominator if denominator != 0 else np.nan
    return pbias, rmse, mae, mape, nse, r_squared


def configure_plot_style(save_dpi: int) -> None:
    """Apply compact journal-style plotting settings."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "savefig.dpi": save_dpi,
        "figure.dpi": 900,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def get_station_label(file_path: Path, config: dict) -> str:
    """Create the station label used in Q-Q plot panels."""
    mode = str(config.get("station_label_mode", "first_letter_plus_second_token")).lower()
    parts = file_path.stem.split("_")
    if mode == "filename":
        return file_path.stem
    if mode == "first_token":
        return parts[0] if parts else file_path.stem
    if mode == "first_letter_plus_second_token" and len(parts) >= 2 and parts[0].strip():
        return f"{parts[0].strip()[0].upper()}{parts[1].strip()}"
    return file_path.stem


def get_group_name(file_path: Path, config: dict) -> str:
    """Create catchment/group name for grouped plots."""
    mode = str(config.get("group_name_mode", "first_token")).lower()
    parts = file_path.stem.split("_")
    if mode == "none":
        return "All"
    if mode == "filename":
        return file_path.stem
    return parts[0].strip() if parts and parts[0].strip() else "Unknown"


def select_value_column(sheet_data: pd.DataFrame, config: dict) -> pd.Series:
    """Select the column containing flow/discharge values."""
    column_name = config.get("value_column_name")
    column_index = int(config.get("value_column_index", 0))
    if column_name:
        if column_name not in sheet_data.columns:
            raise KeyError(f"Configured value_column_name '{column_name}' not found. Available columns: {list(sheet_data.columns)}")
        return sheet_data[column_name]
    if column_index >= sheet_data.shape[1]:
        raise IndexError(f"value_column_index={column_index} is outside sheet width {sheet_data.shape[1]}")
    return sheet_data.iloc[:, column_index]


def transform_values(values: np.ndarray, config: dict) -> np.ndarray:
    """Transform normalized values into log space."""
    log_base = str(config.get("log_base", "10")).lower()
    if log_base in {"10", "log10"}:
        return np.log10(values)
    if log_base in {"e", "natural", "ln"}:
        return np.log(values)
    raise ValueError("log_base must be one of: '10', 'log10', 'e', 'natural', or 'ln'.")


def save_master_combined_qq_figure(
    plot_items: list[dict],
    output_png: Path,
    output_pdf: Path,
    panel_width_cm: float,
    panel_height_cm: float,
    save_dpi: int,
    title_text: str | None,
    xlabel_fontsize: int,
    ylabel_fontsize: int,
) -> None:
    """Save a multi-panel Q-Q figure."""
    if not plot_items:
        print(f"No plot items available for {output_png.name}")
        return

    n_plots = len(plot_items)
    if n_plots <= 4:
        ncols = 2
    elif n_plots <= 12:
        ncols = 3
    elif n_plots <= 24:
        ncols = 4
    else:
        ncols = 5
    nrows = math.ceil(n_plots / ncols)

    fig_width_cm = ncols * panel_width_cm
    fig_height_cm = nrows * panel_height_cm + 1.2
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width_cm * CM_TO_INCH, fig_height_cm * CM_TO_INCH))
    axes = np.atleast_1d(axes).flatten()

    all_theoretical = np.concatenate([item["theoretical_quantiles"] for item in plot_items])
    all_observed = np.concatenate([item["observed_quantiles"] for item in plot_items])
    global_min = min(np.min(all_theoretical), np.min(all_observed))
    global_max = max(np.max(all_theoretical), np.max(all_observed))
    padding = 0.05 * (global_max - global_min) if global_max > global_min else 0.1
    axis_min = global_min - padding
    axis_max = global_max + padding

    for axis, item in zip(axes, plot_items):
        axis.scatter(item["theoretical_quantiles"], item["observed_quantiles"], s=4, facecolors="white", edgecolors="black", linewidths=0.4)
        axis.plot([axis_min, axis_max], [axis_min, axis_max], color="black", linewidth=0.9)
        axis.set_xlim(axis_min, axis_max)
        axis.set_ylim(axis_min, axis_max)
        axis.grid(False)
        fig.canvas.draw()
        x_ticks = axis.get_xticks()
        axis.set_yticks(x_ticks[(x_ticks >= axis_min) & (x_ticks <= axis_max)])
        axis.text(0.03, 0.97, item["station_label"], transform=axis.transAxes, ha="left", va="top", fontsize=7, fontweight="bold")
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)

    for axis in axes[n_plots:]:
        axis.axis("off")

    fig.supxlabel("Theoretical quantiles", fontsize=xlabel_fontsize, y=0.04)
    fig.supylabel("Observed quantiles", fontsize=ylabel_fontsize, x=0.04)
    if title_text:
        fig.suptitle(title_text, fontsize=18, fontweight="bold", y=0.995)
        fig.tight_layout(rect=[0.05, 0.05, 1, 0.97])
    else:
        fig.tight_layout(rect=[0.05, 0.05, 1, 1.0])
    fig.savefig(output_png, dpi=save_dpi)
    fig.savefig(output_pdf)
    plt.close(fig)


def save_overlay_qq_plot(plot_items: list[dict], output_png: Path, output_pdf: Path, save_dpi: int) -> None:
    """Save all station Q-Q points in a single axis."""
    if not plot_items:
        print(f"No plot items available for {output_png.name}")
        return

    fig, axis = plt.subplots(figsize=(14 * CM_TO_INCH, 12 * CM_TO_INCH))
    all_theoretical = np.concatenate([item["theoretical_quantiles"] for item in plot_items])
    all_observed = np.concatenate([item["observed_quantiles"] for item in plot_items])
    global_min = min(np.min(all_theoretical), np.min(all_observed))
    global_max = max(np.max(all_theoretical), np.max(all_observed))
    padding = 0.05 * (global_max - global_min) if global_max > global_min else 0.1
    axis_min = global_min - padding
    axis_max = global_max + padding

    for item in plot_items:
        axis.scatter(item["theoretical_quantiles"], item["observed_quantiles"], s=2, alpha=0.35, linewidths=0, edgecolors="none")
    axis.plot([axis_min, axis_max], [axis_min, axis_max], color="black", linewidth=1.0)
    axis.set_xlim(axis_min, axis_max)
    axis.set_ylim(axis_min, axis_max)
    axis.set_xlabel("Theoretical quantiles", fontsize=14)
    axis.set_ylabel("Observed quantiles", fontsize=14)
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)
    fig.tight_layout()
    fig.savefig(output_png, dpi=save_dpi)
    fig.savefig(output_pdf)
    plt.close(fig)


def run_lognormal_fitting(config: dict) -> None:
    input_dir = resolve_path(config["input_dir"])
    output_dir = ensure_dir(resolve_path(config["output_dir"]))
    file_pattern = str(config.get("input_file_pattern", "*.xlsx"))
    sheet_filter = set(as_list(config.get("sheet_names"), "sheet_names"))
    save_dpi = int(config.get("save_dpi", 600))
    panel_width_cm = float(config.get("panel_width_cm", 4.8))
    panel_height_cm = float(config.get("panel_height_cm", 4.2))
    normalize_by_mean = bool(config.get("normalize_by_mean", True))
    use_positive_only = bool(config.get("use_positive_only", True))

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    configure_plot_style(save_dpi)

    group_plot_dir = ensure_dir(output_dir / "Catchment_Group_Plots")
    group_map_dir = ensure_dir(output_dir / "Catchment_Group_Panel_Maps")

    all_plot_items: list[dict] = []
    panel_map_records: list[dict] = []
    summary_records: list[dict] = []
    group_plot_items: dict[str, list[dict]] = {}
    group_panel_map_records: dict[str, list[dict]] = {}

    for file_path in sorted(input_dir.glob(file_pattern)):
        if file_path.name.startswith("~$"):
            continue
        print(f"Processing file: {file_path.name}")
        try:
            workbook = pd.ExcelFile(file_path)
            station_label = get_station_label(file_path, config)
            group_name = get_group_name(file_path, config)
            group_plot_items.setdefault(group_name, [])
            group_panel_map_records.setdefault(group_name, [])

            for sheet_name in workbook.sheet_names:
                if sheet_filter and sheet_name not in sheet_filter:
                    continue
                try:
                    sheet = workbook.parse(sheet_name)
                    values = pd.to_numeric(select_value_column(sheet, config), errors="coerce").dropna()
                    if use_positive_only:
                        values = values[values > 0]
                    if values.empty:
                        print(f"Skipping empty/invalid sheet: {file_path.name} / {sheet_name}")
                        continue

                    sorted_values = np.sort(values.to_numpy(dtype=float))
                    divisor = np.mean(sorted_values) if normalize_by_mean else 1.0
                    if divisor == 0 or not np.isfinite(divisor):
                        print(f"Skipping sheet with invalid mean: {file_path.name} / {sheet_name}")
                        continue

                    normalized_values = sorted_values / divisor
                    log_values = transform_values(normalized_values, config)
                    probabilities = np.arange(1, len(sorted_values) + 1) / (len(sorted_values) + 1)
                    fitted_mu, fitted_sigma = norm.fit(log_values)
                    if np.isfinite(fitted_sigma) and fitted_sigma > 0:
                        theoretical_quantiles = norm.ppf(probabilities, loc=fitted_mu, scale=fitted_sigma)
                    else:
                        theoretical_quantiles = np.full_like(probabilities, fill_value=fitted_mu, dtype=float)

                    pbias, rmse, mae, mape, nse, r_squared = calculate_metrics(log_values, theoretical_quantiles)
                    plot_item = {
                        "theoretical_quantiles": theoretical_quantiles,
                        "observed_quantiles": log_values,
                        "station_label": station_label,
                    }
                    all_plot_items.append(plot_item)
                    group_plot_items[group_name].append(plot_item)

                    panel_record = {
                        "Station Label": station_label,
                        "Group": group_name,
                        "File Name": file_path.name,
                        "Sheet Name": sheet_name,
                    }
                    panel_map_records.append(panel_record)
                    group_panel_map_records[group_name].append(panel_record)
                    summary_records.append({
                        **panel_record,
                        "N positive values": int(len(sorted_values)),
                        "Normalized by mean": bool(normalize_by_mean),
                        "Log base": str(config.get("log_base", "10")),
                        "PBIAS (%)": pbias,
                        "RMSE": rmse,
                        "MAE": mae,
                        "MAPE (%)": mape,
                        "NSE": nse,
                        "R2": r_squared,
                        "Fitted mean in log space": fitted_mu,
                        "Fitted std dev in log space": fitted_sigma,
                    })
                except Exception as sheet_error:
                    print(f"Error processing {file_path.name} / {sheet_name}: {sheet_error}")
        except Exception as file_error:
            print(f"Error processing file {file_path.name}: {file_error}")

    save_master_combined_qq_figure(
        all_plot_items,
        output_dir / "All_QQ_Plots_Combined.png",
        output_dir / "All_QQ_Plots_Combined.pdf",
        panel_width_cm,
        panel_height_cm,
        save_dpi,
        None,
        20,
        20,
    )
    save_overlay_qq_plot(
        all_plot_items,
        output_dir / "All_Stations_Overlay_QQ_Plot.png",
        output_dir / "All_Stations_Overlay_QQ_Plot.pdf",
        save_dpi,
    )
    pd.DataFrame(panel_map_records).to_csv(output_dir / "All_QQ_Plots_Panel_Map.csv", index=False)

    for group_name, items in group_plot_items.items():
        if not items:
            continue
        group_safe = safe_name(group_name)
        save_master_combined_qq_figure(
            items,
            group_plot_dir / f"{group_safe}_QQ_Plots_Combined.png",
            group_plot_dir / f"{group_safe}_QQ_Plots_Combined.pdf",
            panel_width_cm,
            panel_height_cm,
            save_dpi,
            None,
            14,
            14,
        )
        pd.DataFrame(group_panel_map_records[group_name]).to_csv(group_map_dir / f"{group_safe}_Panel_Map.csv", index=False)

    summary = pd.DataFrame(summary_records)
    summary.to_csv(output_dir / "Lognormal_Fitting_Summary.csv", index=False)
    summary.to_excel(output_dir / "Lognormal_Fitting_Summary.xlsx", index=False, engine="openpyxl")
    print("Log-normal fitting analysis finished successfully.")
    print(f"Outputs saved in: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit log-normal distribution to normalized non-zero flow data.")
    parser.add_argument("--config", default="config_template.json", help="Path to JSON configuration file.")
    parser.add_argument("--section", default="lognormal_distribution_fitting", help="Configuration section name.")
    args = parser.parse_args()
    run_lognormal_fitting(load_config(args.config, args.section))


if __name__ == "__main__":
    main()
