"""Shared DV-PC regression workflow for capped and non-capped analyses."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler

from utils import (
    as_list,
    build_equation_label,
    build_input_equation_label,
    ensure_dir,
    find_station_col,
    get_station_column,
    load_config,
    numeric_frame,
    require_columns,
    resolve_path,
    safe_name,
    zscore,
)


DEFAULT_STATION_COLORS = {
    "napoli": "red",
    "catanzaro": "blue",
    "bari": "green",
}


def plot_corr_heatmap(corr_df: pd.DataFrame, title: str, out_path: Path, cbar_label: str) -> None:
    """Save a blue-white-red heatmap of correlations."""
    nrows, ncols = corr_df.shape
    fig_w = max(6.0, 1.6 * ncols)
    fig_h = max(3.0, 0.75 * nrows + 1.5)
    scale = max(nrows, ncols, 1)
    cell_fs = max(10, int(min(18, 220 / scale)))

    plt.figure(figsize=(fig_w, fig_h))
    image = plt.imshow(corr_df.values, aspect="auto", interpolation="nearest", cmap="bwr", vmin=-1, vmax=1)
    plt.colorbar(image, label=cbar_label)
    plt.xticks(range(ncols), corr_df.columns, rotation=45, ha="right")
    plt.yticks(range(nrows), corr_df.index)
    plt.title(title, fontsize=14, fontweight="bold")

    for i in range(nrows):
        for j in range(ncols):
            value = corr_df.iat[i, j]
            if np.isfinite(value):
                plt.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=cell_fs, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def infer_parameter_columns_from_loadings(loadings_file: Path) -> list[str]:
    """Read PCA loadings index and use it as the parameter list."""
    if not loadings_file.exists():
        raise FileNotFoundError(f"PCA loadings file not found: {loadings_file}")
    loadings = pd.read_excel(loadings_file, index_col=0)
    return [str(item) for item in loadings.index.tolist()]


def infer_pc_columns(loadings_file: Path, configured: list[str]) -> list[str]:
    """Use configured PC columns, or infer columns named PC1, PC2, ... from loadings."""
    if configured:
        return configured
    loadings = pd.read_excel(loadings_file, index_col=0)
    pc_columns = [str(col) for col in loadings.columns if str(col).lower().startswith("pc")]
    if not pc_columns:
        raise ValueError("No pc_columns were configured, and no PC columns were found in the PCA loadings file.")
    return pc_columns


def prepare_pca_transform_information(
    pca_input_file: Path,
    loadings_file: Path,
    parameter_columns: list[str],
    log_columns: list[str],
    pc_columns: list[str],
    output_dir: Path,
    skip_bad_rows: bool,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, int, pd.DataFrame | None]:
    """Load PCA loadings and reconstruct the input means/stds used for PCA."""
    if not pca_input_file.exists():
        raise FileNotFoundError(f"PCA input parameter file not found: {pca_input_file}")
    if not loadings_file.exists():
        raise FileNotFoundError(f"PCA loadings file not found: {loadings_file}")

    loadings_all = pd.read_excel(loadings_file, index_col=0)
    missing_loadings_vars = [var for var in parameter_columns if var not in loadings_all.index]
    missing_pcs = [pc for pc in pc_columns if pc not in loadings_all.columns]
    if missing_loadings_vars:
        raise KeyError(f"The PCA loadings file does not contain these parameter rows: {missing_loadings_vars}")
    if missing_pcs:
        raise KeyError(f"The PCA loadings file does not contain these PC columns: {missing_pcs}")
    loadings = loadings_all.loc[parameter_columns, pc_columns].copy()

    pca_input = pd.read_excel(pca_input_file)
    require_columns(pca_input, parameter_columns, str(pca_input_file))
    station_column = get_station_column(pca_input, None)
    x_pca = numeric_frame(pca_input, parameter_columns)

    invalid_reasons = pd.Series("", index=x_pca.index, dtype=object)
    for col in parameter_columns:
        invalid_reasons.loc[x_pca[col].isna()] += f"{col}: missing/non-numeric; "
    for col in log_columns:
        invalid_reasons.loc[x_pca[col].notna() & (x_pca[col] <= 0)] += f"{col}: <= 0, cannot apply ln; "

    bad_rows = invalid_reasons.str.len() > 0
    skipped_rows = None
    if bad_rows.any():
        skipped_rows = pca_input.loc[bad_rows, [station_column] + parameter_columns].copy()
        skipped_rows.insert(1, "Reason_skipped", invalid_reasons.loc[bad_rows].str.rstrip("; "))
        skipped_rows.to_excel(output_dir / "PCA_input_rows_SKIPPED_or_INVALID.xlsx", index=False)
        if not skip_bad_rows:
            raise ValueError(
                f"The PCA input parameter file contains {int(bad_rows.sum())} invalid rows. "
                "Fix these rows or set skip_bad_rows_with_missing_or_invalid_log=true."
            )

    x_valid = x_pca.loc[~bad_rows].copy()
    if x_valid.shape[0] < 2:
        raise ValueError("At least two valid parameter rows are required to reconstruct PCA preprocessing.")
    for col in log_columns:
        x_valid[col] = np.log(x_valid[col])

    scaler = StandardScaler()
    scaler.fit(x_valid.values)
    x_mean = pd.Series(scaler.mean_, index=parameter_columns, name="mean")
    x_std = pd.Series(scaler.scale_, index=parameter_columns, name="std")
    return loadings, x_mean, x_std, int(bad_rows.sum()), skipped_rows


def should_apply_unit_cap(dv_name: str, y: pd.Series, config: dict) -> bool:
    """Determine whether capped regression should clip a DV to configured bounds."""
    mode = str(config.get("cap_mode", "all")).lower()
    if mode == "all":
        return True
    if mode in {"none", "off", "false"}:
        return False

    keywords = [k.lower() for k in as_list(config.get("unit_interval_keywords"), "unit_interval_keywords")]
    tolerance = float(config.get("unit_range_tolerance", 0.05))
    dv_lower = str(dv_name).lower()
    by_name = any(k in dv_lower for k in keywords)
    y_num = pd.to_numeric(y, errors="coerce")
    y_min = float(np.nanmin(y_num.values)) if np.isfinite(np.nanmin(y_num.values)) else np.nan
    y_max = float(np.nanmax(y_num.values)) if np.isfinite(np.nanmax(y_num.values)) else np.nan
    by_range = np.isfinite(y_min) and np.isfinite(y_max) and y_min >= -tolerance and y_max <= 1.0 + tolerance
    return bool(by_name or by_range)


def load_and_combine_dv_pc(config: dict, pc_columns: list[str]) -> tuple[pd.DataFrame, str, list[str], str]:
    """Load the DV table and PCA score table, then merge by station if possible."""
    dv_file = resolve_path(config["dv_file"])
    pc_scores_file = resolve_path(config["pc_scores_file"])
    if not dv_file.exists():
        raise FileNotFoundError(f"DV file not found: {dv_file}")
    if not pc_scores_file.exists():
        raise FileNotFoundError(f"PCA scores file not found: {pc_scores_file}")

    dv_df = pd.read_excel(dv_file)
    dv_station_col = get_station_column(dv_df, config.get("dv_station_column"))
    dv_df[dv_station_col] = dv_df[dv_station_col].astype(str).str.strip()
    dv_columns = [col for col in dv_df.columns if col != dv_station_col]

    pc_header = pd.read_excel(pc_scores_file, nrows=0)
    pc_station_col = config.get("pc_station_column") or find_station_col(pc_header.columns)
    usecols = list(pc_columns)
    if pc_station_col is not None:
        usecols = [pc_station_col] + usecols
    scores = pd.read_excel(pc_scores_file, usecols=usecols)

    if pc_station_col is not None:
        scores[pc_station_col] = scores[pc_station_col].astype(str).str.strip()
        scores = scores.rename(columns={pc_station_col: dv_station_col})
        combined = dv_df[[dv_station_col] + dv_columns].merge(scores, on=dv_station_col, how="inner")
        combine_mode = "MERGE on station/name column (INNER)"
    else:
        combined = pd.concat([dv_df[[dv_station_col] + dv_columns], scores], axis=1)
        combine_mode = "CONCAT by row order because no station/name column was found in PCA scores"

    for col in dv_columns + pc_columns:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
    return combined, dv_station_col, dv_columns, combine_mode


def make_correlation_outputs(combined: pd.DataFrame, dv_columns: list[str], pc_columns: list[str], heatmap_dir: Path, dv_tag: str):
    """Compute correlations and save heatmaps/bar plots."""
    corr_pearson = combined[dv_columns + pc_columns].corr(method="pearson")
    dv_pc_pearson = corr_pearson.loc[dv_columns, pc_columns]
    corr_spearman = combined[dv_columns + pc_columns].corr(method="spearman")
    dv_pc_spearman = corr_spearman.loc[dv_columns, pc_columns]

    spearman_heatmap = heatmap_dir / f"{dv_tag}_DV_vs_PC_SPEARMAN_heatmap.png"
    pearson_heatmap = heatmap_dir / f"{dv_tag}_DV_vs_PC_PEARSON_heatmap.png"
    plot_corr_heatmap(dv_pc_spearman, "Spearman correlation (DV vs PCs)", spearman_heatmap, "Spearman rho")
    plot_corr_heatmap(dv_pc_pearson, "Pearson correlation (DV vs PCs)", pearson_heatmap, "Pearson r")

    pearson_each_dir = ensure_dir(heatmap_dir / "PEARSON_EACH_DV")
    spearman_each_dir = ensure_dir(heatmap_dir / "SPEARMAN_EACH_DV")
    pearson_bar_dir = ensure_dir(heatmap_dir / "PEARSON_BAR_PLOTS")
    spearman_bar_dir = ensure_dir(heatmap_dir / "SPEARMAN_BAR_PLOTS")

    for dv in dv_columns:
        plot_corr_heatmap(
            dv_pc_pearson.loc[[dv], pc_columns],
            f"{dv} - Pearson correlation (DV vs PCs)",
            pearson_each_dir / f"{dv_tag}_{safe_name(dv)}_PEARSON.png",
            "Pearson r",
        )
        plot_corr_heatmap(
            dv_pc_spearman.loc[[dv], pc_columns],
            f"{dv} - Spearman correlation (DV vs PCs)",
            spearman_each_dir / f"{dv_tag}_{safe_name(dv)}_SPEARMAN.png",
            "Spearman rho",
        )
        for corr_df, out_dir, ylabel, suffix in [
            (dv_pc_spearman, spearman_bar_dir, "Spearman rho", "spearman"),
            (dv_pc_pearson, pearson_bar_dir, "Pearson r", "pearson"),
        ]:
            values = corr_df.loc[dv, pc_columns].astype(float)
            colors = plt.cm.bwr(plt.Normalize(-1, 1)(values.values))
            plt.figure(figsize=(6, 4))
            plt.bar(pc_columns, values.values, color=colors)
            plt.ylim(-1, 1)
            plt.axhline(0, linewidth=1)
            plt.ylabel(ylabel)
            plt.title(f"{dv} - {ylabel} correlation (DV vs PCs)", fontsize=12, fontweight="bold")
            plt.tight_layout()
            plt.savefig(out_dir / f"{dv_tag}_{safe_name(dv)}_{suffix}_bar.png", dpi=300)
            plt.close()

    return corr_pearson, corr_spearman, dv_pc_pearson, dv_pc_spearman, spearman_heatmap, pearson_heatmap


def run_regression_from_config(config_path: str, section: str, force_capped: bool) -> None:
    """Entry point used by the two regression scripts."""
    config = load_config(config_path, section)
    run_regression_analysis(config, force_capped=force_capped)


def run_regression_analysis(config: dict, force_capped: bool = False) -> None:
    """Run DV-PC correlation and OLS regression analysis."""
    output_dir = ensure_dir(resolve_path(config["output_dir"]))
    plot_dir = ensure_dir(output_dir / "OBS_vs_PRED_PLOTS")
    heatmap_dir = ensure_dir(output_dir / "CORRELATION_HEATMAPS")
    excel_out = output_dir / "DV_PC_correlation_and_REGRESSION_REPORT.xlsx"

    pca_loadings_file = resolve_path(config["pca_loadings_file"])
    pca_input_file = resolve_path(config["pca_input_parameters"])

    parameter_columns = as_list(config.get("parameter_columns"), "parameter_columns")
    if not parameter_columns:
        parameter_columns = infer_parameter_columns_from_loadings(pca_loadings_file)
    log_columns = as_list(config.get("log_transform_columns"), "log_transform_columns")
    missing_log = [col for col in log_columns if col not in parameter_columns]
    if missing_log:
        raise KeyError(f"log_transform_columns must be included in parameter_columns. Problem columns: {missing_log}")

    pc_columns = infer_pc_columns(pca_loadings_file, as_list(config.get("pc_columns"), "pc_columns"))
    skip_bad_rows = bool(config.get("skip_bad_rows_with_missing_or_invalid_log", True))

    loadings, x_mean, x_std, n_bad_rows, skipped_rows = prepare_pca_transform_information(
        pca_input_file=pca_input_file,
        loadings_file=pca_loadings_file,
        parameter_columns=parameter_columns,
        log_columns=log_columns,
        pc_columns=pc_columns,
        output_dir=output_dir,
        skip_bad_rows=skip_bad_rows,
    )

    combined, dv_station_col, dv_columns, combine_mode = load_and_combine_dv_pc(config, pc_columns)
    dv_tag = safe_name(Path(config["dv_file"]).stem)
    corr_pearson, corr_spearman, dv_pc_pearson, dv_pc_spearman, spearman_heatmap, pearson_heatmap = make_correlation_outputs(
        combined=combined,
        dv_columns=dv_columns,
        pc_columns=pc_columns,
        heatmap_dir=heatmap_dir,
        dv_tag=dv_tag,
    )

    station_colors = dict(DEFAULT_STATION_COLORS)
    station_colors.update(config.get("station_colors", {}))
    apply_nonnegative_q = bool(config.get("clip_qnx_qnz_at_zero", True))
    cap_predictions = force_capped or bool(config.get("cap_predictions", False))
    cap_lower = float(config.get("prediction_lower_bound", 0.0))
    cap_upper = float(config.get("prediction_upper_bound", 1.0))

    reg_rows: list[dict] = []
    coef_rows: list[dict] = []
    pval_rows: list[dict] = []
    stdbeta_rows: list[dict] = []
    plot_info_rows: list[dict] = []
    formula_rows: list[dict] = []
    input_stdcoef_rows: list[dict] = []
    input_transcoef_rows: list[dict] = []

    for dv in dv_columns:
        data = combined[[dv_station_col, dv] + pc_columns].dropna()
        if len(data) < 5:
            reg_rows.append({"DV": dv, "N": len(data), "note": "Not enough data for OLS regression"})
            continue

        stations = data[dv_station_col].astype(str).str.strip()
        y_clean = pd.to_numeric(data[dv], errors="coerce")
        x_clean = data[pc_columns].apply(pd.to_numeric, errors="coerce")
        x_const = sm.add_constant(x_clean)
        model = sm.OLS(y_clean, x_const).fit()

        reg_rows.append({
            "DV": dv,
            "N": int(model.nobs),
            "R2": model.rsquared,
            "Adj_R2": model.rsquared_adj,
            "F_stat": model.fvalue,
            "F_pvalue": model.f_pvalue,
            "AIC": model.aic,
            "BIC": model.bic,
        })

        coefs = model.params.reindex(["const"] + pc_columns)
        pvals = model.pvalues.reindex(["const"] + pc_columns)
        coef_rows.append({"DV": dv, **{f"{key}_coef": value for key, value in coefs.items()}})
        pval_rows.append({"DV": dv, **{f"{key}_p": value for key, value in pvals.items()}})

        y_z = zscore(y_clean)
        x_z = pd.DataFrame({pc: zscore(x_clean[pc]) for pc in pc_columns})
        model_z = sm.OLS(y_z, x_z).fit()
        stdbeta_rows.append({"DV": dv, **{f"{pc}_beta_std": model_z.params[pc] for pc in pc_columns}})

        eq_pc = build_equation_label(str(dv), model.params, pc_columns)
        b_pc = model.params.reindex(pc_columns).astype(float).fillna(0.0)
        b0 = float(model.params.get("const", 0.0))
        alpha_s = pd.Series(loadings.values @ b_pc.values, index=parameter_columns, name="alpha_on_Z")
        gamma_s = (alpha_s / x_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        c0 = b0 - float(((alpha_s * x_mean) / x_std).replace([np.inf, -np.inf], np.nan).fillna(0.0).sum())
        eq_input_std = build_equation_label(str(dv), pd.Series({"const": b0, **alpha_s.to_dict()}), parameter_columns)
        eq_input_trans = build_input_equation_label(str(dv), c0, gamma_s, log_columns)

        formula_rows.append({
            "DV": dv,
            "Formula_DV_vs_PCs": eq_pc,
            "Formula_DV_vs_inputs_STANDARDIZED": eq_input_std,
            "Formula_DV_vs_inputs_TRANSFORMED": eq_input_trans,
        })
        input_stdcoef_rows.append({"DV": dv, "const_b0": b0, **{f"{var}_alphaZ": float(alpha_s[var]) for var in parameter_columns}})
        input_transcoef_rows.append({
            "DV": dv,
            "const_c0": c0,
            **{f"{var}_coef_on_{'ln' if var in log_columns else 'raw'}": float(gamma_s[var]) for var in parameter_columns},
        })

        y_pred = model.predict(x_const)
        dv_lower = str(dv).lower()
        is_qnx_qnz = "qnx" in dv_lower or "qnz" in dv_lower
        if apply_nonnegative_q and is_qnx_qnz:
            y_pred = y_pred.clip(lower=0)

        cap_applied = False
        if cap_predictions and should_apply_unit_cap(dv, y_clean, config):
            y_pred = y_pred.clip(lower=cap_lower, upper=cap_upper)
            y_obs_plot = y_clean.clip(lower=cap_lower, upper=cap_upper)
            cap_applied = True
            pad = 0.02 * (cap_upper - cap_lower) if cap_upper > cap_lower else 0.02
            x_min = cap_lower
            y_min = cap_lower
            x_max = cap_upper + pad
            y_max = cap_upper + pad
        elif apply_nonnegative_q and is_qnx_qnz:
            vmax = float(np.nanmax([y_clean.max(), y_pred.max(), 0.0]))
            pad = 0.05 * vmax if vmax > 0 else 1.0
            x_min = 0.0 if "qnx" in dv_lower else -pad
            y_min = 0.0 if "qnx" in dv_lower else -pad
            x_max = vmax + pad
            y_max = vmax + pad
            y_obs_plot = y_clean
        else:
            vmin = float(np.nanmin([y_clean.min(), y_pred.min()]))
            vmax = float(np.nanmax([y_clean.max(), y_pred.max()]))
            pad = 0.05 * (vmax - vmin) if vmax > vmin else 1.0
            x_min = vmin - pad
            y_min = vmin - pad
            x_max = vmax + pad
            y_max = vmax + pad
            y_obs_plot = y_clean

        plt.figure(figsize=(6, 6))
        axis = plt.gca()
        station_lower = stations.str.lower()
        matched = pd.Series(False, index=data.index)
        for key, color in station_colors.items():
            mask = station_lower.str.contains(str(key).lower(), na=False)
            matched |= mask
            if mask.any():
                axis.scatter(y_obs_plot[mask], y_pred[mask], c=color, label=str(key).title(), alpha=0.85)
        other = ~matched
        if other.any():
            axis.scatter(y_obs_plot[other], y_pred[other], c="0.7", label="_nolegend_", alpha=0.6)

        axis.plot([x_min, x_max], [x_min, x_max])
        axis.set_xlabel("Observed", fontsize=12)
        axis.set_ylabel("Predicted", fontsize=12)
        axis.set_title(f"{dv} - Observed vs Predicted", fontsize=14, fontstyle="italic", fontweight="bold")
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        if cap_applied and cap_lower == 0.0 and cap_upper == 1.0:
            axis.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        elif apply_nonnegative_q and "qnz" in dv_lower:
            axis.set_xticks([tick for tick in axis.get_xticks() if tick >= 0])
            axis.set_yticks([tick for tick in axis.get_yticks() if tick >= 0])

        station_handles, station_labels = axis.get_legend_handles_labels()
        if station_handles:
            station_legend = axis.legend(station_handles, station_labels, title="Station", loc="upper left", fontsize=9, title_fontsize=10, frameon=True)
            axis.add_artist(station_legend)

        r2_label = f"R2 = {model.rsquared:.3f}"
        dummy_eq = Line2D([], [], linestyle="none", marker=None, color="none", label=eq_pc)
        dummy_r2 = Line2D([], [], linestyle="none", marker=None, color="none", label=r2_label)
        model_legend = axis.legend(
            [dummy_eq, dummy_r2],
            [eq_pc, r2_label],
            title="Model (PCs)",
            loc="lower right",
            fontsize=9,
            title_fontsize=10,
            frameon=True,
            handlelength=0,
            handletextpad=0,
        )
        if model_legend.get_texts():
            model_legend.get_texts()[0].set_fontweight("bold")
            model_legend.get_texts()[0].set_fontsize(12)
        try:
            model_legend._legend_box.align = "left"
        except Exception:
            pass

        plt.grid(True)
        plt.tight_layout()
        plot_path = plot_dir / f"{dv_tag}_{safe_name(dv)}_obs_vs_pred.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()

        plot_info_rows.append({
            "DV": dv,
            "N": int(model.nobs),
            "R2": model.rsquared,
            "Plot_file": str(plot_path),
            "Prediction_cap_applied": bool(cap_applied),
            "Cap_lower": cap_lower if cap_applied else np.nan,
            "Cap_upper": cap_upper if cap_applied else np.nan,
        })

    reg_summary = pd.DataFrame(reg_rows)
    if "Adj_R2" in reg_summary.columns:
        reg_summary = reg_summary.sort_values(["Adj_R2"], ascending=False, na_position="last")
    coef_df = pd.DataFrame(coef_rows)
    pval_df = pd.DataFrame(pval_rows)
    stdbeta_df = pd.DataFrame(stdbeta_rows)
    plot_info_df = pd.DataFrame(plot_info_rows)
    formulas_df = pd.DataFrame(formula_rows)
    input_stdcoef_df = pd.DataFrame(input_stdcoef_rows)
    input_transcoef_df = pd.DataFrame(input_transcoef_rows)

    reg_full = reg_summary.merge(coef_df, on="DV", how="left").merge(pval_df, on="DV", how="left").merge(stdbeta_df, on="DV", how="left")
    scaler_info = pd.DataFrame({
        "Variable": parameter_columns,
        "Transform": ["ln(x)" if var in log_columns else "raw(x)" for var in parameter_columns],
        "Mean_used_in_Z": x_mean.values,
        "Std_used_in_Z": x_std.values,
    })
    combine_info = pd.DataFrame({
        "Item": [
            "DV file",
            "PCA scores file",
            "PCA loadings file",
            "PCA input parameters file",
            "Combine mode",
            "Rows in combined data",
            "Parameter columns",
            "Log-transformed columns",
            "PC columns",
            "Rows skipped in PCA input file",
            "Prediction cap enabled for this run",
            "Cap mode",
            "Obs vs Pred plots folder",
            "Spearman heatmap all DV vs PCs",
            "Pearson heatmap all DV vs PCs",
        ],
        "Value": [
            str(resolve_path(config["dv_file"])),
            str(resolve_path(config["pc_scores_file"])),
            str(pca_loadings_file),
            str(pca_input_file),
            combine_mode,
            len(combined),
            ", ".join(parameter_columns),
            ", ".join(log_columns),
            ", ".join(pc_columns),
            n_bad_rows,
            bool(cap_predictions),
            str(config.get("cap_mode", "all")),
            str(plot_dir),
            str(spearman_heatmap),
            str(pearson_heatmap),
        ],
    })

    with pd.ExcelWriter(excel_out, engine="openpyxl") as writer:
        dv_pc_pearson.to_excel(writer, sheet_name="DV_vs_PC_pearson")
        dv_pc_spearman.to_excel(writer, sheet_name="DV_vs_PC_spearman")
        corr_pearson.to_excel(writer, sheet_name="Pearson_full")
        corr_spearman.to_excel(writer, sheet_name="Spearman_full")
        reg_summary.to_excel(writer, sheet_name="Regression_results", index=False)
        reg_full.to_excel(writer, sheet_name="Regression_full", index=False)
        coef_df.to_excel(writer, sheet_name="Coefficients", index=False)
        pval_df.to_excel(writer, sheet_name="Pvalues", index=False)
        stdbeta_df.to_excel(writer, sheet_name="Std_Beta", index=False)
        plot_info_df.to_excel(writer, sheet_name="Obs_vs_Pred_plots", index=False)
        formulas_df.to_excel(writer, sheet_name="Formulas", index=False)
        input_stdcoef_df.to_excel(writer, sheet_name="DV_vs_inputs_alphaZ", index=False)
        input_transcoef_df.to_excel(writer, sheet_name="DV_vs_inputs_transformed", index=False)
        scaler_info.to_excel(writer, sheet_name="PCA_scaler_info", index=False)
        combine_info.to_excel(writer, sheet_name="Combine_info", index=False)
        if skipped_rows is not None:
            skipped_rows.to_excel(writer, sheet_name="Skipped_PCA_input_rows", index=False)

    print("Regression analysis finished successfully.")
    print(f"Excel report: {excel_out}")
    print(f"Plots folder: {plot_dir}")
