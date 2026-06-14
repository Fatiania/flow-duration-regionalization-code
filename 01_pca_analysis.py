"""Script 01: PCA analysis of catchment descriptors.

The parameter columns and log-transformed columns are read from config_template.json.
No variable names or local Windows paths are hard-coded in this script.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils import as_list, ensure_dir, get_station_column, load_config, require_columns, resolve_path, safe_name


def abbreviate_station(name: object) -> str:
    """Create a compact station label from a station name or ID."""
    parts = str(name).split("_")
    if len(parts) >= 2 and parts[0].strip() and parts[-1].strip():
        return parts[0].strip()[0].upper() + parts[-1].strip()
    return safe_name(name)


def preprocess_parameters(
    df: pd.DataFrame,
    parameter_columns: list[str],
    log_columns: list[str],
    station_column: str,
    skip_bad_rows: bool,
    output_dir,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series | None]:
    """Select variables, apply ln() where requested, and optionally skip invalid rows."""
    require_columns(df, [station_column] + parameter_columns, "PCA input parameter file")
    missing_log = [col for col in log_columns if col not in parameter_columns]
    if missing_log:
        raise KeyError(f"log_transform_columns must be included in parameter_columns. Problem columns: {missing_log}")

    raw = df[parameter_columns].copy()
    x = raw.copy()
    for col in parameter_columns:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    invalid_reasons = pd.Series("", index=x.index, dtype=object)
    for col in parameter_columns:
        invalid_reasons.loc[x[col].isna()] += f"{col}: missing/non-numeric; "
    for col in log_columns:
        invalid_reasons.loc[x[col].notna() & (x[col] <= 0)] += f"{col}: <= 0, cannot apply ln; "

    bad_rows = invalid_reasons.str.len() > 0
    if bad_rows.any():
        skipped = df.loc[bad_rows, [station_column] + parameter_columns].copy()
        skipped.insert(1, "Reason_skipped", invalid_reasons.loc[bad_rows].str.rstrip("; "))
        skipped.to_excel(output_dir / "PCA_input_rows_SKIPPED_or_INVALID.xlsx", index=False)
        if not skip_bad_rows:
            raise ValueError(
                f"The input parameter file contains {int(bad_rows.sum())} invalid rows. "
                "Fix the data or set skip_bad_rows_with_missing_or_invalid_log=true in the config."
            )

    station_names = df.loc[~bad_rows, station_column].astype(str).str.strip()
    raw_valid = raw.loc[~bad_rows].copy()
    transformed = x.loc[~bad_rows].copy()
    if transformed.shape[0] < 2:
        raise ValueError("At least two valid stations are required for PCA.")

    for col in log_columns:
        transformed[col] = np.log(transformed[col])

    return station_names, raw_valid, transformed, transformed.copy(), invalid_reasons.loc[bad_rows] if bad_rows.any() else None


def run_pca(config: dict) -> None:
    input_path = resolve_path(config["input_parameters"])
    output_dir = ensure_dir(resolve_path(config["output_dir"]))
    parameter_columns = as_list(config.get("parameter_columns"), "parameter_columns")
    log_columns = as_list(config.get("log_transform_columns"), "log_transform_columns")
    station_column_config = config.get("station_column")
    skip_bad_rows = bool(config.get("skip_bad_rows_with_missing_or_invalid_log", False))

    if not parameter_columns:
        raise ValueError("The PCA config must define parameter_columns.")
    if not input_path.exists():
        raise FileNotFoundError(f"Input parameter file not found: {input_path}")

    df = pd.read_excel(input_path)
    station_column = get_station_column(df, station_column_config)

    station_names, raw_valid, transformed, transformed_for_output, _ = preprocess_parameters(
        df=df,
        parameter_columns=parameter_columns,
        log_columns=log_columns,
        station_column=station_column,
        skip_bad_rows=skip_bad_rows,
        output_dir=output_dir,
    )

    scaler = StandardScaler()
    x_std = scaler.fit_transform(transformed.values)

    pca = PCA()
    x_pca = pca.fit_transform(x_std)
    pc_names = [f"PC{i + 1}" for i in range(x_pca.shape[1])]

    scores = pd.DataFrame(x_pca, columns=pc_names)
    scores.insert(0, "Station_full", station_names.values)
    scores.insert(1, "Station_abbr", scores["Station_full"].apply(abbreviate_station))
    scores.insert(2, "Group", scores["Station_abbr"].astype(str).str[0])

    explained_variance = pd.DataFrame({
        "PC": pc_names,
        "Eigenvalue": pca.explained_variance_,
        "Explained_Variance_%": pca.explained_variance_ratio_ * 100,
        "Cumulative_Variance_%": np.cumsum(pca.explained_variance_ratio_) * 100,
    })

    loadings = pd.DataFrame(pca.components_.T, columns=pc_names, index=parameter_columns)

    scores.to_excel(output_dir / "PCA_scores_ALL_PC.xlsx", index=False)
    explained_variance.to_excel(output_dir / "PCA_explained_variance_ALL_PC.xlsx", index=False)
    loadings.to_excel(output_dir / "PCA_loadings_ALL_PC.xlsx")

    with pd.ExcelWriter(output_dir / "PCA_results_ALL_PC.xlsx", engine="openpyxl") as writer:
        scores.to_excel(writer, sheet_name="Scores", index=False)
        explained_variance.to_excel(writer, sheet_name="ExplainedVariance", index=False)
        loadings.to_excel(writer, sheet_name="Loadings", index=True)

    rename_map = {col: f"ln_{col}" for col in log_columns}
    output_columns = [rename_map.get(col, col) for col in parameter_columns]
    raw_for_output = raw_valid.copy()
    transformed_renamed = transformed_for_output.rename(columns=rename_map)
    zscores = pd.DataFrame(x_std, columns=parameter_columns).rename(columns=rename_map)

    loadings_renamed = loadings.copy()
    loadings_renamed.index = [rename_map.get(col, col) for col in loadings_renamed.index]
    scaled_loadings = pd.DataFrame(
        pca.components_.T * np.sqrt(pca.explained_variance_),
        columns=pc_names,
        index=[rename_map.get(col, col) for col in parameter_columns],
    )

    preprocessing = pd.DataFrame({
        "Variable_in_output": output_columns,
        "Original_variable": parameter_columns,
        "Transform": ["ln" if col in log_columns else "none" for col in parameter_columns],
        "Standardized": "yes: z = (x_transformed - mean) / std",
        "Mean_after_transform_before_zscore": scaler.mean_,
        "Std_after_transform_before_zscore": scaler.scale_,
    })

    readme = pd.DataFrame({
        "README": [
            "PCA results with explicit preprocessing information.",
            "All parameter names are controlled by config_template.json.",
            "Variables listed in log_transform_columns were transformed using natural logarithm ln(x).",
            "All variables were then standardized before PCA.",
            "Scores, explained variance, loadings, preprocessing statistics, and transformed matrices are included.",
        ]
    })

    with pd.ExcelWriter(output_dir / "PCA_results_ALL_PC_CLEAR_PREPROCESSING.xlsx", engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        preprocessing.to_excel(writer, sheet_name="Preprocessing", index=False)
        raw_for_output.to_excel(writer, sheet_name="RawData", index=False)
        transformed_renamed[output_columns].to_excel(writer, sheet_name="AfterTransform", index=False)
        zscores[output_columns].to_excel(writer, sheet_name="ZscoresUsedForPCA", index=False)
        scores.to_excel(writer, sheet_name="Scores", index=False)
        explained_variance.to_excel(writer, sheet_name="ExplainedVariance", index=False)
        loadings_renamed.to_excel(writer, sheet_name="Weights_transform_names", index=True)
        scaled_loadings.to_excel(writer, sheet_name="ScaledLoadings", index=True)

    print("PCA analysis finished successfully.")
    print(f"Outputs saved in: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PCA analysis using configurable parameter columns.")
    parser.add_argument("--config", default="config_template.json", help="Path to the JSON configuration file.")
    parser.add_argument("--section", default="pca_analysis", help="Configuration section name.")
    args = parser.parse_args()
    run_pca(load_config(args.config, args.section))


if __name__ == "__main__":
    main()
