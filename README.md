# Data folder

Place the Excel input files here, or edit `config_template.json` to point to another folder.

Expected examples:

```text
data/
├── input_parameters_for_pca.xlsx
├── input_parameters_for_regression_noncapped.xlsx
├── input_parameters_for_regression_capped.xlsx
├── DV_noncapped.xlsx
├── DV_capped_0_1.xlsx
└── non_zero_flow_station_files/
    ├── Napoli_001.xlsx
    ├── Bari_002.xlsx
    └── ...
```

The names above are examples only. You can use different names as long as `config_template.json` is updated.

For PCA and regression, the first column is treated as the station/name column when `station_column`, `dv_station_column`, or `pc_station_column` is set to `null`.
