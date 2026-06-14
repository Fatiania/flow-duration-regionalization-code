"""Script 03: DV-PC regression with prediction capping.

Use this script for bounded dependent variables such as intermittency/fraction
parameters. By default, predictions are clipped to [0, 1]. Change the bounds or
cap_mode in config_template.json if needed.
"""

from __future__ import annotations

import argparse

from regression_workflow import run_regression_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run capped DV-PC regression analysis.")
    parser.add_argument("--config", default="config_template.json", help="Path to JSON configuration file.")
    parser.add_argument("--section", default="dv_regression_capped_0_1", help="Configuration section name.")
    args = parser.parse_args()
    run_regression_from_config(args.config, args.section, force_capped=True)


if __name__ == "__main__":
    main()
