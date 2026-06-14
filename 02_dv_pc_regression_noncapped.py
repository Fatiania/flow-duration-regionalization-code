"""Script 02: DV-PC regression without upper/lower prediction capping.

Use this script for dependent variables that do not have a physical [0, 1]
bound. The parameter columns are configurable and are not hard-coded.
"""

from __future__ import annotations

import argparse

from regression_workflow import run_regression_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run non-capped DV-PC regression analysis.")
    parser.add_argument("--config", default="config_template.json", help="Path to JSON configuration file.")
    parser.add_argument("--section", default="dv_regression_noncapped", help="Configuration section name.")
    args = parser.parse_args()
    run_regression_from_config(args.config, args.section, force_capped=False)


if __name__ == "__main__":
    main()
