# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load the example agent package and print the load report as JSON.

With no config, the package is verified against its lock and nothing is mapped.
With ``--config``, the package is also mapped onto that NeMo Fabric config.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from nemo_fabric import FabricConfig, FabricConfigError, load_agent_package

PACKAGE_DIR = Path(__file__).resolve().parent / "release-notes-agent"
VARIANT_ENTRY = "variants/short/agent-definition.json"


def _variable(text: str) -> tuple[str, str | int | float | bool]:
    name, separator, raw = text.partition("=")
    if not separator or not name:
        raise argparse.ArgumentTypeError("write a variable as NAME=VALUE")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return name, raw
    if isinstance(value, (str, int, float, bool)):
        return name, value
    return name, raw


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m examples.agent_package", description=__doc__)
    parser.add_argument("--variant", action="store_true", help="load the short variant, a definition, instead of the agent")
    parser.add_argument("--config", type=Path, help="a NeMo Fabric config file in JSON; the package is mapped onto it")
    parser.add_argument("--base-dir", type=Path, help="the folder that relative paths in the config are resolved from")
    parser.add_argument("--var", action="append", default=[], type=_variable, metavar="NAME=VALUE", help="a value for one of the agent's variables; repeat for more")
    parser.add_argument("--if-unsupported", choices=("stop", "warn"), default="stop", help="what to do about a required thing that is not mapped or not met")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load the package, print the report, and return the process exit code."""
    args = _parser().parse_args(argv)
    config = None
    if args.config is not None:
        config = FabricConfig.from_mapping(json.loads(args.config.read_text(encoding="utf-8")))
    try:
        report = load_agent_package(
            PACKAGE_DIR,
            config,
            base_dir=args.base_dir,
            entry=VARIANT_ENTRY if args.variant else None,
            variables=dict(args.var),
            if_unsupported=args.if_unsupported,
        )
    except FabricConfigError as error:
        print(f"refused: {error.code}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report.to_mapping(), indent=2, default=str))
    return 0 if report.outcome != "stopped" else 2


if __name__ == "__main__":
    raise SystemExit(main())
