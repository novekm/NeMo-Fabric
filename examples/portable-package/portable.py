# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load one OADL package on each harness and say what each harness took of it.

usage: portable.py PACKAGE BINDINGS_DIR [--run "PROMPT"] [--only a,b]

BINDINGS_DIR holds one NAME.json per harness: a Fabric config naming the adapter and a
model that harness accepts. For each, this prints the lock digest Fabric verified, the
harness, what the adapter accepts, every part of the package that was not mapped
(with the status Fabric gave it), and, with --run, the answer.
"""
import asyncio, json, pathlib, sys
from nemo_fabric import Fabric, FabricConfig, load_agent_package


def lines(report):
    for item in report["declared"]:
        if item["status"] not in ("mapped", "bound"):
            yield item


def show(name, package, binding, prompt):
    print(f"\n=== {name} ===")
    config = FabricConfig.from_mapping(binding)
    try:
        report = load_agent_package(package, config, if_unsupported="warn")
    except Exception as error:
        print(f"load failed: {type(error).__name__}: {str(error)[:400]}")
        return
    adapter = report["adapter"]
    print("outcome:", report["outcome"])
    print("lock sha256 verified:", report["lock"]["sha256"])
    print("harness:", adapter["adapter_id"] if adapter else binding["harness"]["adapter_id"])
    if adapter:
        print("adapter accepts:", ", ".join(adapter["accepts"]))
    counts = {}
    for item in report["declared"]:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    print("parts by status:", counts)
    for item in lines(report):
        print(f"  {item['status']:<10} {item['id']}: {str(item['detail'] or item['reason'] or item['gap'])[:150]}")
    if prompt:
        asyncio.run(run(package, config, prompt))


async def run(package, config, prompt):
    run = await Fabric().start_agent_package(package, config, options={"if_unsupported": "warn"})
    try:
        result = await run.invoke(input=prompt)
        out = result.output
        print("answer:", getattr(out, "response", None) or str(out)[:1500])
    except Exception as error:
        print(f"run failed: {type(error).__name__}: {str(error)[:400]}")
    finally:
        try:
            await run.stop()
        except Exception:
            pass


def main(argv):
    package, bindings = argv[1], pathlib.Path(argv[2])
    prompt = argv[argv.index("--run") + 1] if "--run" in argv else None
    only = argv[argv.index("--only") + 1].split(",") if "--only" in argv else None
    for path in sorted(bindings.glob("*.json")):
        if only and path.stem not in only:
            continue
        binding = json.load(open(path))
        binding.setdefault("metadata", {"name": json.load(open(pathlib.Path(package, "agent.json")))["name"]})
        show(path.stem, package, binding, prompt)


main(sys.argv)
