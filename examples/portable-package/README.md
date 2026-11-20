<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 -->

# One package, each harness

`portable.py` loads one OADL package on each harness named in `bindings/` and prints,
for each: the lock digest Fabric verified, the harness, what its adapter accepts, every
part of the package it did not take (`not_mapped`/`skipped`, with the reason), and, with
`--run`, the answer.

```bash
# check only (no model call)
uv run --no-sync python examples/portable-package/portable.py PACKAGE examples/portable-package/bindings
# run the same prompt on Kiro (build the adapter first: just install-typescript-kiro, then npm run build in adapters/typescript/kiro)
uv run --no-sync python examples/portable-package/portable.py PACKAGE examples/portable-package/bindings --only kiro --run "PROMPT"
```

Each binding is that harness's own Fabric config (adapter id and a model it accepts):
the package is the same, the model is not. A part a harness cannot take is reported, not
dropped silently; with `if_unsupported="stop"` (Fabric's default) the load refuses.
