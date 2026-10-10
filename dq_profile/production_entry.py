from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from dq_profile.production_cli import resolve_training_cli
from dq_profile.production_runner import (
    DEFAULT_OUTPUT_BASE,
    ProductionRunOptions,
    run_profile_request,
)


def run_profile_mode(
    training_argv: Sequence[str],
    *,
    preset_name: str = "canonical-v1",
    execution_mode_name: str = "standard",
    output_base: Path = DEFAULT_OUTPUT_BASE,
    profile_name: str | None = None,
    preflight_only: bool = False,
    dry_run: bool = False,
    open_report: bool = False,
    data_diagnostics: str | None = None,
    group_map: Path | None = None,
    te_quantized: bool = True,
    policy_grid_file: Path | None = None,
    dropout_on: bool = False,
    uniform_only: bool = False,
) -> int:
    request = resolve_training_cli(
        training_argv,
        preset_name=preset_name,
        execution_mode_name=execution_mode_name,
    )
    standard = preset_name == "canonical-v2"
    if standard:
        from dq_profile.spatial import UNIFORM_MULS
        if not te_quantized or policy_grid_file is not None:
            raise ValueError("canonical-v2 fixes UNet+TE quantization and its candidate family; use --dq-profile-preset canonical-v1 for legacy TE-off or custom policy grids")
        request = replace(request, execution_mode=replace(request.execution_mode, core_grid=UNIFORM_MULS,
            max_edge_extension_rounds=0, edge_policy="fixed_five_plus_body_spatial_no_edge_extension"))
        # Fail before any model work if offline report support is missing.
        from plotly.offline import get_plotlyjs
    elif dropout_on or uniform_only:
        raise ValueError("dropout confirmation and uniform-only selection require canonical-v2")
    data_diagnostics = data_diagnostics or ("warmup" if standard else "off")
    if data_diagnostics not in {"off", "local", "warmup"}:
        raise ValueError("unknown data diagnostics mode")
    if group_map is not None:
        from dq_profile.dataset_diagnostics import load_group_map
        load_group_map(group_map)
        group_map = group_map.resolve(strict=True)
    if policy_grid_file is not None:
        policy_grid_file = policy_grid_file.resolve(strict=True)
    request = replace(request, data_diagnostics=data_diagnostics, group_map=group_map,
                      te_quantized=bool(te_quantized), policy_grid_file=policy_grid_file,
                      dropout_on=bool(dropout_on), uniform_only=bool(uniform_only))
    policy_contract = request.fixed_policy_contract()  # Validate before GPU setup.
    if not te_quantized and policy_contract is not None:
        if any(policy["te_quantized"] for policy in policy_contract["resolved_policy_grid"].values()):
            raise ValueError("--dq-profile-no-te-quantized conflicts with a TE-on policy in the grid")
    if policy_grid_file is not None:
        request = replace(request, execution_mode=replace(request.execution_mode,
            max_edge_extension_rounds=0, edge_policy="explicit_policy_grid_no_automatic_extrapolation"))
    result = run_profile_request(
        request,
        ProductionRunOptions(
            output_base=output_base,
            profile_name=profile_name,
            preflight_only=preflight_only,
            dry_run=dry_run,
            open_report=open_report,
        ),
    )
    print(
        json.dumps(
            {
                "status": result.status,
                "run_dir": str(result.run_dir),
                "report": str(result.report) if result.report else None,
                "local_te_quantized": request.te_quantized,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0
