"""Explicit fixed-policy inputs for the ordinary Local diagnostic entry point."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

from library.dq_mul_policy import MulPolicy, _unique_json_object, positive_mul


def resolve_policy_grid(args):
    path = getattr(args, "dq_profile_policy_grid_file", None)
    te = bool(getattr(args, "dq_profile_te_quantized", False))
    if not path and not te:
        return {}
    if args.dq_profile_protocol != "v24-acceptance-local":
        raise ValueError("Explicit TE/policy options currently require v24-acceptance-local")
    if args.dq_quantize_z or args.dq_delta_stat != "rms" or not args.dq_delta_bits:
        raise ValueError("Explicit Local policies require RMS delta bits quantization")
    grid = tuple(float(value) for value in args.dq_profile_range_muls_resolved)
    if path:
        raw = Path(path).read_bytes()
        args.dq_profile_policy_grid_sha256 = hashlib.sha256(raw).hexdigest()
        value = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_json_object)
        if not isinstance(value, dict):
            raise ValueError("Policy grid must map numeric grid labels to policy objects")
        numeric = {}
        for label, declaration in value.items():
            key = positive_mul(label)
            if key in numeric:
                raise ValueError("Duplicate numeric policy-grid labels")
            policy = MulPolicy.from_dict(declaration)
            if te and not policy.te_quantized:
                raise ValueError("TE-inclusive mode conflicts with a TE-off policy")
            numeric[key] = policy.to_dict()
        if set(numeric) != set(grid):
            raise ValueError("Policy-grid labels must exactly match dq_profile_range_muls")
    else:
        numeric = {mul: MulPolicy.from_dict({"base_mul": mul, "te_quantized": True}).to_dict() for mul in grid}
    return {f"{mul:.12g}": numeric[mul] for mul in grid}
