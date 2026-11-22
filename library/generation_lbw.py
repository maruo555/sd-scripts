"""SDXL block-weight presets and validation; also used without importing Torch."""
import math

PRESETS = {
    "ALL": (1,) * 12,
    "NONE": (0,) * 12,
    "XLMIDD": (1,0,1,1,1,1,1,1,1,0,0,0),
    "XLMLT1": (1,1,0,0,1,1,0.6,1,1,1,0,1),
}


def parse_sdxl_lbw_weights(spec):
    if spec is None:
        return None
    if isinstance(spec, str):
        stripped = spec.strip()
        spec = PRESETS.get(stripped.upper(), stripped.split(","))
    values = [float(v) for v in spec]
    if len(values) != 12 or not all(math.isfinite(x) for x in values):
        raise ValueError("SDXL LoRA block weights require a preset or 12 finite values")
    return values
