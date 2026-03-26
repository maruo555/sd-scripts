import random
from collections import defaultdict
from typing import Dict, Iterator, List, Optional

from torch.utils.data import Sampler

from library import self_distill_cache


class VariantQuotaSampler(Sampler[int]):
    def __init__(self, entries: List[Dict], variant_quota: Optional[Dict[str, float]] = None, num_samples: Optional[int] = None, seed: int = 0):
        self.entries = entries
        self.variant_quota = self._normalize_quota(variant_quota or {})
        self.num_samples = num_samples or len(entries)
        self.seed = seed

        self.indices_by_variant = defaultdict(list)
        for index, entry in enumerate(entries):
            self.indices_by_variant[entry["variant_type"]].append(index)

        self.available_variants = sorted(self.indices_by_variant.keys())

    def _normalize_quota(self, raw: Dict[str, float]) -> Dict[str, float]:
        if not raw:
            return {}
        total = sum(max(0.0, float(value)) for value in raw.values())
        if total <= 0:
            raise ValueError("variant_quota must have a positive sum.")
        return {key: max(0.0, float(value)) / total for key, value in raw.items()}

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed)
        if not self.variant_quota:
            if not self.entries:
                return
            emitted = 0
            while emitted < self.num_samples:
                shuffled = list(range(len(self.entries)))
                rng.shuffle(shuffled)
                for index in shuffled:
                    yield index
                    emitted += 1
                    if emitted >= self.num_samples:
                        break
            return

        variant_names = [name for name in self.variant_quota.keys() if self.indices_by_variant.get(name)]
        weights = [self.variant_quota[name] for name in variant_names]
        if not variant_names:
            raise ValueError("variant_quota does not match any variants in the dataset.")

        for _ in range(self.num_samples):
            variant = rng.choices(variant_names, weights=weights, k=1)[0]
            yield rng.choice(self.indices_by_variant[variant])


def quota_from_args(args) -> Dict[str, float]:
    return self_distill_cache.parse_mapping_arg(getattr(args, "variant_quota", None))
