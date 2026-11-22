import hashlib
import json

from library.dq_mul_policy import MulPolicy
from library.lora_run_schema import HEADERS
from tools.lora_diagnostic_dq import analyze_dq
from tools.lora_diagnostic_rank import analyze_rank
from tools.lora_run_reader import rank_observations


def test_fixed_site_mul_counts_use_verified_assignment():
    names = ["lora_unet_input_blocks_0_to_q", "lora_unet_input_blocks_2_attn2_to_q", "lora_te1_text_model_x", "lora_te2_text_model_x"]
    policy = MulPolicy(2.70, attn2=3.75, te=3.75)
    record = policy.record(policy.expand(names))
    data = analyze_dq([], [], {"dq":{"mul":record}})
    assert data["multiplier_counts"] == [(2.70,1),(3.75,3)] and not data["issues"]
    bad = analyze_dq([], [], {"dq":{"mul":dict(record,assignment_digest="0"*64)}})
    assert bad["multiplier_counts"] is None and any("digest" in i["reason"] for i in bad["issues"])


def test_partial_parameters_do_not_supply_a_normalization_denominator():
    names = ["lora_unet_input_blocks_1_to_q", "lora_unet_input_blocks_2_to_q", "lora_unet_output_blocks_1_to_k"]
    class Reader:
        index = {"artifacts":{"rank":{"expected_modules_by_scope":{"unet":names}}}}
        def rank_observations(self, rows):
            return rank_observations(rows, {"unet":names})
    def make_row(name, rank):
        return dict(dict.fromkeys(HEADERS["rank"]), epoch=2,attempt_step=100,scope="unet",module=name,
                    rank_dim=rank,rank_energy=1.0,rank_sat=.8,rank_top1=.7)
    rows=[make_row(n,i+1) for i,n in enumerate(names)]
    data=analyze_rank(Reader(), rows, {names[0]:2,names[2]:4})
    path_groups=data["grouped"]["path"]["rows"][0]["groups"]
    assert path_groups["down"]["RankEnergySum"]==2.0 and path_groups["down"]["RankEnergyPerParam"] is None
    assert path_groups["up"]["RankEnergyPerParam"]==.25
    assert all(group["RankEnergySharePerParam"] is None for group in path_groups.values())
    assert data["missing_param_count_modules"]==[names[1]]
    dims={"unet":dict(zip(names,[1,2,3]))}
    expected=hashlib.sha256(json.dumps(dims,ensure_ascii=False,sort_keys=True,allow_nan=False).encode()).hexdigest()
    assert data["expected_modules_sha256"]==expected
    assert analyze_rank(Reader(),list(reversed(rows)))["expected_modules_sha256"]==expected
    assert analyze_rank(Reader(),rows[:-1])["expected_modules_sha256"] is None
    changed = [dict(row, attempt_step=200, rank_dim=4) for row in rows]
    assert analyze_rank(Reader(),rows+changed)["expected_modules_sha256"] is None
