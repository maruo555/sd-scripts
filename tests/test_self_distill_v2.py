import json
import unittest
from types import SimpleNamespace

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

try:
    import numpy  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    numpy = None


class SelfDistillV2Tests(unittest.TestCase):
    @unittest.skipIf(torch is None or numpy is None, "torch/numpy is not installed")
    def test_prompt_bank_generates_holdout_and_variants(self):
        from tools import build_prompt_bank

        args = SimpleNamespace(
            keep_triggers="style_a",
            suppress_triggers="style_b,style_c",
            support_tags="soft light,wet surface",
            frontier_tags="backlight",
            carrier_families="portrait,product",
            shot_types="close-up,bust shot",
            lighting_envs="studio,overcast",
            seed_list="101,102",
            template_suffix="",
            negative_prompt="",
            num_templates=4,
            max_support_tags_per_prompt=2,
            holdout_ratio=0.25,
            width=640,
            height=640,
            sample_steps=12,
            guidance_scale=7.5,
            sample_sampler="euler_a",
            prediction_target="eps",
            variant_quota='{"keep_strong":0.4,"off_null":0.3}',
            suppress_conditioning_source="teacher",
            prompt_seed=42,
        )
        payload = build_prompt_bank.build_prompt_bank(args)
        variants = {record["variant_type"] for record in payload["records"]}
        self.assertTrue({"keep_strong", "keep_weak", "off_null", "frontier", "suppress_trigger_style_b", "suppress_trigger_style_c"} <= variants)
        self.assertTrue({record["split"] for record in payload["records"]} <= {"train", "holdout"})

    @unittest.skipIf(torch is None or numpy is None, "torch/numpy is not installed")
    def test_sample_target_step_indices_custom(self):
        from library import self_distill_targets

        result = self_distill_targets.sample_target_step_indices(10, 3, "custom", [8, 2, 20, -1])
        self.assertEqual(result, [2, 8])

    @unittest.skipIf(torch is None or numpy is None, "torch/numpy is not installed")
    def test_manifest_validation_rejects_mismatch(self):
        import tempfile
        from pathlib import Path
        from library import self_distill_cache

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model_path = tmp_path / "base.safetensors"
            teacher_path = tmp_path / "teacher.safetensors"
            prompt_bank = tmp_path / "prompt_bank.json"
            model_path.write_bytes(b"base")
            teacher_path.write_bytes(b"teacher")
            prompt_bank.write_text(json.dumps({"version": 2, "records": [], "metadata": {}}), encoding="utf-8")

            args = SimpleNamespace(
                pretrained_model_name_or_path=str(model_path),
                teacher_lora_weights=str(teacher_path),
                student_init_weights=str(teacher_path),
                lbw_profile=None,
                prediction_target="eps",
                v_parameterization=False,
                resolution=640,
                sample_sampler="euler_a",
                xt_source_mode="teacher_rollout",
                timestep_sampling_mode="uniform",
            )
            header = self_distill_cache.build_manifest_header(args, teacher_te_included=False, prompt_bank_path=str(prompt_bank)).to_dict()
            header["resolution"] = 768
            with self.assertRaises(ValueError):
                self_distill_cache.validate_manifest_header(header, args)

    @unittest.skipIf(torch is None or numpy is None, "torch/numpy is not installed")
    def test_compute_self_distill_loss_keep_and_suppress(self):
        from library import self_distill_losses

        student = torch.zeros(1, 4, 2, 2)
        teacher = torch.ones(1, 4, 2, 2)
        base = torch.zeros(1, 4, 2, 2)
        args = SimpleNamespace(
            per_variant_loss_weight="",
            per_block_anchor_weight="",
            use_keep_delta_loss=True,
            use_suppress_to_base_loss=True,
            use_weight_anchor_loss=False,
            use_coarse_preservation_loss=False,
            use_high_pass_delta_loss=False,
            use_low_pass_delta_loss=False,
            use_sparse_loss=False,
            keep_delta_loss_weight=1.0,
            suppress_to_base_loss_weight=1.0,
        )
        keep_loss, _ = self_distill_losses.compute_self_distill_loss(student, teacher, base, "keep_strong", "keep", args)
        suppress_loss, _ = self_distill_losses.compute_self_distill_loss(student, teacher, base, "off_null", "off", args)
        self.assertGreater(keep_loss.item(), 0.0)
        self.assertAlmostEqual(suppress_loss.item(), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
