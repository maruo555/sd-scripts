"""Settings provenance and compatibility tests, independent of torch/accelerate."""
import ast
import copy
import json
from pathlib import Path
import random
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from library.training_settings import dataset_batch_settings, finish_record, optimizer_groups, snapshot, snapshot_metadata, start_record
from tools.lora_training_settings import load_training_settings, normalize_metadata, read_metadata
from tools.lora_training_settings_display import render_settings
from tools.make_lora_diagnostic_report import build_html


class TensorMustNotBeRead:
    def __repr__(self):
        raise AssertionError("tensor repr")

    def item(self):
        raise AssertionError("tensor item")


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "test.safetensors"
        self.metadata = {"ss_session_id": "123", "ss_training_started_at": "456.5", "ss_output_name": "test",
                         "ss_unet_lr": "None", "ss_full_fp16": "False", "ss_learning_rate": "0.00035",
                         "ss_seed": "0", "ss_text_encoder_lr": "0.0002", "ss_epoch": "7",
                         "ss_steps": "100", "ss_network_args": '{"conv_dim": 4}'}
        self.header(self.metadata)
        self.args = SimpleNamespace(output_name="test", output_dir=str(self.root), learning_rate=0.00035,
                                    text_encoder_lr1=0.0003, unet_lr=None, avg_cp=False,
                                    mixed_precision="fp16", seed=0, wandb_api_key="secret-value")
        self.optimizer = SimpleNamespace(param_groups=[{"params": [TensorMustNotBeRead()], "lr": 0.0003, "betas": (0.9, 0.99)}])

    def header(self, metadata):
        header = json.dumps({"__metadata__": metadata}).encode()
        self.model.write_bytes(struct.pack("<Q", len(header)) + header)

    def record(self, session=123, started=456.5, resolve=True):
        record = start_record(self.root, "test", session, started, snapshot(vars(self.args)))
        if resolve:
            finish_record(record, self.args, self.optimizer, ["textencoder 1"], optimizer_groups(self.optimizer, ["textencoder 1"]),
                          self.metadata, {"dq_auto_preset": "clip_rate_high"})
        return record

    def load(self, explicit=None):
        return load_training_settings(self.root, "test", self.model, explicit)

    def test_record_preserves_rng_and_args_and_never_reads_tensors(self):
        before = copy.deepcopy(vars(self.args))
        state = random.getstate()
        record = self.record()
        self.assertEqual(random.getstate(), state)
        self.assertEqual(vars(self.args), before)
        self.assertEqual(self.optimizer.param_groups[0]["lr"], 0.0003)
        contents = "".join(p.read_text(encoding="utf-8") for p in record[0].rglob("*.json"))
        self.assertNotIn("secret-value", contents)
        data = self.load()
        self.assertEqual(data["status"], "recorded")
        self.assertEqual(data["association"], "session_metadata")
        self.assertFalse(data["values"]["avg_cp"])
        self.assertIsNone(data["values"]["unet_lr"])
        self.assertEqual(data["resolved"]["optimizer_groups_created"][0]["options"]["lr"], 0.0003)

    def test_requested_and_resolved_capture_different_stages(self):
        record = self.record(resolve=False)
        self.args.learning_rate = 0.01
        created = optimizer_groups(self.optimizer)
        self.optimizer.param_groups[0]["lr"] = 0
        finish_record(record, self.args, self.optimizer, None, created, self.metadata, {})
        data = self.load()
        self.assertEqual(data["requested_args"]["learning_rate"], 0.00035)
        self.assertEqual(data["values"]["learning_rate"], 0.01)
        self.assertEqual(data["resolved"]["optimizer_groups_created"][0]["options"]["lr"], 0.0003)
        self.assertEqual(data["resolved"]["optimizer_groups_at_start"][0]["options"]["lr"], 0)

    def test_trainer_hooks_only_write_on_main_and_resolve_after_resume(self):
        # Exercise the actual inserted trainer statements without loading CUDA dependencies.
        source = (Path(__file__).resolve().parents[1] / "train_network.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NetworkTrainer")
        train = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == "train")
        assignments = {n.targets[0].id: n for n in train.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
        start_guard = next(n for n in train.body if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
                           and n.test.id == "is_main_process" and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                           and c.func.attr == "start_record" for c in ast.walk(n)))
        finish_guard = next(n for n in train.body if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "settings_record"
                            and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                                    and c.func.attr == "finish_record" for c in ast.walk(n)))
        resume = next(n for n in ast.walk(train) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                      and n.func.attr == "resume_from_local_or_hf_if_specified")
        scheduler = next(n for n in ast.walk(train) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                         and n.func.attr == "get_scheduler_fix")
        self.assertLess(assignments["settings_created_groups"].lineno, scheduler.lineno)
        self.assertLess(resume.lineno, finish_guard.lineno)
        nodes = [assignments["requested_training_args"], assignments["settings_record"], start_guard,
                 assignments["settings_created_groups"]]
        code = compile(ast.Module(body=nodes, type_ignores=[]), "trainer-settings-hooks", "exec")
        from library import training_settings
        env = dict(args=self.args, training_settings=training_settings, is_main_process=False,
                   train_util=SimpleNamespace(default_if_none=lambda value, default: default if value is None else value,
                                              DEFAULT_LAST_OUTPUT_NAME="last"),
                   session_id=123, training_started_at=456.5, optimizer=self.optimizer, lr_descriptions=["TE1"])
        exec(code, env)
        self.assertIsNone(env["settings_record"])
        self.assertFalse((self.root / "run_records").exists())
        env["is_main_process"] = True
        exec(code, env)
        self.assertIsNotNone(env["settings_record"])
        self.assertEqual(env["settings_created_groups"][0]["label"], "TE1")

    def test_dataset_batches_use_dataset_values_and_preserve_multiple_sizes(self):
        datasets = [SimpleNamespace(batch_size=4), SimpleNamespace(batch_size=2)]
        before = [d.batch_size for d in datasets]
        batches = dataset_batch_settings(datasets, num_processes=2, accumulation_steps=3)
        self.assertEqual(batches, [
            {"dataset_index": 0, "batch_size_per_device": 4, "nominal_effective_batch_size": 24},
            {"dataset_index": 1, "batch_size_per_device": 2, "nominal_effective_batch_size": 12},
        ])
        self.assertEqual([d.batch_size for d in datasets], before)
        self.assertEqual(dataset_batch_settings([SimpleNamespace(batch_size=1)], 1, 1)[0]["nominal_effective_batch_size"], 1)

    def test_dataset_batch_hook_reads_before_dataset_is_deleted(self):
        source = (Path(__file__).resolve().parents[1] / "train_network.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "dataset_batch_settings")
        deletion = next(n for n in ast.walk(tree) if isinstance(n, ast.Delete)
                        and any(isinstance(t, ast.Name) and t.id == "train_dataset_group" for t in n.targets))
        self.assertLess(call.lineno, deletion.lineno)
        self.assertIn('"dataset_batch_sizes": settings_dataset_batches', source)
        self.assertNotIn('"total_batch_size": total_batch_size', source)

    def test_structured_metadata_secrets_do_not_reappear_in_record_or_report(self):
        secret = "DEMO_ONLY_REVIEW_SECRET"
        self.args.network_args = ["api_key=" + secret, "conv_dim=4"]
        self.metadata["ss_network_args"] = json.dumps({"api_key": secret, "conv_dim": 4,
                                                       "nested": {"access_token": secret}})
        original = copy.deepcopy(self.metadata)
        record = self.record()
        for path in record[0].rglob("*.json"):
            self.assertNotIn(secret, path.read_text(encoding="utf-8"))
        data = self.load()
        self.assertEqual(data["status"], "recorded")
        network = json.loads(data["resolved"]["metadata"]["ss_network_args"])
        self.assertEqual(network, {"api_key": "[redacted]", "conv_dim": 4,
                                  "nested": {"access_token": "[redacted]"}})
        self.assertNotIn(secret, json.dumps(data))
        self.assertNotIn(secret, build_html({"charts": {}, "training_settings": data}))
        self.assertEqual(self.metadata, original, "checkpoint metadata itself must not be changed")

    def test_normal_metadata_is_preserved_and_malformed_structured_fields_are_omitted(self):
        self.assertEqual(snapshot_metadata(self.metadata), self.metadata)
        bad = {"ss_network_args": '{"api_key":"DEMO_ONLY_SECRET"'}
        self.assertNotIn("DEMO_ONLY_SECRET", json.dumps(snapshot_metadata(bad)))

    def test_bad_manifest_identity_does_not_stop_report_generation(self):
        record = self.record()
        path = record[0] / "manifest.json"
        for key in ("output_name", "session_id", "training_started_at", "run_id"):
            for invalid in ([], {}, None, 123):
                with self.subTest(key=key, invalid=invalid):
                    path.write_text(json.dumps({**record[1], "settings_status": "resolved", key: invalid}), encoding="utf-8")
                    data = self.load()
                    self.assertEqual((data["source"], data["status"]), ("record", "error"))
                    self.assertEqual(data["values"], {})
                    self.assertIn("学習設定", build_html({"charts": {}, "training_settings": data}))

    def test_bad_unrelated_manifest_does_not_hide_valid_match(self):
        good = self.record()
        other = self.record(session=456)
        (other[0] / "manifest.json").write_text(json.dumps({**other[1], "output_name": []}), encoding="utf-8")
        self.assertEqual(self.load()["run_id"], good[1]["run_id"])
        self.assertEqual(self.load()["status"], "recorded")

    def test_invalid_resolved_structures_become_error_without_breaking_html(self):
        record = self.record()
        path = record[0] / "inputs/resolved_config.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        cases = [(key, value) for key in ("optimizer_groups_created", "optimizer_groups_at_start")
                 for value in (None, {}, [None], [{"options": None}], [{"options": []}],
                               [{"options": {}, "label": []}], [{"options": {}, "index": "bad"}])]
        cases += [(key, value) for key in ("runtime", "metadata") for value in (None, [], "invalid")]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                path.write_text(json.dumps({**original, key: value}), encoding="utf-8")
                data = self.load()
                self.assertEqual((data["source"], data["status"]), ("record", "error"))
                self.assertEqual(data["values"], {})
                self.assertIn("学習設定", build_html({"charts": {}, "training_settings": data}))
        original["optimizer_groups_created"] = {"unrecorded": "optimizer_groups"}
        path.write_text(json.dumps(original), encoding="utf-8")
        self.assertEqual(self.load()["status"], "recorded")
        self.assertIn("学習設定", build_html({"charts": {}, "training_settings": self.load()}))

    def test_metadata_fallback_redacts_both_normalized_and_raw_values(self):
        secret = "FALLBACK_DUMMY_SECRET"
        self.metadata["ss_network_args"] = json.dumps({"api_key": secret, "conv_dim": 4})
        self.header(self.metadata)
        original = self.model.read_bytes()
        data = self.load()
        self.assertEqual(data["source"], "metadata")
        self.assertEqual(data["values"]["network_args"], {"api_key": "[redacted]", "conv_dim": 4})
        self.assertNotIn(secret, json.dumps(data))
        self.assertNotIn(secret, build_html({"charts": {}, "training_settings": data}))
        self.assertEqual(self.model.read_bytes(), original)

    def test_legacy_sidecar_secrets_are_redacted_at_read_time(self):
        record = self.record()
        secret = "LEGACY_SIDECAR_DUMMY_SECRET"
        originals = {}
        for name in ("requested_args", "resolved_config"):
            path = record[0] / "inputs" / (name + ".json")
            value = json.loads(path.read_text(encoding="utf-8"))
            value["args"]["wandb_api_key"] = secret
            value["args"]["network_args"] = ["api_key=" + secret, "conv_dim=4"]
            if name == "resolved_config":
                value["metadata"]["ss_network_args"] = json.dumps({"api_key": secret, "conv_dim": 4})
                value["runtime"]["access_token"] = secret
            path.write_text(json.dumps(value), encoding="utf-8")
            originals[path] = path.read_bytes()
        data = self.load()
        self.assertEqual(data["status"], "recorded")
        self.assertNotIn(secret, json.dumps(data))
        self.assertNotIn(secret, build_html({"charts": {}, "training_settings": data}))
        self.assertEqual({p: p.read_bytes() for p in originals}, originals)
        manifest = {**record[1], "settings_status": "requested"}
        (record[0] / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.assertEqual(self.load()["status"], "partial")
        self.assertNotIn(secret, json.dumps(self.load()))

    def test_known_metadata_types_match_settings_without_guessing_unknown_values(self):
        expected = {"caption_dropout_rate": 0.0, "caption_dropout_every_n_epochs": 0,
                    "caption_tag_dropout_rate": 0.25, "noise_offset_random_strength": False,
                    "ip_noise_gamma_random_strength": True, "noise_offset_random_min_ratio": 0.0,
                    "noise_offset_random_max_ratio": 1.0, "huber_c": 0.1}
        metadata = {"ss_" + k: str(v) for k, v in expected.items()}
        actual = normalize_metadata(metadata)
        self.assertEqual(actual, expected)
        for key, value in expected.items():
            self.assertIs(type(actual[key]), type(value))
        actual = normalize_metadata({"ss_future_name": "0001", "ss_learning_rate": "invalid",
                                     "ss_unet_lr": "None", "ss_noise_offset_random_strength": "invalid"})
        self.assertEqual(actual, {"future_name": "0001", "learning_rate": "invalid", "unet_lr": None,
                                  "noise_offset_random_strength": "invalid"})
        self.assertNotIn("caption_dropout_rate", actual)

    def test_metadata_fallback_only_without_record(self):
        data = self.load()
        self.assertEqual(data["source"], "metadata")
        self.assertEqual(data["values"]["learning_rate"], 0.00035)
        self.assertIs(data["values"]["full_fp16"], False)
        self.assertIsNone(data["values"]["unet_lr"])
        self.assertEqual(data["values"]["seed"], 0)
        self.assertNotIn("text_encoder_lr1", data["values"])
        self.assertNotIn("epoch", data["values"])
        self.assertEqual(data["checkpoint"]["metadata"]["ss_epoch"], "7")

    def test_no_per_field_metadata_fill(self):
        self.record()
        data = self.load()
        self.assertNotIn("text_encoder_lr", data["values"])
        self.assertNotIn("raw_metadata", data)

    def test_requested_only_is_partial_not_false_completion(self):
        self.record(resolve=False)
        data = self.load()
        self.assertEqual((data["status"], data["value_stage"]), ("partial", "requested"))
        self.assertNotIn("text_encoder_lr", data["values"])

    def test_matching_session_wins_over_same_name(self):
        first = self.record()
        other = self.record(session=456)
        self.assertNotEqual(first[0], other[0])
        self.assertEqual(self.load()["run_id"], first[1]["run_id"])
        self.assertEqual(self.load(other[0] / "manifest.json")["status"], "mismatch")

    def test_no_latest_guess_and_explicit_choice_for_missing_checkpoint(self):
        first = self.record()
        self.record(session=456)
        self.model.unlink()
        self.assertEqual(self.load()["status"], "ambiguous")
        data = self.load(first[0] / "manifest.json")
        self.assertEqual(data["status"], "recorded")
        self.assertEqual(data["association"], "explicit")

    def test_mismatched_record_does_not_fallback(self):
        self.record(session=999)
        data = self.load()
        self.assertEqual(data["status"], "mismatch")
        self.assertEqual(data["values"], {})

    def test_corrupt_or_missing_record_file_does_not_fallback(self):
        record = self.record()
        path = record[0] / "inputs/resolved_config.json"
        path.write_text("{bad", encoding="utf-8")
        self.assertEqual(self.load()["status"], "error")
        self.assertEqual(self.load()["values"], {})
        path.unlink()
        self.assertEqual(self.load()["status"], "error")
        (record[0] / "manifest.json").write_text("{bad", encoding="utf-8")
        self.assertEqual(self.load()["source"], "record")
        self.assertEqual(self.load()["status"], "error")

    def test_unknown_version_and_cross_run_payload_are_rejected(self):
        record = self.record()
        path = record[0] / "inputs/resolved_config.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["schema_version"] = 2
        path.write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(self.load()["status"], "error")
        data.update(schema_version=1, run_id="different-run")
        path.write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(self.load()["status"], "error")

    def test_manifest_cannot_read_outside_run_directory(self):
        record = self.record()
        path = record[0] / "manifest.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["requested_args"] = "../../private.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        with patch("tools.lora_training_settings._read_json", wraps=__import__('tools.lora_training_settings', fromlist=['_read_json'])._read_json) as reader:
            self.assertEqual(self.load()["status"], "error")
            self.assertFalse(any("private" in str(call) for call in reader.call_args_list))

    def test_missing_checkpoint_and_minimal_metadata_are_not_errors(self):
        self.model.unlink()
        self.assertEqual(self.load()["status"], "unavailable")
        self.header({})
        self.assertEqual(self.load()["status"], "unavailable")
        self.record()
        data = self.load()
        self.assertEqual(data["association"], "unique_output_name")
        self.assertIn("未検証", "".join(data["notes"]))

    def test_invalid_safetensors_header_is_bounded(self):
        for data in [b"", struct.pack("<Q", 2**63), struct.pack("<Q", 1024)+b"{}"]:
            self.model.write_bytes(data)
            with self.assertRaises(ValueError):
                read_metadata(self.model)
            self.assertEqual(self.load()["status"], "unavailable")

    def test_write_failure_is_best_effort_and_leaves_partial_state(self):
        with patch("library.training_settings._write", side_effect=OSError("not writable")):
            self.assertIsNone(self.record(resolve=False))
        record = self.record(resolve=False)
        with patch("library.training_settings._write", side_effect=OSError("not writable")):
            finish_record(record, self.args, self.optimizer, None, [], self.metadata, {})
        self.assertEqual(self.load()["value_stage"], "requested")

    def test_sensitive_keys_and_nonjson_values(self):
        data = snapshot({"huggingface_token": "secret", "network_args": ["api_key=secret", "class_tokens=abc"],
                         "class_tokens": "character", "max_token_length": 225, "tensor": TensorMustNotBeRead()})
        self.assertNotIn("secret", json.dumps(data))
        self.assertEqual(data["class_tokens"], "character")
        self.assertEqual(data["max_token_length"], 225)
        self.assertEqual(data["tensor"], {"unrecorded": "non_json_value"})

    def test_display_is_escaped_and_diagnostics_are_unchanged(self):
        self.args.network_dim = "</script><img src=x onerror=alert(1)>"
        self.record()
        settings = self.load()
        html = render_settings(settings)
        self.assertIn("&lt;img", html)
        self.assertNotIn("<img", html)
        self.assertIn("None（指定なし）", html)
        self.assertIn("false", html)
        report = {"base_name": "test", "charts": {}, "training_settings": settings, "diagnostics": {"score": 50}}
        before = copy.deepcopy(report)
        self.assertIn("学習設定", build_html(report))
        self.assertEqual(report, before)


if __name__ == "__main__":
    unittest.main()
