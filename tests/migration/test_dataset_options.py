"""Tests for standard SDXL LoRA dataset settings before model loading."""
import argparse
from dataclasses import asdict

import pytest
from PIL import Image
from voluptuous import MultipleInvalid

from library import config_util
from library.subset import ControlNetSubset, DreamBoothSubset, FineTuningSubset


def blueprint(config, *, support_group=True):
    sanitizer = config_util.ConfigSanitizer(True, True, True, True, support_subset_group=support_group)
    return config_util.BlueprintGenerator(sanitizer).generate(config, argparse.Namespace()).dataset_group


def test_group_inheritance_and_precedence_across_dataset_types():
    config = {
        "general": {"group": " inherited "},
        "datasets": [
            {"subsets": [{"image_dir": "images1"}]},
            {"group": "dataset", "subsets": [{"metadata_file": "a.json"}, {"metadata_file": "b.json", "group": "subset"}]},
            {"subsets": [{"image_dir": "images2", "conditioning_data_dir": "controls", "group": "   "}]},
        ],
    }
    resolved = blueprint(config)
    constructors = [DreamBoothSubset, FineTuningSubset, ControlNetSubset]
    names = [
        [constructor(**asdict(subset.params)).group for subset in dataset.subsets]
        for constructor, dataset in zip(constructors, resolved.datasets)
    ]
    assert names == [["inherited"], ["dataset", "subset"], [None]]
    with pytest.raises(MultipleInvalid):
        blueprint(config, support_group=False)


def test_training_and_validation_keep_the_same_group(tmp_path):
    for i in range(4):
        Image.new("RGB", (64, 64)).save(tmp_path / f"image{i}.png")
    resolved = blueprint({
        "general": {"resolution": 64, "validation_split": 0.5, "validation_seed": 39, "group": "paired"},
        "datasets": [{"subsets": [{"image_dir": str(tmp_path), "class_tokens": "sample"}]}],
    })
    train, validation = config_util.generate_dataset_group_by_blueprint(resolved)
    assert train.num_train_images == validation.num_train_images == 2
    assert train.datasets[0].subsets[0].group == validation.datasets[0].subsets[0].group == "paired"


def test_controlnet_preserves_group_in_its_dreambooth_delegate(tmp_path):
    main_dir, control_dir = tmp_path / "source", tmp_path / "controls"
    for directory in (main_dir, control_dir):
        directory.mkdir()
        Image.new("RGB", (64, 64)).save(directory / "image.png")
    resolved = blueprint({
        "general": {"resolution": 64},
        "datasets": [{"subsets": [{
            "image_dir": str(main_dir), "conditioning_data_dir": str(control_dir), "group": "control"
        }]}],
    })
    train, _ = config_util.generate_dataset_group_by_blueprint(resolved)
    assert train.datasets[0].dreambooth_dataset_delegate.subsets[0].group == "control"


def test_multiplier_checks_resolved_values_and_identifies_dataset():
    config = {
        "general": {"network_multiplier": 0.5},
        "datasets": [
            {"network_multiplier": 1.0, "subsets": [{"image_dir": "ordinary"}]},
            {"subsets": [{"metadata_file": "custom.json"}]},
        ],
    }
    resolved = blueprint(config)
    assert [d.params.network_multiplier for d in resolved.datasets] == [1.0, 0.5]
    with pytest.raises(ValueError, match=r"datasets\[1\].*custom\.json.*network_multiplier=0\.5"):
        config_util.validate_custom_training_datasets(resolved)
    config["datasets"][1]["network_multiplier"] = 1.0
    config_util.validate_custom_training_datasets(blueprint(config))
    config_util.validate_custom_training_datasets(blueprint({"datasets": [{"subsets": [{"image_dir": "default"}]}]}))


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_other_non_unit_multipliers_are_rejected_only_by_explicit_validation(value):
    resolved = blueprint({"datasets": [{"network_multiplier": value, "subsets": [{"image_dir": "outside"}]}]}, support_group=False)
    with pytest.raises(ValueError, match=r"datasets\[0\].*outside"):
        config_util.validate_custom_training_datasets(resolved)


def test_batch_size_uses_resolved_dataset_value():
    config = {
        "general": {"batch_size": 2},
        "datasets": [
            {"batch_size": 1, "subsets": [{"image_dir": "first"}]},
            {"subsets": [{"metadata_file": "second.json"}]},
        ],
    }
    with pytest.raises(ValueError, match=r"datasets\[1\].*second\.json.*batch_size=2"):
        config_util.validate_custom_training_datasets(blueprint(config))
    config["datasets"][1]["batch_size"] = 1
    config_util.validate_custom_training_datasets(blueprint(config))


@pytest.mark.parametrize("value", [True, 1, "false"])
def test_differential_output_preservation_uses_upstream_truthiness(value):
    config = {
        "general": {"custom_attributes": {"diff_output_preservation": value}},
        "datasets": [{"subsets": [
            {"image_dir": "disabled", "custom_attributes": {"diff_output_preservation": False}},
            {"image_dir": "inherited"},
        ]}],
    }
    with pytest.raises(ValueError, match=r"datasets\[0\]\.subsets\[1\].*inherited.*diff_output_preservation="):
        config_util.validate_custom_training_datasets(blueprint(config))


@pytest.mark.parametrize("attributes", [{}, {"other": True}, {"diff_output_preservation": False},
                                        {"diff_output_preservation": 0}, {"diff_output_preservation": ""}])
def test_inactive_preservation_does_not_block_custom_training(attributes):
    config = {"datasets": [{"subsets": [{"image_dir": "train", "custom_attributes": attributes}]}]}
    config_util.validate_custom_training_datasets(blueprint(config))


@pytest.mark.parametrize("options", [[], ["--rank_log"], ["--group_loss_log"]])
def test_observation_or_standard_learning_delegates_dataset_settings(monkeypatch, options):
    import sdxl_train_network
    import train_network
    from library import maruo_training_config, strategy_base

    args = sdxl_train_network.setup_parser().parse_args(["--network_module", "networks.lora", "--dataset_config", "test.toml", *options])
    trainer = sdxl_train_network.SdxlNetworkTrainer()
    trainer.maruo_config = maruo_training_config.resolve_training_config(args)
    assert not trainer.maruo_config.changes_training
    dataset = {"general": {"batch_size": 2, "network_multiplier": 0.5}, "datasets": [
        {"subsets": [{"image_dir": "standard", "custom_attributes": {"diff_output_preservation": True}}]}
    ]}
    monkeypatch.setattr(config_util, "load_user_config", lambda path: dataset)
    monkeypatch.setattr(trainer, "get_tokenize_strategy", lambda args: None)
    monkeypatch.setattr(trainer, "get_tokenizers", lambda strategy: [])
    monkeypatch.setattr(trainer, "get_latents_caching_strategy", lambda args: None)
    monkeypatch.setattr(strategy_base.TokenizeStrategy, "set_strategy", lambda strategy: None)
    monkeypatch.setattr(strategy_base.LatentsCachingStrategy, "set_strategy", lambda strategy: None)
    monkeypatch.setattr(train_network, "set_seed", lambda seed: None)
    monkeypatch.setattr(train_network, "setup_logging", lambda *args, **kwargs: None)

    class UpstreamDatasetReached(Exception):
        pass

    def check_native_dataset(resolved):
        data = resolved.datasets[0]
        assert data.params.batch_size == 2 and data.params.network_multiplier == 0.5
        assert data.subsets[0].params.custom_attributes == {"diff_output_preservation": True}
        raise UpstreamDatasetReached

    monkeypatch.setattr(config_util, "generate_dataset_group_by_blueprint", check_native_dataset)
    # Test the dataset consumer even at revisions where a later observation feature is not wired yet.
    with pytest.raises(UpstreamDatasetReached):
        train_network.NetworkTrainer.train(trainer, args)
