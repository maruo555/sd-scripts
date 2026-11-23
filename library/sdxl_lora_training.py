"""SDXL LoRA features at dataset, optimizer, checkpoint and epoch boundaries."""
from dataclasses import dataclass, replace
from functools import partial
import logging
import time

from torch import Tensor

from library import checkpoint_io, config_util, logging_util
from library.avg_training import EpochAveraging, save_final_raw
from library.grad_norm_guardian import GradNormGuardian
from library.promote_bank import BankItem
from library.training_run_control import create_run_controller
from library.training_run_output import RunOutput, reserve_run
from library.training_run_weights import publish_weight, record_weight, remove_weight, weight_args
from library.training_settings import optimizer_groups, snapshot

logger = logging.getLogger(__name__)


@dataclass
class ForwardInputs:
    """Intermediate values reused when this training batch enters the promote bank."""

    noisy_latents: Tensor | None = None
    weighted_tokens: tuple | None = None


# sdxl_train_network.py が学習1回につき1つ作り、train_network.py の共通ループが各メソッドを呼ぶ。
# 同じインスタンスを使い続けるので、Guardianの履歴やDQの倍率、promoteの採点用入力を次のステップに引き継げる。
class SdxlLoraTraining:
    """One training invocation. Keep feature state out of the shared NetworkTrainer."""

    def __init__(self, args, config):
        self.started_at = time.perf_counter()
        self.requested = snapshot(vars(args))
        self.config = config
        self.averaging = EpochAveraging(config.values) if "averaging" in config.training_features else None
        self.run = self.output = None
        self.dq = self.guardian = None
        self.bank = self.capture = self.score = None
        self.group_timesteps = self.created_groups = None

    def validate_dataset(self, blueprint):
        if self.config.changes_training:
            config_util.validate_custom_training_datasets(blueprint)

    def prepare_dataset(self, group):
        if self.averaging is not None and self.averaging.bank is not None:
            for dataset in getattr(group, "datasets", []):
                getattr(dataset, "dreambooth_dataset_delegate", dataset).record_promote_labels = True

    def open_run(self, args, accelerator, session_id, started_at):
        if self.config.changes_training and accelerator.state.deepspeed_plugin is not None:
            raise ValueError("DeepSpeed from the Accelerate configuration is unsupported with custom SDXL training")
        self.run = reserve_run(args, accelerator, session_id, started_at)
        if accelerator.is_main_process:
            self.output = RunOutput(self.run, self.requested)
            logger.info("SDXL LoRA run: %s", self.run.directory)

    def checkpoint_args(self, args):
        return weight_args(args, self.run)

    def controller(self, save_args, is_main_process):
        return create_run_controller(save_args, is_main_process, self.config)

    def optimizer_created(self, optimizer, descriptions):
        if self.output is not None:
            self.created_groups = optimizer_groups(optimizer, descriptions)

    def prepare_network(self, args, network, num_processes):
        if "dq" in self.config.training_features:
            from library.dq_training import DQTraining

            self.dq = DQTraining(self.config, network, total_steps=args.max_train_steps,
                                 lr_warmup_steps=args.lr_warmup_steps, num_processes=num_processes)

    def prepare_loop(self, *, args, trainer, accelerator, network, optimizer, descriptions, dataset, epochs,
                     unet, text_encoders, noise_scheduler, weight_dtype, text_encoding_strategy, tokenize_strategy):
        if "guardian" in self.config.training_features:
            self.guardian = GradNormGuardian(self.config.values["grad_norm_mode"], network.parameters())
        if self.output is not None:
            self.output.configure(args=args, config=self.config, dq=self.dq, network=accelerator.unwrap_model(network),
                                  optimizer=optimizer, descriptions=descriptions, dataset=dataset, metadata=trainer._metadata,
                                  epochs=epochs, diffusion_steps=noise_scheduler.config.num_train_timesteps,
                                  averaging=self.averaging, created_groups=self.created_groups)
        if self.averaging is not None and self.averaging.bank is not None:
            self.score = partial(trainer.score_promote_bank, args=args, accelerator=accelerator, unet=unet,
                                 text_encoders=text_encoders, noise_scheduler=noise_scheduler, weight_dtype=weight_dtype,
                                 text_encoding_strategy=text_encoding_strategy, tokenize_strategy=tokenize_strategy)

    def start_epoch(self, epoch, epochs):
        if self.averaging is not None:
            self.bank = self.averaging.bank if self.averaging.eligible(epoch, epochs) else None

    def start_progress(self, progress, is_main_process):
        if self.started_at is not None:
            logging_util.start_step_progress(progress, time.perf_counter() - self.started_at, main_process=is_main_process)
            self.started_at = None

    def validate_batch(self, batch):
        if self.output is not None and self.output.group is not None:
            self.output.group.validate_batch(batch)

    def before_forward(self, completed_steps):
        if self.dq is not None:
            self.dq.begin_attempt(completed_steps)

    def observe_target(self, batch, target, timesteps, huber_c, weighting):
        if "group_loss_log" in self.config.observation_features:
            self.group_timesteps = timesteps.detach()
        if self.capture is not None:
            self.bank.add(BankItem.capture(batch, self.capture.noisy_latents, target, timesteps,
                                           huber_c, weighting, self.capture.weighted_tokens))

    def after_backward(self, loss, epoch, batch_index, accelerator):
        observation = self.guardian.observe() if self.guardian is not None else None
        loss_value = None
        if self.output is not None and (observation is not None or self.output.group is not None):
            loss_value = loss.detach().item()
            if observation is not None:
                self.output.gradient(epoch, batch_index, observation, loss_value, accelerator.scaler)
        # この判定を受けた共通ループが、今回のoptimizer更新と学習率スケジューラを進めるか決める。
        return observation is not None and observation.skipped, loss_value

    def after_sync_step(self, batch, loss_value, epoch, step, scheduler, skipped):
        if self.output is not None and self.output.group is not None:
            self.output.group_step(batch, self.group_timesteps, loss_value, epoch, step, skipped)
        if self.dq is not None:
            # stepは更新を試みた回数。Guardianが棄却した回も1回と数え、DQはその回の統計を破棄する。
            self.dq.finish_attempt(step, epoch=epoch, guardian_skipped=skipped)
        if self.output is not None:
            self.output.dq_and_rank(self.dq, scheduler, epoch, step, skipped)

    @property
    def skipped_steps(self):
        return self.guardian.skipped_steps if self.guardian is not None else None

    def after_epoch_save(self, epoch, epochs, network, optimizer, accelerator, diffusion_steps, step):
        if self.averaging is not None:
            # epoch末の定期保存・サンプル生成を先に済ませることで、promote採用前の重みで両方を揃える。
            rows = self.averaging.finish_epoch(epoch, epochs, accelerator.unwrap_model(network), optimizer, self.score,
                                               accelerator.device, diffusion_steps)
            if rows:
                if self.output is not None:
                    self.output.writer.append("promote", rows)
                decision = rows[0]
                logger.info("promote epoch %s: %s, raw=%s, %s=%s, streak=%s, applied=%s",
                            epoch, decision["status"], decision["raw_proxy_loss"], decision["selected_candidate_mode"],
                            decision["selected_proxy_loss"], decision["win_streak"], decision["promote_applied"])
        if self.output is not None:
            self.output.writer.end_epoch(step)

    def save_weight(self, name, moment, network, dtype, metadata):
        metadata = dict(metadata, ss_run_id=self.run.run_id, ss_session_id=self.run.session_id)
        return publish_weight(self.run, name, moment, lambda path: network.save_weights(path, dtype, metadata))

    def remove_weight(self, name):
        remove_weight(self.run, name)

    def save_final_raw(self, save_args, save_dtype, metadata, moment):
        if self.averaging is None or self.averaging.final_raw is None:
            return
        name = checkpoint_io.get_last_ckpt_name(save_args, "_final_raw.safetensors")
        path = self.run.directory / name
        metadata = dict(metadata, ss_steps=str(moment.attempt_step), ss_epoch=str(moment.epoch),
                        ss_run_id=self.run.run_id, ss_session_id=self.run.session_id)
        save_final_raw(path, self.averaging.final_raw, save_dtype, metadata)
        record_weight(self.run, path, replace(moment, role="final_raw", phase="before_epoch_decision"))

    def finish(self, status):
        return self.output is not None and self.output.finish(status)

    def report_after_completion(self):
        if "report_after_training" in self.config.observation_features:
            from library.training_run_report import report_after_training
            report_after_training(self.run)
