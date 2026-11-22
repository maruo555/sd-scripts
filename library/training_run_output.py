"""SDXL run lifecycle. The trainer supplies resolved objects and completed training observations."""
import datetime as dt
import logging
from pathlib import Path

from library.training_run import TrainingRun
from library.training_settings import optimizer_groups, save_settings, snapshot_metadata, source_identity
from library.training_diagnostics import RunCsvWriter

logger = logging.getLogger(__name__)


def reserve_run(args, accelerator, session_id, started_at):
    message = [None]
    if accelerator.is_main_process:
        try:
            run = TrainingRun.create(args.output_dir, args.output_name, session_id,
                                     now=dt.datetime.fromtimestamp(started_at).astimezone())
            message[0] = (str(run.directory), None)
        except (OSError, ValueError) as exc:
            message[0] = (None, str(exc))
    if accelerator.num_processes > 1:
        from accelerate.utils import broadcast_object_list
        broadcast_object_list(message)
    directory, error = message[0]
    if error is not None:
        raise RuntimeError(f"Could not reserve training run: {error}")
    return TrainingRun.open(directory)


class RunOutput:
    def __init__(self, run, requested):
        self.run = run
        self.requested = requested
        self.provenance = source_identity(Path(__file__).resolve().parents[1])
        self.writer = self.rank = self.group = None
        self.finished = False
        save_settings(run, requested, provenance=self.provenance)

    def configure(self, *, args, config, dq, network, optimizer, descriptions, dataset, metadata, epochs, diffusion_steps, averaging=None, created_groups=None):
        from library.dq_auto import EMA_BETA
        from library.group_loss import GroupLossObserver
        from library.lora_rank_stats import RankObserver

        v = config.values
        if v["rank_log"]:
            self.rank = RankObserver(network, optimizer)
        if v["group_loss_log"]:
            self.group = GroupLossObserver(dataset)
        auto = dq.auto_controller if dq is not None else None
        promote = v["avg_cp"] and v["avg_cp_mode"] == "promote"
        active = {"rank": self.rank is not None, "group_loss": self.group is not None, "gradient": bool(v["grad_norm_mode"]),
                  "dq": dq is not None and v["dq_delta_log"], "dq_auto": auto is not None, "promote": promote}
        self.writer = RunCsvWriter(self.run, [k for k, enabled in active.items() if enabled],
                                   expected_modules_by_scope={"unet": self.rank.module_names} if self.rank else {})
        if auto:
            self.writer.append("dq_auto", [auto.init_row])
        dq_settings = dict(enabled=dq is not None, stat=v["dq_delta_stat"], granularity=v["dq_delta_granularity"],
                           rounding_mode=v["dq_delta_mode"], scope=v["dq_delta_scope"], log_scope=v["dq_delta_log_scope"],
                           metric_definition_version=1, mul=dq.assignment_record if dq else None,
                           auto=auto.settings if auto else {"enabled": False, "ema_beta": EMA_BETA,
                                                            "init": {"applied": False, "mul": None, "clip_target": None}})
        bank_size = v["avg_shadow_bank_size"]
        resolved = dict(
            args=vars(args), metadata=snapshot_metadata(metadata), optimizer_groups_created=created_groups,
            optimizer_groups_at_start=optimizer_groups(optimizer, descriptions),
            sdxl_unet=dict(fp16_safe_norms_mode=v["fp16_safe_norms_mode"], timestep_freq_shift=int(v["downscale_freq_shift"])),
            text_encoder_lrs=config.text_encoder_lrs,
            output_root=str(self.run.output_root), run_dir=str(self.run.directory), options=v,
            training=dict(total_epochs=epochs, total_attempt_steps=args.max_train_steps, num_train_timesteps=diffusion_steps,
                          dataset_batch_sizes=[getattr(d, "batch_size", None) for d in getattr(dataset, "datasets", [])]),
            guardian=dict(mode=v["grad_norm_mode"], log_enabled=active["gradient"]), dq=dq_settings,
            rank=dict(enabled=active["rank"], interval_attempts=RankObserver.interval, mode="per_module"),
            group_loss=dict(enabled=active["group_loss"], ema_beta=GroupLossObserver.beta, flush_interval_attempts=100,
                            subsets=self.group.resolved_subsets if self.group else None),
            promote=dict(enabled=promote, mode=v["avg_mode"], pick="fixed", window_size=v["avg_window"],
                         bank_target_size=bank_size, pool_target_size=averaging.bank.pool_limit if promote else None,
                         patience=v["avg_shadow_patience"], margin=v["avg_shadow_margin"], begin_fraction=v["avg_begin"],
                         reset_stats=v["avg_reset_stats"]),
        )
        save_settings(self.run, self.requested, resolved=resolved, provenance=self.provenance)

    def gradient(self, epoch, batch_index, observation, loss, scaler):
        self.writer.append("gradient", [dict(epoch=epoch, batch_index=batch_index, grad_norm=observation.norm,
                                           threshold=observation.threshold, loss=loss,
                                           grad_scale=scaler.get_scale() if scaler else None)])

    def group_step(self, batch, timestep, loss, epoch, step, skipped):
        if self.group:
            if not skipped:
                self.writer.append("group_loss", self.group.observe(batch, timestep, loss, epoch, step))
            self.writer.after_attempt(step)

    def dq_and_rank(self, dq, scheduler, epoch, step, skipped):
        if dq is not None and dq.last_result is not None:
            result = dq.last_result
            if result.auto_row is not None:
                self.writer.append("dq_auto", [result.auto_row])
            if result.log_rows:
                self.writer.append("dq", result.log_rows)
        if self.rank:
            rows = self.rank.observe(epoch, step, scheduler, guardian_skipped=skipped)
            if rows:
                self.writer.append("rank", rows)

    def finish(self, status):
        if not self.finished:
            complete = self.writer.close(interrupted=status == "failed") if self.writer else False
            self.run.finish(status if complete or status != "completed" else "failed", "complete" if complete else "partial")
            self.finished = True
        return bool(status == "completed" and self.writer and not self.writer.failed)
