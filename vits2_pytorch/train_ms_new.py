import argparse
import itertools
import json
import math
import os

import torch
import torch.distributed as dist
# from tensorboardX import SummaryWriter
import torch.multiprocessing as mp
import tqdm
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import commons
import models
import utils
from data_utils import (DistributedBucketSampler, TextAudioSpeakerCollate,
                        TextAudioSpeakerLoader)
from losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from models import (AVAILABLE_DURATION_DISCRIMINATOR_TYPES,
                    AVAILABLE_FLOW_TYPES, 
                    DurationDiscriminatorV1, DurationDiscriminatorV2,
                    MultiPeriodDiscriminator, SynthesizerTrn)
from text.symbols import symbols

torch.backends.cudnn.benchmark = True
global_step = 0


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6060"

    hps = utils.get_hparams()
    # mp.spawn(
    #     run,
    #     nprocs=n_gpus,
    #     args=(
    #         n_gpus,
    #         hps,
    #     ),
    # )
    # Gọi hàm run trực tiếp mà không cần mp.spawn
    run(0, n_gpus, hps)

def run(rank, n_gpus, hps):
    net_dur_disc = None
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=n_gpus, rank=rank
    )
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    if (
        "use_mel_posterior_encoder" in hps.model.keys()
        and hps.model.use_mel_posterior_encoder == True
    ):
        print("Using mel posterior encoder for VITS2")
        posterior_channels = 80  # vits2
        hps.data.use_mel_posterior_encoder = True
    else:
        print("Using lin posterior encoder for VITS1")
        posterior_channels = hps.data.filter_length // 2 + 1
        hps.data.use_mel_posterior_encoder = False

    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    collate_fn = TextAudioSpeakerCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
    )
    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=8,
            shuffle=False,
            batch_size=hps.train.batch_size,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
    # some of these flags are not being used in the code and directly set in hps json file.
    # they are kept here for reference and prototyping.
    if (
        "use_transformer_flows" in hps.model.keys()
        and hps.model.use_transformer_flows == True
    ):
        use_transformer_flows = True
        transformer_flow_type = hps.model.transformer_flow_type
        print(f"Using transformer flows {transformer_flow_type} for VITS2")
        assert (
            transformer_flow_type in AVAILABLE_FLOW_TYPES
        ), f"transformer_flow_type must be one of {AVAILABLE_FLOW_TYPES}"
    else:
        print("Using normal flows for VITS1")
        use_transformer_flows = False

    if (
        "use_spk_conditioned_encoder" in hps.model.keys()
        and hps.model.use_spk_conditioned_encoder == True
    ):
        if hps.data.n_speakers == 0:
            raise ValueError(
                "n_speakers must be > 0 when using spk conditioned encoder to train multi-speaker model"
            )
        use_spk_conditioned_encoder = True
    else:
        print("Using normal encoder for VITS1")
        use_spk_conditioned_encoder = False

    if (
        "use_noise_scaled_mas" in hps.model.keys()
        and hps.model.use_noise_scaled_mas == True
    ):
        print("Using noise scaled MAS for VITS2")
        use_noise_scaled_mas = True
        mas_noise_scale_initial = 0.01
        noise_scale_delta = 2e-6
    else:
        print("Using normal MAS for VITS1")
        use_noise_scaled_mas = False
        mas_noise_scale_initial = 0.0
        noise_scale_delta = 0.0

    if (
        "use_duration_discriminator" in hps.model.keys()
        and hps.model.use_duration_discriminator == True
    ):
        # print("Using duration discriminator for VITS2")
        use_duration_discriminator = True

        # comment - choihkk
        # add duration discriminator type here
        # I think it would be a good idea to come up with a method to input this part accurately, like a hydra
        duration_discriminator_type = getattr(
            hps.model, "duration_discriminator_type", "dur_disc_1"
        )
        print(f"Using duration_discriminator {duration_discriminator_type} for VITS2")
        assert (
            duration_discriminator_type in AVAILABLE_DURATION_DISCRIMINATOR_TYPES
        ), f"duration_discriminator_type must be one of {AVAILABLE_DURATION_DISCRIMINATOR_TYPES}"
        duration_discriminator_type = AVAILABLE_DURATION_DISCRIMINATOR_TYPES
        if duration_discriminator_type == "dur_disc_1":
            net_dur_disc = DurationDiscriminatorV1(
                hps.model.hidden_channels,
                hps.model.hidden_channels,
                3,
                0.1,
                gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
            ).cuda(rank)
        elif duration_discriminator_type == "dur_disc_2":
            net_dur_disc = DurationDiscriminatorV2(
                hps.model.hidden_channels,
                hps.model.hidden_channels,
                3,
                0.1,
                gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
            ).cuda(rank) 
    else:
        print("NOT using any duration discriminator like VITS1")
        net_dur_disc = None
        use_duration_discriminator = False

    net_g = SynthesizerTrn(
        len(symbols),
        posterior_channels,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
        **hps.model,
    ).cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    if net_dur_disc is not None:
        optim_dur_disc = torch.optim.AdamW(
            net_dur_disc.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
    else:
        optim_dur_disc = None

    # comment - choihkk
    # if we comment out unused parameter like DurationDiscriminator's self.pre_out_norm1,2 self.norm_1,2
    # and ResidualCouplingTransformersLayer's self.post_transformer
    # we don't have to set find_unused_parameters=True
    # but I will not proceed with commenting out for compatibility with the latest work for others
    net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
    net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
    if net_dur_disc is not None:
        net_dur_disc = DDP(
            net_dur_disc, device_ids=[rank], find_unused_parameters=True)

    try:
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g
        )
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d
        )
        if net_dur_disc is not None:
            _, _, _, epoch_str = utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "DUR_*.pth"),
                net_dur_disc,
                optim_dur_disc,
            )
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    if net_dur_disc is not None:
        scheduler_dur_disc = torch.optim.lr_scheduler.ExponentialLR(
            optim_dur_disc, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
        )
    else:
        scheduler_dur_disc = None

    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d, net_dur_disc],
                [optim_g, optim_d, optim_dur_disc],
                [scheduler_g, scheduler_d, scheduler_dur_disc],
                scaler,
                [train_loader, eval_loader],
                logger,
                [writer, writer_eval],
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d, net_dur_disc],
                [optim_g, optim_d, optim_dur_disc],
                [scheduler_g, scheduler_d, scheduler_dur_disc],
                scaler,
                [train_loader, None],
                None,
                None,
            )
        scheduler_g.step()
        scheduler_d.step()
        if net_dur_disc is not None:
            scheduler_dur_disc.step()


def train_and_evaluate(
    rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers
):
    net_g, net_d, net_dur_disc = nets
    optim_g, optim_d, optim_dur_disc = optims
    scheduler_g, scheduler_d, scheduler_dur_disc = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    if net_dur_disc is not None:
        net_dur_disc.train()

    train_loss = 0
    val_loss = 0
    num_batches = 0

    if rank == 0:
        loader = tqdm.tqdm(train_loader, desc="Loading train data")
    else:
        loader = train_loader
    for batch_idx, (
        x,
        x_lengths,
        spec,
        spec_lengths,
        y,
        y_lengths,
        speakers,
    ) in enumerate(loader):
        if net_g.module.use_noise_scaled_mas:
            current_mas_noise_scale = (
                net_g.module.mas_noise_scale_initial
                - net_g.module.noise_scale_delta * global_step
            )
            net_g.module.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(
            rank, non_blocking=True
        )
        spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(
            rank, non_blocking=True
        )
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(
            rank, non_blocking=True
        )
        speakers = speakers.cuda(rank, non_blocking=True)

        with autocast(enabled=hps.train.fp16_run):
            (
                y_hat,
                l_length,
                attn,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
                (hidden_x, logw, logw_),
            ) = net_g(x, x_lengths, spec, spec_lengths, speakers)

            if (
                hps.model.use_mel_posterior_encoder
                or hps.data.use_mel_posterior_encoder
            ):
                mel = spec
            else:
                mel = spec_to_mel_torch(
                    spec.float(),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            y = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                loss_disc_all = loss_disc

            # Duration Discriminator
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(
                    hidden_x.detach(), x_mask.detach(), logw_.detach(), logw.detach()
                )
                with autocast(enabled=False):
                    (
                        loss_dur_disc,
                        losses_dur_disc_r,
                        losses_dur_disc_g,
                    ) = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
                    loss_dur_disc_all = loss_dur_disc
                optim_dur_disc.zero_grad()
                scaler.scale(loss_dur_disc_all).backward()
                scaler.unscale_(optim_dur_disc)
                grad_norm_dur_disc = commons.clip_grad_value_(
                    net_dur_disc.parameters(), None
                )
                scaler.step(optim_dur_disc)

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw_, logw)
            with autocast(enabled=False):
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
                if net_dur_disc is not None:
                    loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
                    loss_gen_all += loss_dur_gen

        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        train_loss += loss_gen_all.item()
        num_batches += 1

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc_all,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }

                scalar_dict.update(
                    {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                )
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    scalars=scalar_dict,
                )

            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_g, eval_loader, writer_eval,global_step)
                utils.save_checkpoint(
                    net_g,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
                )
                utils.save_checkpoint(
                    net_d,
                    optim_d,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
                )
                if net_dur_disc is not None:
                    utils.save_checkpoint(
                        net_dur_disc,
                        optim_dur_disc,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(hps.model_dir, "DUR_{}.pth".format(global_step)),
                    )

                utils.remove_old_checkpoints(hps.model_dir, prefixes=["G_*.pth", "D_*.pth", "DUR_*.pth"])
        global_step += 1

    if rank == 0:
        logger.info("====> Epoch: {}".format(epoch))
        avg_train_loss = train_loss / num_batches
        logger.info(f"Average Training Loss for Epoch {epoch}: {avg_train_loss}")



def evaluate(hps, generator, eval_loader, writer_eval, global_step):
    import torch
    import torch.nn as nn
    from utils import summarize, plot_spectrogram_to_numpy
    from mel_processing import mel_spectrogram_torch, spec_to_mel_torch

    generator.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    num_batches = 0
    max_eval_batches = 2         # Giới hạn batch để đánh giá nhanh
    max_input_len = 500          # Giới hạn độ dài input để tránh treo máy

    print("🔍 Running quick evaluation...")

    sample_logged = False

    with torch.no_grad():
        for batch_idx, (
            x,
            x_lengths,
            spec,
            spec_lengths,
            y,
            y_lengths,
            speakers,
        ) in enumerate(eval_loader):

            if batch_idx >= max_eval_batches:
                break

            print(f"📦 Evaluating batch {batch_idx}...")

            # Chuyển dữ liệu lên GPU
            x, x_lengths = x.cuda(0), x_lengths.cuda(0)
            spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
            y, y_lengths = y.cuda(0), y_lengths.cuda(0)
            speakers = speakers.cuda(0)
            print("🛸 Data moved to GPU.")

            # Giới hạn độ dài để tránh quá tải
            x = x[:, :max_input_len]                     # x thường có shape [B, T]
            spec = spec[:, :, :max_input_len]            # spec có shape [B, C, T]
            y = y[:, :, :max_input_len]                  # y có shape [B, C, T]

            print(f"🔢 Shapes — x: {x.shape}, spec: {spec.shape}, y: {y.shape}")

            # Inference
            y_hat, attn, mask, *_ = generator.module.infer(
                x, x_lengths, speakers, max_len=400
            )
            y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length
            print("🎙 Inference done.")

            # Cắt ground truth để tính loss
            min_len = min(y.size(2), y_hat.size(2))  # Tính chiều dài nhỏ nhất giữa y và y_hat
            y_trimmed = y[:, :, :min_len]
            y_hat_trimmed = y_hat[:, :, :min_len]

            loss = criterion(y_hat_trimmed, y_trimmed)
            total_loss += loss.item()
            num_batches += 1

            # Logging mel và audio của 1 mẫu
            if not sample_logged:
                if hps.model.use_mel_posterior_encoder or hps.data.use_mel_posterior_encoder:
                    mel = spec
                else:
                    mel = spec_to_mel_torch(
                        spec,
                        hps.data.filter_length,
                        hps.data.n_mel_channels,
                        hps.data.sampling_rate,
                        hps.data.mel_fmin,
                        hps.data.mel_fmax,
                    )

                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1).float(),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )

                image_dict = {
                    "gen/mel": plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
                    "gt/mel": plot_spectrogram_to_numpy(mel[0].cpu().numpy()),
                }

                audio_dict = {
                    "gen/audio": y_hat[0, :, : y_hat_lengths[0]],
                    "gt/audio": y[0, :, : y_lengths[0]],
                }

                summarize(
                    writer=writer_eval,
                    global_step=global_step,
                    images=image_dict,
                    audios=audio_dict,
                    audio_sampling_rate=hps.data.sampling_rate,
                )
                sample_logged = True
                print("📈 Logged mel + audio to TensorBoard.")

    avg_loss = total_loss / max(1, num_batches)
    writer_eval.add_scalar("eval/mse_loss_avg", avg_loss, global_step)
    print(f"✅ Eval done — Avg MSE Loss: {avg_loss:.6f}")

    generator.train()


if __name__ == "__main__":
    main()
