from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sigpy as sp
import torch
import tyro
from einops import rearrange
from loguru import logger
from mr_sim.data_sim import data_sim, mrf_sequence, quant_phantom
from optimize_masks import in_range_loss, simple_debug_trj
from optimized_mrf.sequences import FISP, BareFISP, FISPParams, SequenceParams
from optimized_mrf.sequences.defaults import fisp_fa, fisp_tr
from test_mrfrf import cartesian_dcf
from torch.optim import Adam

from mr_recon.nufft import gridded_nufft
from tqdm import tqdm

@dataclass
class TrainCfg:
    log_dir = Path("logs")
    exp_name: str = "train-masks"
    n_modes: int = 2
    im_size: (int, int) = (64, 64)
    smoothing_factor: int = 50
    in_range_lambda: float = 1e-3
    n_iter: int = 5000
    lr: float = 1e-3
    device: str = "cuda"


def generate_fa_map(modes, smooth_factor, fa):
    n_modes, h, w = modes.shape
    smoothed_modes = torch.zeros(n_modes * smooth_factor, h, w).to(modes)
    interpolation_coeff = torch.linspace(1, 0, smooth_factor)[:, None, None].to(modes)
    for i in range(n_modes):
        smoothed_modes[i * smooth_factor : (i + 1) * smooth_factor, ...] = (
            interpolation_coeff * modes[i, ...]
            + (1 - interpolation_coeff) * modes[(i + 1) % n_modes, ...]
        )

    smoothed_n_modes, _, _ = smoothed_modes.shape
    fa_map = torch.zeros(smoothed_modes.shape[1:] + (len(fa),))
    modes = torch.tile(smoothed_modes, [int(np.ceil(len(fa) / smoothed_n_modes)), 1, 1])
    modes = rearrange(modes, "n h w -> h w n")
    fa_map = modes[..., : len(fa)] * fa

    return fa_map


def train(train_cfg: TrainCfg):
    modes = torch.ones(
        train_cfg.n_modes, *train_cfg.im_size, device=train_cfg.device
    ).requires_grad_(True)
    optimizer = Adam([modes], lr=train_cfg.lr)
    data = quant_phantom()
    t1 = torch.from_numpy(sp.resize(data["t1"][100], train_cfg.im_size)).to(train_cfg.device)
    t2 = torch.from_numpy(sp.resize(data["t2"][100], train_cfg.im_size)).to(train_cfg.device)
    pd = torch.from_numpy(sp.resize(data["pd"][100], train_cfg.im_size)).to(train_cfg.device)
    tissues = torch.zeros(*t1.shape, 1, 3, dtype=torch.float32).to(train_cfg.device)
    tissues[..., 0, 0] = pd
    tissues[..., 0, 1] = t1
    tissues[..., 0, 2] = t2

    trs = fisp_tr.clone().to(train_cfg.device)
    fas = fisp_fa.clone().to(train_cfg.device)
    seq_params = SequenceParams(
        flip_angles=fas,
        flip_angle_requires_grad=False,
        TR=trs,
        TR_requires_grad=False,
        TE=torch.ones(len(trs), dtype=torch.float32) * 1.75,
        TE_requires_grad=False,
    )
    fisp_params = FISPParams(seq_params)
    seq = BareFISP(fisp_params).to(train_cfg.device)

    trj = simple_debug_trj(train_cfg.im_size).to(train_cfg.device)
    dcf = cartesian_dcf(trj, train_cfg.im_size).to(train_cfg.device).squeeze()
    nufft_ob = gridded_nufft(train_cfg.im_size, train_cfg.device, grid_oversamp=1.0).to(
        train_cfg.device
    )
    trj = rearrange(
        nufft_ob.rescale_trajectory(trj),
        "nro npe ntr d -> ntr nro npe d",
    )
    trj = trj.expand(len(trs), -1, -1, -1)

    for i in tqdm(range(train_cfg.n_iter)):
        optimizer.zero_grad()

        # gen fa maps
        fa_maps = generate_fa_map(modes, train_cfg.smoothing_factor, fas)

        # gen imgs
        imgs = seq(tissues, fa_maps).squeeze()

        # gridded recon loss
        imgs = rearrange(imgs, "h w ntr -> ntr h w")
        ksp = nufft_ob(imgs, trj)
        recon = nufft_ob.adjoint(ksp * dcf, trj)
        gridded_recon_loss = torch.sum(torch.abs(recon - imgs) ** 2)

        # loss -- ims recon, FAs in range, CRLB of some subsequences
        range_loss = in_range_loss(0, 1, modes)
        loss = gridded_recon_loss + train_cfg.in_range_lambda * range_loss

        # TODO: add CRLB loss

        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            logger.info(f"iter: {i}/{train_cfg.n_iter} loss: {loss.item()}")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    train_cfg = tyro.cli(TrainCfg)
    train(train_cfg)
