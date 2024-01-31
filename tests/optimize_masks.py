import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import tyro
from einops import rearrange
from loguru import logger
from mr_sim.forward_model import forward_model
from mr_sim.trj_lib import trj_lib
from torch.optim import Adam
from mr_recon.nufft import gridded_nufft

from mr_recon.utils.general import create_exp_dir, seed_everything

from test_mrfrf import cartesian_dcf

from pathlib import Path
import matplotlib.pyplot as plt


@dataclass
class TrainCfg:
    log_dir = Path("logs")
    exp_name: str = "train-masks"
    n_masks: int = 2
    im_size: (int, int) = (64, 64)
    in_range_lambda: float = 1e-3
    n_iter: int = 5000
    lr: float = 1e-3
    device: str = "cuda"

def in_range_loss(lower, upper, x):
    # lower log barrier:
    lower_log_barrier = - torch.log(x - (lower - 1e-3))
    upper_log_barrier = - torch.log(upper + 1e-3 - x)

    return torch.sum(lower_log_barrier + upper_log_barrier)

def train(psf: Callable, cfg: TrainCfg):
    masks = torch.rand(cfg.n_masks, *cfg.im_size, device=cfg.device).requires_grad_(
        True
    )  # N, H, W
    optimizer = Adam([masks], lr=cfg.lr)
    all_elements_but_diag_mask = ~torch.eye(
        cfg.n_masks, dtype=torch.bool, device=cfg.device
    )

    for i in range(cfg.n_iter):
        optimizer.zero_grad()
        masks_after_psf = torch.abs(psf(masks))  # N, H, W
        interference = masks_after_psf[:, None, ...] * masks[None, ...]  # N, N, H, W
        interference = interference.sum(dim=(-1, -2))  # N, N
        interference_loss = interference[all_elements_but_diag_mask].sum()
        range_loss = in_range_loss(0, 1, masks)
        loss = interference_loss + cfg.in_range_lambda * range_loss
        loss.backward()
        optimizer.step()
        masks.data = torch.clamp(masks.data, 0, 1)

        if i % 100 == 0:
            print(f"Iter {i}/{cfg.n_iter}: {loss.item()}")

    return masks.detach().cpu().numpy()


def plot_masks(masks: np.ndarray, save_dir: Path):
    N = len(masks)
    side_length = int(np.ceil(np.sqrt(N)))

    fig, axes = plt.subplots(side_length, side_length, figsize=(12, 12))

    for i in range(N):
        ax = axes[i // side_length, i % side_length]
        ax.imshow(masks[i], cmap="gray")
        ax.set_title(f"Mask {i + 1}")
        ax.axis("off")

    # Remove empty subplots, if any
    for i in range(N, side_length * side_length):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout()
    plt.savefig(save_dir / "masks.png")
    plt.close()

def simple_debug_trj(im_size):
    x, y = torch.meshgrid(torch.arange(im_size[0]) - im_size[0] // 2, torch.arange(im_size[1]) - im_size[1] // 2)
    trj_y = y[:, ::2]
    trj_x = x[:, ::2]
    trj = torch.stack([trj_x, trj_y], dim=-1)
    trj = rearrange(trj, 'npe nro d -> nro npe 1 d')

    return trj

def main():
    seed_everything(42)

    train_cfg = tyro.cli(TrainCfg)
    exp_dir = create_exp_dir(train_cfg.log_dir,train_cfg.exp_name)
    logger.add(exp_dir / "log-{time}.log")

    trj_obj = trj_lib(train_cfg.im_size)
    trj = np.round(trj_obj.gen_radial_MRF_trj(ntr=1, n_shots=100, R=1))
    # TODO: take just one interleaf
    # trj = torch.from_numpy(trj).type(torch.float32).to(train_cfg.device)
    trj = simple_debug_trj(train_cfg.im_size).to(train_cfg.device)
    dcf = cartesian_dcf(trj, train_cfg.im_size).to(train_cfg.device).squeeze()
    nufft_ob = gridded_nufft(
        train_cfg.im_size, train_cfg.device, grid_oversamp=1.0
    ).to(train_cfg.device)
    trj = rearrange(
        nufft_ob.rescale_trajectory(trj),
        "nro npe ntr d -> ntr nro npe d",
    )
    trj = trj.expand(train_cfg.n_masks, -1, -1, -1)

    def psf(masks: torch.Tensor):
        Fm = nufft_ob.forward(masks, trj)
        return nufft_ob.adjoint(Fm * dcf, trj)

    masks = train(psf, train_cfg)

    plot_masks(masks, exp_dir)


if __name__ == "__main__":
    main()
