import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from dataclasses import dataclass

import matplotlib
import numpy as np
import torch
import tyro
from torch.optim import Adam
import torch.distributions as D

matplotlib.use("webagg")
import matplotlib.pyplot as plt
from einops import rearrange
from mr_sim.trj_lib import trj_lib
from optimize_masks import in_range_loss, simple_debug_trj
from sklearn.cluster import DBSCAN
from test_mrfrf import cartesian_dcf
from tqdm import tqdm

from mr_recon.nufft import gridded_nufft
from torch.distributions.mixture_same_family import MixtureSameFamily


@dataclass
class Config:
    im_size: (int, int) = (64, 64)
    seq_len: int = 500
    n_avgs: int = 1
    device: str = "cuda"
    threshold: float = 0.1


def random_psd_matrix(dim):
    lower_triangular = torch.randn((dim, dim)).tril()
    psd_matrix = lower_triangular @ lower_triangular.t()

    return psd_matrix


def random_psd_matrices(n, dim):
    psd_matrices = torch.zeros((n, dim, dim))
    for i in range(n):
        psd_matrix = random_psd_matrix(dim)
        psd_matrices[i, :, :] = psd_matrix

    return psd_matrices


def find_optimal_labels(corr, n_labels: int = 2, n_mix: int = 6):
    mixing = torch.rand(n_labels, n_mix).to(corr.device).requires_grad_(True)
    mus = (64 * torch.rand(n_labels, n_mix, 2).to(corr.device)).requires_grad_(True)
    scale_tril = (5 * torch.rand(n_labels, n_mix, 2, 2).tril().to(corr.device)).requires_grad_(True)
    # sigmas = 10 * random_psd_matrices(n_labels * n_mix, 2)
    # sigmas = (
    #     rearrange(sigmas, "(n nmix) d1 d2 -> n nmix d1 d2", n=n_labels, nmix=n_mix)
    #     .to(corr.device)
    #     .requires_grad_(True)
    # )
    labels = (
        torch.randn(corr.shape[0], corr.shape[1], n_labels)
        .type(torch.float32)
        .to(corr.device)
        .requires_grad_(True)
    )

    def corr_loss(labels):
        loss = torch.sum(corr * torch.einsum("abc, dec", labels, labels))
        return loss

    def smoothness_loss(labels):
        pass

    optimizer = Adam([mixing, mus, scale_tril], lr=1e-2)
    # optimizer = Adam([labels], lr=1e-3)
    grid_x, grid_y = torch.meshgrid(
        torch.arange(corr.shape[0]), torch.arange(corr.shape[1])
    )
    # DEBUGGING: SEE IF THIS WORKS WHEN JUST USING TWO PIXELS, DOES THIS RESULTS IN DIFFERENT/SAME LABELS?
    for i in tqdm(range(100000)):
        optimizer.zero_grad()
        cords = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2).type(torch.float32).to(corr.device)
        mix = D.Categorical(torch.clip(mixing, 0))
        comp = D.MultivariateNormal(mus, scale_tril=torch.clip(scale_tril, 0.1).tril())
        gmm = MixtureSameFamily(mix, comp)
        labels = gmm.log_prob(cords)
        loss = corr_loss(torch.nn.functional.softmax(labels, dim=-1))
        loss.backward()
        optimizer.step()
        if i % 5000 == 0:
            plot_labels(labels)
            print(loss.item())

    return labels


def plot_labels(labels):
    plt.imshow(torch.argmax(labels, dim=-1).detach().cpu().numpy())
    plt.show()
    plt.close()


def find_best_separation(locs, labels):
    # -1 corresponds to outliers/noise
    n_labels = len(np.unique(labels[labels != -1]))
    for label in range(n_labels):
        label_locs = locs[labels == label]
        source_locs = label_locs[:, :2]
        target_locs = label_locs[:, 2:]
        im = np.zeros((64, 64))
        im[source_locs[:, 0], source_locs[:, 1]] = 1
        im[target_locs[:, 0], target_locs[:, 1]] = 2
        plt.imshow(im)
        plt.colorbar()
        plt.show()
        plt.close()


def main(cfg: Config):
    corrs = []
    for _ in tqdm(range(cfg.n_avgs)):
        random_sig = torch.randn(*cfg.im_size, 500).type(torch.complex64).to(cfg.device)
        # TODO: mask brain
        en = torch.linalg.norm(random_sig, dim=-1)
        cross_en = torch.einsum("ab, cd", en, en)
        # TODO: should this be normalized?
        prior_corr_mat = (
            torch.abs(torch.einsum("abc, dec", random_sig, random_sig.conj()))
            / cross_en
        )
        prior_corr_2_vis = rearrange(prior_corr_mat, "h1 w1 h2 w2 -> (h1 w1) (h2 w2)")
        # plt.imshow(prior_corr_2_vis.cpu().numpy())
        # plt.colorbar()
        # plt.show()
        # plt.close()

        # trj_obj = trj_lib(cfg.im_size)
        # trj = np.round(trj_obj.gen_radial_MRF_trj(ntr=1, n_shots=100, R=1))
        # TODO: take just one interleaf
        # trj = torch.from_numpy(trj).type(torch.float32).to(cfg.device)
        trj = simple_debug_trj(cfg.im_size).to(cfg.device)
        dcf = cartesian_dcf(trj, cfg.im_size).to(cfg.device).squeeze()
        nufft_ob = gridded_nufft(cfg.im_size, cfg.device, grid_oversamp=1.0).to(
            cfg.device
        )
        trj = rearrange(
            nufft_ob.rescale_trajectory(trj),
            "nro npe ntr d -> ntr nro npe d",
        )
        trj = trj.expand(cfg.seq_len, -1, -1, -1)

        ksp = nufft_ob(rearrange(random_sig, "h w ntr -> ntr h w"), trj)
        recon = rearrange(nufft_ob.adjoint(ksp * dcf, trj), "ntr h w -> h w ntr")
        recon_en = torch.linalg.norm(recon, dim=-1)
        cross_en = torch.einsum("ab, cd", en, recon_en)
        recon_corr_mat = (
            torch.abs(torch.einsum("abc, dec", random_sig, recon.conj())) / cross_en
        )
        corrs.append(recon_corr_mat)

    corrs = torch.stack(corrs, dim=0)
    corrs = torch.mean(corrs, dim=0)
    corrs_for_vis = rearrange(corrs, "h1 w1 h2 w2 -> (h1 w1) (h2 w2)")
    plt.imshow(corrs_for_vis.cpu().numpy())
    # plt.show()
    plt.close()

    # keep only relevant part -- FIXME: this is not completely accurate since this is a cross correlation
    a, b, c, d = torch.meshgrid(
        torch.arange(cfg.im_size[0]),
        torch.arange(cfg.im_size[1]),
        torch.arange(cfg.im_size[0]),
        torch.arange(cfg.im_size[1]),
    )
    # corrs[(c <= a) & (d <= b)] = 0
    # a, b, c, d = torch.where(corrs > cfg.threshold)
    # im = torch.zeros((64, 64, 64, 64))
    # im[a, b, c, d] = 1
    # im2vis = rearrange(im, "h1 w1 h2 w2 -> (h1 w1) (h2 w2)")
    # im = torch.zeros((64, 64))
    # im[a, b] = 1
    # im2vis = rearrange(im, "h1 w1 h2 w2 -> (h1 w1) (h2 w2)")
    # plt.imshow(im.cpu().numpy())
    # plt.show()
    # plt.close()
    # locs = torch.stack([a, b, c, d], dim=1)
    # locs = torch.stack(
    #     [
    #         a,
    #         b,
    #     ],
    #     dim=1,
    # )
    # clustering = DBSCAN(eps=4, min_samples=3).fit(locs.cpu().numpy())
    # find_best_separation(locs.cpu().numpy(), clustering.labels_)
    optimized_labels = find_optimal_labels(corrs)
    plt.imshow(optimized_labels.detach().cpu().numpy())
    plt.show()
    plt.close()


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
