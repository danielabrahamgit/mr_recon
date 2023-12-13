import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
import sigpy.mri as mr
import torch
import tyro
from einops import rearrange
from loguru import logger
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mr_sim.data_sim import data_sim, mrf_sequence, quant_phantom
from mr_sim.trj_lib import trj_lib
from optimized_mrf.sequences import FISP, BareFISP
from scipy.special import j1
from sklearn.cluster import KMeans

from mr_recon.linop import multi_subspace_linop
from mr_recon.recon import recon
from mr_recon.utils.general import create_exp_dir


@dataclass
class MRParams:
    n_coils: int = 12
    R: int = 16
    dt: float = 4e-6
    excitation_ty: str = "flat"


@dataclass
class ReconParams:
    n_clusters: int = 1
    n_fas_per_cluster: int = 1

    n_coeffs: int = 5

    n_iters: int = 20


@dataclass
class GeneratedData:
    gt: torch.Tensor
    ksp: torch.Tensor
    trj: torch.Tensor
    mps: torch.Tensor
    phis: List[torch.Tensor]
    dcts: List[torch.Tensor]
    cmprsd_dcts: List[torch.Tensor]
    tissues: torch.Tensor
    dcf: torch.Tensor
    voxel_labels: torch.Tensor
    masks: torch.Tensor
    brain_mask: torch.Tensor


@dataclass
class Config:
    exp_name: str = "debug"
    log_dir: Path = Path("logs")
    saved_data_dir: Optional[Path] = Path("./test_data")

    generate_data: bool = True
    generate_fa_map: bool = True

    run_recon: bool = True
    noise_std: float = 0.0

    device_idx = 0
    im_size = (220, 220)

    mr_params: MRParams = field(default_factory=lambda: MRParams())

    recon_params: ReconParams = field(default_factory=lambda: ReconParams())


def main():
    args = tyro.cli(Config)
    args.exp_dir = create_exp_dir(args.log_dir, args.exp_name)
    logger.add(args.exp_dir / "log-{time}.log")

    logger.info(f"Experiment directory: {args.exp_dir}")

    if args.generate_data:
        data = generate_data(args)
    else:
        assert (
            args.saved_data_dir is not None
        ), "Wanted to load saved data, but no path was given"
        logger.info(f"Loading data from: {args.saved_data_dir}")

        ksp = torch.from_numpy(np.load(args.saved_data_dir / "ksp.npy")).to(
            args.device_idx
        )
        trj = torch.from_numpy(np.load(args.saved_data_dir / "trj.npy")).to(
            args.device_idx
        )
        mps = torch.from_numpy(np.load(args.saved_data_dir / "mps.npy")).to(
            args.device_idx
        )
        phis = torch.from_numpy(np.load(args.saved_data_dir / "phi.npy")).to(
            args.device_idx
        )
        phis = [phis[i].to(torch.complex64) for i in range(len(phis))]
        dcts = torch.from_numpy(np.load(args.saved_data_dir / "dcts.npy")).to(
            args.device_idx
        )
        dcts = [dcts[i].to(torch.complex64) for i in range(len(dcts))]
        tissues = torch.from_numpy(np.load(args.saved_data_dir / "tissues.npy")).to(
            args.device_idx
        )
        dcf = torch.from_numpy(np.load(args.saved_data_dir / "dcf.npy")).to(
            args.device_idx
        )
        voxel_labels = torch.from_numpy(
            np.load(args.saved_data_dir / "voxel_labels.npy")
        ).to(args.device_idx)

        data = quant_phantom()
        t1 = torch.from_numpy(sp.resize(data["t1"][100], args.im_size)).to(
            args.device_idx
        )
        t2 = torch.from_numpy(sp.resize(data["t2"][100], args.im_size)).to(
            args.device_idx
        )
        pd = torch.from_numpy(sp.resize(data["pd"][100], args.im_size)).to(
            args.device_idx
        )
        gt = torch.stack([pd, t1, t2], axis=-1).to(args.device_idx)
        brain_mask = pd > 0.1

        masks = torch.stack(
            [voxel_labels == i for i in range(args.recon_params.n_clusters)]
        ).to(args.device_idx)

        # compress the dictionary
        cmprsd_dcts = [dcts[i] @ phis[i].T for i in range(len(dcts))]

        data = GeneratedData(
            gt=gt,
            ksp=ksp,
            trj=trj,
            mps=mps,
            phis=phis,
            dcts=dcts,
            cmprsd_dcts=cmprsd_dcts,
            tissues=tissues,
            dcf=dcf,
            voxel_labels=voxel_labels,
            masks=masks,
            brain_mask=brain_mask,
        )

    if args.run_recon:
        recon_coeffs, est_tissues = run_recon(args, data)
        post_process(est_tissues, recon_coeffs, data.gt, data.brain_mask, args.exp_dir)

    logger.info(
        f"Finished running experiments. See results and logs in: {args.exp_dir}"
    )


def generate_data(args: Config) -> GeneratedData:
    logger.info("Generating data")

    device = args.device_idx

    # Load phantom data
    data = quant_phantom()
    t1 = torch.from_numpy(sp.resize(data["t1"][100], args.im_size)).to(device)
    t2 = torch.from_numpy(sp.resize(data["t2"][100], args.im_size)).to(device)
    pd = torch.from_numpy(sp.resize(data["pd"][100], args.im_size)).to(device)
    gt = torch.stack([pd, t1, t2], axis=-1).to(device)
    brain_mask = pd > 0.1
    logger.info(f"Image size: {pd.shape}")

    # Load default sequence
    seq_data = mrf_sequence()
    trs = torch.from_numpy(seq_data["TR_init"][0].astype(np.float32)).to(device)
    fas = torch.deg2rad(
        torch.from_numpy(seq_data["FA_init"][0].astype(np.float32)).to(device)
    )
    logger.info(f"Sequence length: {len(fas)}")

    if args.generate_fa_map:
        logger.info("Generating FA map")
        modes = get_modes(args.im_size, args.mr_params.excitation_ty).to(device)
        logger.info(f"Modes shape: {modes.shape}")

        # get the flip angle maps and cluster them
        fa_map, clusters, voxel_labels = generate_fa_map(
            fas, modes, args.recon_params.n_clusters
        )

        # calculate the subspace per cluster
        phis, dcts, tissues = get_subspaces(
            voxel_labels,
            fa_map,
            clusters,
            args.recon_params.n_fas_per_cluster,
            sing_val_thresh=0.95,
            n_coeffs=args.recon_params.n_coeffs,
        )

        # compress the dictionary
        cmprsd_dcts = [dcts[i] @ phis[i].T for i in range(len(dcts))]

        masks = torch.stack(
            [voxel_labels == i for i in range(args.recon_params.n_clusters)]
        ).to(fa_map)

    ds = data_sim(
        im_size=args.im_size, rfs=fas, trs=trs, te=1.75, device_idx=args.device_idx
    )
    mps = mr.birdcage_maps((args.mr_params.n_coils, *args.im_size), r=1.25).astype(
        np.complex64
    )
    mps = torch.from_numpy(mps).to(device)

    trj_obj = trj_lib(args.im_size)
    trj = trj_obj.gen_MRF_trj(ntr=len(fas), n_shots=16, R=args.mr_params.R)
    trj = torch.from_numpy(trj).to(device)

    ksp, _, _, _ = ds.sim_ksp(
        t1_map=t1,
        t2_map=t2,
        pd_map=pd,
        mps=mps,
        trj=trj,
        coil_batch_size=args.mr_params.n_coils,
        seg_batch_size=1,
        fa_map=fa_map,
    )

    ksp = ksp.to(device)
    dcf = ds.est_dcf(trj).to(device)

    # TODO: Save data
    # np.save(args.exp_dir / "trj.npy", trj.numpy())
    # np.save(args.exp_dir / "dcf.npy", dcf.numpy())
    # np.save(args.exp_dir / "ksp.npy", ksp.numpy())
    # np.save(args.exp_dir / "mps.npy", mps.numpy())
    # np.save(args.exp_dir / "imgs.npy", imgs.detach().cpu().numpy())
    # ds.seq.save(args.exp_dir / "seq")

    data = GeneratedData(
        gt=gt,
        ksp=ksp,
        trj=trj,
        mps=mps,
        phis=phis,
        dcts=dcts,
        cmprsd_dcts=cmprsd_dcts,
        tissues=tissues,
        dcf=dcf,
        voxel_labels=voxel_labels,
        masks=masks,
        brain_mask=brain_mask,
    )

    return data


def run_recon(
    args: Config,
    data: GeneratedData,
):
    logger.info(f"Running reconstruction, adding noise with std: {args.noise_std}")
    noise = torch.randn_like(data.ksp) * args.noise_std
    noisy_ksp = data.ksp + noise

    # TODO: there are other parameters that we can tune
    A = multi_subspace_linop(
        im_size=args.im_size,
        trj=data.trj,
        mps=data.mps,
        phis=data.phis,
        masks=data.masks,
        dcf=data.dcf,
        # use_toeplitz=True,
        # grog_grid_oversamp=None,
    )

    rcn = recon(0)
    img_mr_recon = rcn.run_recon(
        A_linop=A,
        ksp=noisy_ksp,
        max_eigen=1.0,
        max_iter=args.recon_params.n_iters,
        lamda_l2=0,
    )
    img_mr_recon = torch.from_numpy(img_mr_recon).to(noisy_ksp)

    est_tissues_recon = dict_matching(
        rearrange(img_mr_recon, "a b c -> b c a"),
        data.cmprsd_dcts,
        data.tissues,
        data.masks,
        brain_mask=data.brain_mask,
    )

    return img_mr_recon, est_tissues_recon


def post_process(est_tissues, subspace_coeffs, gt_tissues, brain_mask, save_path):
    plot_recons([est_tissues], gt_tissues, ["MRFRF"], brain_mask, save_path)
    plot_coeffs([subspace_coeffs], ["MRFRF"], save_path)


def load_clusters():
    pass


def get_modes(im_size, ty="flat"):
    if ty == "ring":
        return get_ring_modes(im_size)
    elif ty == "flat":
        return torch.ones(1, *im_size).type(torch.float32)
    else:
        raise ValueError(f"Unknown mode type: {ty}")


def jinc(x):
    # Ensure we don't divide by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        result = j1(np.pi * x) / (np.pi * x)
        result[x == 0] = 0.5  # handle the singularity
    return result


def get_ring_modes(im_size, scale: float = 1.3):
    rows = torch.linspace(-im_size[0] // 2, im_size[0] // 2, im_size[0])
    cols = torch.linspace(-im_size[1] // 2, im_size[1] // 2, im_size[1])

    # Create a grid of distances from the center
    x, y = torch.meshgrid(rows, cols, indexing="ij")
    r = torch.sqrt(x**2 + y**2)

    modes = []
    rf = torch.abs(jinc(r / (0.5 * max(im_size))))
    modes.append(scale * (rf / rf.max()))

    rf = torch.abs(jinc((r - 47) / (0.5 * max(im_size))))
    modes.append(scale * (rf / rf.max()))

    rf = torch.abs(jinc((r - 95) / (0.5 * max(im_size))))
    modes.append(scale * (rf / rf.max()))

    return torch.stack(modes)


def get_subspaces(
    labeled_voxels, fa_map, clusters, n_fas=1, sing_val_thresh=0.95, n_coeffs=10
):
    logger.info(
        f"Finding subspaces for each cluster. Number of coefficients per cluster: {n_coeffs}, "
        f"number of FA trains per cluster: {n_fas}"
    )

    # Estimate subspace from range of t1/t2 from xiaozhi's paper
    t1_vals = np.arange(20, 3000, 20)
    t1_vals = np.append(t1_vals, np.arange(3200, 5000, 200))
    t2_vals = np.arange(10, 200, 2)
    t2_vals = np.append(t2_vals, np.arange(220, 1000, 20))
    t2_vals = np.append(t2_vals, np.arange(1050, 2000, 50))
    t2_vals = np.append(t2_vals, np.arange(2100, 4000, 100))
    t1_vals, t2_vals = np.meshgrid(t1_vals, t2_vals, indexing="ij")
    t1_vals, t2_vals = t1_vals.flatten(), t2_vals.flatten()
    pd_vals = np.ones_like(t1_vals)
    nom_tissues = (
        torch.tensor(np.array([pd_vals, t1_vals, t2_vals]), dtype=torch.float32)
        .T.unsqueeze(-2)
        .to(fa_map)
    )

    seq = BareFISP().to("cuda")

    phis = []
    dcts = []

    # construct a subspaces for each cluster
    for i, cluster in enumerate(clusters):
        # get the voxels in the cluster
        voxels = labeled_voxels == i
        clustered_fa = fa_map[voxels]
        # per fa train we take from the cluster, we build a full dictionary
        tissues = nom_tissues.unsqueeze(1).expand(-1, n_fas, -1, -1)

        if n_fas > 1:
            # TODO: make that pick smarter
            idx = np.random.choice(len(clustered_fa), n_fas - 1, replace=False)
            fas = clustered_fa[idx]
            # those are the fa trains we will use for computing the dictionary and subspace
            fas = torch.concatenate((fas, cluster[None, ...]), axis=0)
            # assign each tissues with a fa (each tissue will have n_fas different fas)
            fas = fas[None, ...] * torch.ones((tissues.shape[0], 1, 1)).to(fas)
        else:
            fas = cluster[None, ...]
            fas = fas[None, ...] * torch.ones((tissues.shape[0], 1, 1)).to(fas)

        # This simulates the different tissues with the different fas and generates a big dictionary
        dct_torch = seq(tissues, fa_map=fas, batch_size=None)
        dct_torch = dct_torch.squeeze(-2)
        dct_torch = rearrange(dct_torch, "n c j -> (n c) j")
        dcts.append(dct_torch.type(torch.complex64))

        # normalize the dictionary before SVD
        norms = torch.linalg.norm(dct_torch, ord=2, axis=-1, keepdims=True)
        d = dct_torch / norms
        _, s, vh = torch.linalg.svd(d, full_matrices=False)
        cmsm = torch.cumsum(s, dim=0)
        n_th = int(torch.argwhere(cmsm > sing_val_thresh * cmsm[-1]).flatten()[0])
        phis.append(vh[:n_coeffs, :].conj().type(torch.complex64))
        logger.info(
            f"Dictionary {i}: To get {sing_val_thresh} of the signal energy we need to use: {n_th} coeffs. "
            f"We compressed to {n_coeffs} coefficients"
        )

    # keep the same number of subspaces for all clusters
    # phis = [p[:n_coeffs, ...] for p in phis]

    return phis, dcts, rearrange(tissues, "n c ... -> (n c) ...")


def generate_fa_map(fa, modes, n_clusters):
    """
    Generate FA map, clusters, and voxel labels.

    Args:
        fa (list or numpy.ndarray): List or array of FA values.
        modes (torch.Tensor): Tensor of shape (n_modes, h, w) representing the modes.
        n_clusters (int): Number of clusters for FA clustering.

    Returns:
        tuple: A tuple containing the following:
            - fa_map (torch.Tensor): Tensor of shape (h, w, len(fa)) representing the FA map.
            - clusters (torch.Tensor): Tensor of shape (n_clusters, len(fa)) representing the cluster centers.
            - voxels_labels (torch.Tensor): Tensor of shape (h, w) representing the voxel labels.
    """

    assert modes.min() > 0

    n_modes, h, w = modes.shape

    logger.info(f"Creating alternating excitation pattern with {n_modes} modes")

    # explicit FA map -- FA per voxel
    fa_map = torch.zeros(modes.shape[1:] + (len(fa),))
    modes = torch.tile(modes, [int(np.ceil(len(fa) / n_modes)), 1, 1])
    modes = rearrange(modes, "n h w -> h w n")
    fa_map = modes[..., : len(fa)] * fa

    logger.info(f"Clustering the different FAs into {n_clusters} clusters")
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(rearrange(fa_map, "h w n -> (h w) n").cpu().numpy())

    voxels_labels = torch.from_numpy(
        rearrange(kmeans.labels_, "(h w) -> h w", h=h, w=w)
    ).to(fa)
    clusters = torch.from_numpy(kmeans.cluster_centers_).to(fa)

    return fa_map, clusters, voxels_labels


def _dict_matching(signal, dct, tissues, device="cpu"):
    signal = signal.to(device)
    dct = dct.to(device)
    tissues = tissues.to(device)

    norm_vals = torch.linalg.norm(dct, ord=2, axis=-1, keepdims=True)
    norm_dict = dct / norm_vals

    corr = torch.abs(signal @ norm_dict.squeeze().T)

    est_idx = torch.argmax(corr, axis=-1)

    est_tissues = tissues[est_idx].squeeze()

    est_amp = corr[torch.arange(corr.shape[0]), est_idx] / norm_vals[est_idx].squeeze()
    est_tissues[:, 0] = est_amp

    return est_tissues


def dict_matching(signal, dcts, tissues, masks=None, brain_mask=None, avg_pd=None):
    est_tissues = torch.zeros(signal.shape[:2] + (3,)).to(signal.device)
    if masks == None:
        mask = torch.ones(1, *signal.shape[:2]).to(signal)

    for i, mask in enumerate(masks):
        mask_bool = mask == 1
        masked_sig = signal[mask_bool]
        cluster_dict = dcts[i]
        est_tissues[mask_bool] = _dict_matching(
            masked_sig, cluster_dict, tissues[:, 0]
        ).to(est_tissues)

    if brain_mask is not None:
        # Normalize PD
        est_tissues[brain_mask == False, ...] = 0
        if avg_pd is not None:
            scale = avg_pd / np.mean(est_tissues[brain_mask, 0])
            est_tissues[brain_mask, 0] *= scale

    return est_tissues


def plot_recons(
    recon_list, ground_truth, titles, mask, save_path_prefix, modalites_append=None
):
    if len(recon_list) != len(titles):
        raise ValueError(
            "The number of titles must match the number of images in the list."
        )

    # Move everything to CPU and numpy for plotting
    recon_list = [img.detach().cpu().numpy() for img in recon_list]
    ground_truth = ground_truth.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()

    modalities = ["PD", "T1", "T2"]
    if modalites_append is not None:
        for i in range(len(modalities)):
            modalities[i] += modalites_append

    for index, modality in enumerate(modalities):
        gs = gridspec.GridSpec(
            2, len(recon_list) + 1, width_ratios=[5] * len(recon_list) + [0.5]
        )
        fig = plt.figure(figsize=(5 * len(recon_list), 10))
        fig.suptitle(modality, fontsize=20, y=0.99)

        # For consistent vmin, vmax across original images
        vmin_orig = min([np.min(img[:, :, index]) for img in recon_list])
        vmax_orig = max([np.max(img[:, :, index]) for img in recon_list])

        for col, img in enumerate(recon_list):
            original_image = img[:, :, index]
            gt_image = ground_truth[:, :, index]

            normalized_loss = np.abs(original_image - gt_image) / (gt_image + 1e-8)
            avg_err = np.mean(normalized_loss[mask])

            ax_orig = fig.add_subplot(gs[0, col])
            ax_loss = fig.add_subplot(gs[1, col])

            im_orig = ax_orig.imshow(
                original_image, cmap="hot", vmin=vmin_orig, vmax=vmax_orig
            )
            ax_orig.axis("off")
            ax_orig.set_title(titles[col], fontsize=20)

            im_loss = ax_loss.imshow(normalized_loss, cmap="hot", vmin=0, vmax=1)
            ax_loss.axis("off")
            ax_loss.text(
                0.73 * original_image.shape[1],
                0.95 * original_image.shape[0],
                f"Avg.Err: {avg_err * 100:.1f}%",
                color="white",
                ha="center",
                va="center",
                fontsize=19,
            )

            if col == len(recon_list) - 1:
                ax_cbar_orig = fig.add_subplot(gs[0, -1])
                ax_cbar_loss = fig.add_subplot(gs[1, -1])

                cbar_orig = fig.colorbar(im_orig, cax=ax_cbar_orig)
                cbar_orig.ax.set_title("ms", pad=10, fontsize=18)
                cbar_orig.ax.tick_params(labelsize=16)

                cbar_loss = fig.colorbar(im_loss, cax=ax_cbar_loss)
                cbar_loss.set_ticks([0, 0.25, 0.5, 0.75, 1])
                cbar_loss.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
                cbar_loss.ax.tick_params(labelsize=16)

        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.tight_layout()
        plt.savefig(
            save_path_prefix / f"{modality}_Recon.png",
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close(fig)

        # Ground truth plotting
        fig_gt = plt.figure(figsize=(5, 5))
        plt.imshow(
            ground_truth[:, :, index], cmap="hot", vmin=vmin_orig, vmax=vmax_orig
        )
        plt.axis("off")
        plt.title(f"Ground Truth {modality}", fontsize=22)
        plt.savefig(
            save_path_prefix / f"{modality}_GT.png", bbox_inches="tight", pad_inches=0.1
        )
        plt.close(fig_gt)


def plot_coeffs(image_list, titles, save_path):
    C = image_list[0].shape[0]
    num_columns = len(image_list)

    # move everything to CPU and numpy for plotting
    image_list = [img.abs().detach().cpu().numpy() for img in image_list]

    fig, axarr = plt.subplots(C, num_columns, figsize=(5 * num_columns, 5 * C))
    if num_columns == 1:
        axarr = axarr[:, None]

    for c in range(C):
        # Determine the vmin and vmax for the current channel across all images in the list
        vmin = min([img[c].min() for img in image_list])
        vmax = max([img[c].max() for img in image_list])

        for col, img in enumerate(image_list):
            # Display the image
            im = axarr[c, col].imshow(img[c], cmap="gray", vmin=vmin, vmax=vmax)
            axarr[c, col].axis("off")

            if c == 0:
                axarr[c, col].set_title(titles[col])

            # If it's the last column, add a colorbar to the right
            if col == num_columns - 1:
                divider = make_axes_locatable(axarr[c, col])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
    plt.savefig(save_path / f"recon_coeffs", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


if __name__ == "__main__":
    main()
