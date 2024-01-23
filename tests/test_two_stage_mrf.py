import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import pickle
import pprint
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import sigpy as sp
import sigpy.mri as mr
import torch
import tyro
from einops import rearrange
from fast_pytorch_kmeans import KMeans
from loguru import logger
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mr_sim.data_sim import data_sim, mrf_sequence, quant_phantom
from mr_sim.trj_lib import trj_lib
from optimized_mrf.sequences import FISP, BareFISP
from optimized_mrf.sequences.defaults import fisp_fa, fisp_tr
from optimized_mrf.tissues import get_typical_tissues
from scipy.special import j1
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchkbnufft import KbNufft
from tqdm import tqdm

from mr_recon.linop import multi_subspace_linop
from mr_recon.nufft import gridded_nufft, sigpy_nufft, torchkb_nufft
from mr_recon.recon import recon
from mr_recon.utils.general import create_exp_dir, seed_everything
from mr_recon.utils.indexing import ravel


@dataclass
class MRParams:
    n_coils: int = 1
    n_interleaves: int = 100
    R: int = 100
    dt: float = 4e-6
    excitation_ty: str = "random"
    smooth_factor: Optional[int] = 30
    trj_type: str = "radial"


@dataclass
class ReconParams:
    n_clusters: int = 1
    n_fas_per_cluster: int = 1

    n_coeffs: int = 5
    n_iters: int = 40
    use_toeplitz: bool = False
    grog_grid_oversamp: Optional[float] = 1.0
    coil_batch_size: int = 1
    sub_batch_size: int = 1
    seg_batch_size: int = 1

    lambda_l2: float = 0.0


@dataclass
class ReconData:
    tissues: torch.Tensor
    coeffs: torch.Tensor


@dataclass
class GeneratedData:
    gt: torch.Tensor
    ksp: torch.Tensor
    trj: torch.Tensor
    mps: torch.Tensor
    phis: List[torch.Tensor]
    cmprsd_dcts: List[torch.Tensor]
    tissues: torch.Tensor
    dcf: torch.Tensor
    voxel_labels: torch.Tensor
    masks: torch.Tensor
    brain_mask: torch.Tensor
    fa_map: torch.Tensor
    avg_pd: Optional[float] = None

    # Store this for debugging
    gt_coeffs: Optional[torch.Tensor] = None
    gt_ims: Optional[torch.Tensor] = None


@dataclass
class DebugConfig:
    guassian_lpf_sigma: float = 1
    num_avgs: int = 1
    # This overrides the sampling rate and runs a fully sampled experiment
    fully_sampled: bool = False


@dataclass
class Config:
    exp_name: str = "debug"
    log_dir: Path = Path("logs")
    saved_data_dir: Optional[Path] = Path("logs/debug/2023-12-15_14-22")

    generate_data: bool = True
    save_generated_data: bool = True

    run_classic_recon: bool = True
    save_classic_recon: bool = True
    run_stochastic_recon: bool = True
    init_stochastic_w_classic: bool = True

    noise_std: float = 0  # 3e-3

    device_idx = 0
    im_size = (220, 220)

    mr_params: MRParams = field(default_factory=lambda: MRParams())

    recon_params: ReconParams = field(default_factory=lambda: ReconParams())

    debug: bool = True
    debug_cfg: DebugConfig = field(default_factory=lambda: DebugConfig())


def main():
    seed_everything(42)

    args = tyro.cli(Config)
    exp_name = (
        f"{args.exp_name}-excitation_type={args.mr_params.excitation_ty}-sampling={args.mr_params.trj_type}-"
        f"n_interleaves={args.mr_params.n_interleaves}-R={args.mr_params.R}-n_coils={args.mr_params.n_coils}"
    )
    args.exp_dir = create_exp_dir(args.log_dir, exp_name)
    logger.add(args.exp_dir / "log-{time}.log")

    if args.debug:
        import matplotlib

        matplotlib.use("webagg")

    cfg_str = pprint.pformat(args)

    logger.info(f"Experiment directory: {args.exp_dir}")
    logger.info(f"Config:\n{cfg_str}")

    if args.generate_data:
        data = generate_data(args)
    else:
        assert (
            args.saved_data_dir is not None
        ), "Wanted to load saved data, but no path was given"

        logger.info(f"Loading data from: {args.saved_data_dir}")
        with open(args.saved_data_dir / "data", "rb") as file:
            data = pickle.load(file)

    recon_data = None
    if args.run_classic_recon:
        recon_coeffs, est_tissues = run_recon(args, data)
        post_process(est_tissues, recon_coeffs, data.gt, data.brain_mask, args.exp_dir)
        recon_data = ReconData(est_tissues, recon_coeffs)
        if args.save_classic_recon:
            logger.info(f"Saving classic recon to: {args.exp_dir / 'classic_recon'}")
            with open(args.exp_dir / "classic_recon", "wb") as file:
                pickle.dump(recon_data, file)

    if args.run_stochastic_recon:
        if args.init_stochastic_w_classic and recon_data is None:
            assert (
                args.saved_data_dir is not None
            ), "Wanted to load saved data, but no path was given"

            logger.info(f"Loading data from: {args.saved_data_dir}")
            with open(args.saved_data_dir / "classic_recon", "rb") as file:
                recon_data = pickle.load(file)

        stochastic_recon(data, args, recon_data)

    logger.info(
        f"Finished running experiments. See results and logs in: {args.exp_dir}"
    )


def generate_data(args: Config) -> GeneratedData:
    logger.info("Generating data")

    device = args.device_idx

    # Load phantom data
    data = quant_phantom()
    t1 = sp.resize(data["t1"][100], args.im_size)
    t2 = sp.resize(data["t2"][100], args.im_size)
    pd = sp.resize(data["pd"][100], args.im_size)

    if args.debug:
        sigma = args.debug_cfg.guassian_lpf_sigma
        t1 = scipy.ndimage.gaussian_filter(t1, sigma=sigma)
        t2 = scipy.ndimage.gaussian_filter(t2, sigma=sigma)
        pd = scipy.ndimage.gaussian_filter(pd, sigma=sigma)

    t1 = torch.from_numpy(t1).to(device)
    t2 = torch.from_numpy(t2).to(device)
    pd = torch.from_numpy(pd).to(device)
    brain_mask = pd > 0.1
    pd[~brain_mask] = 0

    gt = torch.stack([pd, t1, t2], axis=-1).to(device)
    avg_pd = torch.mean(pd[brain_mask])
    logger.info(f"Image size: {pd.shape}")

    trs = fisp_tr.clone().to(device)
    fas = fisp_fa.clone().to(device)
    logger.info(f"Sequence length: {len(fas)}")

    logger.info("Generating FA map")
    modes = get_modes(args.im_size, args.mr_params.excitation_ty).to(device)
    plot_excitation_modes(modes, save_path=args.exp_dir / "excitation_modes.png")
    logger.info(f"Modes shape: {modes.shape}")

    # get the flip angle maps and cluster them
    fa_map, clusters, voxel_labels = generate_fa_map(
        fas, modes, args.recon_params.n_clusters, args.mr_params.smooth_factor
    )
    fa_examples = rearrange(torch.rad2deg(fa_map[::90, ::90, :]), "h w n -> (h w) n")[
        :3
    ]
    plot_signals(fa_examples, args.exp_dir, "fa_exmp")

    plot_clusters(voxel_labels, args.exp_dir)

    # calculate the subspace per cluster
    phis, cmprsd_dcts, tissues = get_subspaces(
        voxel_labels,
        fa_map,
        clusters,
        args.recon_params.n_fas_per_cluster,
        sing_val_thresh=0.95,
        n_coeffs=args.recon_params.n_coeffs,
        args=args,
    )

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

    if args.debug and args.debug_cfg.fully_sampled:
        K_x, K_y = np.meshgrid(np.arange(args.im_size[0]), np.arange(args.im_size[1]))
        K_x, K_y = K_x - args.im_size[0] // 2, K_y - args.im_size[1] // 2
        valid = np.sqrt(K_x**2 + K_y**2) < 110
        K_x = K_x[valid].flatten()
        K_y = K_y[valid].flatten()
        trj = np.zeros((len(K_x), 1, len(fas), 2), dtype=np.float32)
        trj[..., 0] = np.broadcast_to(K_x[:, None, None], trj.shape[:-1])
        trj[..., 1] = np.broadcast_to(K_y[:, None, None], trj.shape[:-1])
    else:
        trj_obj = trj_lib(args.im_size)
        if args.mr_params.trj_type == "spiral":
            trj = trj_obj.gen_MRF_trj(
                ntr=len(fas), n_shots=args.mr_params.n_interleaves, R=args.mr_params.R
            )
        elif args.mr_params.trj_type == "radial":
            trj = trj_obj.gen_radial_MRF_trj(
                ntr=len(fas), n_shots=args.mr_params.n_interleaves, R=args.mr_params.R
            )
        else:
            raise ValueError(f"Unknown trajectory type: {args.mr_params.trj_type}")

        if args.recon_params.grog_grid_oversamp is not None:
            trj = (
                np.round(trj * args.recon_params.grog_grid_oversamp)
                / args.recon_params.grog_grid_oversamp
            )

    trj = torch.from_numpy(trj).to(device)

    # Plots all groups for the first TR
    plot_trj(trj, args.exp_dir)

    ksp, _, _, imgs = ds.sim_ksp(
        t1_map=t1,
        t2_map=t2,
        pd_map=pd,
        mps=mps,
        trj=trj,
        coil_batch_size=args.mr_params.n_coils,
        seg_batch_size=1,
        fa_map=fa_map,
        grog_grid_oversamp=args.recon_params.grog_grid_oversamp,
    )
    gt_coeffs = rearrange(imgs.type(torch.complex64) @ phis[0].T, "h w c -> c h w")

    ksp = ksp.to(device)
    # FIXME: there's a bug in the regular DCF calculation when using more than one group. The DCF is suppose to be
    # calculated on the combined TR image, and rn it is per group per tr so when summing in the subspace recon it's not
    # correct. For now, we use the cartesian DCF
    # dcf = ds.est_dcf(trj).to(device)
    dcf = cartesian_dcf(trj, args.im_size).to(device)

    data = GeneratedData(
        gt=gt,
        ksp=ksp,
        trj=trj,
        mps=mps,
        phis=phis,
        cmprsd_dcts=cmprsd_dcts,
        tissues=tissues,
        dcf=dcf,
        voxel_labels=voxel_labels,
        masks=masks,
        brain_mask=brain_mask,
        avg_pd=avg_pd,
        fa_map=fa_map,
        gt_coeffs=gt_coeffs,
        gt_ims=imgs,
    )

    if args.save_generated_data:
        logger.info(f'Saving generated data to: {args.exp_dir / "data"}')
        with open(args.exp_dir / "data", "wb") as file:
            pickle.dump(data, file)

    return data


def run_recon(
    args: Config,
    data: GeneratedData,
):
    logger.info(f"Running reconstruction, adding noise with std: {args.noise_std}")

    A = multi_subspace_linop(
        im_size=args.im_size,
        trj=data.trj,
        mps=data.mps,
        phis=data.phis,
        masks=data.masks,
        dcf=data.dcf,
        use_toeplitz=args.recon_params.use_toeplitz,
        grog_grid_oversamp=args.recon_params.grog_grid_oversamp,
        coil_batch_size=args.recon_params.coil_batch_size,
        sub_batch_size=args.recon_params.sub_batch_size,
        seg_batch_size=args.recon_params.seg_batch_size,
    )

    rcn = recon(args.device_idx)

    # Running to recon with multiple noise instances
    estimations = []
    num_avgs = args.debug_cfg.num_avgs if args.debug else 1
    for i in range(num_avgs):
        logger.info(f"Running recon {i + 1}/{num_avgs}")

        noise = torch.randn_like(data.ksp) * (
            args.noise_std / np.sqrt(args.mr_params.R)
        )
        noisy_ksp = data.ksp + noise

        img_mr_recon = rcn.run_recon(
            A_linop=A,
            ksp=noisy_ksp,
            max_eigen=1.0,
            max_iter=args.recon_params.n_iters,
            lamda_l2=args.recon_params.lambda_l2,
        )
        img_mr_recon = torch.from_numpy(img_mr_recon).to(noisy_ksp)

        est_tissues_recon = dict_matching(
            rearrange(img_mr_recon, "a b c -> b c a"),
            data.cmprsd_dcts,
            data.tissues,
            data.masks,
            brain_mask=data.brain_mask,
            avg_pd=data.avg_pd,
        )
        estimations.append(est_tissues_recon)

    estimations = torch.stack(estimations, dim=0)
    bias = torch.mean(estimations, dim=0) - data.gt
    var = torch.var(estimations, dim=0)
    plot_bias_variance(bias, var, args.exp_dir)

    return img_mr_recon, est_tissues_recon


def cartesian_dcf(trj, im_size):
    nro, npe, ntr, d = trj.shape
    rescaled_trj = trj.clone()
    rescaled_trj[..., 0] = torch.clamp(
        rescaled_trj[..., 0] + im_size[0] // 2, 0, im_size[0] - 1
    )
    rescaled_trj[..., 1] = torch.clamp(
        rescaled_trj[..., 1] + im_size[1] // 2, 0, im_size[1] - 1
    )
    idx = ravel(rescaled_trj, im_size, dim=-1).to(rescaled_trj.device).type(torch.int64)
    idx = rearrange(idx, "nro npe ntr -> npe ntr nro")
    dcf = (
        torch.zeros((npe, ntr, *im_size))
        .to(rescaled_trj.device)
        .flatten(start_dim=-2, end_dim=-1)
    )
    val = torch.ones_like(idx).to(rescaled_trj.device).type(torch.float32)
    dcf = dcf.scatter_add_(-1, idx, val)
    # Sum over groups
    dcf = torch.sum(dcf, dim=0, keepdim=True)
    dcf = dcf.expand((npe,) + dcf.shape[1:])
    dcf = torch.gather(dcf, -1, idx)
    dcf = rearrange(dcf, "npe ntr ... -> ... npe ntr")

    return 1 / dcf


def post_process(est_tissues, subspace_coeffs, gt_tissues, brain_mask, save_path):
    plot_recons([est_tissues], gt_tissues, ["MRFRF"], brain_mask, save_path)
    plot_coeffs([subspace_coeffs], ["MRFRF"], save_path)


def get_modes(im_size, ty="flat", n_modes=3):
    if ty == "ring":
        return get_ring_modes(im_size)
    elif ty == "shim-ring":
        assert im_size == (220, 220)
        return get_shim_modes()
    elif ty == "flat":
        return torch.ones(1, *im_size).type(torch.float32)
    elif ty == "random":
        return get_rand_modes(im_size, n_modes=n_modes)
    else:
        raise ValueError(f"Unknown mode type: {ty}")


def jinc(x):
    # Ensure we don't divide by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        result = j1(np.pi * x) / (np.pi * x)
        result[x == 0] = 0.5  # handle the singularity
    return result


def get_rand_modes(im_size, n_modes=3):
    modes = 1 + 0.1 * torch.randn(n_modes, *im_size)

    return modes


def get_shim_modes():
    mode_1 = np.load("./download/shim_map/mode_0.npy")
    mode_2 = np.load("./download/shim_map/mode_1.npy")
    mode_3 = np.load("./download/shim_map/mode_2.npy")
    modes = np.concatenate(
        (mode_1[None, ...], mode_2[None, ...], mode_3[None, ...]), axis=0
    )

    return torch.from_numpy(modes).type(torch.float32)


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
    labeled_voxels,
    fa_map,
    clusters,
    n_fas=1,
    sing_val_thresh=0.95,
    n_coeffs=10,
    args=None,
):
    logger.info(
        f"Finding subspaces for each cluster. Number of coefficients per cluster: {n_coeffs}, "
        f"number of FA trains per cluster: {n_fas}"
    )

    # Estimate subspace from range of t1/t2 from xiaozhi's paper
    nom_tissues = get_typical_tissues().to(fa_map.device)

    seq = BareFISP().to("cuda")

    phis = []
    cmprsd_dcts = []

    # per fa train we take from the cluster, we build a full dictionary
    tissues = nom_tissues.unsqueeze(1).expand(-1, n_fas, -1, -1)

    if args is not None:
        clusters_dir = args.exp_dir / f"clusters"
        os.makedirs(clusters_dir, exist_ok=True)

    # construct a subspaces for each cluster
    for i, cluster in enumerate(clusters):
        # get the voxels in the cluster
        voxels = labeled_voxels == i
        clustered_fa = fa_map[voxels]

        if n_fas > 1:
            fas = pick_fas(clustered_fa, cluster, n_fas)
            fas = fas[None, ...] * torch.ones((tissues.shape[0], 1, 1)).to(fas)
        else:
            fas = cluster[None, ...]
            fas = fas[None, ...] * torch.ones((tissues.shape[0], 1, 1)).to(fas)

        # This simulates the different tissues with the different fas and generates a big dictionary
        dct_torch = seq(tissues, fa_map=fas, batch_size=None)
        dct_torch = dct_torch.squeeze(-2)

        if args is not None:
            cluster_dir = clusters_dir / f"cluster_{i}"
            os.makedirs(cluster_dir, exist_ok=True)
            wm_sigs = dct_torch[3720, :, ...]
            gm_sigs = dct_torch[7234, :, ...]
            plot_signals(wm_sigs, cluster_dir, f"cluster_{i}-wm")
            plot_signals(gm_sigs, cluster_dir, f"cluster_{i}-gm")
            plot_region(voxels, fas[0], cluster_dir, f"cluster_{i}")

        dct_torch = rearrange(dct_torch, "n c j -> (n c) j")

        # normalize the dictionary before SVD
        norms = torch.linalg.norm(dct_torch, ord=2, axis=-1, keepdims=True)
        d = dct_torch / norms
        _, s, vh = torch.linalg.svd(d, full_matrices=False)
        cmsm = torch.cumsum(s, dim=0)
        n_th = int(torch.argwhere(cmsm > sing_val_thresh * cmsm[-1]).flatten()[0])
        phis.append(vh[:n_coeffs, :].conj().type(torch.complex64))
        cmprsd_dcts.append(dct_torch.type(torch.complex64) @ phis[-1].T)
        logger.info(
            f"Dictionary {i}: To get {sing_val_thresh} of the signal energy we need to use: {n_th} coeffs. "
            f"We compressed to {n_coeffs} coefficients"
        )

    return phis, cmprsd_dcts, rearrange(tissues, "n c ... -> (n c) ...")


def pick_fas(fas, cluster, n_fas):
    cluster_fa = torch.mean(fas, dim=0)
    # make sure fas correspond to the cluster
    torch.testing.assert_close(cluster_fa, cluster)

    if fas.max() == 0:
        picked_fas = torch.zeros(n_fas, fas.shape[-1], device=fas.device)
    else:
        kmeans = KMeans(n_clusters=n_fas, init_method="kmeans++")
        _ = kmeans.fit_predict(fas)
        picked_fas = kmeans.centroids.to(fas)

    return picked_fas


def generate_fa_map(fa, modes, n_clusters, smooth_factor: Optional[int] = None):
    """
    Generate FA map, clusters, and voxel labels.

    Args:
        fa (list or numpy.ndarray): List or array of FA values.
        modes (torch.Tensor): Tensor of shape (n_modes, h, w) representing the modes.
        n_clusters (int): Number of clusters for FA clustering.
        smooth_factor (int, optional): Interpolate modes for this number of TRs for smoother signal evolution.

    Returns:
        tuple: A tuple containing the following:
            - fa_map (torch.Tensor): Tensor of shape (h, w, len(fa)) representing the FA map.
            - clusters (torch.Tensor): Tensor of shape (n_clusters, len(fa)) representing the cluster centers.
            - voxels_labels (torch.Tensor): Tensor of shape (h, w) representing the voxel labels.
    """

    assert modes.min() >= 0
    # assert len(modes.shape) == 4, "Modes must be a 4D tensor (n_modes, c, h, w)"

    n_modes, h, w = modes.shape

    logger.info(f"Creating alternating excitation pattern with {n_modes} modes")

    if smooth_factor is not None:
        smoothed_modes = torch.zeros(n_modes * smooth_factor, h, w).to(modes)
        interpolation_coeff = torch.linspace(1, 0, smooth_factor)[:, None, None].to(
            modes
        )
        for i in range(n_modes):
            smoothed_modes[i * smooth_factor : (i + 1) * smooth_factor, ...] = (
                interpolation_coeff * modes[i, ...]
                + (1 - interpolation_coeff) * modes[(i + 1) % n_modes, ...]
            )

            torch.testing.assert_close(modes[i], smoothed_modes[i * smooth_factor])

    else:
        smoothed_modes = modes

    # explicit FA map -- FA per voxel
    smoothed_n_modes, _, _ = smoothed_modes.shape
    fa_map = torch.zeros(smoothed_modes.shape[1:] + (len(fa),))
    modes = torch.tile(smoothed_modes, [int(np.ceil(len(fa) / smoothed_n_modes)), 1, 1])
    modes = rearrange(modes, "n h w -> h w n")
    fa_map = torch.clip(modes[..., : len(fa)] * fa, min=0, max=torch.pi / 2)

    logger.info(f"Clustering the different FAs into {n_clusters} clusters")
    # Pytorch version
    kmeans = KMeans(n_clusters=n_clusters, init_method="kmeans++")
    voxels_labels = kmeans.fit_predict(rearrange(fa_map, "h w n -> (h w) n"))
    voxels_labels = rearrange(voxels_labels, "(h w) -> h w", h=h, w=w).to(fa)
    clusters = kmeans.centroids.to(fa)

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
        # Normalize PD so that the same average PD is maintained, otherwise, just scale so that the max is 1
        est_tissues[brain_mask == False, ...] = 0
        if avg_pd is not None:
            scale = avg_pd / torch.mean(est_tissues[brain_mask][..., 0])
            est_tissues[..., 0] = torch.where(
                brain_mask, est_tissues[..., 0] * scale, est_tissues[..., 0]
            )
        else:
            scale = est_tissues[brain_mask][..., 0].max()
            est_tissues[..., 0] = torch.where(
                brain_mask, est_tissues[..., 0] / scale, est_tissues[..., 0]
            )

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

            err = original_image - gt_image
            err[~mask] = 0
            normalized_loss = np.abs(err) / (gt_image + 1e-8)
            avg_err = np.mean(normalized_loss[mask])
            nrmse = np.linalg.norm(err[mask]) / np.linalg.norm(gt_image[mask])
            logger.info(
                f"NRMSE for {modality}: {nrmse * 100:.1f}%. Avg Err: {avg_err * 100:.1f}%"
            )
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
        plt.colorbar()
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


def plot_clusters(voxel_labels, save_path):
    voxel_labels = voxel_labels.detach().cpu().numpy()
    plt.imshow(voxel_labels)
    plt.colorbar()
    plt.axis("off")
    plt.savefig(save_path / "voxel_labels.png")
    plt.close()


def plot_trj(trj, save_path):
    trj = rearrange(trj, "nro npe ntr d -> npe nro ntr d")
    trj = trj.detach().cpu().numpy()
    plt.figure()
    for t in trj:
        plt.plot(t[:, 0, 0], t[:, 0, 1])
    plt.savefig(save_path / "trj.png")
    plt.close()


def plot_signals(signals, save_path, name):
    signals = signals.detach().cpu().numpy()
    plt.figure()
    for i, signal in enumerate(signals):
        plt.plot(signal)
    plt.savefig(save_path / f"{name}-signals.png")
    plt.close()


def plot_region(region, fas, save_path, name):
    region = region.detach().cpu().numpy()
    plt.figure()
    plt.imshow(1 * region)
    plt.axis("off")
    plt.savefig(save_path / f"{name}-region.png")
    plt.close()

    fas = fas.detach().cpu().numpy()
    plt.figure()
    for fa in fas:
        plt.plot(np.rad2deg(fa))
    plt.savefig(save_path / f"{name}-fas.png")
    plt.close()


def plot_excitation_modes(modes, save_path=None):
    """
    Plots a list of 2D arrays in a single column with the title "Mode {i}" for each.
    Optionally saves the figure to a specified path.

    :param modes: list of 2D arrays to plot
    :param save_path: (optional) path to save the figure
    """
    num_modes = len(modes)

    modes = [mode.detach().cpu().numpy() for mode in modes]

    fig, axs = plt.subplots(num_modes, 1, figsize=(6, num_modes * 3))

    if num_modes == 1:
        axs = [axs]

    for i, mode in enumerate(modes):
        ax = axs[i]
        im = ax.imshow(mode, aspect="equal")
        ax.title.set_text(f"Mode {i+1}")
        fig.colorbar(im, ax=ax)
        ax.axis("off")

    plt.tight_layout()

    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_bias_variance(bias, var, save_path_prefix):
    plt.figure(figsize=(12, 8))
    bias = bias.cpu().numpy()
    var = var.cpu().numpy()
    titles = ["PD", "T1", "T2"]

    for i in range(3):
        ax = plt.subplot(2, 3, i + 1)
        im = ax.imshow(bias[..., i], cmap="viridis")
        plt.title(f"Bias {titles[i]}")
        plt.axis("off")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    # Plotting variances with colorbars
    for i in range(3):
        ax = plt.subplot(2, 3, i + 4)
        im = ax.imshow(var[..., i], cmap="viridis")
        plt.title(f"Variance {titles[i]}")
        plt.axis("off")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    plt.tight_layout()
    plt.savefig(save_path_prefix / "bias_var", bbox_inches="tight", pad_inches=0.1)
    plt.close()


def stochastic_recon(data: GeneratedData, args: Config, recon_data: ReconData):
    device = args.device_idx

    seq = BareFISP().to(device)
    mps_torch = data.mps.to(device)

    if args.recon_params.grog_grid_oversamp is not None:
        nufft = gridded_nufft(
            args.im_size, device, grid_oversamp=args.recon_params.grog_grid_oversamp
        )
    else:
        nufft = torchkb_nufft(args.im_size, device)

    trj = nufft.rescale_trajectory(data.trj.to(device))

    sqrt_dcf = torch.sqrt(data.dcf.to(device))
    data.ksp = data.ksp * sqrt_dcf[None, ...]

    def A(x):
        img_coil = x[:, None, ...] * mps_torch[None, ...]  # T nc 220 220
        trj_rs = rearrange(trj, "nro npe ntr d -> ntr nro npe d")
        ksp = nufft(img_coil, trj_rs)
        ksp_rs = rearrange(ksp, "ntr nc nro npe -> nc nro npe ntr")
        return ksp_rs * sqrt_dcf[None, ...]

    def AH(y):
        # y - nc nro npe ntr
        y_dcf = y * sqrt_dcf[None, ...]
        y_rs = rearrange(y_dcf, "nc nro npe ntr -> ntr nc nro npe")
        trj_rs = rearrange(trj, "nro npe ntr d -> ntr nro npe d")
        img_coil = nufft.adjoint(y_rs, trj_rs)  # T nc 220 220
        img = torch.sum(img_coil * mps_torch[None, ...].conj(), dim=1)
        return img

    if True:
        # Make sure A is correct
        seq = BareFISP().to(device)
        ims_gt = rearrange(
            seq(
                data.gt[..., None, :].type(torch.float32), fa_map=data.fa_map
            ).squeeze(),
            "h w ntr -> ntr h w",
        )
        ksp_gt = A(ims_gt)
        err = torch.sum(torch.abs(ksp_gt - data.ksp))
        logger.debug(f"Error in A: {err}")

    # Randomly intialize param maps
    im_size = data.mps.shape[1:]
    mask = data.brain_mask
    if recon_data is None:
        init_params = torch.ones((*im_size, 1, 3)).type(torch.float32).to(data.gt)
        init_params[..., 0, 0] = torch.maximum(
            data.gt[..., 0]
            + 0.1 * torch.randn_like(init_params[..., 0, 0]).to(data.gt),
            torch.tensor(0),
        )
        init_params[..., 0, 1] = torch.maximum(
            data.gt[..., 1]
            + 100 * torch.randn_like(init_params[..., 0, 1]).to(data.gt),
            torch.tensor(0),
        )
        init_params[..., 0, 2] = torch.maximum(
            data.gt[..., 2] + 30 * torch.randn_like(init_params[..., 0, 2]).to(data.gt),
            torch.tensor(0),
        )

        params = init_params.to(device).type(torch.float32).clone()
    else:
        pd = (
            recon_data.tissues[..., 0]
            .clone()
            .type(torch.float32)
            .to(device)
            .requires_grad_(True)
        )
        t1 = (
            recon_data.tissues[..., 1]
            .clone()
            .type(torch.float32)
            .to(device)
            .requires_grad_(True)
        )
        t2 = (
            recon_data.tissues[..., 2]
            .clone()
            .type(torch.float32)
            .to(device)
            .requires_grad_(True)
        )
        params = torch.stack([pd, t1, t2], dim=-1).unsqueeze(2)
    params.requires_grad = True

    def calc_mape(params, gt):
        mape = torch.abs(params.squeeze() - gt) / (gt + 1e-8)
        mape = 100 * torch.mean(mape[mask == 1], dim=0)
        return mape

    initial_mape = calc_mape(params.detach(), data.gt)
    logger.info(f"initial MAPE: {initial_mape}")

    # Gradient descent
    nsteps = 10000
    params_groups = [
        {"params": pd, "lr": 1e-2},
        {"params": t1, "lr": 1},
        {"params": t2, "lr": 1},
    ]
    optim = torch.optim.Adam(params_groups, lr=1)
    scheduler = ReduceLROnPlateau(
        optim, mode="min", patience=50, factor=0.2, verbose=True, min_lr=1e-4
    )
    loss_func = torch.nn.MSELoss(reduction="sum")
    losses = []

    for i in tqdm(range(nsteps)):
        params = torch.stack([pd, t1, t2], dim=-1).unsqueeze(2)
        if i == 0:
            ims = rearrange(
                seq(data.gt.unsqueeze(2).type(torch.float32), data.fa_map).squeeze(),
                "h w t -> t h w",
            )
            ksp_est = A(ims)

            loss = loss_func(data.ksp.real, ksp_est.real) + loss_func(
                data.ksp.imag, ksp_est.imag
            )

        ims = rearrange(seq(params, data.fa_map).squeeze(), "h w t -> t h w")
        ims.retain_grad()
        ksp_est = A(ims)
        ksp_est.retain_grad()

        loss = loss_func(data.ksp.real, ksp_est.real) + loss_func(
            data.ksp.imag, ksp_est.imag
        )
        # loss += 1e-6 * torch.sum(torch.abs(params) ** 2)
        # loss += 1e-7 * torch.sum(torch.abs(ims) ** 2)

        loss.backward()
        # FIXME: when T1 or T2 = 0 then the relaxation EPG function explodes and the gradients are NaN (this is okay since
        # the PD is also 0 so it doesn't matter but pytorch doesn't like nan. Find some better way to do that.)
        pd.grad.nan_to_num_()
        t1.grad.nan_to_num_()
        t2.grad.nan_to_num_()
        t1.grad = 10 * torch.sign(t1.grad)
        t2.grad = 2 * torch.sign(t2.grad)
        optim.step()
        optim.zero_grad()
        with torch.no_grad():
            pd.clamp_(0, 1)  # PD
            t1.clamp_(0, 5000)  # T1
            t2.clamp_(0, 2000)  # T2
        losses.append(loss.detach().cpu())
        scheduler.step(loss)

        if i % 5 == 0:
            logger.info(loss)
            mape = calc_mape(params.detach(), data.gt)
            logger.info(f"MPAE at {i}: {mape}")
            log_dir = args.exp_dir / f"stochastic_{i}"
            if i % 20 == 0:
                os.makedirs(log_dir, exist_ok=True)
                plot_recons(
                    [params.squeeze()], data.gt, ["MRFRF"], data.brain_mask, log_dir
                )

    plt.imshow(params[:, :, 0, 0].detach().cpu())
    plt.colorbar()
    plt.savefig(args.exp_dir / "amps.png")
    plt.close()
    plt.imshow(params[:, :, 0, 1].detach().cpu())
    plt.colorbar()
    plt.savefig(args.exp_dir / "t1.png")
    plt.close()
    plt.imshow(params[:, :, 0, 2].detach().cpu())
    plt.colorbar()
    plt.savefig(args.exp_dir / "t2.png")
    plt.close()
    plt.plot(losses)
    plt.savefig(args.exp_dir / "loss.png")
    plt.close()
    quit()


if __name__ == "__main__":
    main()
