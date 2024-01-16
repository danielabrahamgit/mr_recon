# This is a very ugly solution to store some code that I used for debugging

## DEBUG
# dcf_ksp = noisy_ksp * data.dcf
# im_recon_debug = torch.zeros(220, 220, 500).to(args.device_idx).type(torch.complex64)
# for i in range(500):
#     trj = rearrange(
#         A.linops[0].nufft.rescale_trajectory(data.trj[..., i : i + 1, :]),
#         "nro npe ntr d -> ntr nro npe d",
#     )
#     im_recon_debug[..., i] = (
#         A.linops[0]
#         .nufft.adjoint(dcf_ksp[..., i][None, ...], trj)
#         .squeeze()
#         .to(args.device_idx)
#     )
#     im_recon_debug[..., i] = torch.sum(im_recon_debug[..., i] * data.mps.conj(), dim=-3)
#     err = im_recon_debug[..., i] - data.ims[..., i]
#     print(f"error {i}: max: {err.abs().max()} mean {err.abs().mean()}")

# debug_coeffs = rearrange(
#     im_recon_debug.type(torch.complex64) @ data.phis[0].T, "h w c -> c h w"
# )
# coeff_err = torch.abs(debug_coeffs - data.gt_coeffs)
# print(f"coeff error: max: {coeff_err.max()} mean {coeff_err.mean()}")
# plt.figure(figsize=(15, 10))
# for i in range(5):
#     plt.subplot(5, 3, i * 3 + 1)
#     plt.imshow(
#         data.ims[..., i * 100].cpu(), cmap="gray"
#     )  # Displaying ith image from array1
#     plt.axis("off")

#     plt.subplot(5, 3, i * 3 + 2)
#     plt.imshow(
#         im_recon_debug[..., i * 100].abs().cpu(), cmap="gray"
#     )  # Displaying ith image from array2
#     plt.axis("off")

#     plt.subplot(5, 3, i * 3 + 3)
#     plt.imshow(
#         (data.ims[..., i * 100] - im_recon_debug[..., i * 100]).cpu().abs(), cmap="gray"
#     )  # Displaying ith image from array2
#     plt.axis("off")
# plt.show()
# plt.close()

## DEBUG
        
# img_mr_recon = data.gt_coeffs.cpu().numpy()
# plt.figure(figsize=(15, 10))

# for i in range(5):
#     plt.subplot(5, 3, i * 3 + 1)
#     plt.imshow(
#         data.gt_coeffs[i].abs().cpu(), cmap="gray"
#     )  # Displaying ith image from array1
#     plt.axis("off")

#     plt.subplot(5, 3, i * 3 + 2)
#     plt.imshow(
#         debug_coeffs[i].abs().cpu(), cmap="gray"
#     )  # Displaying ith image from array2
#     plt.axis("off")

#     plt.subplot(5, 3, i * 3 + 3)
#     plt.imshow(
#         np.abs(img_mr_recon[i]), cmap="gray"
#     )  # Displaying ith image from array2
#     plt.axis("off")
# plt.show()
# plt.close()
