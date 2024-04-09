# WIP: This is a work in progress and is not yet functional

import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

def make_gif():

    fig, (ax1, ax2) = plt.subplots(1,2)

    ims = []

    ndim = len(self.im_size)
    if r_slice is None:
        if ndim == 3:            
            slc = self.im_size[-1]//2
            r_slice = (slice(None),) * (ndim - 1) + (slc,)
        else:
            r_slice = (slice(None),) * ndim
    phase = self.phase_est_slice(r_slice, t_slice, lowrank=False).cpu()
    phase_LR = self.phase_est_slice(r_slice, t_slice, lowrank=True).cpu()
    vmin = torch.angle(phase).min()
    vmax = torch.angle(phase).max()

    ax1.set_title('True Phase')
    ax2.set_title('Time-Segmented Phase Model')
    ax1.axis('off')
    ax2.axis('off')
    for i in tqdm(range(phase.shape[-1]), 'Making Movie'):
        im1 = ax1.imshow(torch.angle(phase[..., i]), vmin=vmin, vmax=vmax, animated=True, cmap='jet')
        im2 = ax2.imshow(torch.angle(phase_LR[..., i]), vmin=vmin, vmax=vmax, cmap='jet')
        ims.append([im1,im2])
    width = 0.75
    cb_ax = fig.add_axes([(1-width)/2,.13,width,.04])
    fig.colorbar(im2,orientation='horizontal',cax=cb_ax)
    fig.tight_layout()
    ani = animation.ArtistAnimation(fig, ims, interval=20, blit=True,
                            repeat_delay=500)
    writer = animation.PillowWriter(fps=15,
                            metadata=dict(artist='Me'),
                            bitrate=1800)
    ani.save(name, writer=writer)