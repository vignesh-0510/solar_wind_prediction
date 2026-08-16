import os
import re
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

LABEL_CONFIG = {
    "vt": {"cmap": "gnuplot", "title": r"$v_\theta$"},
    "vp": {"cmap": "gnuplot", "title": r"$v_\phi$"},
    "bt": {"cmap": "coolwarm", "title": r"$\mathbf{B}_\theta$"},
    "bp": {"cmap": "coolwarm", "title": r"$\mathbf{B}_\phi$"},
    "jt": {"cmap": "inferno", "title": r"$\mathbf{J}_\theta$"},
    "jp": {"cmap": "inferno", "title": r"$\mathbf{J}_\phi$"},
    "jr": {"cmap": "inferno", "title": r"$\mathbf{J}_r$"},
    "rho": {"cmap": "viridis", "title": r"$\rho$"},
    "p": {"cmap": "viridis", "title": r"$p$"}
}

LABEL_CONFIG = {
    "vt": {"cmap": "viridis", "title": r"$v_\theta$"},
    "vp": {"cmap": "viridis", "title": r"$v_\phi$"},
    "bt": {"cmap": "viridis", "title": r"$\mathbf{B}_\theta$"},
    "bp": {"cmap": "viridis", "title": r"$\mathbf{B}_\phi$"},
    "jt": {"cmap": "viridis", "title": r"$\mathbf{J}_\theta$"},
    "jp": {"cmap": "viridis", "title": r"$\mathbf{J}_\phi$"},
    "jr": {"cmap": "viridis", "title": r"$\mathbf{J}_r$"},
    "rho": {"cmap": "viridis", "title": r"$\rho$"},
    "p": {"cmap": "viridis", "title": r"$p$"}
}


def create_gif_from_array(array, folder_path='./', file_name='image.gif', duration=100):
    """
    Create and save a GIF from a numpy array of shape (H, W, T).
    Args:
        array: numpy array of shape (H, W, T)
        folder_path: folder to save the GIF
        file_name: name of the GIF file
        duration: frame duration in ms
    """
    import numpy as np
    import matplotlib.pyplot as plt

    output_gif = os.path.join(folder_path, file_name)
    images = []
    T, H, W = array.shape
    # array = array.astype(np.uint8)  # Convert to uint8 for image representation
    for t in range(T):
        img = array[t]
        # Normalize to 0-1 for colormap
        img_norm = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        # Apply viridis colormap
        img_color = plt.get_cmap('viridis')(img_norm)
        # Convert to uint8 RGB
        img_rgb = (img_color[:, :, :3] * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_rgb)
        images.append(img_pil)
    if images:
        images[0].save(
            output_gif,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
        print(f"GIF saved to {output_gif}")
    else:
        print("No images found in array.")

def create_input_output_gif(
    input_array,
    output_array,
    folder_path="./",
    file_name="input_output.gif",
    duration=100,
    cmap="viridis",
    same_scale=True,
    dpi=200,
):
    """
    Create a GIF using matplotlib with input (top) and output (bottom).

    Args:
        input_array:  numpy array (T, H, W)
        output_array: numpy array (T, H, W)
        folder_path:  directory to save GIF
        file_name:    GIF file name
        duration:     frame duration in ms
        cmap:         matplotlib colormap
        same_scale:   share color scale between input & output per frame
        dpi:          figure DPI
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import animation

    assert input_array.shape == output_array.shape, \
        "Input and output arrays must have the same shape (T, H, W)"

    T, H, W = input_array.shape
    output_gif = os.path.join(folder_path, file_name)

    # --- figure ---
    fig, axes = plt.subplots(
        2, 1,
        figsize=(10,8),
        dpi=dpi
    )

    # plt.subplots_adjust(hspace=0.35)  # separation between panels
    plt.tight_layout()
    ax_in, ax_out = axes
    ax_in.set_title("Input")
    ax_out.set_title("Output")

    for ax in axes:
        ax.axis("off")

    # --- initial normalization ---
    if same_scale:
        vmin = min(input_array[0].min(), output_array[0].min())
        vmax = max(input_array[0].max(), output_array[0].max())
    else:
        vmin, vmax = None, None

    im_in = ax_in.imshow(
        input_array[0],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        animated=True,
    )

    im_out = ax_out.imshow(
        output_array[0],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        animated=True,
    )

    def update(t):
        inp = input_array[t]
        out = output_array[t]

        if same_scale:
            vmin = min(inp.min(), out.min())
            vmax = max(inp.max(), out.max())
            im_in.set_clim(vmin, vmax)
            im_out.set_clim(vmin, vmax)

        im_in.set_array(inp)
        im_out.set_array(out)
        return [im_in, im_out]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=T,
        interval=duration,
        blit=True,
    )

    anim.save(output_gif, writer="pillow")
    plt.close(fig)

    print(f"GIF saved to {output_gif}")

def create_initial_param_plot(y_true, pred, labels, folder_path="./", file_name="initial_param_comparison.png", cmap="viridis", dpi=200):
    assert y_true.shape == pred.shape, "y_true and pred must have the same shape (C, H, W)"
    
    output_path = os.path.join(folder_path, file_name)
    n_components = y_true.shape[0]
    fig, ax = plt.subplots(3, n_components, figsize=(n_components*5, 11))
    for i in range(n_components):
        vmin = np.min([y_true[i].min(), pred[i].min()])
        vmax = np.max([y_true[i].max(), pred[i].max()])

        im0 = ax[0, i].imshow(y_true[i], cmap=f'{LABEL_CONFIG[labels[i]]["cmap"]}', vmin=vmin, vmax=vmax)
        # ax[0, i].axis('off')
        ax[0, i].set_title(f'{LABEL_CONFIG[labels[i]]["title"]}', fontsize=28)

        im1 = ax[1, i].imshow(pred[i], cmap=f'{LABEL_CONFIG[labels[i]]["cmap"]}', vmin=vmin, vmax=vmax)
        # ax[1, i].axis('off')
        # ax[1, i].set_title(f'{labels[i]} pred')
        
        diff = y_true[i] - pred[i]
        im2 = ax[2, i].imshow(diff, cmap='coolwarm', vmin=diff.min(), vmax=diff.max())
        # ax[2, i].axis('off')
        # ax[2, i].set_title(f'{labels[i]} diff [T - P]')
        for row in range(3):
            ax[row, i].set_xticks([])
            ax[row, i].set_yticks([])

        if i == 0:
            ax[0, 0].set_ylabel("GT", fontsize=38)
            ax[1, 0].set_ylabel("P", fontsize=38)
            ax[2, 0].set_ylabel("Diff", fontsize=38)
        # colorbar for each image, same scale (vmin/vmax) within the row
        fig.colorbar(im0, ax=ax[0, i], fraction=0.04, pad=0.04)
        fig.colorbar(im1, ax=ax[1, i], fraction=0.04, pad=0.04)
        fig.colorbar(im2, ax=ax[2, i], fraction=0.04, pad=0.04)

    plt.tight_layout()
    plt.savefig(f'{output_path}.png', dpi=dpi)
    plt.savefig(f'{output_path}.pdf', dpi=dpi)
    plt.close(fig)

    print(f"Initial parameter comparison plot saved to {output_path}.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create GIF from image{index}.png files in a folder.")
    parser.add_argument('folder', type=str, help='Folder containing image{index}.png files')
    parser.add_argument('output', type=str, help='Output GIF file path')
    parser.add_argument('--duration', type=int, default=100, help='Frame duration in ms')
    args = parser.parse_args()
    create_gif_from_folder(args.folder, args.output, args.duration)
