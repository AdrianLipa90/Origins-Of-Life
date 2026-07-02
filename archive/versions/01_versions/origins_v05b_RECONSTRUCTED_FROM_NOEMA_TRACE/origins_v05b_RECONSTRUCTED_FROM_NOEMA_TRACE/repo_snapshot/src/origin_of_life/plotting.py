import matplotlib.pyplot as plt


def save_figure(fig, path: str, dpi: int = 160) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
