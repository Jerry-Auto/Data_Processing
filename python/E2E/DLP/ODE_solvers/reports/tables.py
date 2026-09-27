"""Static, publication-friendly table images from experiment rows."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_table_image(rows, path, title, methods, display):
    """Write a compact table showing deployed cost, numerical and distribution quality."""
    columns = ['prediction', 'method', 'nfe', 'latency_ms', 'reference_mse', 'w2', 'component_std']
    labels = ['prediction', 'solver', 'NFE', 'ms', 'ref MSE', 'Gaussian W2', 'std']
    cell_text = [[str(r['prediction']), display.get(r['method'], r['method']), str(r['nfe']),
                  f"{r['latency_ms']:.2f}", f"{r['reference_mse']:.2e}",
                  f"{r['w2']:.3f}", f"{r['component_std']:.3f}"] for r in rows]
    fig_h = max(3.0, .29*len(rows)+1.2)
    fig, ax = plt.subplots(figsize=(11, fig_h)); ax.axis('off')
    table = ax.table(cellText=cell_text, colLabels=labels, cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.18)
    ax.set_title(title, pad=12); fig.tight_layout(); path=Path(path); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=180, bbox_inches='tight'); plt.close(fig)
