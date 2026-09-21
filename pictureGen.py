import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_architecture_png(output_path="arqflow.png"):
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)
    ax.axis('off')

    # Color Palette
    bg_blue = "#EBF5FB"
    border_blue = "#2980B9"
    bg_green = "#EAFAF1"
    border_green = "#27AE60"
    text_color = "#2C3E50"

    # Define Block Box Drawing Function
    def draw_box(x, y, width, height, title, subtitle, color_bg, color_border, is_ellipse=False):
        if is_ellipse:
            patch = patches.Ellipse((x + width/2, y + height/2), width, height, 
                                    facecolor=color_bg, edgecolor=color_border, linewidth=2)
        else:
            patch = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1",
                                          facecolor=color_bg, edgecolor=color_border, linewidth=2)
        ax.add_patch(patch)
        ax.text(x + width/2, y + height/2 + 0.12, title, ha='center', va='center', 
                fontsize=11, fontweight='bold', color=text_color)
        ax.text(x + width/2, y + height/2 - 0.15, subtitle, ha='center', va='center', 
                fontsize=9, style='italic', color="#566573")

    # Draw Top Nodes
    draw_box(0.5, 2.7, 2.4, 1.1, "Raw Data Sources", "(.xlsx Datasets)", bg_blue, border_blue)
    draw_box(3.8, 2.7, 2.4, 1.1, "RDF Graph Engine", "(ciatec_basquete.ttl)", bg_blue, border_blue)
    draw_box(7.1, 2.7, 2.4, 1.1, "SPARQL Engine", "& TransE Embeddings", bg_blue, border_blue)

    # Draw Bottom Nodes
    draw_box(7.1, 0.5, 2.4, 1.1, "Local LLM", "(Qwen2.5-0.5B)", bg_blue, border_blue)
    draw_box(3.8, 0.5, 2.4, 1.1, "Streamlit UI App", "(app.py Interface)", bg_green, border_green, is_ellipse=False)

    # Draw Connector Arrows
    arrow_props = dict(arrowstyle="-|>", color="#2980B9", lw=2, mutation_scale=15)
    green_arrow = dict(arrowstyle="-|>", color="#27AE60", lw=2, mutation_scale=15)
    dashed_arrow = dict(arrowstyle="-|>", color="#2980B9", lw=1.8, linestyle="--", mutation_scale=15)

    # Main flow arrows
    ax.annotate("", xy=(3.8, 3.25), xytext=(2.9, 3.25), arrowprops=arrow_props)
    ax.annotate("", xy=(7.1, 3.25), xytext=(6.2, 3.25), arrowprops=arrow_props)
    ax.annotate("", xy=(8.3, 1.6), xytext=(8.3, 2.7), arrowprops=arrow_props)
    ax.annotate("", xy=(6.2, 1.05), xytext=(7.1, 1.05), arrowprops=green_arrow)

    # Non-intersecting diagonal feed arrow
    ax.annotate("", xy=(6.2, 1.25), xytext=(7.1, 2.7), arrowprops=dashed_arrow)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Image successfully generated and saved to {output_path}")

if __name__ == "__main__":
    generate_architecture_png()