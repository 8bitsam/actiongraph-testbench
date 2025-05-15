import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pymatgen.core import Composition
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Data"))
FEATURIZED_DATA_DIR = os.path.join(DATA_DIR, "featurized-data-baseline/")
PLOTS_DIR = os.path.join(DATA_DIR, "feature_plots_base_pub/")
ELEMENT_PROPS_LIST = [
    "atomic_mass",
    "atomic_radius",
    "melting_point",
    "X",
    "electron_affinity",
    "ionization_energy",
]

PROPERTY_X_LABELS = {
    "atomic_mass": "Average Atomic Mass (amu)",
    "atomic_radius": "Average Atomic Radius (pm)",
    "melting_point": "Average Melting Point (K)",
    "X": "Average Electronegativity (Pauling)",
    "electron_affinity": "Average Electron Affinity (eV)",
    "ionization_energy": "Average Ionization Energy (eV)",
}
DEFAULT_X_LABEL = "Average Property Value"
plt.style.use("seaborn-v0_8-whitegrid")
PUB_FONT_SIZE_SMALL = 12
PUB_FONT_SIZE_MEDIUM = 14
PUB_FIGURE_DPI = 300
plt.rcParams.update(
    {
        "font.size": PUB_FONT_SIZE_MEDIUM,
        "axes.labelsize": PUB_FONT_SIZE_MEDIUM,
        "xtick.labelsize": PUB_FONT_SIZE_SMALL,
        "ytick.labelsize": PUB_FONT_SIZE_SMALL,
        "legend.fontsize": PUB_FONT_SIZE_SMALL,
    }
)


def load_base_featurized_data(feature_dir):
    print(f"Loading base featurized data from: {feature_dir}")
    try:
        features = np.load(os.path.join(feature_dir, "features.npy"))
        with open(os.path.join(feature_dir, "element_map.json"), "r") as f:
            element_map = json.load(f)
        with open(os.path.join(feature_dir, "targets.json"), "r") as f:
            targets = json.load(f)
        print(
            f"Loaded {features.shape[0]} samples with {features.shape[1]} \
              features."
        )
        return features, element_map, targets
    except FileNotFoundError as e:
        print(
            f"Error: Base featurized data not found in {feature_dir}. \
                File: {e.filename}",
            file=sys.stderr,
        )
        return None, None, None
    except Exception as e:
        print(f"Error loading base data: {e}", file=sys.stderr)
        return None, None, None


def plot_property_distributions_base(features, element_map, props_list, plots_dir):
    print(
        "\nPlotting individual distributions of averaged elemental \
            properties (Base k-NN)..."
    )
    num_elements_in_map = len(element_map)
    num_props = len(props_list)
    expected_dims = num_elements_in_map + num_props
    if features.shape[1] != expected_dims:
        print(
            f"Error: Feature dimension mismatch. Expected {expected_dims}, \
                got {features.shape[1]}.",
            file=sys.stderr,
        )
        return

    prop_features_data = features[:, num_elements_in_map:]
    os.makedirs(plots_dir, exist_ok=True)
    for i in range(num_props):
        prop_name = props_list[i]
        data_col = prop_features_data[:, i]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        if len(data_col) > 0:
            sns.histplot(
                data_col,
                kde=True,
                ax=ax,
                color="skyblue",
                edgecolor="black",
                linewidth=0.5,
            )
            ax.set_xlabel(PROPERTY_X_LABELS.get(prop_name, DEFAULT_X_LABEL))
            ax.set_ylabel("Frequency")
        else:
            ax.text(
                0.5,
                0.5,
                "No Data Available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xlabel(PROPERTY_X_LABELS.get(prop_name, DEFAULT_X_LABEL))
            ax.set_ylabel("Frequency")
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        sanitized_prop_name = prop_name.replace("/", "_").replace("\\", "_")
        save_path = os.path.join(plots_dir, f"base_prop_dist_{sanitized_prop_name}.png")
        plt.savefig(save_path, dpi=PUB_FIGURE_DPI)
        print(
            f"  Saved property distribution for \
              '{prop_name}' to {save_path}"
        )
        plt.close(fig)


def plot_tsne_projection_base(
    features, targets, plots_dir, n_samples_tsne=2000, perplexity_tsne=30
):
    print(
        f"\nPerforming t-SNE projection on Base features (on max \
            {n_samples_tsne} samples)..."
    )
    os.makedirs(plots_dir, exist_ok=True)
    if features.shape[0] == 0:
        return
    actual_n_samples = min(n_samples_tsne, features.shape[0])
    if features.shape[0] > actual_n_samples:
        indices = np.random.choice(features.shape[0], actual_n_samples, replace=False)
        features_subset = features[indices]
        targets_subset = [targets[i] for i in indices]
    else:
        features_subset = features
        targets_subset = targets
    current_perplexity = perplexity_tsne
    if len(features_subset) <= current_perplexity:
        current_perplexity = max(5, len(features_subset) - 2)
        if current_perplexity < 5:
            print(
                f"  Skipping t-SNE: Samples ({len(features_subset)}) \
                    too few for perplexity {current_perplexity}.",
                file=sys.stderr,
            )
            return
        print(f"  Adjusted t-SNE perplexity to {current_perplexity}.")
    print("  Scaling Base data for t-SNE using StandardScaler...")
    scaler = StandardScaler()
    try:
        scaled_features = scaler.fit_transform(features_subset)
        if np.any(~np.isfinite(scaled_features)):
            scaled_features = np.nan_to_num(
                scaled_features, nan=0.0, posinf=0.0, neginf=0.0
            )
    except Exception as e:
        print(f"Error scaling Base data for t-SNE: {e}", file=sys.stderr)
        return
    print(
        f"  Running t-SNE on Base features (perplexity={current_perplexity}, \
            samples={scaled_features.shape[0]})..."
    )
    start_tsne_time = time.time()
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=current_perplexity,
        n_iter=300,
        init="pca",
        learning_rate="auto",
    )
    try:
        tsne_results = tsne.fit_transform(scaled_features)
    except Exception as e_tsne:
        print(f"Error during t-SNE: {e_tsne}", file=sys.stderr)
        return
    print(f"  t-SNE finished in {time.time() - start_tsne_time:.2f} seconds.")
    plt.figure(figsize=(7, 6))
    num_elements_list = []
    for formula in targets_subset:
        try:
            num_elements_list.append(len(Composition(formula).elements))
        except:
            num_elements_list.append(0)
    scatter = plt.scatter(
        tsne_results[:, 0],
        tsne_results[:, 1],
        c=num_elements_list,
        cmap="viridis",
        alpha=0.65,
        s=15,
        edgecolor="none",
    )
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.xticks([])
    plt.yticks([])
    try:
        cbar = plt.colorbar(
            scatter, label="# Elements in Target Formula", fraction=0.046, pad=0.04
        )
        cbar.ax.tick_params(labelsize=PUB_FONT_SIZE_SMALL)
        cbar.set_label("# Elements in Target Formula", size=PUB_FONT_SIZE_SMALL)
    except Exception:
        pass

    plt.tight_layout()
    save_path = os.path.join(plots_dir, "tsne_projection_base.png")
    plt.savefig(save_path, dpi=PUB_FIGURE_DPI)
    print(f"Saved Base k-NN t-SNE plot to {save_path}")
    plt.close()


if __name__ == "__main__":
    print("--- Running Base Feature Visualization (Publication Style) ---")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    features_arr, elem_map, target_formulas = load_base_featurized_data(
        FEATURIZED_DATA_DIR
    )

    if (
        features_arr is not None
        and elem_map is not None
        and target_formulas is not None
    ):
        plot_property_distributions_base(
            features_arr, elem_map, ELEMENT_PROPS_LIST, PLOTS_DIR
        )

        plot_tsne_projection_base(
            features_arr, target_formulas, PLOTS_DIR, n_samples_tsne=2000
        )

        print("\nBase Feature Visualization script finished.")
    else:
        print(
            "\nExiting Base feature visualization due to data \
              loading failure."
        )
        sys.exit(1)
