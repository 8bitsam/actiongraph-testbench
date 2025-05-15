import json
import os
import sys
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pymatgen.core import Composition
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Data"))
FEATURIZED_DATA_DIR_AG = os.path.join(DATA_DIR, "featurized-data-actiongraph/")
PLOTS_DIR_AG = os.path.join(DATA_DIR, "feature_plots_ag_pca_pub/")

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


def load_ag_featurized_data(feature_dir_ag):
    print(f"Loading AG-PCA featurized data from: {feature_dir_ag}")
    try:
        features_ag = np.load(os.path.join(feature_dir_ag, "features_ag.npy"))
        with open(os.path.join(feature_dir_ag, "element_map_ag.json"), "r") as f:
            element_map_ag = json.load(f)
        with open(
            os.path.join(feature_dir_ag, "targets_ag_output_formula.json"), "r"
        ) as f:
            targets_ag = json.load(f)

        pca_model_path = os.path.join(feature_dir_ag, "adj_pca.joblib")
        pca_n_components = 0
        if os.path.exists(pca_model_path):
            pca_model = joblib.load(pca_model_path)
            if hasattr(pca_model, "n_components_"):
                pca_n_components = pca_model.n_components_
        else:
            print(
                f"Warning: PCA model 'adj_pca.joblib' not found. \
                    PCA related plots might be affected or \
                        pca_n_components inferred."
            )

        print(
            f"Loaded {features_ag.shape[0]} AG samples with \
                {features_ag.shape[1]} features."
        )
        if pca_n_components > 0:
            print(f"  PCA components from loaded model: {pca_n_components}")
        elif os.path.exists(pca_model_path):
            print(
                f"  PCA model 'adj_pca.joblib' loaded, but n_components_ is \
                    0 or not set as expected."
            )
        return features_ag, element_map_ag, targets_ag, pca_n_components
    except FileNotFoundError as e:
        print(
            f"Error: Featurized AG data or PCA model not found in \
                {feature_dir_ag}. File: {e.filename}",
            file=sys.stderr,
        )
        return None, None, None, None
    except Exception as e:
        print(f"Error loading AG data: {e}", file=sys.stderr)
        return None, None, None, None


def plot_target_property_distributions_ag(
    features_ag, element_map, props_list, plots_dir
):
    print(
        "\nPlotting individual distributions of TARGET averaged \
            elemental properties..."
    )
    num_elements_in_map = len(element_map)
    num_props = len(props_list)
    target_prop_features_start_idx = num_elements_in_map
    target_prop_features_end_idx = num_elements_in_map + num_props
    if target_prop_features_end_idx > features_ag.shape[1]:
        print(
            f"Error: Index for target properties \
                ({target_prop_features_end_idx}) exceeds feature \
                    dimension ({features_ag.shape[1]}).",
            file=sys.stderr,
        )
        return
    target_prop_features = features_ag[
        :, target_prop_features_start_idx:target_prop_features_end_idx
    ]
    os.makedirs(plots_dir, exist_ok=True)
    for i in range(num_props):
        prop_name = props_list[i]
        data_col = target_prop_features[:, i]

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
        save_path = os.path.join(plots_dir, f"prop_dist_{sanitized_prop_name}.png")
        plt.savefig(save_path, dpi=PUB_FIGURE_DPI)
        print(
            f"  Saved property distribution for \
              '{prop_name}' to {save_path}"
        )
        plt.close(fig)


def plot_pca_explained_variance(pca_model_path, plots_dir):
    print("\nPlotting PCA Explained Variance...")
    if not os.path.exists(pca_model_path):
        print(
            f"  PCA model file not found at {pca_model_path}. \
                Skipping explained variance plot."
        )
        return
    try:
        pca = joblib.load(pca_model_path)
        if not hasattr(pca, "explained_variance_ratio_"):
            print(
                "  Loaded PCA model does not have \
                    'explained_variance_ratio_'. Skipping plot."
            )
            return
        explained_variance_ratio = pca.explained_variance_ratio_
        if len(explained_variance_ratio) == 0:
            print("  PCA explained variance ratio is empty. Skipping plot.")
            return
        plt.figure(figsize=(8, 5))
        plt.bar(
            range(1, len(explained_variance_ratio) + 1),
            explained_variance_ratio,
            alpha=0.8,
            align="center",
            label="Individual",
            color="steelblue",
        )
        plt.step(
            range(1, len(explained_variance_ratio) + 1),
            np.cumsum(explained_variance_ratio),
            where="mid",
            label="Cumulative",
            color="firebrick",
            linewidth=2,
        )
        plt.ylabel("Explained Variance Ratio")
        plt.xlabel("Principal Component Index")
        plt.legend(loc="best")
        num_components_to_show_ticks = len(explained_variance_ratio)
        tick_step = max(
            1,
            (
                num_components_to_show_ticks // 10
                if num_components_to_show_ticks > 0
                else 1
            ),
        )
        if num_components_to_show_ticks < 10 and num_components_to_show_ticks > 0:
            tick_step = 1
        if num_components_to_show_ticks > 0:
            plt.xticks(
                ticks=np.arange(1, num_components_to_show_ticks + 1, step=tick_step)
            )
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        save_path = os.path.join(plots_dir, "ag_pca_explained_variance.png")
        plt.savefig(save_path, dpi=PUB_FIGURE_DPI)
        print(f"Saved PCA explained variance plot to {save_path}")
        plt.close()
    except Exception as e:
        print(f"Error plotting PCA explained variance: {e}", file=sys.stderr)


def plot_tsne_projection_ag(
    features_ag, targets_ag, plots_dir, n_samples_tsne=2000, perplexity_tsne=30
):
    print(
        f"\nPerforming t-SNE projection on AG features (on max \
            {n_samples_tsne} samples)..."
    )
    os.makedirs(plots_dir, exist_ok=True)
    if features_ag.shape[0] == 0:
        return
    actual_n_samples = min(n_samples_tsne, features_ag.shape[0])
    if features_ag.shape[0] > actual_n_samples:
        indices = np.random.choice(
            features_ag.shape[0], actual_n_samples, replace=False
        )
        features_subset = features_ag[indices]
        targets_subset = [targets_ag[i] for i in indices]
    else:
        features_subset = features_ag
        targets_subset = targets_ag
    current_perplexity = perplexity_tsne
    if len(features_subset) <= current_perplexity:
        current_perplexity = max(5, len(features_subset) - 2)
        if current_perplexity < 5:
            print(
                f"  Skipping t-SNE: Samples ({len(features_subset)}) too few \
                    for perplexity {current_perplexity}.",
                file=sys.stderr,
            )
            return
        print(f"  Adjusted t-SNE perplexity to {current_perplexity}.")
    print("  Scaling AG data for t-SNE...")
    scaler = StandardScaler()
    try:
        scaled_features = scaler.fit_transform(features_subset)
        if np.any(~np.isfinite(scaled_features)):
            scaled_features = np.nan_to_num(
                scaled_features, nan=0.0, posinf=0.0, neginf=0.0
            )
    except Exception as e:
        print(f"Error scaling for t-SNE: {e}", file=sys.stderr)
        return
    print(
        f"  Running t-SNE (perplexity={current_perplexity}, \
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
    save_path = os.path.join(plots_dir, "tsne_projection_ag_pca.png")
    plt.savefig(save_path, dpi=PUB_FIGURE_DPI)
    print(f"Saved AG t-SNE plot to {save_path}")
    plt.close()


if __name__ == "__main__":
    print(
        "--- Running ActionGraph Feature Visualization (PCA version - \
            No PCA Component Distributions) ---"
    )
    os.makedirs(PLOTS_DIR_AG, exist_ok=True)

    features_ag_arr, elem_map_ag, target_formulas_ag, pca_n_components_loaded = (
        load_ag_featurized_data(FEATURIZED_DATA_DIR_AG)
    )

    if (
        features_ag_arr is not None
        and elem_map_ag is not None
        and target_formulas_ag is not None
    ):
        num_element_map_features = len(elem_map_ag)
        num_prop_features = len(ELEMENT_PROPS_LIST)
        num_total_target_chem_features = num_element_map_features + num_prop_features
        if pca_n_components_loaded is None:
            pca_n_components_loaded = 0
        if (
            features_ag_arr.shape[1]
            != num_total_target_chem_features + pca_n_components_loaded
        ):
            print(
                f"CRITICAL ERROR: Loaded feature dimension \
                    ({features_ag_arr.shape[1]}) does not match expected \
                    sum of chemical features \
                    ({num_total_target_chem_features}) and \
                    PCA components ({pca_n_components_loaded})."
            )
            print(
                "  Please check feature generation script (featurize.py) and constants here."
            )
            sys.exit(1)

        plot_target_property_distributions_ag(
            features_ag_arr, elem_map_ag, ELEMENT_PROPS_LIST, PLOTS_DIR_AG
        )
        pca_model_filepath = os.path.join(FEATURIZED_DATA_DIR_AG, "adj_pca.joblib")
        plot_pca_explained_variance(pca_model_filepath, PLOTS_DIR_AG)
        plot_tsne_projection_ag(
            features_ag_arr, target_formulas_ag, PLOTS_DIR_AG, n_samples_tsne=2000
        )
        print("\nAG Feature Visualization script finished.")
    else:
        print(
            "\nExiting AG feature visualization due to data loading \
              failure."
        )
        sys.exit(1)
