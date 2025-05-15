import argparse
import json
import os
import sys
import time
import warnings

import joblib
import networkx as nx
import numpy as np
from pymatgen.core import Composition, Element
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from actiongraph.actiongraph import ActionGraph

# Path configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Data"))
MP_DATA_DIR = os.path.join(DATA_DIR, "filtered-mp-data/")
AG_DATA_DIR = os.path.join(DATA_DIR, "filtered-ag-data/")
FEATURIZED_DATA_DIR_AG = os.path.join(DATA_DIR, "featurized-data-actiongraph/")

# Featurizer configuration
ELEMENT_PROPS = [
    "atomic_mass",
    "atomic_radius",
    "melting_point",
    "X",
    "electron_affinity",
    "ionization_energy",
]
MAX_NODES = 31
PCA_COMPONENTS_DEFAULT = 20

warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")
warnings.filterwarnings("ignore", category=RuntimeWarning)


def load_mp_and_ag_data_pairs(ag_dir_path, mp_dir_path):
    if not os.path.isdir(ag_dir_path):
        print(
            f"Error: AG data path is not a valid directory: {ag_dir_path}",
            file=sys.stderr,
        )
        return []
    if not os.path.isdir(mp_dir_path):
        print(
            f"Error: MP data path is not a valid directory: {mp_dir_path}",
            file=sys.stderr,
        )
        return []

    paired_data = []
    print(
        f"Scanning AG directory: {ag_dir_path} and MP directory: \
          {mp_dir_path}"
    )
    start_time = time.time()
    ag_files_found = 0
    pairs_loaded = 0
    mp_file_missing_count = 0
    mp_load_errors = 0
    ag_load_errors = 0

    for root, _, files in os.walk(ag_dir_path):
        ag_json_files = [f for f in files if f.lower().endswith("_ag.json")]
        if not ag_json_files:
            continue

        for ag_filename in ag_json_files:
            ag_files_found += 1
            mp_filename_base = ag_filename[:-8]
            mp_filename = mp_filename_base + ".json"

            ag_filepath = os.path.join(root, ag_filename)
            mp_filepath = os.path.join(mp_dir_path, mp_filename)

            mp_data_dict, ag_data_dict = None, None

            if not os.path.exists(mp_filepath):
                mp_file_missing_count += 1
                continue

            try:
                with open(mp_filepath, "r", encoding="utf-8") as f:
                    mp_data_content = json.load(f)
                if isinstance(mp_data_content, dict):
                    mp_data_dict = mp_data_content
                    mp_data_dict["_source_file"] = mp_filename
                else:
                    mp_load_errors += 1
            except Exception:
                mp_load_errors += 1

            if mp_data_dict is None:
                continue

            try:
                with open(ag_filepath, "r", encoding="utf-8") as f:
                    ag_data_content = json.load(f)
                if isinstance(ag_data_content, dict):
                    ag_data_dict = ag_data_content
                else:
                    ag_load_errors += 1
            except Exception:
                ag_load_errors += 1

            if mp_data_dict and ag_data_dict:
                paired_data.append((mp_data_dict, ag_data_dict, ag_filename))
                pairs_loaded += 1

    end_time = time.time()
    print(f"Finished scanning. Found {ag_files_found} AG files.")
    print(f"Successfully loaded {pairs_loaded} AG-MP pairs.")
    if mp_file_missing_count > 0:
        print(
            f"Skipped {mp_file_missing_count} pairs due to \
              missing MP files."
        )
    if mp_load_errors > 0:
        print(
            f"Skipped {mp_load_errors} pairs due to MP file \
              loading errors."
        )
    if ag_load_errors > 0:
        print(
            f"Skipped {ag_load_errors} pairs due to AG file \
              loading errors."
        )
    print(f"Data pair loading took {end_time - start_time:.2f} seconds.")

    if not paired_data:
        print(f"Error: No valid AG-MP JSON data pairs loaded.", file=sys.stderr)
    return paired_data


def get_element_map_from_ag_data(payloads_for_map_creation):
    all_elements = set()
    ag_processed_for_map = 0
    formula_parse_errors_map = 0
    ag_deserialize_errors_map = 0

    print(
        f"Creating element map from {len(payloads_for_map_creation)} \
            unique AG structures for map."
    )
    for i, (ag_data_dict, _, _, ag_filename_map) in enumerate(
        payloads_for_map_creation
    ):
        try:
            ag = ActionGraph.deserialize(ag_data_dict)
            found_chem_node_in_current_ag = False
            for _node_id, node_attrs in ag.nodes(data=True):
                if node_attrs.get("type") == "chemical":
                    formula = node_attrs.get("formula")
                    if formula and isinstance(formula, str):
                        try:
                            comp = Composition(formula)
                            if comp.num_atoms >= 1:
                                for el in comp.elements:
                                    all_elements.add(el.symbol)
                                found_chem_node_in_current_ag = True
                        except Exception:
                            formula_parse_errors_map += 1
            if found_chem_node_in_current_ag:
                ag_processed_for_map += 1
        except Exception:
            ag_deserialize_errors_map += 1

    if ag_deserialize_errors_map > 0:
        print(
            f"Warning: Failed to deserialize {ag_deserialize_errors_map} \
                AGs during element map creation."
        )
    if formula_parse_errors_map > 0:
        print(
            f"Warning: Could not parse {formula_parse_errors_map} chemical \
                formulas during element map creation."
        )
    print(
        f"Successfully extracted elements from {ag_processed_for_map} \
          AGs for map."
    )

    try:
        common_elements = {el.symbol for el in Element}
        all_elements.update(common_elements)
    except Exception as e:
        print(
            f"Warning: Could not retrieve all Pymatgen elements: {e}", file=sys.stderr
        )

    sorted_elements = sorted(list(all_elements))
    if not sorted_elements:
        print(
            "Error: No valid elements found from ActionGraphs for \
                element map.",
            file=sys.stderr,
        )
        return None
    return {element: i for i, element in enumerate(sorted_elements)}


def get_recipe_hash(entry):
    try:
        target_formula = entry.get("target", {}).get("material_formula", "")
        precursors = entry.get("precursors", [])
        operations = entry.get("operations", [])
        prec_formulas = sorted(
            [p.get("material_formula", "") for p in precursors if isinstance(p, dict)]
        )
        op_strings_list = []
        for op in operations:
            if isinstance(op, dict):
                try:
                    op_strings_list.append(
                        json.dumps(op, sort_keys=True, separators=(",", ":"))
                    )
                except TypeError:
                    op_strings_list.append(str(op))
            else:
                op_strings_list.append(str(op))
        op_strings_sorted = sorted(op_strings_list)
        hash_content_str = f"TARGET_FORMULA::{target_formula}||\
            PRECURSORS::{'#'.join(prec_formulas)}||\
            OPERATIONS::{'&&'.join(op_strings_sorted)}"
        return hash(hash_content_str)
    except Exception:
        fallback_str = str(entry.get("_source_file", str(entry)))
        return hash(fallback_str)


def featurize_target_formula(formula, element_map, element_props_list):
    if not formula or not isinstance(formula, str):
        return None
    if not element_map:
        return None
    try:
        comp = Composition(formula)
        if comp.num_atoms < 1:
            return None
        el_amt_dict = comp.get_el_amt_dict()
        total_atoms = comp.num_atoms

        comp_vector = np.zeros(len(element_map))
        for element_symbol, amount in el_amt_dict.items():
            if element_symbol in element_map:
                comp_vector[element_map[element_symbol]] = amount / total_atoms

        prop_vector = np.zeros(len(element_props_list))
        elements_in_comp = list(el_amt_dict.keys())

        for i, prop_name in enumerate(element_props_list):
            weighted_sum = 0.0
            total_amount_for_prop = 0.0
            prop_available = False
            for el_symbol in elements_in_comp:
                amount = el_amt_dict[el_symbol]
                try:
                    el_obj = Element(el_symbol)
                    attr_or_method = getattr(el_obj, prop_name, None)
                    prop_value = None
                    if callable(attr_or_method):
                        try:
                            prop_value = attr_or_method()
                        except Exception:
                            prop_value = None
                    else:
                        prop_value = attr_or_method

                    if prop_value is not None and isinstance(prop_value, (int, float)):
                        weighted_sum += prop_value * amount
                        total_amount_for_prop += amount
                        prop_available = True
                except Exception:
                    pass
            if prop_available and total_amount_for_prop > 0:
                prop_vector[i] = weighted_sum / total_amount_for_prop

        return np.nan_to_num(
            np.concatenate((comp_vector, prop_vector)), nan=0.0, posinf=0.0, neginf=0.0
        )
    except Exception:
        return None


def featurize_adj_matrix(ag_object, max_nodes):
    if not isinstance(ag_object, nx.DiGraph):
        return np.zeros(max_nodes * max_nodes, dtype=np.float32)
    if ag_object.number_of_nodes() == 0:
        return np.zeros(max_nodes * max_nodes, dtype=np.float32)

    try:
        nodes = sorted(list(ag_object.nodes()))
        adj = nx.to_numpy_array(ag_object, nodelist=nodes, dtype=np.float32)

        if adj.ndim != 2:
            return np.zeros(max_nodes * max_nodes, dtype=np.float32)

        padded_adj = np.zeros((max_nodes, max_nodes), dtype=np.float32)
        num_actual_nodes = adj.shape[0]
        if num_actual_nodes == 0:
            return np.zeros(max_nodes * max_nodes, dtype=np.float32)

        copy_rows = min(num_actual_nodes, max_nodes)
        copy_cols = min(adj.shape[1] if adj.ndim == 2 else 0, max_nodes)

        if copy_rows > 0 and copy_cols > 0:
            padded_adj[:copy_rows, :copy_cols] = adj[:copy_rows, :copy_cols]

        return padded_adj.flatten()
    except Exception:
        return np.zeros(max_nodes * max_nodes, dtype=np.float32)


def run_featurize_ag(pca_n_components_arg=None):
    if pca_n_components_arg is not None:
        current_pca_components = pca_n_components_arg
        print(
            f"--- Running ActionGraph Featurization with PCA \
                (Components: {current_pca_components} from arg) ---"
        )
    else:
        current_pca_components = PCA_COMPONENTS_DEFAULT
        print(
            f"--- Running ActionGraph Featurization with PCA \
                (Components: {current_pca_components} from default) ---"
        )

    os.makedirs(FEATURIZED_DATA_DIR_AG, exist_ok=True)

    loaded_pairs = load_mp_and_ag_data_pairs(AG_DATA_DIR, MP_DATA_DIR)
    if not loaded_pairs:
        print("CRITICAL ERROR: No data loaded. Exiting.")
        return False
    print(f"Loaded {len(loaded_pairs)} AG-MP pairs initially.")

    payloads_for_map_creation = []
    seen_ag_content_hashes_for_map = set()
    for _, ag_d, ag_fn in loaded_pairs:
        try:
            ag_data_str_canonical = json.dumps(
                ag_d, sort_keys=True, separators=(",", ":")
            )
        except TypeError:
            ag_data_str_canonical = str(ag_d)
        current_ag_content_hash = hash(ag_data_str_canonical)

        if current_ag_content_hash not in seen_ag_content_hashes_for_map:
            payloads_for_map_creation.append((ag_d, {}, "", ag_fn))
            seen_ag_content_hashes_for_map.add(current_ag_content_hash)

    if not payloads_for_map_creation:
        print("CRITICAL ERROR: No unique AG data for element map. Exiting.")
        return False
    element_map = get_element_map_from_ag_data(payloads_for_map_creation)
    if not element_map:
        print("CRITICAL ERROR: Element map creation failed. Exiting.")
        return False
    print(f"Element map created with {len(element_map)} elements.")

    chemical_features_list = []
    adjacency_matrices_list = []
    recipes_for_eval_list = []
    targets_list = []
    original_ag_ids_list = []
    skipped_count_total = 0

    print(
        f"\nStarting main processing loop for {len(loaded_pairs)} \
          loaded pairs..."
    )
    for i, payload_tuple in enumerate(loaded_pairs):
        if len(payload_tuple) != 3:
            skipped_count_total += 1
            continue

        mp_data, ag_data, ag_filename = payload_tuple

        try:
            ag = ActionGraph.deserialize(ag_data)
            if ag is None:
                skipped_count_total += 1
                continue
            if not ag.output_nodes:
                skipped_count_total += 1
                continue
            output_node_id = ag.output_nodes[0]
            if output_node_id not in ag.nodes or not ag.nodes[output_node_id].get(
                "formula"
            ):
                skipped_count_total += 1
                continue
            formula = ag.nodes[output_node_id]["formula"]

            chem_features = featurize_target_formula(
                formula, element_map, ELEMENT_PROPS
            )
            if chem_features is None:
                skipped_count_total += 1
                continue

            adj_features_flat = featurize_adj_matrix(ag, MAX_NODES)
            if adj_features_flat is None:
                skipped_count_total += 1
                continue

            chemical_features_list.append(chem_features)
            adjacency_matrices_list.append(adj_features_flat)
            recipes_for_eval_list.append(
                {
                    "precursors": mp_data.get("precursors", []),
                    "operations": mp_data.get("operations", []),
                }
            )
            targets_list.append(formula)
            original_ag_ids_list.append(ag_filename)
        except Exception:
            skipped_count_total += 1
            continue

    print(
        f"\nFinished main processing loop. Successfully processed: \
            {len(chemical_features_list)}"
    )
    print(f"Total skipped: {skipped_count_total}")

    if not adjacency_matrices_list:
        print("CRITICAL ERROR: No adjacency matrices collected. Exiting.")
        return False

    adj_array = np.array(adjacency_matrices_list)
    print(f"Adjacency matrices NumPy array shape: {adj_array.shape}")

    if adj_array.ndim == 1 and adj_array.size > 0:
        if len(adjacency_matrices_list) == 1:
            adj_array = adj_array.reshape(1, -1)
        else:
            print(
                f"CRITICAL ERROR: adj_array 1D. Shape: {adj_array.shape}. \
                  Exiting."
            )
            return False
    if adj_array.shape[0] == 0:
        print("CRITICAL ERROR: adj_array empty. Exiting.")
        return False
    if adj_array.ndim != 2:
        print(
            f"CRITICAL ERROR: adj_array not 2D. Shape: {adj_array.shape}. \
              Exiting."
        )
        return False

    adj_scaler = StandardScaler()
    n_pca_effective = min(
        current_pca_components, adj_array.shape[0], adj_array.shape[1]
    )

    pca = None
    reduced_adj = np.empty((adj_array.shape[0], 0))

    if n_pca_effective > 0:
        if n_pca_effective < current_pca_components:
            print(
                f"Warning: Requested PCA components \
                    ({current_pca_components}) reduced to {n_pca_effective}."
            )
        pca = PCA(n_components=n_pca_effective)
        scaled_adj = adj_scaler.fit_transform(adj_array)
        reduced_adj = pca.fit_transform(scaled_adj)
    else:
        print(
            f"Warning: PCA not performed (effective components is 0). \
                PCA features will be empty."
        )

    print(f"Adjacency matrices PCA reduced. Shape: {reduced_adj.shape}")

    final_features_list = []
    if len(chemical_features_list) != reduced_adj.shape[0]:
        print(
            f"CRITICAL ERROR: Mismatch btw chemical features \
                ({len(chemical_features_list)}) & PCA'd adj \
                    ({reduced_adj.shape[0]}). Exiting."
        )
        return False

    for chem_feat_vec, adj_feat_vec_pca in zip(chemical_features_list, reduced_adj):
        combined = np.concatenate([chem_feat_vec, adj_feat_vec_pca])
        final_features_list.append(combined)

    if not final_features_list:
        print("CRITICAL ERROR: No final features combined. Exiting.")
        return False

    final_feature_array = np.array(final_features_list)
    print(f"Final combined feature array shape: {final_feature_array.shape}")

    print(f"\nSaving artifacts to: {FEATURIZED_DATA_DIR_AG}")
    np.save(
        os.path.join(FEATURIZED_DATA_DIR_AG, "features_ag.npy"), final_feature_array
    )
    if pca is not None:
        joblib.dump(pca, os.path.join(FEATURIZED_DATA_DIR_AG, "adj_pca.joblib"))
    joblib.dump(adj_scaler, os.path.join(FEATURIZED_DATA_DIR_AG, "adj_scaler.joblib"))
    with open(os.path.join(FEATURIZED_DATA_DIR_AG, "element_map_ag.json"), "w") as f:
        json.dump(element_map, f)
    with open(
        os.path.join(FEATURIZED_DATA_DIR_AG, "recipes_mp_style_for_ag.json"), "w"
    ) as f:
        json.dump(recipes_for_eval_list, f)
    with open(
        os.path.join(FEATURIZED_DATA_DIR_AG, "targets_ag_output_formula.json"), "w"
    ) as f:
        json.dump(targets_list, f)
    with open(os.path.join(FEATURIZED_DATA_DIR_AG, "original_ag_ids.json"), "w") as f:
        json.dump(original_ag_ids_list, f)

    actual_saved_pca_components = reduced_adj.shape[1]
    print(
        f"Final feature dimension: {final_feature_array.shape[1]} \
            (Chem: {chemical_features_list[0].shape[0]} + \
                Adj_PCA: {actual_saved_pca_components})"
    )
    print(
        f"--- ActionGraph Featurization with PCA (Components: \
            {current_pca_components}) Completed ---"
    )
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Featurize \
                                     ActionGraph data with PCA."
    )
    parser.add_argument(
        "--pca_components",
        type=int,
        default=PCA_COMPONENTS_DEFAULT,
        help=f"Number of principal components for adjacency matrix \
            PCA (default: {PCA_COMPONENTS_DEFAULT}).",
    )
    args = parser.parse_args()

    if not run_featurize_ag(pca_n_components_arg=args.pca_components):
        sys.exit(1)
