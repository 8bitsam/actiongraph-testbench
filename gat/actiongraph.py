# actiongraph.py

import networkx as nx
from networkx.readwrite import json_graph
from pymatgen.core import Composition
from emmet.core.synthesis import OperationTypeEnum
from typing import Dict, List, Optional, Union, Any
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json

class ActionGraph(nx.DiGraph):
    """
    Enhanced graph structure for materials synthesis workflows.

    Node Types:
    - 'chemical': Input/output materials with composition data.
    - 'operation': Synthesis steps with type/token/conditions.

    Edge Types:
    - 'material_flow': Connects an operation to its output chemical.
    - 'temporal': Connects an input chemical to an operation,
                  or an operation to a subsequent operation.
    """

    def __init__(self, input_chemicals: Optional[List[Dict]] = None):
        super().__init__()
        self.input_nodes: List[str] = []
        self.operation_nodes: List[str] = []
        self.output_nodes: List[str] = []

        if input_chemicals:
            for chem_data in input_chemicals:
                try:
                    if isinstance(chem_data, dict) and 'material_formula' in chem_data:
                        self.add_input(chem_data)
                    # else:
                    #     print(f"Warning: Invalid chemical_data structure in __init__: {chem_data}. Skipping.")
                except ValueError as ve: # Catch specific ValueErrors from add_input
                    # print(f"Warning: Error adding initial chemical {chem_data.get('material_formula', 'Unknown')}: {ve}")
                    pass # Continue if one precursor is problematic
                except Exception as e:
                    # print(f"Warning: Unexpected error adding initial chemical: {chem_data.get('material_formula', 'Unknown')}. Error: {e}")
                    pass

    def _parse_amount(self, amount_data: Any) -> float:
        """Helper to parse amount which can be float, str, or dict."""
        if isinstance(amount_data, (int, float)):
            return float(amount_data)
        elif isinstance(amount_data, str):
            try:
                return float(amount_data)
            except ValueError:
                return 1.0 # Default if string is not a valid float
        elif isinstance(amount_data, dict):
            if 'value' in amount_data:
                try: return float(amount_data['value'])
                except (ValueError, TypeError): return 1.0
            elif 'values' in amount_data and isinstance(amount_data['values'], list) and amount_data['values']:
                try: return float(amount_data['values'][0])
                except (ValueError, TypeError, IndexError): return 1.0
            elif 'min_value' in amount_data and 'max_value' in amount_data:
                try:
                    min_val = float(amount_data['min_value'])
                    max_val = float(amount_data['max_value'])
                    return (min_val + max_val) / 2.0
                except (ValueError, TypeError): return 1.0
            return 1.0 # Fallback for other dict structures
        return 1.0 # Default for other unexpected types

    def _add_node_safe(self, node_id: str, **attrs):
        """Add node, allowing attribute updates if node exists."""
        if node_id in self.nodes:
            # print(f"Info: Node {node_id} already exists. Updating attributes.")
            self.nodes[node_id].update(attrs) # Update existing node's attributes
        else:
            self.add_node(node_id, **attrs)

    def add_input(self, chemical_data: Dict, node_id: Optional[str] = None) -> str:
        """Add chemical precursor node with full material data."""
        if not isinstance(chemical_data, dict) or 'material_formula' not in chemical_data:
            raise ValueError(f"Invalid chemical_data for input node: Must be dict with 'material_formula'. Got: {chemical_data}")
        if not chemical_data['material_formula'] or not isinstance(chemical_data['material_formula'], str):
             raise ValueError("Input chemical_data 'material_formula' cannot be empty or non-string.")

        if node_id is None:
            node_id = f"chem_in_{len(self.input_nodes)}"

        comp_list = chemical_data.get('composition', [])
        # Ensure comp_list is a list and first_comp_dict is a dict before accessing
        first_comp_dict = {}
        if isinstance(comp_list, list) and comp_list:
            if isinstance(comp_list[0], dict):
                first_comp_dict = comp_list[0]

        amount_raw = first_comp_dict.get('amount', 1.0)
        elements_dict = first_comp_dict.get('elements', {}) # Original string-based counts from MP
        amount = self._parse_amount(amount_raw)

        try:
            composition_obj = Composition(chemical_data['material_formula'])
        except Exception as e:
            raise ValueError(f"Invalid Pymatgen formula for input '{chemical_data['material_formula']}': {e}")

        features = {
            'type': 'chemical',
            'formula': chemical_data['material_formula'],
            'composition': composition_obj, # Store Pymatgen Composition object
            'amount': amount,
            'elements': elements_dict
        }
        self._add_node_safe(node_id, **features)
        if node_id not in self.input_nodes: # Avoid duplicates if node updated
            self.input_nodes.append(node_id)
        return node_id

    def _parse_single_condition_value_entry(self, entry: Any) -> List[float]:
        """
        Parses a single entry from a condition list (e.g., one item from heating_temperature list).
        An entry can be a number, a string convertible to number, or a dict.
        Returns a list of extracted float values.
        """
        values = []
        if isinstance(entry, (int, float)):
            values.append(float(entry))
        elif isinstance(entry, str):
            try: values.append(float(entry))
            except ValueError: pass # Ignore non-numeric strings
        elif isinstance(entry, dict):
            if 'values' in entry and isinstance(entry['values'], list) and entry['values']:
                for val_item in entry['values']:
                    try: values.append(float(val_item))
                    except (ValueError, TypeError): pass
            elif 'value' in entry:
                try: values.append(float(entry['value']))
                except (ValueError, TypeError): pass
            elif 'min_value' in entry or 'max_value' in entry: # Check if either key exists
                min_v_raw, max_v_raw = entry.get('min_value'), entry.get('max_value')
                min_v, max_v = None, None
                try:
                    if min_v_raw is not None: min_v = float(min_v_raw)
                    if max_v_raw is not None: max_v = float(max_v_raw)

                    if min_v is not None and max_v is not None:
                        values.append(min_v)
                        if min_v != max_v: values.append(max_v) # Add max only if different
                    elif min_v is not None: values.append(min_v)
                    elif max_v is not None: values.append(max_v)
                except (ValueError, TypeError): pass
        return values

    def _parse_conditions(self, cond_input: Dict) -> Dict:
        """
        Normalize condition data types from MP JSON format.
        Correctly extracts temperatures and times, including min/max ranges.
        Resulting temperature/time are flat lists of floats.
        """
        parsed_cond = {
            'temperature_K': [], 'time_s': [], 'atmosphere': [],
            'device': None, 'media': None, 'other_conditions': {}
        }
        if not isinstance(cond_input, dict): return parsed_cond

        raw_temps_list = cond_input.get('heating_temperature', [])
        if isinstance(raw_temps_list, list):
            for t_entry_item in raw_temps_list:
                parsed_cond['temperature_K'].extend(self._parse_single_condition_value_entry(t_entry_item))
        # Ensure unique, sorted for consistency in serialized output
        if parsed_cond['temperature_K']:
             parsed_cond['temperature_K'] = sorted(list(set(parsed_cond['temperature_K'])))


        raw_times_list = cond_input.get('heating_time', [])
        if isinstance(raw_times_list, list):
            for t_entry_item in raw_times_list:
                # TODO: Implement unit conversion to seconds if 'units' field is present and varies
                parsed_cond['time_s'].extend(self._parse_single_condition_value_entry(t_entry_item))
        if parsed_cond['time_s']:
             parsed_cond['time_s'] = sorted(list(set(parsed_cond['time_s'])))

        atm_raw = cond_input.get('heating_atmosphere', [])
        if isinstance(atm_raw, list):
            parsed_cond['atmosphere'] = [str(a) for a in atm_raw if isinstance(a, (str, int, float, bool)) or a is None]
        elif isinstance(atm_raw, (str, int, float, bool)) or atm_raw is None:
            parsed_cond['atmosphere'] = [str(atm_raw)]

        parsed_cond['device'] = cond_input.get('mixing_device')
        parsed_cond['media'] = cond_input.get('mixing_media')

        # Store any other conditions that were not explicitly parsed
        for key, value in cond_input.items():
            if key not in ['heating_temperature', 'heating_time', 'heating_atmosphere', 'mixing_device', 'mixing_media']:
                # Ensure value is JSON serializable or convert
                if isinstance(value, (list, dict, str, int, float, bool)) or value is None:
                    parsed_cond['other_conditions'][key] = value
                else:
                    parsed_cond['other_conditions'][key] = str(value)
        return parsed_cond

    def add_operation(self, op_data: Dict, source_nodes: List[str],
                      node_id: Optional[str] = None) -> str:
        if not isinstance(op_data, dict): raise ValueError(f"Invalid op_data for operation: {op_data}")
        if not isinstance(source_nodes, list): raise ValueError("source_nodes must be a list.")
        if node_id is None: node_id = f"op_{len(self.operation_nodes)}"

        op_type_str = op_data.get('type')
        try: op_type_enum = OperationTypeEnum(op_type_str)
        except ValueError: op_type_enum = OperationTypeEnum.unknown

        conditions_from_op_data = op_data.get('conditions', {})
        parsed_conditions = self._parse_conditions(conditions_from_op_data)

        features = {'type': 'operation', 'op_type': op_type_enum,
                    'token': op_data.get('token', 'unknown_token'),
                    'conditions': parsed_conditions}
        self._add_node_safe(node_id, **features)
        if node_id not in self.operation_nodes:
            self.operation_nodes.append(node_id)
        for src in source_nodes:
            if src in self.nodes: self.add_edge(src, node_id, edge_type='temporal')
        return node_id

    def add_final_operation_step(self, source_nodes: List[str],
                                 step_description: str = "FinalCollectionStep",
                                 node_id: Optional[str] = None) -> str:
        if not isinstance(source_nodes, list): raise ValueError("source_nodes must be a list.")
        if node_id is None:
            node_id = f"op_final_collection_{len([n for n in self.operation_nodes if 'final_collection' in n])}"
        op_data = {'type': OperationTypeEnum.unknown.value, # Use .value for string
                   'token': step_description, 'conditions': {}}
        return self.add_operation(op_data, source_nodes, node_id=node_id)

    def add_output(self, target_data: Dict, source_operation_node: str,
                     node_id: Optional[str] = None) -> str:
        if not isinstance(target_data, dict) or 'material_formula' not in target_data:
             raise ValueError(f"Invalid target_data for output: {target_data}")
        if not target_data['material_formula'] or not isinstance(target_data['material_formula'], str):
             raise ValueError("Output 'material_formula' cannot be empty or non-string.")
        if source_operation_node not in self.nodes or self.nodes[source_operation_node].get('type') != 'operation':
            raise ValueError(f"Source '{source_operation_node}' for output must be existing op node.")
        if node_id is None: node_id = f"chem_out_{len(self.output_nodes)}"

        comp_list = target_data.get('composition', [])
        first_comp_dict = {}
        if isinstance(comp_list, list) and comp_list:
             if isinstance(comp_list[0], dict): first_comp_dict = comp_list[0]
        amount_raw = first_comp_dict.get('amount', 1.0)
        elements_dict = first_comp_dict.get('elements', {})
        amount = self._parse_amount(amount_raw)
        try: composition_obj = Composition(target_data['material_formula'])
        except Exception as e: raise ValueError(f"Invalid Pymatgen formula for output '{target_data['material_formula']}': {e}")

        features = {'type': 'chemical', 'formula': target_data['material_formula'],
                    'composition': composition_obj, 'amount': amount, 'elements': elements_dict}
        self._add_node_safe(node_id, **features)
        if node_id not in self.output_nodes:
            self.output_nodes.append(node_id)
        self.add_edge(source_operation_node, node_id, edge_type='material_flow')
        return node_id

    def display(self):
        plt.figure(figsize=(14, 10))
        if self.number_of_nodes() == 0:
            plt.text(0.5, 0.5, "Empty Graph", ha="center", va="center"); plt.axis('off'); plt.show(); return
        try: pos = nx.nx_agraph.graphviz_layout(self, prog="dot")
        except ImportError:
            # print("Warning: pygraphviz not found, using spring_layout for display.")
            pos = nx.spring_layout(self, seed=42, k=0.8 / (self.number_of_nodes()**0.4) if self.number_of_nodes() > 0 else 0.8, iterations=50)

        edge_colors_map = {'material_flow': 'green', 'temporal': 'gray', 'unknown': 'black'}
        edge_color_list = [edge_colors_map.get(d.get('edge_type', 'unknown'), 'black') for u,v,d in self.edges(data=True)]
        nx.draw_networkx_edges(self, pos, edgelist=list(self.edges(data=True)), edge_color=edge_color_list, arrows=True, arrowsize=15, arrowstyle='-|>', width=1.5, node_size=2000)

        node_colors_map_dict = {'input': '#1f77b4', 'output': '#ff7f0e', 'operation_std': '#d62728', 'operation_final': '#2ca02c', 'unknown': '#9467bd'}
        node_colors_list_display = []
        node_ids_list_display = list(self.nodes())

        for node_id_disp in node_ids_list_display:
            attrs_disp = self.nodes[node_id_disp]
            node_main_type_disp = attrs_disp.get('type')
            if node_id_disp in self.input_nodes: node_colors_list_display.append(node_colors_map_dict['input'])
            elif node_id_disp in self.output_nodes: node_colors_list_display.append(node_colors_map_dict['output'])
            elif node_main_type_disp == 'operation':
                if 'final_collection' in node_id_disp.lower() or attrs_disp.get('token', '').lower() == "finalcollectionstep": node_colors_list_display.append(node_colors_map_dict['operation_final'])
                else: node_colors_list_display.append(node_colors_map_dict['operation_std'])
            else: node_colors_list_display.append(node_colors_map_dict['unknown'])

        nx.draw_networkx_nodes(self, pos, nodelist=node_ids_list_display, node_color=node_colors_list_display, node_size=2000, edgecolors='black', linewidths=1.5)
        labels_disp = {}
        for node_id_disp in node_ids_list_display:
            attrs_disp = self.nodes[node_id_disp]
            short_id_disp = node_id_disp.split('_')[-1] if '_' in node_id_disp else node_id_disp
            if attrs_disp.get('type') == 'chemical': labels_disp[node_id_disp] = f"{short_id_disp}\n{attrs_disp.get('formula', '?')}"
            elif attrs_disp.get('type') == 'operation':
                op_type_val_disp = attrs_disp.get('op_type')
                op_type_str_disp = op_type_val_disp.name if isinstance(op_type_val_disp, OperationTypeEnum) else str(op_type_val_disp)
                labels_disp[node_id_disp] = f"{short_id_disp}\n{op_type_str_disp}"
            else: labels_disp[node_id_disp] = f"{short_id_disp}\n(Unk)"
        nx.draw_networkx_labels(self, pos, labels=labels_disp, font_size=7, font_weight='normal', font_color='black')
        legend_elements_disp = [mpatches.Patch(color=node_colors_map_dict['input'], label='Input Chemical'), mpatches.Patch(color=node_colors_map_dict['operation_std'], label='Operation'), mpatches.Patch(color=node_colors_map_dict['operation_final'], label='Final/Coll. Op'), mpatches.Patch(color=node_colors_map_dict['output'], label='Output Chemical')]
        plt.legend(handles=legend_elements_disp, loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=max(1, len(legend_elements_disp)//2), fontsize=9)
        plt.title("Action Graph Visualization", fontsize=16); plt.axis('off'); plt.tight_layout(rect=[0, 0.05, 1, 1]); plt.show()

    def serialize(self) -> Dict:
        temp_graph = nx.DiGraph()
        # Deep copy node attributes to avoid modifying the live graph
        for n, d in self.nodes(data=True):
            temp_graph.add_node(n, **json.loads(json.dumps(d, default=str))) # Force basic types

        # Convert Pymatgen and Enum objects in the copied attributes
        for node_id, attrs in temp_graph.nodes(data=True):
            if 'composition' in attrs and isinstance(attrs['composition'], dict): # It would have been made dict by json cycle
                # This assumes it's already a Pymatgen-like dict if it came from a Composition object
                pass # Already a serializable dict if Pymatgen's as_dict() was used
            elif 'composition' in attrs and isinstance(attrs['composition'], Composition): # Should not happen if using deepcopy trick
                 attrs['composition'] = attrs['composition'].as_dict()

            if 'op_type' in attrs and isinstance(attrs['op_type'], OperationTypeEnum):
                attrs['op_type'] = attrs['op_type'].value
            # Ensure conditions dict itself is serializable (values should be lists of floats/strings)
            if 'conditions' in attrs and isinstance(attrs['conditions'], dict):
                 attrs['conditions'] = json.loads(json.dumps(attrs['conditions'],default=str))


        temp_graph.add_edges_from(self.edges(data=True))

        data = json_graph.node_link_data(temp_graph)
        data['custom_input_nodes'] = self.input_nodes
        data['custom_operation_nodes'] = self.operation_nodes
        data['custom_output_nodes'] = self.output_nodes
        return data

    @classmethod
    def deserialize(cls, data: Dict) -> 'ActionGraph':
        try:
            edge_key = 'links' if 'links' in data else ('edges' if 'edges' in data else None)
            if edge_key is None: raise KeyError("Missing 'links' or 'edges' key for graph edges.")
            graph_data_for_nx = {'nodes': data.get('nodes', []), edge_key: data.get(edge_key, []), 'directed': True, 'multigraph': False, 'graph': data.get('graph', {})}
            reconstructed_graph = json_graph.node_link_graph(graph_data_for_nx)
        except Exception as e:
            reconstructed_graph = nx.DiGraph(); reconstructed_graph.graph.update(data.get('graph', {}))
            node_id_map = {}
            for node_data_item in data.get('nodes', []):
                node_id = str(node_data_item['id']); node_id_map[node_data_item['id']] = node_id
                attrs = {k: v for k, v in node_data_item.items() if k != 'id'}; reconstructed_graph.add_node(node_id, **attrs)
            links_data = data.get('links', data.get('edges', [])) # Try both keys
            for link_item in links_data:
                source_orig, target_orig = link_item.get('source'), link_item.get('target')
                source, target = node_id_map.get(source_orig, str(source_orig)), node_id_map.get(target_orig, str(target_orig))
                edge_attrs = {k: v for k, v in link_item.items() if k not in ['source', 'target', 'key']}; reconstructed_graph.add_edge(source, target, **edge_attrs)

        ag = cls(); ag.add_nodes_from(reconstructed_graph.nodes(data=True)); ag.add_edges_from(reconstructed_graph.edges(data=True)); ag.graph.update(reconstructed_graph.graph)
        for node_id, attrs in ag.nodes(data=True):
            if attrs.get('type') == 'chemical' and 'composition' in attrs and isinstance(attrs['composition'], dict):
                try: attrs['composition'] = Composition.from_dict(attrs['composition'])
                except Exception: pass
            if attrs.get('type') == 'operation' and 'op_type' in attrs and isinstance(attrs['op_type'], str):
                try: attrs['op_type'] = OperationTypeEnum(attrs['op_type'])
                except ValueError: attrs['op_type'] = OperationTypeEnum.unknown

        ag.input_nodes = data.get('custom_input_nodes', [])
        ag.operation_nodes = data.get('custom_operation_nodes', [])
        ag.output_nodes = data.get('custom_output_nodes', [])
        # Validation/Recovery logic for node lists
        final_inputs, final_ops, final_outputs = [], [], []
        all_node_ids_set = set(ag.nodes())
        for node_id in ag.input_nodes: # Trust custom lists if present and valid
            if node_id in all_node_ids_set and ag.nodes[node_id].get('type') == 'chemical': final_inputs.append(node_id)
        for node_id in ag.operation_nodes:
            if node_id in all_node_ids_set and ag.nodes[node_id].get('type') == 'operation': final_ops.append(node_id)
        for node_id in ag.output_nodes:
            if node_id in all_node_ids_set and ag.nodes[node_id].get('type') == 'chemical': final_outputs.append(node_id)

        # Infer if custom lists were empty or incomplete
        if not (data.get('custom_input_nodes') or data.get('custom_operation_nodes') or data.get('custom_output_nodes')):
            for node_id, attrs in ag.nodes(data=True):
                node_type = attrs.get('type')
                if node_type == 'chemical':
                    is_output_by_edge = any(ag.nodes[u].get('type') == 'operation' and ed.get('edge_type') == 'material_flow' for u,v,ed in ag.in_edges(node_id, data=True) if u in all_node_ids_set)
                    is_input_by_degree = ag.in_degree(node_id) == 0
                    if is_output_by_edge and node_id not in final_outputs : final_outputs.append(node_id)
                    elif is_input_by_degree and node_id not in final_inputs and node_id not in final_outputs: final_inputs.append(node_id) # Avoid classifying an output as input
                elif node_type == 'operation' and node_id not in final_ops: final_ops.append(node_id)
        ag.input_nodes = sorted(list(set(final_inputs))); ag.operation_nodes = sorted(list(set(final_ops))); ag.output_nodes = sorted(list(set(final_outputs)))
        return ag

    @classmethod
    def from_mp_synthesis(cls, mp_data: dict) -> Union["ActionGraph", None]:
        required_keys = ["precursors", "operations", "target"]
        for key in required_keys:
            if key not in mp_data or not mp_data[key]: return None
        try:
            if not isinstance(mp_data["precursors"], list) or \
               not all(isinstance(p, dict) and 'material_formula' in p and p.get('material_formula') for p in mp_data["precursors"]):
                return None
            ag = cls(input_chemicals=mp_data["precursors"])
            current_source_nodes = ag.input_nodes[:]
            if not current_source_nodes and mp_data["precursors"]: return None
            if not isinstance(mp_data["operations"], list) or \
               not all(isinstance(op, dict) and 'type' in op for op in mp_data["operations"]):
                return None

            for op_idx, operation_data in enumerate(mp_data["operations"]):
                if not current_source_nodes and op_idx > 0 and ag.input_nodes : break
                op_id = ag.add_operation(op_data=operation_data, source_nodes=current_source_nodes if current_source_nodes else [], node_id=f"op_{op_idx}")
                current_source_nodes = [op_id]

            if not current_source_nodes:
                 if ag.input_nodes:
                      final_op_id = ag.add_final_operation_step(source_nodes=ag.input_nodes)
                      current_source_nodes = [final_op_id]
                 else: return None
            last_operation_node_id = current_source_nodes[0]

            if not isinstance(mp_data["target"], dict) or 'material_formula' not in mp_data["target"] or not mp_data["target"].get("material_formula"):
                 return None
            ag.add_output(target_data=mp_data["target"], source_operation_node=last_operation_node_id)
            return ag
        except ValueError: return None
        except Exception: return None
