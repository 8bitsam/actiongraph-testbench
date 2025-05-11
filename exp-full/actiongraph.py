# actiongraph.py

import networkx as nx
from networkx.readwrite import json_graph
from pymatgen.core import Composition
from emmet.core.synthesis import OperationTypeEnum # Ensure this import works and enum is defined
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# import sys # Not strictly needed here but often useful


class ActionGraph(nx.DiGraph):
    """
    Enhanced graph structure for materials synthesis workflows.

    Node Types:
    - 'chemical': Input/output materials with composition data.
    - 'operation': Synthesis steps with type/token/conditions.

    Edge Types:
    - 'material_flow': Connects an operation to its output chemical.
    - 'temporal': Connects an input chemical to an operation, or an operation to a subsequent operation.
    """

    def __init__(self, input_chemicals: Optional[List[Dict]] = None):
        super().__init__()
        self.input_nodes = []
        self.operation_nodes = []
        self.output_nodes = []

        if input_chemicals:
            for chem_data in input_chemicals:
                try:
                    if isinstance(chem_data, dict) and 'material_formula' in chem_data:
                        self.add_input(chem_data)
                    # else:
                        # print(f"Warning: Invalid chemical_data structure in __init__: {chem_data}. Skipping.")
                except Exception as e: # Catch errors during initial input addition
                    # print(f"Warning: Error adding initial chemical: {chem_data.get('material_formula', 'Unknown')}. Error: {e}")
                    pass # Allow __init__ to continue if one precursor fails


    def add_input(self, chemical_data: Dict, node_id: Optional[str] = None) -> str:
        """Add chemical precursor node with full material data."""
        if not isinstance(chemical_data, dict) or 'material_formula' not in chemical_data:
            raise ValueError(f"Invalid chemical_data for input node: {chemical_data}")
        if not chemical_data['material_formula']: # Check for empty formula string
             raise ValueError("Input chemical_data 'material_formula' cannot be empty.")


        if node_id is None:
            node_id = f"chem_in_{len(self.input_nodes)}"

        # Robustly get 'composition' list, then first element, then 'amount'
        comp_list = chemical_data.get('composition', [])
        first_comp_dict = comp_list[0] if comp_list and isinstance(comp_list[0], dict) else {}
        amount_raw = first_comp_dict.get('amount', 1.0) # Default to 1.0 if any key is missing
        elements_dict = first_comp_dict.get('elements', {})

        amount = self._parse_amount(amount_raw)

        try:
            composition_obj = Composition(chemical_data['material_formula'])
        except Exception as e:
            raise ValueError(f"Invalid formula for input '{chemical_data['material_formula']}': {e}")


        features = {
            'type': 'chemical',
            'formula': chemical_data['material_formula'],
            'composition': composition_obj, # Store Pymatgen Composition object
            'amount': amount,
            'elements': elements_dict # Store original elements dict from MP data
        }

        self._add_node_safe(node_id, **features)
        self.input_nodes.append(node_id)
        return node_id

    def _parse_amount(self, amount_data) -> float:
        """Helper to parse amount which can be float or dict."""
        if isinstance(amount_data, (int, float)):
            return float(amount_data)
        elif isinstance(amount_data, str): # Handle amount as string
            try:
                return float(amount_data)
            except ValueError:
                return 1.0
        elif isinstance(amount_data, dict):
            if 'value' in amount_data:
                try: return float(amount_data['value'])
                except (ValueError, TypeError): return 1.0
            elif 'values' in amount_data and amount_data['values'] and isinstance(amount_data['values'],list):
                try: return float(amount_data['values'][0]) # Take first if list
                except (ValueError, TypeError, IndexError): return 1.0
            elif 'min_value' in amount_data and 'max_value' in amount_data:
                try:
                    min_val = float(amount_data['min_value'])
                    max_val = float(amount_data['max_value'])
                    return (min_val + max_val) / 2
                except (ValueError, TypeError):
                    return 1.0
            return 1.0
        return 1.0

    def _add_node_safe(self, node_id: str, **attrs):
        """Add node, allowing attribute updates if node exists."""
        if node_id in self.nodes:
            self.nodes[node_id].update(attrs)
        else:
            self.add_node(node_id, **attrs)

    def _parse_conditions(self, cond_input: Dict) -> Dict:
        """
        Normalize condition data types from MP JSON format.
        Specifically extracts temperatures and times into flat lists of floats.
        """
        parsed_cond = {
            'temperature': [], 'time': [], 'atmosphere': [],
            'device': None, 'media': None
        }
        if not isinstance(cond_input, dict):
            return parsed_cond

        def _extract_floats_from_list_or_dict(data_entries):
            """Helper to extract all possible float values from a list of items
               where items can be numbers, strings, or dicts with value/values/min_max."""
            extracted_values = []
            if not isinstance(data_entries, list): # Ensure we iterate over a list
                return extracted_values

            for entry in data_entries:
                if isinstance(entry, (int, float)):
                    extracted_values.append(float(entry))
                elif isinstance(entry, str):
                    try: extracted_values.append(float(entry))
                    except ValueError: pass # Ignore non-numeric strings
                elif isinstance(entry, dict):
                    if 'values' in entry and isinstance(entry['values'], list):
                        for val in entry['values']:
                            try: extracted_values.append(float(val))
                            except (ValueError, TypeError): pass
                    elif 'value' in entry:
                        try: extracted_values.append(float(entry['value']))
                        except (ValueError, TypeError): pass
                    # This part specifically handles the min_value/max_value case
                    elif 'min_value' in entry and 'max_value' in entry:
                        min_v, max_v = entry.get('min_value'), entry.get('max_value')
                        try:
                            if min_v is not None: extracted_values.append(float(min_v))
                            # Append max_v only if different from min_v to represent a range, or if only max_v is present
                            if max_v is not None and max_v != min_v: extracted_values.append(float(max_v))
                            elif max_v is not None and min_v is None: extracted_values.append(float(max_v))
                        except (ValueError, TypeError): pass
            return extracted_values

        parsed_cond['temperature'] = _extract_floats_from_list_or_dict(cond_input.get('heating_temperature', []))
        parsed_cond['time'] = _extract_floats_from_list_or_dict(cond_input.get('heating_time', []))

        atm_raw = cond_input.get('heating_atmosphere', [])
        if isinstance(atm_raw, list):
            parsed_cond['atmosphere'] = [str(a) for a in atm_raw] # Ensure all are strings
        elif isinstance(atm_raw, str):
            parsed_cond['atmosphere'] = [atm_raw]
        # else: it remains []

        parsed_cond['device'] = cond_input.get('mixing_device')
        parsed_cond['media'] = cond_input.get('mixing_media')

        return parsed_cond

    def add_operation(self, op_data: Dict, source_nodes: List[str],
                      node_id: Optional[str] = None) -> str:
        """Add synthesis operation node."""
        if not isinstance(op_data, dict):
            raise ValueError(f"Invalid op_data for operation node: {op_data}")
        if not isinstance(source_nodes, list):
             raise ValueError(f"source_nodes for operation must be a list.")

        if node_id is None:
            node_id = f"op_{len(self.operation_nodes)}"

        op_type_str = op_data.get('type')
        try:
            op_type_enum = OperationTypeEnum(op_type_str)
        except ValueError:
            op_type_enum = OperationTypeEnum.unknown

        features = {
            'type': 'operation',
            'op_type': op_type_enum, # Store Enum member
            'token': op_data.get('token', 'unknown_token'),
            'conditions': self._parse_conditions(op_data.get('conditions', {}))
        }
        self._add_node_safe(node_id, **features)
        self.operation_nodes.append(node_id)
        for src in source_nodes:
            if src in self.nodes:
                self.add_edge(src, node_id, edge_type='temporal')
            # else:
                # print(f"Warning: Source node '{src}' for op '{node_id}' not found. Skipping edge.")
        return node_id

    def add_final_operation_step(self, source_nodes: List[str],
                                 step_description: str = "FinalCollectionStep",
                                 node_id: Optional[str] = None) -> str:
        """Adds a conceptual final operation step."""
        if not isinstance(source_nodes, list):
             raise ValueError("source_nodes for final operation step must be a list.")
        if node_id is None:
            node_id = f"op_final_collection_{len([n for n in self.operation_nodes if 'final_collection' in n])}"
        op_data = {
            'type': OperationTypeEnum.unknown.value, # Use .value for the string if op_data expects strings
            'token': step_description,
            'conditions': {}
        }
        return self.add_operation(op_data, source_nodes, node_id=node_id)

    def add_output(self, target_data: Dict, source_operation_node: str,
                     node_id: Optional[str] = None) -> str:
        """Add target chemical node."""
        if not isinstance(target_data, dict) or 'material_formula' not in target_data:
             raise ValueError(f"Invalid target_data for output node: {target_data}")
        if not target_data['material_formula']:
             raise ValueError("Output target_data 'material_formula' cannot be empty.")

        if source_operation_node not in self.nodes or self.nodes[source_operation_node].get('type') != 'operation':
            raise ValueError(f"Source node '{source_operation_node}' for output must be an existing operation node.")

        if node_id is None:
            node_id = f"chem_out_{len(self.output_nodes)}"

        comp_list = target_data.get('composition', [])
        first_comp_dict = comp_list[0] if comp_list and isinstance(comp_list[0], dict) else {}
        amount_raw = first_comp_dict.get('amount', 1.0)
        elements_dict = first_comp_dict.get('elements', {})
        amount = self._parse_amount(amount_raw)

        try:
            composition_obj = Composition(target_data['material_formula'])
        except Exception as e:
            raise ValueError(f"Invalid formula for output '{target_data['material_formula']}': {e}")

        features = {
            'type': 'chemical',
            'formula': target_data['material_formula'],
            'composition': composition_obj, # Store Pymatgen Composition object
            'amount': amount,
            'elements': elements_dict # Store original elements dict
        }

        self._add_node_safe(node_id, **features)
        self.output_nodes.append(node_id)
        self.add_edge(source_operation_node, node_id, edge_type='material_flow')
        return node_id

    def display(self):
        """Plot the ActionGraph with labels."""
        plt.figure(figsize=(14, 10))
        if self.number_of_nodes() == 0:
            plt.text(0.5, 0.5, "Empty Graph", ha="center", va="center"); plt.axis('off'); plt.show(); return
        try: pos = nx.nx_agraph.graphviz_layout(self, prog="dot")
        except ImportError:
            # print("Warning: pygraphviz not found, using spring_layout for display.")
            pos = nx.spring_layout(self, seed=42, k=0.8 / (self.number_of_nodes()**0.4) if self.number_of_nodes() > 0 else 0.8, iterations=50)

        edge_colors = []
        for u,v,d in self.edges(data=True):
            edge_type = d.get('edge_type', 'unknown')
            if edge_type == 'material_flow': edge_colors.append('green')
            elif edge_type == 'temporal': edge_colors.append('gray')
            else: edge_colors.append('black') # Default for unknown edge types

        nx.draw_networkx_edges(self, pos, edgelist=list(self.edges(data=True)), edge_color=edge_colors, arrows=True, arrowsize=15, arrowstyle='-|>', width=1.5, node_size=2000)
        node_colors_map = {'input': '#1f77b4', 'output': '#ff7f0e', 'operation_std': '#d62728', 'operation_final': '#2ca02c', 'unknown': '#9467bd'}
        node_colors_list = []
        node_ids_list = list(self.nodes()) # Get a list for consistent ordering

        for node_id in node_ids_list:
            attrs = self.nodes[node_id]
            node_main_type = attrs.get('type')
            if node_id in self.input_nodes: node_colors_list.append(node_colors_map['input'])
            elif node_id in self.output_nodes: node_colors_list.append(node_colors_map['output'])
            elif node_main_type == 'operation':
                if 'final_collection' in node_id.lower() or attrs.get('token', '').lower() == "finalcollectionstep": node_colors_list.append(node_colors_map['operation_final'])
                else: node_colors_list.append(node_colors_map['operation_std'])
            else: node_colors_list.append(node_colors_map['unknown'])

        nx.draw_networkx_nodes(self, pos, nodelist=node_ids_list, node_color=node_colors_list, node_size=2000, edgecolors='black', linewidths=1.5)
        labels = {}
        for node_id in node_ids_list:
            attrs = self.nodes[node_id]
            short_id = node_id.split('_')[-1] # Get part after last underscore for brevity
            if attrs.get('type') == 'chemical': labels[node_id] = f"{short_id}\n{attrs.get('formula', '?')}"
            elif attrs.get('type') == 'operation':
                op_type_val = attrs.get('op_type')
                op_type_str = op_type_val.name if isinstance(op_type_val, OperationTypeEnum) else str(op_type_val) # Use .name for Enum
                labels[node_id] = f"{short_id}\n{op_type_str}"
            else: labels[node_id] = f"{short_id}\n(Unk)"
        nx.draw_networkx_labels(self, pos, labels=labels, font_size=7, font_weight='normal', font_color='black')
        legend_elements = [mpatches.Patch(color=node_colors_map['input'], label='Input Chemical'), mpatches.Patch(color=node_colors_map['operation_std'], label='Operation'), mpatches.Patch(color=node_colors_map['operation_final'], label='Final/Coll. Op'), mpatches.Patch(color=node_colors_map['output'], label='Output Chemical')]
        plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=max(1, len(legend_elements)//2), fontsize=9)
        plt.title("Action Graph Visualization", fontsize=16); plt.axis('off'); plt.tight_layout(rect=[0, 0.05, 1, 1]); plt.show()


    def serialize(self) -> Dict:
        """Convert an ActionGraph into a node-link dictionary format."""
        temp_graph = nx.DiGraph()
        temp_graph.add_nodes_from((n, d.copy()) for n, d in self.nodes(data=True))
        temp_graph.add_edges_from(self.edges(data=True))

        for node_id, attrs in temp_graph.nodes(data=True):
            if 'composition' in attrs and isinstance(attrs['composition'], Composition):
                attrs['composition'] = attrs['composition'].as_dict()
            if 'op_type' in attrs and isinstance(attrs['op_type'], OperationTypeEnum):
                attrs['op_type'] = attrs['op_type'].value # Serialize enum as its string value

        data = json_graph.node_link_data(temp_graph)
        data['custom_input_nodes'] = self.input_nodes
        data['custom_operation_nodes'] = self.operation_nodes
        data['custom_output_nodes'] = self.output_nodes
        return data

    @classmethod
    def deserialize(cls, data: Dict) -> 'ActionGraph':
        """Reconstruct an ActionGraph from a dictionary."""
        try:
            edge_key = 'links' if 'links' in data else 'edges'
            graph_data_for_nx = {'nodes': data.get('nodes', []), edge_key: data.get(edge_key, []), 'directed': True, 'multigraph': False, 'graph': data.get('graph', {})}
            reconstructed_graph = json_graph.node_link_graph(graph_data_for_nx, directed=True, multigraph=False)
        except Exception as e:
            # print(f"Warning: node_link_graph failed during deserialization: {e}. Falling back to manual.", file=sys.stderr)
            reconstructed_graph = nx.DiGraph(); reconstructed_graph.graph.update(data.get('graph', {}))
            node_id_map = {}
            for node_data_item in data.get('nodes', []):
                node_id = str(node_data_item['id']); node_id_map[node_data_item['id']] = node_id
                attrs = {k: v for k, v in node_data_item.items() if k != 'id'}; reconstructed_graph.add_node(node_id, **attrs)
            for link_item in data.get('links', []): # Assuming 'links' for fallback
                source_orig, target_orig = link_item.get('source'), link_item.get('target')
                source, target = node_id_map.get(source_orig, str(source_orig)), node_id_map.get(target_orig, str(target_orig))
                edge_attrs = {k: v for k, v in link_item.items() if k not in ['source', 'target', 'key']}; reconstructed_graph.add_edge(source, target, **edge_attrs)

        ag = cls(); ag.add_nodes_from(reconstructed_graph.nodes(data=True)); ag.add_edges_from(reconstructed_graph.edges(data=True)); ag.graph.update(reconstructed_graph.graph)

        for node_id, attrs in ag.nodes(data=True):
            if 'composition' in attrs and isinstance(attrs['composition'], dict):
                try: attrs['composition'] = Composition.from_dict(attrs['composition'])
                except Exception: pass
            if 'op_type' in attrs and isinstance(attrs['op_type'], str):
                try: attrs['op_type'] = OperationTypeEnum(attrs['op_type'])
                except ValueError: attrs['op_type'] = OperationTypeEnum.unknown

        ag.input_nodes = data.get('custom_input_nodes', [])
        ag.operation_nodes = data.get('custom_operation_nodes', [])
        ag.output_nodes = data.get('custom_output_nodes', [])

        # Validation/Recovery logic for node lists (more robust)
        final_inputs, final_ops, final_outputs = [], [], []
        all_node_ids = set(ag.nodes()) # For quick check

        # Populate from custom lists if they exist and nodes are valid
        for node_id in ag.input_nodes:
            if node_id in all_node_ids and ag.nodes[node_id].get('type') == 'chemical': final_inputs.append(node_id)
        for node_id in ag.operation_nodes:
            if node_id in all_node_ids and ag.nodes[node_id].get('type') == 'operation': final_ops.append(node_id)
        for node_id in ag.output_nodes:
            if node_id in all_node_ids and ag.nodes[node_id].get('type') == 'chemical': final_outputs.append(node_id)

        # If lists are still empty, try to infer (especially for older serialized formats)
        if not final_inputs and not final_outputs and not final_ops:
            for node_id, attrs in ag.nodes(data=True):
                node_type = attrs.get('type')
                if node_type == 'chemical':
                    is_output = any(ag.nodes[u].get('type') == 'operation' and ed.get('edge_type') == 'material_flow' for u,v,ed in ag.in_edges(node_id, data=True) if u in all_node_ids)
                    is_input = ag.in_degree(node_id) == 0
                    if is_output : final_outputs.append(node_id)
                    elif is_input: final_inputs.append(node_id)
                elif node_type == 'operation': final_ops.append(node_id)

        ag.input_nodes = sorted(list(set(final_inputs)))
        ag.operation_nodes = sorted(list(set(final_ops)))
        ag.output_nodes = sorted(list(set(final_outputs)))
        return ag


    @classmethod
    def from_mp_synthesis(cls, mp_data: dict) -> Union["ActionGraph", None]:
        """
        Construct ActionGraph from Materials Project synthesis data.
        Returns None if essential data is missing, empty, or causes construction errors.
        """
        required_keys = ["precursors", "operations", "target"]
        for key in required_keys:
            if key not in mp_data or not mp_data[key]:
                return None # Skip if key missing or value is "falsy" (empty list/dict)

        try:
            # Validate main structures are lists/dicts as expected
            if not isinstance(mp_data["precursors"], list) or \
               not isinstance(mp_data["operations"], list) or \
               not isinstance(mp_data["target"], dict):
                return None

            # Validate content of precursors and target before even creating AG instance
            if not all(isinstance(p, dict) and 'material_formula' in p and p['material_formula'] for p in mp_data["precursors"]):
                return None
            if 'material_formula' not in mp_data["target"] or not mp_data["target"]['material_formula']:
                return None

            ag = cls(input_chemicals=mp_data["precursors"]) # May raise ValueError if a precursor is invalid
            current_source_nodes = ag.input_nodes[:]

            if not current_source_nodes and mp_data["precursors"]: # All precursors failed to add
                 return None

            # Validate operations list content
            if not all(isinstance(op, dict) and 'type' in op for op in mp_data["operations"]): # Basic check
                return None

            for op_idx, operation_data in enumerate(mp_data["operations"]):
                # Allow first operation to not have chemical sources (e.g. setup step)
                # But subsequent operations in a linear chain usually depend on previous
                if not current_source_nodes and op_idx > 0 and ag.input_nodes :
                    # This implies a broken chain if inputs existed but were not connected to ops
                    # print(f"Warning: Operation op_{op_idx} has no sources but inputs existed. Potential break in chain.")
                    break # Stop processing further operations if flow seems broken

                op_id = ag.add_operation(
                    op_data=operation_data,
                    source_nodes=current_source_nodes if current_source_nodes else [],
                    node_id=f"op_{op_idx}"
                )
                current_source_nodes = [op_id] # Assume linear flow for next step

            if not current_source_nodes: # No operations added, or all failed
                 if ag.input_nodes: # If we had inputs, make them source for output via a final step
                      final_op_id = ag.add_final_operation_step(source_nodes=ag.input_nodes)
                      current_source_nodes = [final_op_id]
                 else: # No inputs and no operations, cannot form a meaningful graph for this task
                      return None
            last_operation_node_id = current_source_nodes[0]

            ag.add_output(
                target_data=mp_data["target"],
                source_operation_node=last_operation_node_id
            )
            return ag
        except ValueError as ve: # Catch ValueErrors from add_input/op/output
             # print(f"Warning: ValueError during AG construction: {ve}. Skipping.")
             return None
        except Exception as e: # Catch other unexpected errors
             # print(f"Warning: Unexpected error during AG construction: {e}. Skipping.")
             return None
