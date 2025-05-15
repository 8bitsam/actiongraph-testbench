import json
from typing import Any, Dict, List, Optional, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
from emmet.core.synthesis import OperationTypeEnum
from networkx.readwrite import json_graph
from pymatgen.core import Composition

# Display settings
ACTIONGRAPH_PLOT_STYLE = "seaborn-v0_8-whitegrid"
ACTIONGRAPH_FONT_SIZE_SMALL = 12
ACTIONGRAPH_FONT_SIZE_MEDIUM = 16
ACTIONGRAPH_NODE_SIZE = 2400
ACTIONGRAPH_EDGE_WIDTH = 2.5
ACTIONGRAPH_ARROW_SIZE = 6
ACTIONGRAPH_EDGE_COLOR = "dimgray"


class ActionGraph(nx.DiGraph):
    def __init__(self, input_chemicals: Optional[List[Dict]] = None):
        super().__init__()
        self.input_nodes: List[str] = []
        self.operation_nodes: List[str] = []
        self.output_nodes: List[str] = []

        if input_chemicals:
            for chem_data in input_chemicals:
                try:
                    if isinstance(chem_data, dict) and "material_formula" in chem_data:
                        self.add_input(chem_data)
                except ValueError:
                    pass
                except Exception:
                    pass

    def _parse_amount(self, amount_data: Any) -> float:
        if isinstance(amount_data, (int, float)):
            return float(amount_data)
        elif isinstance(amount_data, str):
            try:
                return float(amount_data)
            except ValueError:
                return 1.0
        elif isinstance(amount_data, dict):
            if "value" in amount_data:
                try:
                    return float(amount_data["value"])
                except (ValueError, TypeError):
                    return 1.0
            if (
                "values" in amount_data
                and isinstance(amount_data["values"], list)
                and amount_data["values"]
            ):
                try:
                    return float(amount_data["values"][0])
                except (ValueError, TypeError, IndexError):
                    return 1.0
            if "min_value" in amount_data and "max_value" in amount_data:
                try:
                    min_val, max_val = float(amount_data["min_value"]), float(
                        amount_data["max_value"]
                    )
                    return (min_val + max_val) / 2.0
                except (ValueError, TypeError):
                    return 1.0
            return 1.0
        return 1.0

    def _add_node_safe(self, node_id: str, **attrs):
        if node_id in self.nodes:
            self.nodes[node_id].update(attrs)
        else:
            self.add_node(node_id, **attrs)

    def add_input(self, chemical_data: Dict, node_id: Optional[str] = None) -> str:
        if not (
            isinstance(chemical_data, dict)
            and "material_formula" in chemical_data
            and chemical_data["material_formula"]
            and isinstance(chemical_data["material_formula"], str)
        ):
            raise ValueError("Invalid chemical_data for input node.")
        if node_id is None:
            node_id = f"chem_in_{len(self.input_nodes)}"
        try:
            composition_obj = Composition(chemical_data["material_formula"])
        except Exception as e:
            raise ValueError(f"Invalid Pymatgen formula: {e}")
        amount, elements_data = 1.0, {}
        if (
            "composition" in chemical_data
            and isinstance(chemical_data["composition"], list)
            and chemical_data["composition"]
        ):
            first_comp = chemical_data["composition"][0]
            if isinstance(first_comp, dict):
                amount = self._parse_amount(first_comp.get("amount", 1.0))
                elements_data = first_comp.get("elements", {})
        features = {
            "type": "chemical",
            "formula": chemical_data["material_formula"],
            "composition": composition_obj,
            "amount": amount,
            "elements": elements_data,
        }
        self._add_node_safe(node_id, **features)
        if node_id not in self.input_nodes:
            self.input_nodes.append(node_id)
        return node_id

    def _parse_single_condition_value_entry(self, entry: Any) -> List[float]:
        values = []
        if isinstance(entry, (int, float)):
            values.append(float(entry))
        elif isinstance(entry, str):
            try:
                values.append(float(entry))
            except ValueError:
                pass
        elif isinstance(entry, dict):
            if "values" in entry and isinstance(entry["values"], list):
                for val_item in entry["values"]:
                    try:
                        values.append(float(val_item))
                    except (ValueError, TypeError):
                        pass
            elif "value" in entry:
                try:
                    values.append(float(entry["value"]))
                except (ValueError, TypeError):
                    pass
            elif "min_value" in entry and "max_value" in entry:
                try:
                    min_v, max_v = (float(entry["min_value"]),)
                    float(entry["max_value"])
                    values.extend(sorted(list(set([min_v, max_v]))))
                except (ValueError, TypeError):
                    pass
        return values

    def _parse_conditions(self, cond_input: Dict) -> Dict:
        parsed_cond = {
            "temperature_K": [],
            "time_s": [],
            "atmosphere": [],
            "device": None,
            "media": None,
            "other_conditions": {},
        }
        if not isinstance(cond_input, dict):
            return parsed_cond
        for temp_entry in cond_input.get("heating_temperature", []):
            parsed_cond["temperature_K"].extend(
                self._parse_single_condition_value_entry(temp_entry)
            )
        if parsed_cond["temperature_K"]:
            parsed_cond["temperature_K"] = sorted(
                list(set(parsed_cond["temperature_K"]))
            )
        for time_entry in cond_input.get("heating_time", []):
            parsed_cond["time_s"].extend(
                self._parse_single_condition_value_entry(time_entry)
            )
        if parsed_cond["time_s"]:
            parsed_cond["time_s"] = sorted(list(set(parsed_cond["time_s"])))
        atm_raw = cond_input.get("heating_atmosphere", [])
        if isinstance(atm_raw, list):
            parsed_cond["atmosphere"] = [str(a) for a in atm_raw if a is not None]
        elif atm_raw is not None:
            parsed_cond["atmosphere"] = [str(atm_raw)]
        parsed_cond["device"] = cond_input.get("mixing_device")
        parsed_cond["media"] = cond_input.get("mixing_media")
        for key, value in cond_input.items():
            if key not in [
                "heating_temperature",
                "heating_time",
                "heating_atmosphere",
                "mixing_device",
                "mixing_media",
            ]:
                try:
                    json.dumps(value)
                    parsed_cond["other_conditions"][key] = value
                except TypeError:
                    parsed_cond["other_conditions"][key] = str(value)
        return parsed_cond

    def add_operation(
        self, op_data: Dict, source_nodes: List[str], node_id: Optional[str] = None
    ) -> str:
        if not isinstance(op_data, dict):
            raise ValueError("Invalid op_data.")
        if not isinstance(source_nodes, list):
            raise ValueError("source_nodes must be a list.")
        if node_id is None:
            node_id = f"op_{len(self.operation_nodes)}"
        op_type_str = op_data.get("type")
        try:
            op_type_enum = OperationTypeEnum(op_type_str)
        except ValueError:
            op_type_enum = OperationTypeEnum.unknown
        conditions = self._parse_conditions(op_data.get("conditions", {}))
        features = {
            "type": "operation",
            "op_type": op_type_enum,
            "token": op_data.get("token", "unknown_token"),
            "conditions": conditions,
        }
        self._add_node_safe(node_id, **features)
        if node_id not in self.operation_nodes:
            self.operation_nodes.append(node_id)
        for src_node_id in source_nodes:
            if src_node_id in self.nodes:
                self.add_edge(src_node_id, node_id)
        return node_id

    def add_output(
        self,
        target_data: Dict,
        source_operation_node_id: str,
        node_id: Optional[str] = None,
    ) -> str:
        if not (
            isinstance(target_data, dict)
            and "material_formula" in target_data
            and target_data["material_formula"]
            and isinstance(target_data["material_formula"], str)
        ):
            raise ValueError("Invalid target_data for output node.")
        if (
            source_operation_node_id not in self.nodes
            or self.nodes[source_operation_node_id].get("type") != "operation"
        ):
            raise ValueError(
                f"Source '{source_operation_node_id}' must be an existing operation node."
            )
        if node_id is None:
            node_id = f"chem_out_{len(self.output_nodes)}"
        try:
            composition_obj = Composition(target_data["material_formula"])
        except Exception as e:
            raise ValueError(f"Invalid Pymatgen formula for output: {e}")
        amount, elements_data = 1.0, {}
        if (
            "composition" in target_data
            and isinstance(target_data["composition"], list)
            and target_data["composition"]
        ):
            first_comp = target_data["composition"][0]
            if isinstance(first_comp, dict):
                amount = self._parse_amount(first_comp.get("amount", 1.0))
                elements_data = first_comp.get("elements", {})
        features = {
            "type": "chemical",
            "formula": target_data["material_formula"],
            "composition": composition_obj,
            "amount": amount,
            "elements": elements_data,
        }
        self._add_node_safe(node_id, **features)
        if node_id not in self.output_nodes:
            self.output_nodes.append(node_id)
        self.add_edge(source_operation_node_id, node_id)
        return node_id

    def display(self, figsize=(10, 7), prog="dot"):
        plt.style.use(ACTIONGRAPH_PLOT_STYLE)
        plt.rcParams.update(
            {
                "font.size": ACTIONGRAPH_FONT_SIZE_MEDIUM,
                "axes.labelsize": ACTIONGRAPH_FONT_SIZE_MEDIUM,
                "xtick.labelsize": ACTIONGRAPH_FONT_SIZE_SMALL,
                "ytick.labelsize": ACTIONGRAPH_FONT_SIZE_SMALL,
                "legend.fontsize": ACTIONGRAPH_FONT_SIZE_SMALL,
            }
        )
        plt.figure(figsize=figsize)
        if self.number_of_nodes() == 0:
            plt.text(
                0.5,
                0.5,
                "Empty Graph",
                ha="center",
                va="center",
                fontsize=ACTIONGRAPH_FONT_SIZE_MEDIUM,
            )
            plt.axis("off")
            plt.show()
            return
        try:
            pos = nx.nx_agraph.graphviz_layout(self, prog=prog)
        except ImportError:
            pos = nx.spring_layout(
                self,
                seed=42,
                k=(
                    0.9 / (self.number_of_nodes() ** 0.5)
                    if self.number_of_nodes() > 0
                    else 0.8
                ),
                iterations=50,
            )

        chem_color, op_color = "#1f77b4", "#ff7f0e"
        node_colors_list = [
            (
                chem_color
                if self.nodes[node_id].get("type") == "chemical"
                else (
                    op_color
                    if self.nodes[node_id].get("type") == "operation"
                    else "#9467bd"
                )
            )
            for node_id in self.nodes()
        ]

        nx.draw_networkx_nodes(
            self,
            pos,
            node_color=node_colors_list,
            node_size=ACTIONGRAPH_NODE_SIZE,
            edgecolors="black",
            linewidths=1,
        )
        nx.draw_networkx_edges(
            self,
            pos,
            edge_color=ACTIONGRAPH_EDGE_COLOR,
            arrows=True,
            arrowsize=ACTIONGRAPH_ARROW_SIZE,
            arrowstyle="-|>",
            width=ACTIONGRAPH_EDGE_WIDTH,
            node_size=ACTIONGRAPH_NODE_SIZE,
        )
        labels = {}
        for node_id, attrs in self.nodes(data=True):
            short_id = node_id.split("_")[-1] if "_" in node_id else node_id
            if attrs.get("type") == "chemical":
                formula_str = attrs.get("formula", "?")
                try:
                    pretty_formula = Composition(
                        formula_str
                    ).get_reduced_formula_and_factor(iupac_ordering=True)[0]
                except:
                    pretty_formula = formula_str
                labels[node_id] = f"{short_id}\n{pretty_formula}"
            elif attrs.get("type") == "operation":
                op_type_obj = attrs.get("op_type")
                op_type_str = (
                    op_type_obj.name
                    if isinstance(op_type_obj, OperationTypeEnum)
                    else str(op_type_obj)
                )
                if "Operation" in op_type_str:
                    op_type_str = op_type_str.replace("Operation", "Op.")
                if "Synthesis" in op_type_str:
                    op_type_str = op_type_str.replace("Synthesis", "Synth.")
                labels[node_id] = f"{short_id}\n{op_type_str}"
            else:
                labels[node_id] = f"{short_id}\n(Unk)"
        nx.draw_networkx_labels(
            self, pos, labels=labels, font_size=ACTIONGRAPH_FONT_SIZE_SMALL
        )
        legend_elements = [
            mpatches.Patch(color=chem_color, label="Chemical Node"),
            mpatches.Patch(color=op_color, label="Operation Node"),
        ]
        plt.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            fancybox=True,
            frameon=False,
            fontsize=ACTIONGRAPH_FONT_SIZE_SMALL,
        )
        plt.axis("off")
        plt.tight_layout(pad=0.1)
        plt.show()

    def serialize(self) -> Dict:
        graph_copy = self.copy()
        for node_id, attrs in graph_copy.nodes(data=True):
            if "composition" in attrs and isinstance(attrs["composition"], Composition):
                attrs["composition"] = attrs["composition"].formula
            if "op_type" in attrs and isinstance(attrs["op_type"], OperationTypeEnum):
                attrs["op_type"] = attrs["op_type"].value
        data = json_graph.node_link_data(graph_copy)
        data["custom_input_nodes"] = self.input_nodes
        data["custom_operation_nodes"] = self.operation_nodes
        data["custom_output_nodes"] = self.output_nodes
        return data

    @classmethod
    def deserialize(cls, data: Dict) -> "ActionGraph":
        edge_key = (
            "links" if "links" in data else ("edges" if "edges" in data else None)
        )
        if edge_key is None:
            raise KeyError("Edge key not found in serialized data.")
        graph_data_for_nx = {
            "nodes": data.get("nodes", []),
            edge_key: data.get(edge_key, []),
            "directed": data.get("directed", True),
            "multigraph": data.get("multigraph", False),
            "graph": data.get("graph", {}),
        }
        try:
            reconstructed_graph = json_graph.node_link_graph(graph_data_for_nx)
        except Exception:
            reconstructed_graph = nx.DiGraph()
            reconstructed_graph.graph.update(data.get("graph", {}))
            node_id_map = {}
            for node_data_item in data.get("nodes", []):
                node_id = str(node_data_item["id"])
                node_id_map[node_data_item["id"]] = node_id
                attrs = {k: v for k, v in node_data_item.items() if k != "id"}
                reconstructed_graph.add_node(node_id, **attrs)
            links_data_list = data.get(edge_key, [])
            for link_item in links_data_list:
                source_orig, target_orig = (link_item.get("source"),)
                link_item.get("target")
                source_id, target_id = node_id_map.get(
                    source_orig, str(source_orig)
                ), node_id_map.get(target_orig, str(target_orig))
                edge_attrs = {
                    k: v
                    for k, v in link_item.items()
                    if k not in ["source", "target", "key", "edge_type"]
                }
                reconstructed_graph.add_edge(source_id, target_id, **edge_attrs)

        ag = cls()
        ag.add_nodes_from(reconstructed_graph.nodes(data=True))
        ag.add_edges_from(reconstructed_graph.edges(data=True))
        ag.graph.update(reconstructed_graph.graph)

        for node_id, attrs in ag.nodes(data=True):
            if (
                attrs.get("type") == "chemical"
                and "composition" in attrs
                and isinstance(attrs["composition"], str)
            ):
                try:
                    attrs["composition"] = Composition(attrs["composition"])
                except:
                    try:
                        attrs["composition"] = Composition.from_dict(
                            json.loads(attrs["composition"])
                        )
                    except:
                        pass
            elif (
                attrs.get("type") == "chemical"
                and "formula" in attrs
                and "composition" not in attrs
            ):
                try:
                    attrs["composition"] = Composition(attrs["formula"])
                except:
                    pass
            if (
                attrs.get("type") == "operation"
                and "op_type" in attrs
                and isinstance(attrs["op_type"], str)
            ):
                try:
                    attrs["op_type"] = OperationTypeEnum(attrs["op_type"])
                except ValueError:
                    attrs["op_type"] = OperationTypeEnum.unknown

        ag.input_nodes = data.get("custom_input_nodes", [])
        ag.operation_nodes = data.get("custom_operation_nodes", [])
        ag.output_nodes = data.get("custom_output_nodes", [])

        if not any([ag.input_nodes, ag.operation_nodes, ag.output_nodes]):
            inferred_inputs, inferred_ops, inferred_outputs = [], [], []
            for node_id, attrs_node in ag.nodes(data=True):
                node_type = attrs_node.get("type")
                if node_type == "chemical":
                    is_output = False
                    if ag.out_degree(node_id) == 0:
                        for u, _ in ag.in_edges(node_id):
                            if ag.nodes[u].get("type") == "operation":
                                is_output = True
                                break
                    if is_output:
                        inferred_outputs.append(node_id)
                    elif ag.in_degree(node_id) == 0 and not is_output:
                        inferred_inputs.append(node_id)
                elif node_type == "operation":
                    inferred_ops.append(node_id)
            ag.input_nodes = sorted(list(set(inferred_inputs)))
            ag.operation_nodes = sorted(list(set(inferred_ops)))
            ag.output_nodes = sorted(list(set(inferred_outputs)))
        return ag

    @classmethod
    def from_mp_synthesis(cls, mp_data: dict) -> Union["ActionGraph", None]:
        required_keys = ["precursors", "operations", "target"]
        if not all(key in mp_data and mp_data[key] for key in required_keys):
            return None
        try:
            if not (
                isinstance(mp_data["precursors"], list)
                and all(
                    isinstance(p, dict) and p.get("material_formula")
                    for p in mp_data["precursors"]
                )
            ):
                return None
            ag = cls(input_chemicals=mp_data["precursors"])
            current_source_node_ids = ag.input_nodes[:]
            if not current_source_node_ids and mp_data["precursors"]:
                return None

            if (
                not mp_data.get("operations")
                or not isinstance(mp_data["operations"], list)
                or not all(
                    isinstance(op, dict) and "type" in op
                    for op in mp_data["operations"]
                )
            ):
                if ag.input_nodes and mp_data.get("target"):
                    generic_op_id = ag.add_operation(
                        op_data={"type": "HeatingOperation", "token": "AssumedProcess"},
                        source_nodes=ag.input_nodes,
                        node_id="op_assumed_0",
                    )
                    current_source_node_ids = [generic_op_id]
                else:
                    return None
            else:
                for op_idx, operation_data in enumerate(mp_data["operations"]):
                    if not current_source_node_ids and op_idx > 0:
                        return None
                    op_id = ag.add_operation(
                        op_data=operation_data,
                        source_nodes=(
                            current_source_node_ids if current_source_node_ids else []
                        ),
                        node_id=f"op_{op_idx}",
                    )
                    current_source_node_ids = [op_id]

            if not current_source_node_ids:
                return None
            last_operation_node_id = current_source_node_ids[0]
            if not (
                isinstance(mp_data["target"], dict)
                and mp_data["target"].get("material_formula")
            ):
                return None
            ag.add_output(
                target_data=mp_data["target"],
                source_operation_node_id=last_operation_node_id,
            )
            return ag
        except ValueError:
            return None
        except Exception:
            return None
