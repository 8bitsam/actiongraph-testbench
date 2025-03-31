import networkx as nx
from networkx.readwrite import json_graph
from pymatgen.core import Composition
from emmet.core.synthesis import OperationTypeEnum
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class ActionGraph(nx.DiGraph):
    """
    Enhanced graph structure for materials synthesis workflows with MP integration
    
    Node Types:
    - 'chemical': Input/output materials with composition data
    - 'operation': Synthesis steps with type/token/conditions
    
    Edge Types:
    - 'material_flow': Chemical transformation relationships
    - 'temporal': Step sequencing dependencies
    """
    
    def __init__(self, input_chemicals: Optional[List[Dict]] = None):
        super().__init__()
        self.input_nodes = []
        self.operation_nodes = []
        self.output_nodes = []
        self.terminal_node = None
        if input_chemicals:
            for chem in input_chemicals:
                self.add_input(chem)

    def add_input(self, chemical_data: Dict, node_id: Optional[str] = None):
        """Add chemical precursor node with full material data"""
        if node_id is None:
            node_id = f"chem_in_{len(self.input_nodes)}"
        
        # Safely extract amount value, handling dictionary structures
        amount = chemical_data['composition'][0]['amount']
        if isinstance(amount, dict):
            # Try different possible dict structures
            if 'value' in amount:
                amount = amount['value']
            elif 'values' in amount and amount['values']:
                amount = amount['values'][0]
            elif 'min_value' in amount and 'max_value' in amount:
                # Use average of min and max
                amount = (float(amount['min_value']) + float(amount['max_value'])) / 2
            else:
                # Default fallback
                amount = 1.0
        
        features = {
            'type': 'chemical',
            'formula': chemical_data['material_formula'],
            'composition': Composition(chemical_data['material_formula']),
            'amount': float(amount),
            'elements': chemical_data['composition'][0]['elements']
        }
        
        self._add_node_safe(node_id, **features)
        self.input_nodes.append(node_id)
        return node_id

    def add_operation(self, op_data: Dict, source_nodes: List[str], 
                    node_id: Optional[str] = None):
        """
        Add synthesis operation node with MP-specific features
        
        Parameters
        ----------
        op_data : dict
            MP operation format with keys:
            - type (OperationTypeEnum)
            - token
            - conditions
        """
        if node_id is None:
            node_id = f"op_{len(self.operation_nodes)}"
        features = {
            'type': 'operation',
            'op_type': OperationTypeEnum(op_data['type']),
            'token': op_data['token'],
            'conditions': self._parse_conditions(op_data['conditions'])
        }
        self._add_node_safe(node_id, **features)
        self.operation_nodes.append(node_id)
        for src in source_nodes:
            self.add_edge(src, node_id, edge_type='temporal')
        return node_id

    def merge_to_terminal(self, source_nodes: List[str], 
                        terminal_desc: str = "Final Step",
                        node_id: str = "terminal"):
        """Connect final operations to terminal node"""
        self._add_node_safe(node_id, 
                          type='operation',
                          op_type=OperationTypeEnum.starting,
                          token=terminal_desc,
                          conditions={})
        self.terminal_node = node_id
        for src in source_nodes:
            self.add_edge(src, node_id, edge_type='temporal')
        return node_id

    def add_output(self, target_data: Dict, node_id: Optional[str] = None):
        """Add target chemical node with composition data"""
        if not self.terminal_node:
            raise ValueError("Create terminal node first")
        if node_id is None:
            node_id = f"chem_out_{len(self.output_nodes)}"
        
        # Safely extract amount value, handling dictionary structures
        amount = target_data['composition'][0]['amount']
        if isinstance(amount, dict):
            # Try different possible dict structures
            if 'value' in amount:
                amount = amount['value']
            elif 'values' in amount and amount['values']:
                amount = amount['values'][0]
            elif 'min_value' in amount and 'max_value' in amount:
                # Use average of min and max
                amount = (float(amount['min_value']) + float(amount['max_value'])) / 2
            else:
                # Default fallback
                amount = 1.0
        
        features = {
            'type': 'chemical',
            'formula': target_data['material_formula'],
            'composition': Composition(target_data['material_formula']),
            'amount': float(amount),
            'elements': target_data['composition'][0]['elements']
        }
        
        self._add_node_safe(node_id, **features)
        self.output_nodes.append(node_id)
        self.add_edge(self.terminal_node, node_id, edge_type='material_flow')
        return node_id

    def _add_node_safe(self, node_id: str, **attrs):
        """Validate node creation"""
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} exists")
        self.add_node(node_id, **attrs)

    def _parse_conditions(self, cond: Dict) -> Dict:
        """Normalize condition data types"""
        temperatures = []
        times = []
        
        # Extract temperature values from potentially nested structures
        for t in cond.get('heating_temperature', []):
            if isinstance(t, (int, float)):
                temperatures.append(float(t))
            elif isinstance(t, dict):
                # Handle structured temperature data
                if 'values' in t and t['values']:
                    temperatures.extend([float(val) for val in t['values']])
                elif 'value' in t:
                    temperatures.append(float(t['value']))
                elif 'min_value' in t and 'max_value' in t:
                    temperatures.append(float(t['min_value']))
                    temperatures.append(float(t['max_value']))
        
        # Similar logic for time values
        for t in cond.get('heating_time', []):
            if isinstance(t, (int, float)):
                times.append(float(t))
            elif isinstance(t, dict):
                if 'values' in t and t['values']:
                    times.extend([float(val) for val in t['values']])
                elif 'value' in t:
                    times.append(float(t['value']))
                elif 'min_value' in t and 'max_value' in t:
                    times.append(float(t['min_value']))
                    times.append(float(t['max_value']))
        
        return {
            'temperature': temperatures,
            'time': times,
            'atmosphere': cond.get('heating_atmosphere', []),
            'device': cond.get('mixing_device'),
            'media': cond.get('mixing_media')
        }

    def display(self):
        """Plot the ActionGraph with labels."""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self, seed=42, k=0.5)
        nx.draw_networkx_edges(
            self, pos,
            edgelist=self.edges(),
            edge_color='#7f7f7f',
            arrows=True,
            arrowsize=20,
            arrowstyle='-|>',
            width=2,
            node_size=1500
        )
        node_colors = []
        for node in self.nodes():
            if node in self.input_nodes:
                node_colors.append('#1f77b4')
            elif node in self.output_nodes:
                node_colors.append('#ff7f0e')
            elif node == self.terminal_node:
                node_colors.append('#2ca02c')
            else:
                node_colors.append('#d62728')
        nx.draw_networkx_nodes(
            self, pos,
            node_color=node_colors,
            node_size=1500,
            edgecolors='black',
            linewidths=2
        )
        nx.draw_networkx_labels(
            self, pos,
            font_size=12,
            font_weight='bold',
            font_color='black'
        )
        legend_elements = [
            mpatches.Patch(color='#1f77b4', label='Input Chemicals'),
            mpatches.Patch(color='#d62728', label='Operations'),
            mpatches.Patch(color='#2ca02c', label='Terminal Node'),
            mpatches.Patch(color='#ff7f0e', label='Output Chemicals'),
        ]
        plt.legend(handles=legend_elements, loc='best', fontsize=10)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def serialize(self) -> Dict:
        """Convert an ActionGraph into a dictionary."""
        serializable_graph = nx.DiGraph()
        for node_id, attrs in self.nodes(data=True):
            node_attrs = {}
            for key, value in attrs.items():
                if key == 'composition' and hasattr(value, 'as_dict'):
                    node_attrs[key] = value.as_dict()
                elif key == 'op_type' and hasattr(value, 'value'):
                    node_attrs[key] = value.value
                else:
                    node_attrs[key] = value
            serializable_graph.add_node(node_id, **node_attrs)
        for src, tgt, attrs in self.edges(data=True):
            serializable_graph.add_edge(src, tgt, **attrs)
        data = json_graph.node_link_data(serializable_graph)
        data['input_nodes'] = self.input_nodes
        data['operation_nodes'] = self.operation_nodes
        data['output_nodes'] = self.output_nodes
        data['terminal_node'] = self.terminal_node
        return data

    @classmethod
    def deserialize(cls, data: Dict) -> 'ActionGraph':
        """Reconstruct an ActionGraph from a dictionary."""
        ag = cls()
        ag.graph = data.get('graph', {})
        
        # Process nodes
        for node in data['nodes']:
            node_id = node['id']
            
            # Extract attributes from node
            attrs = {}
            for k, v in node.items():
                if k != 'id':
                    attrs[k] = v
            
            # Convert specific attributes
            if 'composition' in attrs and isinstance(attrs['composition'], dict):
                attrs['composition'] = Composition.from_dict(attrs['composition'])
            if 'op_type' in attrs and isinstance(attrs['op_type'], str):
                attrs['op_type'] = OperationTypeEnum(attrs['op_type'])
            
            # Add node with attributes
            ag.add_node(node_id, **attrs)
        
        # Process edges
        for link in data['links']:
            source = link.get('source')
            target = link.get('target')
            
            # Handle various source/target formats
            if isinstance(source, dict):
                source = source.get('id')
            if isinstance(target, dict):
                target = target.get('id')
            
            # Extract edge attributes
            edge_attrs = {}
            for k, v in link.items():
                if k not in ['source', 'target']:
                    edge_attrs[k] = v
            
            ag.add_edge(source, target, **edge_attrs)
        
        # Set node lists with validation
        ag.input_nodes = data.get('input_nodes', [])
        ag.operation_nodes = data.get('operation_nodes', [])
        ag.output_nodes = data.get('output_nodes', [])
        ag.terminal_node = data.get('terminal_node')
        
        # Validate node lists - ensure they reference actual nodes
        ag.input_nodes = [node_id for node_id in ag.input_nodes if node_id in ag.nodes]
        ag.operation_nodes = [node_id for node_id in ag.operation_nodes if node_id in ag.nodes]
        ag.output_nodes = [node_id for node_id in ag.output_nodes if node_id in ag.nodes]
        
        # If input_nodes is empty but we have nodes that look like inputs, try to recover them
        if not ag.input_nodes:
            potential_inputs = [node_id for node_id, attrs in ag.nodes(data=True) 
                            if node_id.startswith('chem_in_') or 
                                (attrs.get('type') == 'chemical' and ag.in_degree(node_id) == 0)]
            if potential_inputs:
                print(f"Recovered {len(potential_inputs)} input nodes that were missing from input_nodes list")
                ag.input_nodes = potential_inputs
        
        return ag

    @classmethod
    def from_mp_synthesis(cls, mp_data: dict) -> "ActionGraph":
        """
        Construct ActionGraph from Materials Project synthesis data
        
        Parameters
        ----------
        mp_data : dict
            Materials Project synthesis entry format with keys:
            - precursors: list of precursor material dicts
            - operations: list of operation dicts
            - target: target material dict

        Returns
        -------
        ActionGraph
            Constructed reaction graph with:
            - Input nodes from precursors
            - Step nodes from operations
            - Output node from target
        """
        ag = cls(input_chemicals=mp_data["precursors"])
        prev_nodes = ag.input_nodes.copy()
        for op_idx, operation in enumerate(mp_data["operations"]):
            op_id = ag.add_operation(
                op_data=operation,
                source_nodes=prev_nodes,
                node_id=f"op_{op_idx}"
            )
            prev_nodes = [op_id]
        ag.merge_to_terminal(source_nodes=prev_nodes)
        ag.add_output(target_data=mp_data["target"])
        return ag
