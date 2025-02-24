import networkx as nx
from networkx.readwrite import json_graph
from pymatgen.core import Composition
from emmet.core.synthesis import OperationTypeEnum
from typing import Dict, List, Optional

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
        """
        Add chemical precursor node with full material data
        
        Parameters
        ----------
        chemical_data : dict
            MP precursor format with keys:
            - material_formula
            - composition
            - elements
            - amount
        """
        if node_id is None:
            node_id = f"chem_in_{len(self.input_nodes)}"
        features = {
            'type': 'chemical',
            'formula': chemical_data['material_formula'],
            'composition': Composition(chemical_data['material_formula']),
            'amount': float(chemical_data['composition'][0]['amount']),
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
        features = {
            'type': 'chemical',
            'formula': target_data['material_formula'],
            'composition': Composition(target_data['material_formula']),
            'amount': float(target_data['composition'][0]['amount']),
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
        return {
            'temperature': [float(t) for t in cond.get('heating_temperature', [])],
            'time': [float(t) for t in cond.get('heating_time', [])],
            'atmosphere': cond.get('heating_atmosphere', []),
            'device': cond.get('mixing_device'),
            'media': cond.get('mixing_media')
        }

    def serialize(self) -> Dict:
        """Networkx-compatible serialization with enum handling"""
        data = json_graph.node_link_data(self)
        for node in data['nodes']:
            if 'op_type' in node:
                node['op_type'] = node['op_type'].value
        return data

    @classmethod
    def deserialize(cls, data: Dict) -> 'ActionGraph':
        """Reconstruct from serialized data"""
        ag = cls()
        ag.graph = data['graph']
        for node in data['nodes']:
            if 'op_type' in node:
                node['op_type'] = OperationTypeEnum(node['op_type'])
            ag.add_node(node['id'], **node['attributes'])
        for link in data['links']:
            ag.add_edge(link['source'], link['target'], **link.get('attributes', {}))
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
