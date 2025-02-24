import networkx as nx
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt
import json

class ActionGraph(nx.DiGraph):
    """
    A graph representing a chemical reaction flowchart.
    
    Node Types:
      - 'chemical': Represents a chemical (input or output).
      - 'step': Represents an intermediate reaction step.
    
    Construction Workflow:
      1. Input chemicals are added as isolated nodes.
      2. Reaction steps (operations) are added via add_step(), which connects 
         provided source nodes (which can be chemicals or previous steps) to the step node.
      3. A terminal (final) step node is created via merge_to_terminal() to merge 
         preceding step nodes. This is the de facto terminal node.
      4. Output chemicals are added as nodes and connected from the terminal node.
    """
    
    def __init__(self, input_chemicals=None):
        """
        Parameters
        ----------
        input_chemicals : list of str, optional
            List of chemical names/formulas to be added as input nodes.
        """
        super().__init__()
        self.input_nodes = []    # IDs for input chemical nodes
        self.step_nodes = []     # IDs for intermediate reaction step nodes
        self.output_nodes = []   # IDs for output chemical nodes
        self.terminal_node = None  # ID for the de facto terminal step node
        
        if input_chemicals:
            for i, chem in enumerate(input_chemicals):
                node_id = f"input_{i}"
                self.add_node(node_id, type="chemical", data=chem)
                self.input_nodes.append(node_id)
    
    def add_input(self, chemical, node_id=None):
        """
        Add an input chemical node.
        
        Parameters
        ----------
        chemical : str
            The chemical data (e.g., name or formula).
        node_id : str, optional
            Custom node identifier. If not provided, a default is generated.
        
        Returns
        -------
        str
            The identifier of the added chemical node.
        """
        if node_id is None:
            node_id = f"input_{len(self.input_nodes)}"
        if node_id in self.nodes:
            raise ValueError(f"Node '{node_id}' already exists.")
        self.add_node(node_id, type="chemical", data=chemical)
        self.input_nodes.append(node_id)
        return node_id
    
    def add_step(self, step_description, source_nodes, node_id=None):
        """
        Add a reaction step node and connect it from the given source nodes.
        
        Parameters
        ----------
        step_description : str
            Description of the reaction step (e.g., "mixed using anatase TiO2").
        source_nodes : list of str
            List of node IDs from which the reaction step is initiated.
        node_id : str, optional
            Custom identifier for the step node. If not provided, a default is generated.
        
        Returns
        -------
        str
            The identifier of the added step node.
        """
        if node_id is None:
            node_id = f"step_{len(self.step_nodes)}"
        if node_id in self.nodes:
            raise ValueError(f"Node '{node_id}' already exists.")
        self.add_node(node_id, type="step", data=step_description)
        self.step_nodes.append(node_id)
        for src in source_nodes:
            if src not in self.nodes:
                raise ValueError(f"Source node '{src}' does not exist.")
            self.add_edge(src, node_id)
        return node_id
    
    def merge_to_terminal(self, source_nodes, terminal_step_description="Final Reaction Step", terminal_node_id="terminal"):
        """
        Merge one or more reaction steps into a single de facto terminal node.
        This node will serve as the final reaction step from which output chemicals are derived.
        
        Parameters
        ----------
        source_nodes : list of str
            List of node IDs (typically step nodes) to merge.
        terminal_step_description : str, optional
            Description for the terminal reaction step.
        terminal_node_id : str, optional
            Identifier for the terminal node.
        
        Returns
        -------
        str
            The identifier of the terminal node.
        """
        if terminal_node_id in self.nodes:
            self.nodes[terminal_node_id]['data'] = terminal_step_description
        else:
            self.add_node(terminal_node_id, type="step", data=terminal_step_description)
        self.terminal_node = terminal_node_id
        
        for src in source_nodes:
            if src not in self.nodes:
                raise ValueError(f"Source node '{src}' does not exist.")
            self.add_edge(src, terminal_node_id)
        return terminal_node_id
    
    def add_output(self, chemical, node_id=None):
        """
        Add an output chemical node and connect it from the terminal node.
        
        Parameters
        ----------
        chemical : str
            The chemical data for the output.
        node_id : str, optional
            Custom identifier for the output node. If not provided, a default is generated.
        
        Returns
        -------
        str
            The identifier of the added output chemical node.
        """
        if self.terminal_node is None:
            raise ValueError("Terminal node not defined. Call merge_to_terminal() before adding outputs.")
        if node_id is None:
            node_id = f"output_{len(self.output_nodes)}"
        if node_id in self.nodes:
            raise ValueError(f"Node '{node_id}' already exists.")
        self.add_node(node_id, type="chemical", data=chemical)
        self.output_nodes.append(node_id)
        self.add_edge(self.terminal_node, node_id)
        return node_id
    
    def serialize(self):
        """
        Serialize the ActionGraph into a dictionary using networkx's node_link_data.
        
        Returns
        -------
        dict
            The serialized graph.
        """
        return json_graph.node_link_data(self)
