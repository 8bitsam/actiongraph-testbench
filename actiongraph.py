import networkx as nx
from networkx.readwrite import json_graph

class ActionGraph(nx.DiGraph):
    """
    A directed graph representing a chemical reaction flowchart.
    
    Nodes represent chemicals (inputs or intermediate products).
    Edges represent experimental steps (transformations) between chemicals.
    
    The graph is built in three phases:
      1. Initialization: Create floating chemical nodes.
      2. Adding Steps: Manually add an edge (step) between two chemical nodes.
      3. Connecting Output: Connect selected nodes (or all sink nodes) to a single terminal output node.
    """
    
    def __init__(self, chemicals=None):
        """
        Initialize the graph with a list of chemical nodes.
        
        Parameters
        ----------
        chemicals : list of str, optional
            List of chemical names/data to add as floating nodes.
            Each chemical will be added as a node with a generated ID.
        """
        super().__init__()
        if chemicals:
            for i, chem in enumerate(chemicals):
                node_id = f"chem_{i}"
                self.add_node(node_id, type="chemical", data=chem)
    
    def add_chemical(self, chem, node_id=None):
        """
        Add a chemical node to the graph.
        
        Parameters
        ----------
        chem : str
            The chemical data (e.g., a name or formula).
        node_id : str, optional
            Custom identifier for the chemical node; if not provided, a default is generated.
        
        Returns
        -------
        str
            The identifier of the newly added chemical node.
        """
        if node_id is None:
            node_id = f"chem_{len(self.nodes)}"
        if node_id in self.nodes:
            raise ValueError(f"Node '{node_id}' already exists.")
        self.add_node(node_id, type="chemical", data=chem)
        return node_id
    
    def add_step(self, source, destination, step_data):
        """
        Add an experimental step (edge) between two chemical nodes.
        
        Parameters
        ----------
        source : str
            The identifier of the source chemical node.
        destination : str
            The identifier of the destination chemical node.
        step_data : str
            A description of the experimental step (e.g., "heat for 20 mins at 20°C").
        
        Returns
        -------
        tuple
            A tuple (source, destination) representing the edge added.
        """
        if source not in self.nodes:
            raise ValueError(f"Source node '{source}' does not exist.")
        if destination not in self.nodes:
            raise ValueError(f"Destination node '{destination}' does not exist.")
        if self.nodes[source].get("type") != "chemical":
            raise ValueError(f"Source node '{source}' is not a chemical node.")
        if self.nodes[destination].get("type") != "chemical":
            raise ValueError(f"Destination node '{destination}' is not a chemical node.")
        
        self.add_edge(source, destination, step=step_data)
        return (source, destination)
    
    def connect_output(self, output_data, source_nodes=None):
        """
        Connect selected chemical nodes to a single terminal output node.
        If no source_nodes are provided, automatically connect all chemical nodes
        that have no outgoing edges.
        
        Parameters
        ----------
        output_data : any
            Data for the output node (e.g., the final product's name or formula).
        source_nodes : list of str, optional
            List of chemical node identifiers to connect to the output node.
        
        Returns
        -------
        str
            The identifier of the output node.
        """
        output_node = "output"
        if output_node not in self.nodes:
            self.add_node(output_node, type="output", data=output_data)
        else:
            self.nodes[output_node]['data'] = output_data
        
        if source_nodes is None:
            # Automatically choose all chemical nodes with no outgoing edges
            source_nodes = [n for n in self.nodes 
                            if self.out_degree(n) == 0 and self.nodes[n].get("type") == "chemical"]
        
        for src in source_nodes:
            self.add_edge(src, output_node, step="finalize")
        return output_node
    
    def serialize(self):
        """
        Serialize the ActionGraph into a dictionary format using NetworkX's node_link_data.
        
        Returns
        -------
        dict
            A serialized dictionary representation of the graph.
        """
        return json_graph.node_link_data(self)
