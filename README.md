# actiongraph-testbench
### Structure of an ActionGraph
#### Node Types
ChemicalNode : contains parameters for a single material
- material_formula
- composition
- elements
- amount

OperationNode : contains parameters for a single operation
- operation type
- token (the extracted string from the paper, see mp_api for more)
- conditions

#### Action Flow
The graph itself can be thought of as $A \rightarrow B \rightarrow C$, where $A$ is the input and $B$ is the output set of materials/chemicals and $C$ is the set of all operations.
