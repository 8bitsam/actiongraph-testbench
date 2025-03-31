import os
import json
import re
from pymatgen.core import Composition

def has_variable_subscripts(formula_str):
    """Check if a chemical formula has variable subscripts."""
    # Patterns indicating variable subscripts
    patterns = [
        r'\([a-zA-Z0-9]*[_a-zA-Z][a-zA-Z0-9]*\)',  # Parentheses with letters
        r'\[[a-zA-Z0-9]*[_a-zA-Z][a-zA-Z0-9]*\]',  # Brackets with letters
        r'_[a-zA-Z]',                           # Underscore followed by letter
        r'[a-zA-Z]\d*\+',                       # Cation notation
        r'[a-zA-Z]\d*\-',                       # Anion notation
    ]
    
    for pattern in patterns:
        if re.search(pattern, formula_str):
            return True
    
    # Verify with pymatgen's Composition
    try:
        comp = Composition(formula_str)
        return False  # Successfully parsed with fixed stoichiometry
    except:
        return True   # Failed to parse, likely has variable parts

def filter_synthesis_data():
    """Filter synthesis data to include only reactions with constant subscripts."""
    input_dir = "Data/mp-data/"
    output_dir = "Data/filtered-mp-data/"
    os.makedirs(output_dir, exist_ok=True)
    
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    filtered_count = 0
    total_count = len(json_files)
    
    for json_file in json_files:
        with open(os.path.join(input_dir, json_file), 'r') as f:
            data = json.load(f)
        
        # Verify it's a solid-state synthesis
        if data.get("synthesis_type", "").lower() != "solid-state":
            continue
        
        # Check precursors for variable subscripts
        valid_precursors = True
        for precursor in data.get("precursors", []):
            formula = precursor.get("material_formula", "")
            if has_variable_subscripts(formula):
                valid_precursors = False
                break
        
        if not valid_precursors:
            continue
        
        # Check target for variable subscripts
        target = data.get("target", {})
        target_formula = target.get("material_formula", "")
        if has_variable_subscripts(target_formula):
            continue
        
        # Save the filtered data
        with open(os.path.join(output_dir, json_file), 'w') as f:
            json.dump(data, f, indent=2)
        filtered_count += 1
    
    print(f"Filtered {filtered_count} out of {total_count} synthesis reactions.")
