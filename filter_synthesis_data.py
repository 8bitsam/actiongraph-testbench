import os
import json
import re
from pymatgen.core import Composition, Element # Import Element class directly

# Optional: More specific exception handling if needed
# from pymatgen.core.composition import CompositionError

# --- Add verbose flag ---
VERBOSE_FILTERING = True # Set to True to see detailed filtering decisions

def has_variable_subscripts(formula_str):
    """Check if a chemical formula has variable subscripts, uses pseudo-elements, or is invalid."""
    # --- Added Check: Handle empty or None formulas explicitly ---
    if not formula_str:
        if VERBOSE_FILTERING: print(f"  - Formula '{formula_str}' is empty/None -> Variable/Invalid")
        return True # Treat empty/None as invalid/variable

    # Patterns indicating variable subscripts
    # Added pattern for common variables like x, y, z, delta (often lowercase)
    patterns = [
        r'[xyzδ]',                               # Matches standalone x, y, z, or delta
        r'\([a-zA-Z0-9]*[_a-zA-Z][a-zA-Z0-9]*\)',  # Parentheses with letters e.g., (La,Sr)MnO3
        r'\[[a-zA-Z0-9]*[_a-zA-Z][a-zA-Z0-9]*\]',  # Brackets with letters
        r'_[a-zA-Z]',                           # Underscore followed by letter e.g., Fe_xO_y
        r'[a-zA-Z]\d*\+',                       # Cation notation e.g., Fe2+
        r'[a-zA-Z]\d*\-',                       # Anion notation e.g., O2-
    ]

    for pattern in patterns:
        if re.search(pattern, formula_str):
            if VERBOSE_FILTERING: print(f"  - Formula '{formula_str}' matched regex '{pattern}' -> Variable")
            return True

    # Verify with pymatgen's Composition
    try:
        # attempt to parse. If successful, it's likely fixed stoichiometry.
        comp = Composition(formula_str)
        # Check if amounts are numeric (pymatgen might parse 'x' into a non-numeric amount)
        for el, amt in comp.items():
             if not isinstance(amt, (int, float)):
                  if VERBOSE_FILTERING: print(f"  - Formula '{formula_str}' parsed by Composition, but amount '{amt}' for element '{el}' is not numeric -> Variable")
                  return True
        # --- ADDED CHECK FOR VALID ELEMENT SYMBOLS ---
        # Check if all element symbols are valid known elements in pymatgen
        for el in comp.elements:
             # Use pymatgen's built-in validator Element.is_valid_symbol()
             if not Element.is_valid_symbol(el.symbol):
                 if VERBOSE_FILTERING: print(f"  - Formula '{formula_str}' parsed by Composition, but symbol '{el.symbol}' is not a valid Element -> Variable/Pseudo")
                 return True
        # --- END ADDED CHECK ---

        # If parsing works, amounts are numeric, and symbols are valid, assume fixed stoichiometry
        if VERBOSE_FILTERING: print(f"  - Formula '{formula_str}' passed regex and Composition checks -> Fixed")
        return False
    except Exception as e:
        # If Composition parsing fails for any reason, treat it as variable/invalid
        if VERBOSE_FILTERING: print(f"  - Formula '{formula_str}' failed Composition check ({type(e).__name__}: {e}) -> Variable/Invalid")
        return True

def filter_synthesis_data():
    """Filter synthesis data to include only reactions with constant subscripts."""
    input_dir = "Data/mp-data/"
    output_dir = "Data/filtered-mp-data/"
    os.makedirs(output_dir, exist_ok=True)

    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    filtered_count = 0
    skipped_type = 0
    skipped_precursor = 0
    skipped_target = 0
    skipped_load_error = 0
    skipped_write_error = 0
    total_count = len(json_files)

    print(f"Starting filtering of {total_count} files from {input_dir}...")
    if VERBOSE_FILTERING: print("Verbose mode enabled.")

    for i, json_file in enumerate(json_files):
        if VERBOSE_FILTERING and (i % 100 == 0 or i == total_count - 1):
             print(f"\nProcessing file {i+1}/{total_count}: {json_file}")
        elif not VERBOSE_FILTERING and i % 1000 == 0:
             print(f"Processing file {i+1}/{total_count}...")

        file_path = os.path.join(input_dir, json_file)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  - ERROR loading JSON from {json_file}: {e}. Skipping file.")
            skipped_load_error += 1
            continue

        # Verify it's a solid-state synthesis
        synthesis_type = data.get("synthesis_type", "").lower()
        if synthesis_type != "solid-state":
            if VERBOSE_FILTERING: print(f"  - Skipping: Synthesis type is '{synthesis_type}', not 'solid-state'.")
            skipped_type += 1
            continue

        # Check precursors for variable subscripts
        valid_precursors = True
        precursors = data.get("precursors", [])
        if not precursors:
             if VERBOSE_FILTERING: print(f"  - Skipping: No precursors found in data.")
             skipped_precursor += 1
             continue # Skip if no precursors listed

        for precursor in precursors:
            formula = precursor.get("material_formula", "")
            if VERBOSE_FILTERING: print(f"  - Checking precursor formula: '{formula}'")
            if has_variable_subscripts(formula):
                if VERBOSE_FILTERING: print(f"  - Skipping: Precursor '{formula}' identified as variable/pseudo/invalid.")
                valid_precursors = False
                skipped_precursor += 1
                break # Stop checking precursors for this reaction

        if not valid_precursors:
            continue

        # Check target for variable subscripts
        target = data.get("target", {})
        target_formula = target.get("material_formula", "")
        if not target_formula:
            if VERBOSE_FILTERING: print(f"  - Skipping: No target formula found.")
            skipped_target += 1
            continue # Skip if no target formula

        if VERBOSE_FILTERING: print(f"  - Checking target formula: '{target_formula}'")
        if has_variable_subscripts(target_formula):
            if VERBOSE_FILTERING: print(f"  - Skipping: Target '{target_formula}' identified as variable/pseudo/invalid.")
            skipped_target += 1
            continue

        # If all checks passed, save the filtered data
        output_path = os.path.join(output_dir, json_file)
        try:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            filtered_count += 1
            if VERBOSE_FILTERING: print(f"  - PASSED FILTERING. Saved to {output_path}")
        except Exception as e:
             print(f"  - ERROR writing JSON to {output_path}: {e}")
             skipped_write_error += 1


    print("\n--- Filtering Summary ---")
    print(f"Total files processed: {total_count}")
    print(f"Files skipped (load/write error): {skipped_load_error + skipped_write_error}")
    print(f"Files skipped (wrong synthesis type): {skipped_type}")
    print(f"Files skipped (variable/pseudo/invalid precursor): {skipped_precursor}")
    print(f"Files skipped (variable/pseudo/invalid target): {skipped_target}")
    print(f"Files successfully filtered and saved: {filtered_count}")
    print(f"Filtered files saved to: {output_dir}")

# Example usage:
if __name__ == "__main__":
    # Example formulas to test the checker function
    # print("Testing has_variable_subscripts function:")
    # test_formulas = [
    #     "LiCoO2", "Fe3O4", "H2O", "NaCl", # Fixed
    #     "LixCoO2", "La0.8Sr0.2MnO3", "Fe3O4-d", # Variable (letters, delta)
    #     "FePO4_x", "(La,Sr)MnO3", "[Co(NH3)6]Cl3", # Variable (underscore, parens, brackets with letters)
    #     "Fe2+", "O2-", # Variable (charge)
    #     "InvalidFormula", "", None, # Invalid/Empty
    #     "Li(Ni0.5Mn0.3Co0.2)O2", # Complex but fixed
    #     "Ln2O3", "MCO3", "RE2O3", # Pseudo-elements
    #     "Ag2S", "SiS2", "Ag8SiS6" # From example
    # ]
    # # Disable verbose mode just for this test section
    # original_verbose = VERBOSE_FILTERING
    # VERBOSE_FILTERING = False
    # print("Formula -> IsVariableOrPseudo?")
    # for f in test_formulas:
    #     print(f"'{f}': {has_variable_subscripts(f)}")
    # VERBOSE_FILTERING = original_verbose # Restore flag

    print("\nRunning full data filtering...")
    filter_synthesis_data()
