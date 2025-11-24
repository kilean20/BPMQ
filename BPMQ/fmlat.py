import re
import numpy as np
import torch
from pprint import pprint
from .BAL4BPMQ import EnvelopeEnsembleModel

_name_conversions =(
    (':PSQ_' , ':Q_'  ),
    (':PSQ_' , ':QV_' ),
    (':PSQ_' , ':QH_' ),
    (':PSC2_', ':DCH_'),
    (':PSC1_', ':DCV_'),
)
def fmname2mpname(name):
    for mp_ , fm_ in _name_conversions:
        name = name.replace(fm_,mp_)
    return name

def parse_lattice(lattice_text):
    """Efficiently parses the lattice definition and returns dictionaries of element properties and line definitions."""
    
    # Pre-compile regex patterns
    element_pattern = re.compile(r'([a-zA-Z0-9_:]+):\s*([a-zA-Z]+)(.*?);', re.DOTALL)
    line_pattern = re.compile(r'([a-zA-Z0-9_]+):\s*LINE\s*=\s*\((.*?)\);', re.DOTALL)
    property_pattern = re.compile(r'([a-zA-Z0-9_]+)\s*=\s*([+-]?\d*\.?\d+([eE][+-]?\d+)?)')
    
    # Clean and preprocess lines
    cleaned_lines = (re.sub(r'#.*', '', line).strip() for line in lattice_text.splitlines())
    lattice_text = "\n".join(filter(None, cleaned_lines))  # Remove empty lines
    
    elements = {}
    lines = {}
    
    # Parse element definitions
    for match in element_pattern.finditer(lattice_text):
        name, type_, properties_str = match.groups()
        properties = {"name": name, "type": type_, "L": 0}
        for prop_match in property_pattern.finditer(properties_str.strip()):
            prop_name, prop_value = prop_match.group(1), prop_match.group(2)
            properties[prop_name] = float(prop_value) if '.' in prop_value or 'e' in prop_value.lower() else int(prop_value)
        elements[name] = properties
    
    # Parse LINE definitions
    for match in line_pattern.finditer(lattice_text):
        line_name, line_content = match.groups()
        lines[line_name] = [elem.strip() for elem in line_content.split(",")]
    
    return elements, lines

def get_all_line_elements(line_name, lines, elements):
    """
    Recursively retrieves all elements from a nested LINE definition with indices, positions, and original properties.
    """
    visited = set()  # Avoid infinite recursion

    def recursive_helper(line_name):
        if line_name in visited:
            return []  # Avoid infinite recursion
        visited.add(line_name)
        
        all_elements = []
        if line_name in lines:  # If line_name exists in lines
            for elem in lines[line_name]:
                if elem in lines:  # Nested LINE definition
                    all_elements.extend(recursive_helper(elem))
                else:  # Single element or multiplied element
                    match = re.match(r"([a-zA-Z0-9_:]+)\*(\d+)", elem)
                    if match:
                        base_elem, multiplier = match.groups()
                        for _ in range(int(multiplier)):
                            all_elements.append(elements.get(base_elem, {"name": base_elem}))
                    else:
                        all_elements.append(elements.get(elem, {"name": elem}))
        return all_elements

    # Retrieve elements
    flat_elements = recursive_helper(line_name)
    
    # Add indices and positions
    cumulative_pos = 0
    for idx, elem in enumerate(flat_elements):
        elem["index"] = idx
        elem["pos"] = cumulative_pos
        cumulative_pos += elem.get("L", 0)  # Increment cumulative position by the element length (L)
    return flat_elements


def combine_lattice_elements_quads_only(filename, from_element=None, to_element=None, marker_types=['bpm'], Brho=None, line_name=None): 
    with open(filename, 'r') as file:
        lattice_text = file.read()
    
    # Find default line_name if not provided
    if line_name is None:
        use_match = re.search(r'USE:\s*([a-zA-Z0-9_]+);', lattice_text)
        if use_match:
            line_name = use_match.group(1)
        else:
            raise ValueError("No default LINE name found in the lattice file, and no line_name was provided.")

    elements, lines = parse_lattice(lattice_text)
    all_line_elements = get_all_line_elements(line_name, lines, elements)

    # Map from element name to element data
    name_to_data = {elem["name"]: elem for elem in all_line_elements}

    # Ensure from_element and to_element exist in the element list
    
    if from_element is not None and from_element not in name_to_data:
        raise ValueError(f"Starting element '{from_element}' not found in the LINE.")
    if to_element is not None and to_element not in name_to_data:
        raise ValueError(f"Ending element '{to_element}' not found in the LINE.")

    # Get indices of from_element and to_element
    if from_element is None:
        start_index = all_line_elements[1]["index"]
    else:
        start_index = name_to_data[from_element]["index"]
    if to_element is None:
        end_index = all_line_elements[-1]["index"]
    else:
        end_index = name_to_data[to_element]["index"]

    if start_index > end_index:
        raise ValueError("Starting element must come before ending element in the LINE.")

    # Subset of elements between from_element and to_element
    sub_elements = [elem for elem in all_line_elements if start_index <= elem["index"] <= end_index]

    combined_elements = []
    current_drift = None

    for elem in sub_elements:
        elem_type = elem.get("type", "").lower()

        if elem_type == "quadrupole":
            # Handle quadrupoles
            if current_drift:
                combined_elements.append(current_drift)
                current_drift = None     
            elem['Brho'] = Brho
            elem["aper"] = elem.get("aper", 0.1)
            combined_elements.append(elem)

        elif elem_type in marker_types or elem['name']==to_element:
            # Convert marker_types to drifts
            if current_drift:
                combined_elements.append(current_drift)
            elem["type"] = "drift"
            elem["aper"] = elem.get("aper", 0.1)
            elem["L"] = elem.get("L", 0)
            current_drift = elem

        else:
            # Combine other element types into a drift
            if current_drift:
                current_drift["L"] += elem.get("L", 0)
            else:
                current_drift = {
                    "name": elem["name"],
                    "type": "drift",
                    "index": elem["index"],
                    "pos": elem["pos"],
                    "L": elem.get("L", 0),
                    "aper": elem.get("aper", 0.1),   
                }
    # Append the last drift
    if current_drift:
        combined_elements.append(current_drift)
    #elem = sub_elements[-1]
    #elem["aper"] = elem.get("aper", 0.1)
    #combined_elements.append(sub_elements[-1])
    
    return combined_elements



def update_lattice_file(
    read_fname,
    write_fname,
    IonEk=None,
    IonQ=None,
    IonA=None,
    IonChargeStates=None,
    NCharge=None,
    BaryCenter0=None,
    S0=None,
):
    """
    Updates the lattice file with new values for specified parameters.

    Parameters:
        read_fname (str): The path to the lattice file.
        IonEk (float, optional): New kinetic energy [eV/u].
        IonQ (float, optional): New charge state.
        IonA (float, optional): New mass number.
        IonChargeStates (list[float], optional): New charge states.
        NCharge (list[float], optional): New number of charges.
        BaryCenter0 (list, np.array, or torch.Tensor, optional): New barycenter (shape 7).
        S0 (list, np.array, or torch.Tensor, optional): New beam envelope parameters (shape 49 or 7x7).

    Returns:
        None: Modifies the file in place.
    """

    # Read the file
    with open(read_fname, 'r') as file:
        lines = file.readlines()

    updated_lines = []
    skip_until_semicolon = False

    for line in lines:
        stripped = line.strip()

        # Keep commented lines intact
        if stripped.startswith("#"):
            updated_lines.append(line)
            continue

        # Skip lines until the end of the current block (marked by ';')
        if skip_until_semicolon:
            if stripped.endswith(";"):
                skip_until_semicolon = False
            continue

        # Update IonEk
        if IonEk is not None and stripped.startswith("IonEk"):
            updated_lines.append(f"IonEk = {IonEk}; \n")
            continue

        # Update IonQ
        if IonQ is not None and stripped.startswith("IonQ"):
            updated_lines.append(f"IonQ = {IonQ}; \n")
            continue

        # Update IonA
        if IonA is not None and stripped.startswith("IonA"):
            updated_lines.append(f"IonA = {IonA}; \n")
            continue

        # Update IonChargeStates
        if IonChargeStates is not None and stripped.startswith("IonChargeStates"):
            charges = ", ".join(map(str, IonChargeStates))
            updated_lines.append(f"IonChargeStates = [{charges}]; \n")
            continue

        # Update NCharge
        if NCharge is not None and stripped.startswith("NCharge"):
            charges = ", ".join(map(str, NCharge))
            updated_lines.append(f"NCharge = [{charges}]; \n")
            continue

        # Update BaryCenter0
        if BaryCenter0 is not None and stripped.startswith("BaryCenter0"):
            if isinstance(BaryCenter0, (np.ndarray, torch.Tensor)):
                BaryCenter0 = BaryCenter0.tolist()
            barycenter = ", ".join(map(str, BaryCenter0))
            updated_lines.append(f"BaryCenter0 = [{barycenter}]; \n")
            continue

        # Update S0
        if S0 is not None and stripped.startswith("S0"):
            skip_until_semicolon = True  # Skip existing S0 block
            if isinstance(S0, (np.ndarray, torch.Tensor)):
                S0 = S0.flatten().tolist()
            s_values = ",\n    ".join(", ".join(map(str, S0[i:i+7])) for i in range(0, len(S0), 7))
            updated_lines.append(f"S0 = [\n    {s_values}\n]; \n")
            continue

        # Keep other lines intact
        updated_lines.append(line)

    # Combine updated lines
    updated_content = "".join(updated_lines)

    # Write back to the file
    with open(write_fname, 'w') as file:
        file.write(updated_content)


def read_matrix_from_lattice_file(filename, matrix_name="S0"):
    """
    Reads a numpy array of shape (7, 7) from a specified matrix in the lattice file.

    Parameters:
        filename (str): The path to the lattice file.
        matrix_name (str): The name of the matrix to read (e.g., 'S0', 'S1').

    Returns:
        np.ndarray: The matrix as a numpy array.
    """
    with open(filename, 'r') as file:
        lines = file.readlines()

    matrix_lines = []
    inside_matrix = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith(f"{matrix_name} = ["):
            inside_matrix = True
            matrix_lines.append(stripped[len(f"{matrix_name} = ["):].strip())
            continue

        if inside_matrix:
            if stripped.endswith("];"):
                inside_matrix = False
                matrix_lines.append(stripped[:-2].strip())
            else:
                matrix_lines.append(stripped)

    # Clean up empty lines and trailing commas
    matrix_lines = [line.rstrip(',') for line in matrix_lines if line]

    # Convert matrix lines into a numpy array
    matrix_values = [list(map(float, row.split(","))) for row in matrix_lines]
    return np.array(matrix_values)


def update_lattice_file_from_bpmQscan(read_fname,
                                      write_fname,
                                      bpmQscan, 
                                      zero_couplings=True,
                                      from_element="LS3_WD06:BPM_D4699",
                                      to_element="LS3_WD06:BPM_D4699",
#                                       from_element=None, 
#                                       to_element="BDS_BTS:QV_D5501"
                                     ):
    
    if from_element == to_element:
        xcov = bpmQscan.model.xcovs[0].detach().numpy().copy()
        ycov = bpmQscan.model.ycovs[0].detach().numpy().copy()
    else:
        lattice_dicts = combine_lattice_elements_quads_only(
            read_fname, 
            from_element = from_element, 
            to_element   = to_element)
        
        for elem_dic in lattice_dicts:
            elem_dic['name'] = fmname2mpname(elem_dic['name'])
            
        model = EnvelopeEnsembleModel(E_MeV_u=bpmQscan.E_MeV_u, 
                                      mass_number=bpmQscan.mass_number, 
                                      charge_number=bpmQscan.charge_number,
                                      lattice_dicts=lattice_dicts[:-1])
        
        l_xcovs, l_ycovs = model.backward_simulate_beam_covars(bpmQscan.model.xcovs[:1], 
                                                               bpmQscan.model.ycovs[:1], 
                                                               i_monitors=[0,len(lattice_dicts)-1])
        xcov = l_xcovs[0,0].detach().numpy().copy()
        ycov = l_ycovs[0,0].detach().numpy().copy()
        
    xcov[0,0]*=1e6
    ycov[0,0]*=1e6
    xcov[0,1]*=1e3
    ycov[0,1]*=1e3
    xcov[1,0]*=1e3
    ycov[1,0]*=1e3
    S0 = read_matrix_from_lattice_file(read_fname)
    S0[:2,:2] = xcov
    S0[2:4,2:4] = ycov
    if zero_couplings:
        S0[:2,2:4] = 0.0
        S0[2:4,:2] = 0.0
        

    update_lattice_file(read_fname=read_fname, 
                        write_fname=write_fname,
                        S0=S0,
                        IonEk=bpmQscan.E_MeV_u*1e6, 
                        IonQ=bpmQscan.charge_number, 
                        IonA=bpmQscan.mass_number, 
                        IonChargeStates=[bpmQscan.charge_number/bpmQscan.mass_number], 
                       )
    
#     return l_xcovs, l_ycovs
