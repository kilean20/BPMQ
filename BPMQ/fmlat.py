import re
import os
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



def update_quad_B2_from_machine(read_fname, write_fname):
    """
    Reads all quadrupole currents from the live machine via EPICS (I_RD PVs),
    converts each current to a B2 field strength using the machine portal's
    calibration, and writes the updated FLAME lattice file.

    Naming convention
    -----------------
    FLAME lattice  : <section>:QV_D#### / :QH_D#### / :Q_D####
    Machine portal : <section>:PSQ_D####
    EPICS read PV  : <section>:PSQ_D####:I_RD   (never I_CSET)

    Parameters
    ----------
    read_fname  : str  Path to the input FLAME lattice file.
    write_fname : str  Path for the updated output file (may be the same as
                       read_fname to update in-place).

    Returns
    -------
    dict  {flame_element_name: new_B2_value} for every quadrupole updated.
    """
    try:
        from epics import caget_many as epics_caget_many
    except ImportError:
        raise ImportError(
            "The 'epics' package is required for update_quad_B2_from_machine. "
            "Install it or make sure it is on sys.path."
        )

    from .machine_portal_helper import get_MPelem_from_PVnames

    # ------------------------------------------------------------------
    # 1. Read the lattice file
    # ------------------------------------------------------------------
    with open(read_fname, 'r') as fh:
        lines = fh.readlines()

    # ------------------------------------------------------------------
    # 2. Identify every quadrupole element line and collect FLAME names
    #    Pattern:  <elem_name>: quadrupole, B2 = <value>, ...;
    #    (single-line definitions only, comments already start with '#')
    # ------------------------------------------------------------------
    _quad_line_re = re.compile(
        r'^([a-zA-Z0-9_:]+)\s*:\s*quadrupole\b'
    )
    _b2_field_re = re.compile(
        r'(B2\s*=\s*)([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)'
    )

    quad_fm_names  = []   # FLAME element names, in file order
    quad_line_idxs = []   # corresponding line indices in `lines`

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        m = _quad_line_re.match(stripped)
        if m:
            quad_fm_names.append(m.group(1))
            quad_line_idxs.append(idx)

    if not quad_fm_names:
        raise ValueError(
            f"No quadrupole elements found in '{read_fname}'."
        )

    # ------------------------------------------------------------------
    # 3. Convert FLAME element names → machine portal PV base names
    #    e.g. LS3_BTS:QV_D4713  →  LS3_BTS:PSQ_D4713
    # ------------------------------------------------------------------
    quad_mp_names = [fmname2mpname(name) for name in quad_fm_names]

    # ------------------------------------------------------------------
    # 4. Build I_RD PV list and fetch all currents in a single caget_many
    # ------------------------------------------------------------------
    i_rd_pvs = [mp_name + ':I_RD' for mp_name in quad_mp_names]

    currents = epics_caget_many(i_rd_pvs)

    # Validate – caget_many returns None for unreachable PVs
    failed = [pv for pv, val in zip(i_rd_pvs, currents) if val is None]
    if failed:
        raise RuntimeError(
            f"EPICS caget_many failed (returned None) for {len(failed)} PV(s):\n"
            + "\n".join(f"  {pv}" for pv in failed)
        )

    # ------------------------------------------------------------------
    # 5. Get machine-portal element objects and convert I → B2
    # ------------------------------------------------------------------
    mp_elems = get_MPelem_from_PVnames(quad_mp_names)

    b2_values = [
        mp_elem.convert(curr, from_field='I', to_field='B2')
        for mp_elem, curr in zip(mp_elems, currents)
    ]

    # ------------------------------------------------------------------
    # 6. Patch B2 values in the lines list (in-place string substitution)
    # ------------------------------------------------------------------
    for line_idx, b2 in zip(quad_line_idxs, b2_values):
        lines[line_idx] = _b2_field_re.sub(
            lambda m, b2=b2: m.group(1) + repr(float(b2)),
            lines[line_idx],
            count=1,          # only the first B2 = ... on that line
        )

    # ------------------------------------------------------------------
    # 7. Write the updated lattice file
    # ------------------------------------------------------------------
    with open(write_fname, 'w') as fh:
        fh.writelines(lines)

    return dict(zip(quad_fm_names, b2_values))



import os   # add to existing imports at the top of fmlat.py

# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _fetch_quad_B2_from_machine(file_lines):
    """
    Reads all quadrupole currents from the live machine via EPICS (I_RD PVs),
    converts each to a B2 value using the machine-portal calibration, and
    returns both a name→B2 mapping and the file lines patched with new B2s.

    Operates on ALL quadrupole elements found in `file_lines` so that an
    optional file-write covers the entire lattice, not just a sub-range.

    Parameters
    ----------
    file_lines : list[str]
        Raw lines of a FLAME lattice file (as returned by file.readlines()).

    Returns
    -------
    fm_name_to_b2 : dict  {flame_element_name: new_B2_float}
    patched_lines : list[str]
        Copy of `file_lines` with every quadrupole's B2 value replaced.

    Raises
    ------
    ImportError  if the 'epics' package is not available.
    RuntimeError if any I_RD PV returns None (unreachable channel).
    """
    try:
        from epics import caget_many as epics_caget_many
    except ImportError:
        raise ImportError(
            "'epics' is required for update_B2_from_machine. "
            "Install it or ensure it is on sys.path."
        )

    from .machine_portal_helper import get_MPelem_from_PVnames

    _quad_line_re = re.compile(r'^([a-zA-Z0-9_:]+)\s*:\s*quadrupole\b')
    _b2_token_re  = re.compile(r'(B2\s*=\s*)([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)')

    # --- 1. Identify every quadrupole line in the file --------------------
    quad_fm_names  = []   # FLAME element names, in file order
    quad_line_idxs = []   # matching index into file_lines

    for idx, line in enumerate(file_lines):
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        m = _quad_line_re.match(stripped)
        if m:
            quad_fm_names.append(m.group(1))
            quad_line_idxs.append(idx)

    if not quad_fm_names:
        raise ValueError("No quadrupole elements found in the supplied file lines.")

    # --- 2. FLAME names → machine portal PV base names -------------------
    #    e.g.  LS3_BTS:QV_D4713  →  LS3_BTS:PSQ_D4713
    quad_mp_names = [fmname2mpname(name) for name in quad_fm_names]

    # --- 3. Fetch all I_RD values in a single caget_many -----------------
    i_rd_pvs = [mp + ':I_RD' for mp in quad_mp_names]
    currents  = epics_caget_many(i_rd_pvs)

    failed = [pv for pv, val in zip(i_rd_pvs, currents) if val is None]
    if failed:
        raise RuntimeError(
            f"epics.caget_many returned None for {len(failed)} PV(s):\n"
            + "\n".join(f"  {pv}" for pv in failed)
        )

    # --- 4. Convert I → B2 via machine-portal calibration ----------------
    mp_elems = get_MPelem_from_PVnames(quad_mp_names)
    b2_values = [
        mp_elem.convert(curr, from_field='I', to_field='B2')
        for mp_elem, curr in zip(mp_elems, currents)
    ]

    # --- 5. Patch file lines (substitute B2 token, leave everything else) -
    patched_lines = list(file_lines)   # shallow copy — strings are immutable
    for line_idx, b2 in zip(quad_line_idxs, b2_values):
        patched_lines[line_idx] = _b2_token_re.sub(
            lambda m, _b2=b2: m.group(1) + repr(float(_b2)),
            patched_lines[line_idx],
            count=1,
        )

    fm_name_to_b2 = dict(zip(quad_fm_names, b2_values))
    return fm_name_to_b2, patched_lines


# ---------------------------------------------------------------------------
# Public function  (replaces the original combine_lattice_elements_quads_only)
# ---------------------------------------------------------------------------

def combine_lattice_elements_quads_only_w_live_update(
        filename,
        from_element=None,
        to_element=None,
        marker_types=['bpm'],
        Brho=None,
        line_name=None,
        update_B2_from_machine=False,
        update_file=False,
):
    """
    Parse a FLAME lattice file and return a list of element dicts in which
    everything that is not a quadrupole (or a selected marker type) is merged
    into a single drift element.

    Parameters
    ----------
    filename : str
        Path to the FLAME .lat file.
    from_element : str, optional
        Name of the first element to include (default: second element in LINE).
    to_element : str, optional
        Name of the last element to include (default: last element in LINE).
    marker_types : list[str]
        Element types to keep as individual drift-like entries (default: ['bpm']).
    Brho : float, optional
        Magnetic rigidity [T·m] injected into every quadrupole dict.
    line_name : str, optional
        Name of the LINE to use; auto-detected from 'USE:' if omitted.
    update_B2_from_machine : bool
        If True, fetch live I_RD values from EPICS and replace B2 in every
        quadrupole dict before returning.  Requires the 'epics' package and
        a live machine connection.
    update_file : bool
        If True (and update_B2_from_machine is True), write the updated lattice
        to  <filename_without_extension>_updatedB2.lat  with all 
        quadrupole B2 values replaced throughout the whole file.

    Returns
    -------
    combined_elements : list[dict]
        Ordered list of element dicts (quadrupoles and merged drifts/markers).
    """
    with open(filename, 'r') as fh:
        file_lines = fh.readlines()
    lattice_text = "".join(file_lines)

    # ------------------------------------------------------------------
    # Optionally fetch live B2 values from the machine
    # ------------------------------------------------------------------
    fm_name_to_b2 = {}
    if update_B2_from_machine:
        fm_name_to_b2, patched_lines = _fetch_quad_B2_from_machine(file_lines)

        if update_file:
            base, _ = os.path.splitext(filename)
            out_fname = base + '_updatedB2.lat'
            with open(out_fname, 'w') as fh:
                fh.writelines(patched_lines)

        # Rebuild lattice_text from the patched lines so that parse_lattice
        # also sees the updated B2 values (keeps internal state consistent).
        lattice_text = "".join(patched_lines)

    # ------------------------------------------------------------------
    # Parse lattice and resolve LINE hierarchy  (unchanged logic)
    # ------------------------------------------------------------------
    if line_name is None:
        use_match = re.search(r'USE:\s*([a-zA-Z0-9_]+);', lattice_text)
        if use_match:
            line_name = use_match.group(1)
        else:
            raise ValueError(
                "No default LINE name found in the lattice file, "
                "and no line_name was provided."
            )

    elements, lines = parse_lattice(lattice_text)
    all_line_elements = get_all_line_elements(line_name, lines, elements)

    name_to_data = {elem["name"]: elem for elem in all_line_elements}

    if from_element is not None and from_element not in name_to_data:
        raise ValueError(f"Starting element '{from_element}' not found in the LINE.")
    if to_element is not None and to_element not in name_to_data:
        raise ValueError(f"Ending element '{to_element}' not found in the LINE.")

    start_index = (name_to_data[from_element]["index"] if from_element is not None
                   else all_line_elements[1]["index"])
    end_index   = (name_to_data[to_element]["index"]   if to_element   is not None
                   else all_line_elements[-1]["index"])

    if start_index > end_index:
        raise ValueError("Starting element must come before ending element in the LINE.")

    sub_elements = [e for e in all_line_elements if start_index <= e["index"] <= end_index]

    # ------------------------------------------------------------------
    # Merge non-quad, non-marker elements into drifts  (unchanged logic)
    # ------------------------------------------------------------------
    combined_elements = []
    current_drift = None

    for elem in sub_elements:
        elem_type = elem.get("type", "").lower()

        if elem_type == "quadrupole":
            if current_drift:
                combined_elements.append(current_drift)
                current_drift = None
            elem['Brho'] = Brho
            elem["aper"] = elem.get("aper", 0.1)
            # Apply live B2 if available (fm_name_to_b2 is empty when
            # update_B2_from_machine=False, so the lookup is a no-op).
            if elem["name"] in fm_name_to_b2:
                elem["B2"] = fm_name_to_b2[elem["name"]]
            combined_elements.append(elem)

        elif elem_type in marker_types or elem['name'] == to_element:
            if current_drift:
                combined_elements.append(current_drift)
            elem["type"] = "drift"
            elem["aper"] = elem.get("aper", 0.1)
            elem["L"]    = elem.get("L", 0)
            current_drift = elem

        else:
            if current_drift:
                current_drift["L"] += elem.get("L", 0)
            else:
                current_drift = {
                    "name":  elem["name"],
                    "type":  "drift",
                    "index": elem["index"],
                    "pos":   elem["pos"],
                    "L":     elem.get("L", 0),
                    "aper":  elem.get("aper", 0.1),
                }

    if current_drift:
        combined_elements.append(current_drift)

    return combined_elements