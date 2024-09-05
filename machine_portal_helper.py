from phantasy import MachinePortal, disable_warnings
from utils import get_Dnum_from_pv, split_name_field_from_PV, suppress_outputs, _name_conversions
from typing import List, Dict


disable_warnings()
_mp = MachinePortal(machine='FRIB', segment='LINAC')

   
    
def get_MPelem_from_PVnames(
    names: List, 
    mp: MachinePortal = _mp) -> List or None:
    """
    Retrieves MachinePortal elements from a list of PVs.

    Args:
        names (list): List of PV names strings.
        mp (MachinePortal): MachinePortal instance.

    Returns:
        list or None: List of MachinePortal elements corresponding to the PVs.
    """
    # Check if mp is provided, otherwise use the default MachinePortal instance
    if mp is None:
        mp = MachinePortal(machine='FRIB', segment='LINAC')
    mp_names = mp.get_all_names()
    mp_dnums = [get_Dnum_from_pv(mp_name) for mp_name in mp_names]
    elems = []
    for name in names:
        with suppress_outputs():
            elem = mp.get_elements(name=name)
        if len(elem) == 0:
            # Try replacements
            for orig, new in _name_conversions:
                with suppress_outputs():
                    elem = mp.get_elements(name=name.replace(orig, new))
                if len(elem) > 0:
                    break

            # If still not found, get elem from matching dnum
            if len(elem) == 0:
                try:
                    i = mp_dnums.index(get_Dnum_from_pv(name))
                except:
                    try:
                        i = mp_dnums.index(get_Dnum_from_pv(name)+1)
                    except:
                        i = mp_dnums.index(get_Dnum_from_pv(name)-1)        
                if i >= 0:
                    with suppress_outputs():
                        elem = mp.get_elements(name=mp_names[i])
        if len(elem) == 0:
            elems.append(None)
            print(f"MachinePortal element is not found for PV: {name}")
        else:
            elems.append(elem[0])
    return elems    