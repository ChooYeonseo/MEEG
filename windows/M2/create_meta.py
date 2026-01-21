def create_meta(name, memo="", date="", activated_channels=None):
    """
    Create metadata dictionary from filename and additional parameters.
    
    Parameters:
    -----------
    name : str
        Data ID / filename (e.g., 'M1AP01')
    memo : str
        User notes/memo text
    date : str
        Recording date
    activated_channels : list
        List of activated channel names (will be sorted alphabetically)
    
    Returns:
    --------
    dict
        Complete metadata dictionary
    """
    if activated_channels is None:
        activated_channels = []
    
    meta_data = {
        "Species": species_meta(name),
        "Strain": two_digit_meta(name)[0],
        "Background": two_digit_meta(name)[1],
        "Drug": one_digit_meta(name),
        "ExtraData": Unique_Meta(name),
        "Memo": memo,
        "Date": date,
        "ActivatedChannels": sorted(activated_channels)
    }
    
    return meta_data


def species_meta(name):
    key = name[0]
    if key == 'M':
        return 'Mouse'
    if key == 'R':
        return 'Rat'
    elif key == 'H':
        return 'Human'
    elif key == 'K':
        return 'Monkey'
    elif key == 'J':
        return 'Rabbit'
    elif key == 'C':
        return 'Cat'
    elif key == 'D':
        return 'Dog'
    elif key == 'O':
        return 'Other'
    else:
        return 'Unknown'

def two_digit_meta(name):
    if species_meta(name) == 'Mouse':
        key = name[1:3]
        if key == '1A':
            return "Wild_Type", "C57BL/6"
        elif key == '2A':
            return "NR2D_Knock_Out", "C57BL/6"
        else:
            return "Unknown", "Unknown"
    return "Unknown", "Unknown"

def one_digit_meta(name):
    if species_meta(name) == 'Mouse':
        key = name[3]
        if key == 'N':
            return "Drug Not Administered"
        elif key == 'P':
            return "Pilocarpine"
        elif key == "K":
            return "Kainic Acid"
        elif key == "E":
            return "Ethanol"
        elif key == "L":
            return "L-Dopa"
        else:
            return "Unknown"
    return "Unknown"

def Unique_Meta(name):
    """
    Extract the 2-digit ExtraData string from the filename.
    
    For Mouse species, ExtraData is at position 4-5 (0-indexed: 4:6)
    Example: 'M1AP01' -> '01'
    
    Parameters:
    -----------
    name : str
        Data ID / filename
    
    Returns:
    --------
    str
        2-digit ExtraData string, or empty string if not found
    """
    if len(name) >= 6 and species_meta(name) == 'Mouse':
        return name[4:6]
    return ""