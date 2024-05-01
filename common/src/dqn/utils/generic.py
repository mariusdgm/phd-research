def merge_dictionaries(dict1, dict2):
    merged_dict = dict1.copy()  # Create a copy of the first dictionary

    for key, value in dict2.items():
        if key in merged_dict:
            merged_dict[key] += value  # Add the values if the key exists
        else:
            merged_dict[key] = value  # Add the key-value pair if the key doesn't exist

    return merged_dict


def replace_keys(d, original_key, new_key):
    """
    Recursively iterates through a dictionary and its sub-dictionaries,
    replacing a specified key with a new key.
    """
    new_dict = {}
    for key, value in d.items():
        # Replace the key if it matches the specified original key
        updated_key = new_key if key == original_key else key

        # Recursively apply this function if the value is another dictionary
        if isinstance(value, dict):
            new_dict[updated_key] = replace_keys(value, original_key, new_key)
        else:
            new_dict[updated_key] = value
    return new_dict
