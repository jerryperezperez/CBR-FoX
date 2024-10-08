from cbr_fox import cbr_fox


class cbr_fox_factory:
    def __init__(self, techniques):
        # Store techniques as a dictionary, where the key is the technique name and the value is the cbr_fox object
        self.techniques_dict = {technique: cbr_fox(technique, **params) for technique, params in techniques}

    # Override __getitem__ to allow dictionary-like access
    def __getitem__(self, technique_name):
        # Return the corresponding cbr_fox object for the requested technique
        if technique_name in self.techniques_dict:
            return self.techniques_dict[technique_name]
        else:
            raise KeyError(f"Technique '{technique_name}' not found.")


