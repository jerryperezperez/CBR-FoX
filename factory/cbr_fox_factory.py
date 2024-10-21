from cbr_fox import cbr_fox


class cbr_fox_factory:
    def __init__(self, techniques):
        # Store techniques as a dictionary, where the key is the technique name and the value is the cbr_fox object
        self.techniques_dict = dict()
        for item in techniques:
            if isinstance(item, tuple) and len(item) == 2:
                name, config = item
                self.techniques_dict[name] = cbr_fox(name, **config)
            else:
                self.techniques_dict[item] = cbr_fox(item)
    
    def explan_all_techniques(self,  training_windows, target_training_windows, forecasted_window, prediction, num_cases):
        for technique in self.techniques_dict:
            technique.explain(training_windows, target_training_windows, forecasted_window, prediction, num_cases)

    # Override __getitem__ to allow dictionary-like access
    def __getitem__(self, technique_name):
        # Return the corresponding cbr_fox object for the requested technique
        if technique_name in self.techniques_dict:
            return self.techniques_dict[technique_name]
        else:
            raise KeyError(f"Technique '{technique_name}' not found.")


