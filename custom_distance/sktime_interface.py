import numpy as np
from sktime.distances import distance


# TODO Mejorar la estructura de la función
def compute_distance_interface(input_data_dictionary, metric, kwargs):
    correlation_per_window = np.array([])
    try:
        correlation_per_window = np.array(([distance(input_data_dictionary["target"][:, current_component],
                                                     input_data_dictionary["windows"][current_window, :,
                                                     current_component], metric,
                                                     **kwargs)
                                            for current_window in range(input_data_dictionary["windows_len"])
                                            for current_component in
                                            range(input_data_dictionary["components_len"])])).reshape(-1,
                                                                                                      input_data_dictionary[
                                                                                                          "components_len"])
    except ValueError as e:
        print(f"String or callable object is not valid for sktime library: {e}")
        try:
            correlation_per_window = metric(input_data_dictionary, **kwargs)
        except ValueError as e:
            print("The custom callable couldn't be executed")

    return correlation_per_window
