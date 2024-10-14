import numpy as np
from sktime.distances import distance
from typing import Callable, Union


# TODO Mejorar la estructura de la función
def compute_distance_interface(input_data_dictionary,
                               metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
                               kwargs):
    correlation_per_window = np.array([])
    try:
        # Sustituido por distance_process. De todas formas, verificar funcionalidad
        correlation_per_window = np.array(
            ([distance(input_data_dictionary["forecasted_window"][:, current_component],
                               input_data_dictionary["training_windows"][current_window, :,
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


# def distance_process(evaluate_component, target_component, metric, **kwargs):
#     if metric == "pearson":
#         return np.corrcoef(evaluate_component, target_component)[0][1]
#     else:
#         return distance(evaluate_component, target_component, metric, **kwargs)

def pearson(x, y):
    return np.corrcoef(x, y)[0][1]
