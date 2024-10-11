import numpy as np
from sktime.distances import distance


# TODO Mejorar la estructura de la función
def compute_distance_interface(windows: np.ndarray, windows_len, components_len, target: None, metric, kwargs):
    correlation_per_window = np.array([])
    try:
        correlation_per_window = np.array(([distance(target[:, current_component],
                                   windows[current_window, :, current_component], metric,
                                   **kwargs)
                          for current_window in range(windows_len)
                          for current_component in range(components_len)])).reshape(-1, components_len)
    except ValueError as e:
        print(f"String or callable object is not valid for sktime library: {e}")
        try:
            correlation_per_window = metric(windows=windows, targetWindow=target, windowsLen=windows_len, componentsLen=components_len, **kwargs)
        except ValueError as e:
            print("The custom callable couldn't be executed")

    return correlation_per_window
