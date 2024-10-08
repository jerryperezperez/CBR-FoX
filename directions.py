import numpy as np
from sktime.distances import distance
from sktime.distances import dtw_distance
# define arrays
time_series_1 = np.array([1, 2, 3, 4, 5])
time_series_2 = np.array([2, 3, 4, 5, 6])
# define custom function if needed

def funcion (x: np.ndarray, y: np.ndarray, factor: float = 1.0) -> float:
    return factor * np.sqrt(np.sum((x - y) ** 2))
dtw_kwargs = {
    'window': 1.0  # Example of passing the window argument
}
# invoke string-based functions
print(distance(time_series_1, time_series_2, "dtw"))
# passing in extra keyword arguments for custom instances
print(distance(time_series_1, time_series_2, "dtw", **dtw_kwargs))
print(distance(time_series_1, time_series_2, "dtw"))
# this prints out "np.float64(2.0)"
print("another example ", distance(time_series_1, time_series_2, dtw_distance))
# invoke callable-based function
print(distance(time_series_1, time_series_2, funcion))
# this prints out "np.float64(2.23606797749979)"

metric_and_kwargs = [("dtw", {}), ("CCI", {})]
# Define the metric and kwargs as a tuple
metric_and_kwargs = ("dtw", {})

# Unpack the tuple and pass to the distance function
result = distance(
    time_series_1,
    time_series_2,
    metric=metric_and_kwargs[0],  # First element of the tuple (the metric)
    **metric_and_kwargs[1]         # Unpack the second element (the kwargs dictionary)
)

print(f"Distance result: {result}")

# Wrapper class should be so (extra parameters for library purpose are omitted for the sake of simplicity):
# Wrapper function for one technique
def wrapper_function(time_series_1, time_series_2,string_name_technique):
    pass
#Wrapper function for many techniques
def wrapper_many_functions(time_series_1, time_series_2):
    pass