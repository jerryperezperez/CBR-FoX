# Import the factory class
from factory.cbr_fox_factory import cbr_fox_factory
from cbr_fox import cbr_fox
import numpy as np
from custom_distance.cci_distance import cci_distance

training_windows = np.array([
    [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],  # Sample 1
    [[11], [12], [13], [14], [15], [16], [17], [18], [19], [20]],  # Sample 2
    [[21], [22], [23], [24], [25], [26], [27], [28], [29], [30]],  # Sample 3
    [[31], [32], [33], [34], [35], [36], [37], [38], [39], [40]],  # Sample 4
    [[41], [42], [43], [44], [45], [46], [47], [48], [49], [50]]])

target_training_windows = np.array([
    [[10]],  # Target for the first training window
    [[11]],  # Target for the second training window
    [[12]],  # Target for the third training window
    [[13]],  # Target for the fourth training window
    [[14]],  # Target for the fifth training window
])
forecasted_window = np.array([
    [[51]],  # Timestep 1
    [[52]],  # Timestep 2
    [[53]]   # Timestep 3
])
forecasted_window_array = np.array(forecasted_window)
prediction = np.array([[54]])
# Number of cases for training (example value)
num_cases = 2

# Define the target (for example, a label or prediction target)
target = np.random.rand(10)  # Target values for training, aligned with num_cases

instance = cbr_fox("dtw")
instance_2 = cbr_fox("dtw")
# instance_2 = cbr_fox(cci_distance, kwargs={"punishedSumFactor":.5})
instance_2.explain(training_windows, target_training_windows, forecasted_window, prediction, num_cases)
# Example of using the create_multiple method
techniques = [("dtw", {}), ("CCI", {}), ("other_technique", {})]

factory = cbr_fox_factory(techniques)

dtw_technique = factory["dtw"]  # Access "dtw" technique like a dictionary
print(dtw_technique)  # This will print the corresponding cbr_fox instance for "dtw"


