# Import the factory class
from factory.cbr_fox_factory import cbr_fox_factory
from cbr_fox import cbr_fox
import numpy as np
from custom_distance.cci_distance import  cci_distance

cci
# Example time series input for the ANN
inputnn = np.random.rand(11, 100)  # 11 windows of 100 time steps each

# Define windows and target window
windows = inputnn[0:-1]  # Use all windows except the last one
targetWindow = inputnn[-1]  # The last window is the target case

# Number of cases for training (example value)
num_cases = 10

# Define the target (for example, a label or prediction target)
target = np.random.rand(10)  # Target values for training, aligned with num_cases

instance = cbr_fox("dtw")
isntance_2 = cbr_fox(cci_distance)

# Example of using the create_multiple method
techniques = [("dtw", {}), ("CCI", {}), ("other_technique", {})]

factory = cbr_fox_factory(techniques)

dtw_technique = factory["dtw"]  # Access "dtw" technique like a dictionary
print(dtw_technique)  # This will print the corresponding cbr_fox instance for "dtw"


