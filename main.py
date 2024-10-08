# Import the factory class
from factory.cbr_fox_factory import cbr_fox_factory

# Example of using the create_single method
single_instance = cbr_fox_factory.create_single("dtw")
print(single_instance)  # This will print the single `cbr_fox` instance created with the "dtw" technique.
single_instance.explain(time_series_1=None, time_series_2=None)
single_instance.get_analysis_report()


# Example of using the create_multiple method
techniques = [("dtw", {}), ("CCI", {}), ("other_technique", {})]
multiple_instances = cbr_fox_factory.create_multiple(techniques)
multiple_instances.explain(time_series_1=None, time_series_2=None, techniques=techniques)
print(multiple_instances)  # This will print the list of `cbr_fox` instances, each created with different techniques.
