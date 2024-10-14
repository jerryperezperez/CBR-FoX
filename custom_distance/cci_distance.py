import numpy as np
import sktime_interface

def cci_distance(input_data_dictionary, punishedSumFactor):
    print("Calculando correlación de Pearson")

    # pearsonCorrelation = sktime_interface.compute_distance_interface(input_data_dictionary, "pearson")
    pearsonCorrelation = sktime_interface.compute_distance_interface(input_data_dictionary, sktime_interface.pearson)

    print("Calculando distancia Euclidiana")
    euclideanDistance = sktime_interface.compute_distance_interface(input_data_dictionary, "euclidean")
    normalizedEuclideanDistance = (euclideanDistance - np.amin(euclideanDistance, axis=0)) / (np.amax(euclideanDistance, axis=0)-np.amin(euclideanDistance, axis=0))

    normalizedCorrelation = (.5 + (pearsonCorrelation - 2 * normalizedEuclideanDistance + 1) / 4)

    correlationPerWindow = np.sum(((normalizedCorrelation + punishedSumFactor) ** 2), axis=1)
    # Applying scale
    correlationPerWindow = (correlationPerWindow - min(correlationPerWindow)) / (max(correlationPerWindow)-min(correlationPerWindow))
    return correlationPerWindow
