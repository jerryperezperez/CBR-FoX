import numpy as np
from custom_distance import sktime_interface

def cci_distance(input_data_dictionary, punishedSumFactor):
    print("Calculando correlación de Pearson")

    # pearsonCorrelation = sktime_interface.compute_distance_interface(input_data_dictionary, "pearson")
    pearsonCorrelation = sktime_interface.distance_sktime_interface(input_data_dictionary, sktime_interface.pearson)

    print("Calculando distancia Euclidiana")
    euclideanDistance = sktime_interface.distance_sktime_interface(input_data_dictionary, "euclidean")
    normalizedEuclideanDistance = (euclideanDistance - np.amin(euclideanDistance, axis=0)) / (np.amax(euclideanDistance, axis=0)-np.amin(euclideanDistance, axis=0))

    normalizedCorrelation = (.5 + (pearsonCorrelation - 2 * normalizedEuclideanDistance + 1) / 4)

    # TODO Necesario investigar si es la única forma de resolver este problema, así como si se puede atender desde otras secciones del código
    # To overcome 1-d arrays
    print("Prueba")
    correlationPerWindow = np.sum(((normalizedCorrelation + punishedSumFactor) ** 2), axis=1)
    if (correlationPerWindow.ndim == 1):
        correlationPerWindow = correlationPerWindow.reshape(-1, 1)
    # Applying scale
    correlationPerWindow = (correlationPerWindow - min(correlationPerWindow)) / (max(correlationPerWindow)-min(correlationPerWindow))
    # line to simulate correlation. Must be deleted
    #correlationPerWindow = np.sin(np.linspace(0, 10 * np.pi, 35000)).reshape(-1, 1)
    return correlationPerWindow
