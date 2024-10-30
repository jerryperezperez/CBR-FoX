import numpy as np
from custom_distance import sktime_interface
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def cci_distance(input_data_dictionary, punishedSumFactor):
    # print("Calculando correlación de Pearson")
    logging.info("Aplicando Correlación de Pearson")
    pearsonCorrelation = sktime_interface.distance_sktime_interface(input_data_dictionary, sktime_interface.pearson)

    logging.info("Aplicando Correlación Euclidiana")
    euclideanDistance = sktime_interface.distance_sktime_interface(input_data_dictionary, "euclidean")
    normalizedEuclideanDistance = (euclideanDistance - np.amin(euclideanDistance, axis=0)) / (np.amax(euclideanDistance, axis=0)-np.amin(euclideanDistance, axis=0))

    normalizedCorrelation = (.5 + (pearsonCorrelation - 2 * normalizedEuclideanDistance + 1) / 4)

    # To overcome 1-d arrays

    correlationPerWindow = np.sum(((normalizedCorrelation + punishedSumFactor) ** 2), axis=1)
    if (correlationPerWindow.ndim == 1):
        correlationPerWindow = correlationPerWindow.reshape(-1, 1)
    # Applying scale
    correlationPerWindow = (correlationPerWindow - min(correlationPerWindow)) / (max(correlationPerWindow)-min(correlationPerWindow))
    return correlationPerWindow
