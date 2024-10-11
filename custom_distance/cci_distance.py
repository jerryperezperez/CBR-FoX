import numpy as np


def cci_distance(input_data_dictionary, punishedSumFactor):
    print("Calculando correlación de Pearson")


    pearsonCorrelation = np.array(
        (
            [np.corrcoef(input_data_dictionary["windows"][currentWindow, :, currentComponent],
                         input_data_dictionary["target"][:, currentComponent])[
                 0][1]
             for currentWindow in range(input_data_dictionary["windows_len"]) for currentComponent in
             range(input_data_dictionary["components_len"])])).reshape(
        -1,
        input_data_dictionary["components_len"])
    print("Calculando distancia Euclidiana")
    euclideanDistance = np.array(
        ([np.linalg.norm(
            input_data_dictionary["target"][:, currentComponent] - input_data_dictionary["window"][currentWindow, :, currentComponent])
            for currentWindow in range(input_data_dictionary["windows_len"]) for currentComponent in
            range(input_data_dictionary["components_len"])])).reshape(
        -1,
        input_data_dictionary["components_len"])
    normalizedEuclideanDistance = euclideanDistance / np.amax(euclideanDistance, axis=0)

    normalizedCorrelation = (.5 + (pearsonCorrelation - 2 * normalizedEuclideanDistance + 1) / 4)

    correlationPerWindow = np.sum(((normalizedCorrelation + punishedSumFactor) ** 2), axis=1)
    # Applying scale
    correlationPerWindow /= max(correlationPerWindow)
    return correlationPerWindow
