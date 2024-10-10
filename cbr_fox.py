import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sktime.distances import distance
from scipy import signal
from statsmodels.nonparametric.smoothers_lowess import lowess

# TODO Revisar si es conveniente agregar como atributo de clase a correlation_per_windows para facilitar el acceso en los métodos
# o si por tema de memoria sería adecuado solo almacenar smoothed_correlation

class cbr_fox:
    def __init__(self, metric: str or callable = "dtw", smoothness_factor: float = .2, kwargs: dict = {}):
        # Variables for setting

        self.metric = metric
        self.smoothness_factor = smoothness_factor
        self.kwargs = kwargs
        # Variables for results
        # self.outputComponentsLen = len(outputNames)
        self.smoothed_correlation = None
        self.analysisReport = None
        self.best_windows_index = list()
        self.worst_windows_index = list()
        self.bestMAE = list()
        self.worstMAE = list()
        #Private variables for easy access by private methods
        self.__correlation_per_window = None
    # PRIVATE METHODS. ALL THESE METHODS ARE USED INTERNALLY FOR PROCESSING AND ANALYSIS
    def _smoothe_correlation(self):
        return lowess(self.__correlation_per_window, np.arange(len(self.__correlation_per_window)),
                      self.smoothness_factor)[:, 1]

    def _identify_valleys_peaks_indexes(self):
        return signal.argrelextrema(self.smoothed_correlation, np.less)[0], \
            signal.argrelextrema(self.smoothed_correlation, np.greater)[0]

    # TODO Analizar si conviene hacer que retorne los valores y luego asigne, con único fin de seguir el estándar
    def _retreive_concave_convex_segments(self, windows_len):
        self.concaveSegments = np.split(
            np.transpose(np.array((np.arange(windows_len), self.smoothed_correlation))),
            self.valley_index)
        self.convexSegments = np.split(
            np.transpose(np.array((np.arange(windows_len), self.smoothed_correlation))),
            self.peak_index)

    # TODO Analizar si conviene hacer que retorne los valores y luego asigne, con único fin de seguir el estándar
    def _retreive_original_indexes(self):
        for split in self.concaveSegments:
            self.best_windows_index.append(int(split[np.where(split == max(split[:, 1]))[0][0], 0]))
        for split in self.convexSegments:
            self.worst_windows_index.append(int(split[np.where(split == min(split[:, 1]))[0][0], 0]))

    # TODO Analizar si este método puede ser el único que permita realizar asignaciones de variable internamente
    def _compute_statistics(self, target_training_windows: np.ndarray, forecasted_window: np.ndarray, prediction: np.ndarray, num_cases: int):

        self.bestDic = {index: self.__correlation_per_window[index] for index in self.best_windows_index}

        self.worstDic = {index: self.__correlation_per_window[index] for index in self.worst_windows_index}

        self.bestDic = sorted(self.bestDic.items(), reverse=True, key=lambda x: x[1])

        self.worstDic = sorted(self.worstDic.items(), key=lambda x: x[1])

        self.bestDic = self.bestDic[0:num_cases]
        self.worstDic = self.worstDic[0:num_cases]

        print("Calculando MAE para cada ventana")
        for tupla in self.bestDic:
            self.bestMAE.append(
                mean_absolute_error(target_training_windows[tupla[0]], prediction.reshape(-1, 1)))

        for tupla in self.worstDic:
            self.worstMAE.append(
                mean_absolute_error(target_training_windows[tupla[0]], prediction.reshape(-1, 1)))

        print("Generando reporte de análisis")
        # Version with method name as column name
        # d = {'Index': dict(self.bestSorted).keys(), self.method: dict(self.bestSorted).values(), "Best MAE": self.bestMAE,
        #      'Index': dict(self.worstSorted).keys(), self.method: dict(self.worstSorted).values(),
        #      "Worst MAE": self.worstMAE}
        # Version 1.0 when CCI was the only one method defined
        # TODO Atender impresión de nombre de acuerdo su tipo (String o Callable)
        # TODO Verificar correcto funcionamiento de creación del objeto Dataframe
        d = {'index': dict(self.bestDic).keys(), self.method: dict(self.bestDic).values(), "MAE": self.bestMAE}
        # The worst indices were disabled in order to remove the duplicated keys such as index.1 and so forth
        # 'index.1': dict(self.worstSorted).keys(), 'other': dict(self.worstSorted).values(),
        # "MAE.1": self.worstMAE
        self.analysisReport = pd.DataFrame(data=d)

    def _compute_cbr_analysis(self, windows_len: int):
        self.smoothed_correlation = self._smoothe_correlation()
        self.valley_index, self.peak_index = self._identify_valleys_peaks_indexes()
        self._retreive_concave_convex_segments(windows_len)
        self._retreive_original_indexes()


    def _compute_distance(self, windows: np.ndarray,windows_len, components_len, target: None):
        correlation_per_window = np.array(([distance(target[:, current_component],
                                                        windows[current_window, :, current_component], self.metric,
                                                        **self.kwargs)
                                               for current_window in range(windows_len)
                                               for current_component in range(components_len)])).reshape(-1,
                                                                                                             components_len)
        correlation_per_window = np.sum(correlation_per_window, axis=1)
        correlation_per_window = ((correlation_per_window-min(correlation_per_window))/
                                  (max(correlation_per_window)-min(correlation_per_window)))
        return correlation_per_window

    # PUBLIC METHODS. ALL THESE METHODS ARE PROVIDED FOR THE USER

    # Main method. This method allows to the user to perform the primary function. User need to invoke it in order to call others public methods
    def explain(self, training_windows: np.ndarray, target_training_windows: np.ndarray, forecasted_window: np.ndarray, prediction: np.ndarray, num_cases: int):
        # gather some basic data from passed in variables
        components_len = training_windows.shape[2]
        window_len = training_windows.shape[1]
        windows_len = len(training_windows)

        self.__correlation_per_window = self._compute_distance(training_windows, windows_len, components_len, forecasted_window)
        self._compute_cbr_analysis(windows_len)
        self._compute_statistics(target_training_windows, forecasted_window, prediction, num_cases)

    # Method to print a chart or graphic based on results stored in variables. These methods are not strictly necessary
    #   for underlying functionality
    def visualize(self):
        pass

    def get_analysis_report(self):
        return self.analysisReport
