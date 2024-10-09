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
        self.best_windows_index = None
        self.worst_windows_index = None
        self.bestMAE = None
        self.worstMAE = None
    # PRIVATE METHODS. ALL THESE METHODS ARE USED INTERNALLY FOR PROCESSING AND ANALYSIS

    def _smoothe_correlation(self, correlation_per_window):
        return lowess(correlation_per_window, np.arange(len(correlation_per_window)),
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
    # TODO Revisar si debe trabajar con correlation_per_window o con smoothed_correlation
    def _compute_statistics(self, num_cases: int, predictionTargetWindow: np.ndarray, correlation_per_window: np.ndarray, target):

        self.bestDic = {index: correlation_per_window[index] for index in self.best_windows_index}

        self.worstDic = {index: correlation_per_window[index] for index in self.worst_windows_index}

        self.bestSorted = sorted(self.bestDic.items(), reverse=True, key=lambda x: x[1])

        self.worstSorted = sorted(self.worstDic.items(), key=lambda x: x[1])
        # TODO Probablemente solamente guardar bestSorted y tratarlo con el nombre de bestDic o best_windows, además de solo contemplar hasta num_cases
        self.bestSorted = self.bestSorted[0:num_cases]
        self.worstSorted = self.worstSorted[0:num_cases]

        print("Calculando MAE para cada ventana")
        for tupla in self.bestSorted:
            self.bestMAE.append(
                mean_absolute_error(target[tupla[0]], predictionTargetWindow.reshape(-1, 1)))

        for tupla in self.worstSorted:
            self.worstMAE.append(
                mean_absolute_error(target[tupla[0]], predictionTargetWindow.reshape(-1, 1)))

        print("Generando reporte de análisis")
        # Version with method name as column name
        # d = {'Index': dict(self.bestSorted).keys(), self.method: dict(self.bestSorted).values(), "Best MAE": self.bestMAE,
        #      'Index': dict(self.worstSorted).keys(), self.method: dict(self.worstSorted).values(),
        #      "Worst MAE": self.worstMAE}
        # Version 1.0 when CCI was the only one method defined
        # TODO Atender impresión de nombre de acuerdo su tipo (String o Callable)
        # TODO Verificar correcto funcionamiento de creación del objeto Dataframe
        d = {'index': dict(self.bestSorted).keys(), self.method: dict(self.bestSorted).values(), "MAE": self.bestMAE}
        # The worst indices were disabled in order to remove the duplicated keys such as index.1 and so forth
        # 'index.1': dict(self.worstSorted).keys(), 'other': dict(self.worstSorted).values(),
        # "MAE.1": self.worstMAE
        self.analysisReport = pd.DataFrame(data=d)

    def _compute_cbr_analysis(self, correlation_per_window, windows_len: int):
        self.smoothed_correlation = self._smoothe_correlation(correlation_per_window)
        self.valley_index, self.peak_index = self._identify_valleys_peaks_indexes()
        self._retreive_concave_convex_segments(windows_len)
        self._retreive_original_indexes()


    def _compute_distance(self, windows: np.ndarray,windows_len, components_len, target: None):
        # TODO Verificar funcionamiento correcto. Aplicar cualquier cambio necesario
        correlation_per_window = np.array(([distance(target[:, current_component],
                                                        windows[current_window, :, current_component], self.metric,
                                                        **self.kwargs)
                                               for current_window in range(windows_len)
                                               for current_component in range(components_len)])).reshape(-1,
                                                                                                             components_len)
        # TODO Verificar procedimiento. Tomar en cuenta si es necesario aplicar alguna utilería
        correlation_per_window = np.sum(correlation_per_window, axis=1)
        correlation_per_window /= max(correlation_per_window)
        return correlation_per_window

    # PUBLIC METHODS. ALL THESE METHODS ARE PROVIDED FOR THE USER

    # Main method. This method allows to the user to perform the primary function. User need to invoke it in order to call others public methods

    def explain(self, windows: np.ndarray, target: None, target_window: None, num_cases: int):
        # gather some basic data from passed in variables
        components_len = windows.shape[2]
        window_len = windows.shape[1]
        windows_len = len(windows)
        # TODO Revisar si target_window debe ser enviado como argumento a _compute_cbr_analysis
        # TODO Revisar si target debe ser enviado como argumento a _compute_distance
        correlation_per_window = self._compute_distance(windows, windows_len, components_len, target)
        self._compute_cbr_analysis(correlation_per_window, num_cases, target_window, windows_len)
        self._compute_statistics(num_cases, target_window, correlation_per_window, target)

    # Method to print a chart or graphic based on results stored in variables. These methods are not strictly necessary
    #   for underlying functinality
    def visualize(self):
        pass

    def get_analysis_report(self):
        return self.analysisReport
