import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sktime.distances import distance
from scipy import signal
from statsmodels.nonparametric.smoothers_lowess import lowess
from custom_distance import sktime_interface
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
        self.analysisReport_combined = None
        self.best_windows_index = list()
        self.worst_windows_index = list()
        self.bestMAE = list()
        self.worstMAE = list()
        # Private variables for easy access by private methods
        self.correlation_per_window = None
        self.input_data_dictionary = None
        self.records_array = None
        self.records_array_combined = None
        self.dtype = [('index', 'i4'),
                      ('window', 'O'),
                      ('target_window', 'O'),
                      ('correlation', 'f8'),
                      ('MAE', 'f8')]
        # PRIVATE METHODS. ALL THESE METHODS ARE USED INTERNALLY FOR PROCESSING AND ANALYSIS

    def _preprocess_input_data(self, training_windows, target_training_windows, forecasted_window):
        # gather some basic data from passed in variables
        input_data_dictionary = dict()
        input_data_dictionary['training_windows'] = training_windows
        input_data_dictionary['target_training_windows'] = target_training_windows
        input_data_dictionary['forecasted_window'] = forecasted_window
        input_data_dictionary['components_len'] = training_windows.shape[2]
        input_data_dictionary['window_len'] = training_windows.shape[1]
        input_data_dictionary['windows_len'] = len(training_windows)

        return input_data_dictionary

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
        for split in tqdm(self.concaveSegments, desc="Segmentos cóncavos"):
            self.best_windows_index.append(int(split[np.where(split == max(split[:, 1]))[0][0], 0]))
        for split in tqdm(self.convexSegments, desc="Segmentos cóncavos"):
            self.worst_windows_index.append(int(split[np.where(split == min(split[:, 1]))[0][0], 0]))

    def calculate_analysis(self, indexes, input_data_dictionary):
        return np.array([(index,
                                        input_data_dictionary["training_windows"][index],
                                        input_data_dictionary["target_training_windows"][index],
                                        self.correlation_per_window[index],
                                        mean_absolute_error(input_data_dictionary["target_training_windows"][index],
                                                            input_data_dictionary["prediction"].reshape(-1, 1)))
                                       for index in indexes], dtype=self.dtype)


    def calculate_analysis_combined(self, input_data_dictionary):
       return np.array([(-index,
           np.mean(input_data_dictionary["training_windows"][
                   dic[:input_data_dictionary["num_cases"]]], axis=0),
           np.mean(input_data_dictionary["target_training_windows"][
                   dic[:input_data_dictionary["num_cases"]]], axis=0),
           np.mean(self.correlation_per_window[
                   dic[:input_data_dictionary["num_cases"]]]),
           mean_absolute_error(
               np.mean(input_data_dictionary["target_training_windows"][
                       dic[:input_data_dictionary["num_cases"]]], axis=0),
               input_data_dictionary["prediction"].reshape(-1, 1)
           )
          ) for index, dic in enumerate([self.best_windows_index,self.worst_windows_index])], dtype=self.dtype)

    # TODO Analizar si este método puede ser el único que permita realizar asignaciones de variable internamente
    def _compute_statistics(self, input_data_dictionary):

        # self.bestDic = {index: self.__correlation_per_window[index] for index in self.best_windows_index}
        #
        # self.worstDic = {index: self.__correlation_per_window[index] for index in self.worst_windows_index}
        #
        # self.bestDic = sorted(self.bestDic.items(), reverse=True, key=lambda x: x[1])
        #
        # self.worstDic = sorted(self.worstDic.items(), key=lambda x: x[1])
        #
        # self.bestDic = self.bestDic[0:input_data_dictionary['num_cases']]
        # self.worstDic = self.worstDic[0:input_data_dictionary['num_cases']]
        #
        # print("Calculando MAE para cada ventana")
        #
        # for tupla in self.bestDic:
        #     self.bestMAE.append(
        #         mean_absolute_error(input_data_dictionary["target_training_windows"][tupla[0]],
        #                             input_data_dictionary["prediction"].reshape(-1, 1)))
        #
        # for tupla in self.worstDic:
        #     self.worstMAE.append(
        #         mean_absolute_error(input_data_dictionary["target_training_windows"][tupla[0]],
        #                             input_data_dictionary["prediction"].reshape(-1, 1)))

        self.records_array_combined = self.calculate_analysis_combined(input_data_dictionary)

        self.records_array = self.calculate_analysis(self.best_windows_index + self.worst_windows_index,
                                                     input_data_dictionary)

        # Sorting the array
        self.records_array = np.sort(self.records_array, order="correlation")[::-1]

        # Selecting just the number of elements according to num_cases variable
        # The conditional is to avoid duplicity in case records_arrays's shape is not greater than the selected num_cases
        if (self.records_array.shape[0] > (input_data_dictionary["num_cases"]*2)):
            self.records_array = np.concatenate((self.records_array[:input_data_dictionary["num_cases"]], self.records_array[
                                                                                                 -input_data_dictionary[
                                                                                                     "num_cases"]:]))

        print("Generando reporte de análisis")
        self.analysisReport = pd.DataFrame(data=pd.DataFrame.from_records(self.records_array))
        self.analysisReport_combined = pd.DataFrame(data=pd.DataFrame.from_records(self.records_array_combined))

    def _compute_cbr_analysis(self, input_data_dictionary):
        logging.info("Suavizando Correlación")
        self.smoothed_correlation = self._smoothe_correlation()
        logging.info("Extrayendo crestas y valles")
        self.valley_index, self.peak_index = self._identify_valleys_peaks_indexes()
        logging.info("Recuperando segmentos convexos y cóncavos")
        self._retreive_concave_convex_segments(input_data_dictionary['windows_len'])
        logging.info("Recuperando índices originales de correlación")
        self._retreive_original_indexes()

    def _compute_correlation(self, input_data_dictionary):

        # Implementing interface architecture to reduce tight coupling.
        correlation_per_window = sktime_interface.compute_distance_interface(input_data_dictionary, self.metric,
                                                                             self.kwargs)
        correlation_per_window = np.sum(correlation_per_window, axis=1)
        correlation_per_window = ((correlation_per_window - min(correlation_per_window)) /
                                  (max(correlation_per_window) - min(correlation_per_window)))
        self.correlation_per_window = correlation_per_window
        return correlation_per_window

    # PUBLIC METHODS. ALL THESE METHODS ARE PROVIDED FOR THE USER

    def fit(self, training_windows: np.ndarray, target_training_windows: np.ndarray, forecasted_window: np.ndarray):

        logging.info("Analizando conjunto de datos")
        self.input_data_dictionary = self._preprocess_input_data(training_windows, target_training_windows,
                                                                         forecasted_window)
        logging.info("Calculando Correlación")
        self.__correlation_per_window = self._compute_correlation(self.input_data_dictionary)
        logging.info("Computando análisis de CBR")
        self._compute_cbr_analysis(self.input_data_dictionary)
        logging.info("Análisis finalizado")


    def predict(self,prediction, num_cases: int):
        self.input_data_dictionary['prediction'] = prediction
        self.input_data_dictionary['num_cases'] = num_cases
        self._compute_statistics(self.input_data_dictionary)

    # Method to print a chart or graphic based on results stored in variables. These methods are not strictly necessary
    #   for underlying functionality
    def visualize_correlation_per_window(self, plt_oject):
        pass

    def get_analysis_report(self):
        return self.analysisReport

    def get_analysis_report_combined(self):
        return self.analysisReport_combined
