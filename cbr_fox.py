import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sktime.distances import distance
from scipy import signal

class cbr_fox:
    def __init__(self, metric: str or callable="dtw",  smoothness_factor:float =.2, kwargs:dict ={}):
    # Variables for setting
        self.metric = metric
        self.smoothness_factor = smoothness_factor
        self.kwargs = kwargs
    # Variables for results
        # self.outputComponentsLen = len(outputNames)
        self.analysisReport = None
        self.best_correlations = None
        self.worst_correlations = None

    # PRIVATE METHODS

    def smoothe(self):
        self.smoothedCorrelation = lowess(self.correlationPerWindow, np.arange(len(self.correlationPerWindow)),
                                          self.smoothnessFactor)[:, 1]
    def extract_valleys_peaks_indexes(self, smoothedCorrelation):
        self.valleyIndex, self.peakIndex = signal.argrelextrema(smoothedCorrelation, np.less)[0], \
            signal.argrelextrema(smoothedCorrelation, np.greater)[0]
    def retreive_concave_convex_segments(self):
        self.concaveSegments = np.split(
            np.transpose(np.array((np.arange(self.windowsLen), self.correlationPerWindow))),
            self.valleyIndex)
        self.convexSegments = np.split(
            np.transpose(np.array((np.arange(self.windowsLen), self.correlationPerWindow))),
            self.peakIndex)
    def retreive_original_indexes(self):
        for split in self.concaveSegments:
            self.bestWindowsIndex.append(int(split[np.where(split == max(split[:, 1]))[0][0], 0]))
        for split in self.convexSegments:
            self.worstWindowsIndex.append(int(split[np.where(split == min(split[:, 1]))[0][0], 0]))
    def determine_dicts(self, num_cases):
        self.bestDic = {index: self.correlationPerWindow[index] for index in self.bestWindowsIndex}

        self.worstDic = {index: self.correlationPerWindow[index] for index in self.worstWindowsIndex}

        self.bestSorted = sorted(self.bestDic.items(), reverse=True, key=lambda x: x[1])

        self.worstSorted = sorted(self.worstDic.items(), key=lambda x: x[1])

        self.bestSorted = self.bestSorted[0:num_cases]
        self.worstSorted = self.worstSorted[0:num_cases]

        print("Calculando MAE para cada ventana")
        for tupla in self.bestSorted:
            self.bestMAE.append(
                mean_absolute_error(self.target[tupla[0]], self.predictionTargetWindow.reshape(-1, 1)))

        for tupla in self.worstSorted:
            self.worstMAE.append(
                mean_absolute_error(self.target[tupla[0]], self.predictionTargetWindow.reshape(-1, 1)))
        # TODO Modify not all reports are CCI method
        print("Generando reporte de análisis")
        # Version with method name as column name
        # d = {'Index': dict(self.bestSorted).keys(), self.method: dict(self.bestSorted).values(), "Best MAE": self.bestMAE,
        #      'Index': dict(self.worstSorted).keys(), self.method: dict(self.worstSorted).values(),
        #      "Worst MAE": self.worstMAE}
        # Version 1.0 when CCI was the only one method defined
        d = {'index': dict(self.bestSorted).keys(), self.method: dict(self.bestSorted).values(), "MAE": self.bestMAE}
        # The worst indices were disabled in order to remove the duplicated keys such as index.1 and so forth
        # 'index.1': dict(self.worstSorted).keys(), 'other': dict(self.worstSorted).values(),
        # "MAE.1": self.worstMAE

        self.analysisReport = pd.DataFrame(data=d)
    def process(self, num_cases, predictionTargetWindow:np.ndarray):
        self.smoothe()
        self.extract_valleys_peaks_indexes()
        self.retreive_concave_convex_segments()
        self.retreive_original_indexes()
        self.determine_dicts(predictionTargetWindow)
    def compute_distance(self, windows:np.ndarray, target:None, targetWindow:None, num_cases:int):
        componentsLen = windows.shape[2]
        windowLen = windows.shape[1]
        windowsLen = len(self.windows)
        self.correlationPerWindow = np.array(([distance(targetWindow[:, currentComponent],
                                                            windows[currentWindow, :, currentComponent], self.metric, **self.kwargs)
                                               for currentWindow in range(windowsLen)
                                               for currentComponent in range(self.componentsLen)])).reshape(-1,
                                                                                                            componentsLen)
        self.correlationPerWindow = np.sum(self.correlationPerWindow, axis=1)
        self.correlationPerWindow /= max(self.correlationPerWindow)
        self.process(num_cases)

    def determine_dicts(self, num_cases:int, predictionTargetWindow:np.ndarray):
        self.bestDic = {index: self.correlationPerWindow[index] for index in self.bestWindowsIndex}

        self.worstDic = {index: self.correlationPerWindow[index] for index in self.worstWindowsIndex}

        self.bestSorted = sorted(self.bestDic.items(), reverse=True, key=lambda x: x[1])

        self.worstSorted = sorted(self.worstDic.items(), key=lambda x: x[1])

        self.bestSorted = self.bestSorted[0:num_cases]
        self.worstSorted = self.worstSorted[0:num_cases]

        print("Calculando MAE para cada ventana")
        for tupla in self.bestSorted:
            self.bestMAE.append(
                mean_absolute_error(self.target[tupla[0]], predictionTargetWindow.reshape(-1, 1)))

        for tupla in self.worstSorted:
            self.worstMAE.append(
                mean_absolute_error(self.target[tupla[0]], predictionTargetWindow.reshape(-1, 1)))
        # TODO Modify not all reports are CCI method
        print("Generando reporte de análisis")
        # Version with method name as column name
        # d = {'Index': dict(self.bestSorted).keys(), self.method: dict(self.bestSorted).values(), "Best MAE": self.bestMAE,
        #      'Index': dict(self.worstSorted).keys(), self.method: dict(self.worstSorted).values(),
        #      "Worst MAE": self.worstMAE}
        # Version 1.0 when CCI was the only one method defined
        d = {'index': dict(self.bestSorted).keys(), self.method: dict(self.bestSorted).values(), "MAE": self.bestMAE}
        # The worst indices were disabled in order to remove the duplicated keys such as index.1 and so forth
        # 'index.1': dict(self.worstSorted).keys(), 'other': dict(self.worstSorted).values(),
        # "MAE.1": self.worstMAE

        self.analysisReport = pd.DataFrame(data=d)
    def explain(self, windows:np.ndarray, target:None, targetWindow:None, num_cases:int):
        self.compute_distance(windows, target, targetWindow, num_cases)

    # Method to print a chart or graphic based on results stored in variables
    def visualize(self):
        pass

    def get_analysis_report(self):
        return self.analysisReport


