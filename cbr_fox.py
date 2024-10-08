import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sktime.distances import distance


class cbr_fox:
    def __init__(self, windows:np.ndarray, target:None, targetWindow:None, num_cases:int, metric: str or callable="dtw", kwargs:dict ={}):
        self.correlationPerWindow = None
        self.windows = windows
        self.target = target
        self.targetWindow = targetWindow
        self.num_cases = num_cases
        self.metric = metric
        self.kwargs = kwargs
        self.componentsLen = self.windows.shape[2]
        self.windowLen = self.windows.shape[1]
        self.windowsLen = len(self.windows)
        # self.outputComponentsLen = len(outputNames)
        self.analysisReport = None

    def compute_distance(self, time_series_1: np.ndarray, time_series_2: np.ndarray):
        self.correlationPerWindow = np.array(([distance(self.targetWindow[:, currentComponent],
                                                            self.windows[currentWindow, :, currentComponent], self.metric, **self.kwargs)
                                               for currentWindow in range(self.windowsLen)
                                               for currentComponent in range(self.componentsLen)])).reshape(-1,
                                                                                                            self.componentsLen)
        self.correlationPerWindow = np.sum(self.correlationPerWindow, axis=1)
        self.correlationPerWindow /= max(self.correlationPerWindow)

    def determine_dicts(self):
        self.bestDic = {index: self.correlationPerWindow[index] for index in self.bestWindowsIndex}

        self.worstDic = {index: self.correlationPerWindow[index] for index in self.worstWindowsIndex}

        self.bestSorted = sorted(self.bestDic.items(), reverse=True, key=lambda x: x[1])

        self.worstSorted = sorted(self.worstDic.items(), key=lambda x: x[1])

        self.bestSorted = self.bestSorted[0:self.num_cases]
        self.worstSorted = self.worstSorted[0:self.num_cases]

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
    def explain(self, time_series_1: np.ndarray, time_series_2: np.ndarray):
        self.compute_distance(time_series_1, time_series_2)

    # Method to print a chart or graphic based on results stored in variables
    def visualize(self):
        pass

    def get_analysis_report(self):
        return self.analysisReport
