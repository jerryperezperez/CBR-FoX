import pandas as pd
import numpy as np
from scipy import signal
from statsmodels.nonparametric.smoothers_lowess import lowess
from copy import deepcopy
from custom_distance.cci_distance import cci_distance
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
from deprecated import deprecated

from sktime.distances import (
    dtw_distance,
    ddtw_distance,
    msm_distance,
    erp_distance,
    lcss_distance,
    twe_distance,
    edr_distance
)


# TODO Define all technique names
# TODO Reorganize possible new methods for all new techniques
# cci_distance (Combined Correlation Index)...
# dtw_distance (Dynamic Timie Warping)
# ddtw_distance,
# wdtw_distance,
# msm_distance,
# erp_distance,
# lcss_distance,
# twe_distance,
# wddtw_distance,
# edr_distance
class sktime_distance_comparison:
    def __init__(self, windows=None, target=None, targetWindow=None, num_cases=10, smoothnessFactor=.03,
                 inputNames=None,
                 outputNames=None, punishedSumFactor=.5, prediction=None, method="CCI"):
        self.windows = windows
        self.target = target
        self.targetWindow = targetWindow
        self.num_cases = num_cases
        self.smoothnessFactor = smoothnessFactor
        self.inputNames = inputNames
        self.outputNames = outputNames
        self.punishedSumFactor = punishedSumFactor
        self.predictionTargetWindow = prediction

        self.componentsLen = self.windows.shape[2]
        self.windowLen = self.windows.shape[1]
        self.windowsLen = len(self.windows)
        self.outputComponentsLen = len(outputNames)

        # Variables used across all the code
        self.pearsonCorrelation = None
        self.euclideanDistance = None
        self.correlationPerWindow = None
        self.smoothedCorrelation = None
        self.valleyIndex = None
        self.peakIndex = None
        self.concaveSegments = None
        self.convexSegments = None
        self.bestWindowsIndex, self.worstWindowsIndex = list(), list()
        self.worstDic, self.bestDic = dict(), dict()
        self.worstSorted, self.bestSorted = dict(), dict()
        self.bestMAE, self.worstMAE = [], []
        # Added for the version 1.1
        self.method = method
        self.analysisReport = None
        self.dataframes = []

    # New methods for version 1.1
    def process(self):
        self.smoothe()
        self.extract_valleys_peaks_indexes()
        self.retreive_concave_convex_segments()
        self.retreive_original_indexes()
        self.determine_dicts()

    def smoothe(self):
        self.smoothedCorrelation = lowess(self.correlationPerWindow, np.arange(len(self.correlationPerWindow)),
                                          self.smoothnessFactor)[:, 1]

    def extract_valleys_peaks_indexes(self):
        self.valleyIndex, self.peakIndex = signal.argrelextrema(self.smoothedCorrelation, np.less)[0], \
            signal.argrelextrema(self.smoothedCorrelation, np.greater)[0]

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

    # End of new methods for version 1.1

    def explain(self):
        # TWE = np.array(([twe_distance(targetWindow[:,currentComponent],windows[currentWindow,:,currentComponent])for currentWindow in range(windowsLen) for currentComponent in range(componentsLen)]))
        # print("WDDTW")
        if (self.method == "CCI"):
            print("Calculando correlación de Pearson")
            self.correlationPerWindow = cci_distance(self.windows,
                                                     self.targetWindow, self.windowsLen,
                                                     self.componentsLen, self.punishedSumFactor)
        if (self.method == "DTW"):
            self.correlationPerWindow = np.array(([dtw_distance(self.targetWindow[:, currentComponent],
                                                                self.windows[currentWindow, :, currentComponent])
                                                   for currentWindow in range(self.windowsLen)
                                                   for currentComponent in range(self.componentsLen)])).reshape(-1,
                                                                                                                self.componentsLen)
            self.correlationPerWindow = np.sum(self.correlationPerWindow, axis=1)
            self.correlationPerWindow /= max(self.correlationPerWindow)
        if (self.method == "DDTW"):
            self.correlationPerWindow = np.array(([ddtw_distance(self.targetWindow[:, currentComponent],
                                                                 self.windows[currentWindow, :, currentComponent])
                                                   for currentWindow in range(self.windowsLen)
                                                   for currentComponent in range(self.componentsLen)])).reshape(-1,
                                                                                                                self.componentsLen)
            self.correlationPerWindow = np.sum(self.correlationPerWindow, axis=1)
            self.correlationPerWindow /= max(self.correlationPerWindow)
        if (self.method == "MSM"):
            self.correlationPerWindow = np.array(([msm_distance(self.targetWindow[:, currentComponent],
                                                                self.windows[currentWindow, :, currentComponent])
                                                   for currentWindow in range(self.windowsLen)
                                                   for currentComponent in range(self.componentsLen)])).reshape(-1,
                                                                                                                self.componentsLen)
            self.correlationPerWindow = np.sum(self.correlationPerWindow, axis=1)
            self.correlationPerWindow /= max(self.correlationPerWindow)
        if (self.method == "ERP"):
            self.correlationPerWindow = np.array(([erp_distance(self.targetWindow[:, currentComponent],
                                                                self.windows[currentWindow, :, currentComponent])
                                                   for currentWindow in range(self.windowsLen)
                                                   for currentComponent in range(self.componentsLen)])).reshape(-1,
                                                                                                                self.componentsLen)
            self.correlationPerWindow = np.sum(self.correlationPerWindow, axis=1)
            self.correlationPerWindow /= max(self.correlationPerWindow)
        if (self.method == "TWE"):
            self.correlationPerWindow = np.array(([twe_distance(self.targetWindow[:, currentComponent],
                                                                self.windows[currentWindow, :, currentComponent])
                                                   for currentWindow in range(self.windowsLen)
                                                   for currentComponent in range(self.componentsLen)])).reshape(-1,
                                                                                                                self.componentsLen)
            self.correlationPerWindow = np.sum(self.correlationPerWindow, axis=1)
            self.correlationPerWindow /= max(self.correlationPerWindow)
        if (self.method == "EDR"):
            self.correlationPerWindow = np.array(([edr_distance(self.targetWindow[:, currentComponent],
                                                                self.windows[currentWindow, :,
                                                                currentComponent])
                                                   for currentWindow in range(self.windowsLen)
                                                   for currentComponent in range(self.componentsLen)])).reshape(-1,
                                                                                                                self.componentsLen)
            self.correlationPerWindow = np.sum(self.correlationPerWindow, axis=1)
            self.correlationPerWindow /= max(self.correlationPerWindow)
        if (self.method == "LCSS"):
            self.correlationPerWindow = np.array(([lcss_distance(self.targetWindow[:, currentComponent],
                                                                 self.windows[currentWindow, :, currentComponent])
                                                   for currentWindow in range(self.windowsLen)
                                                   for currentComponent in range(self.componentsLen)])).reshape(-1,
                                                                                                                self.componentsLen)
            self.correlationPerWindow = np.sum(self.correlationPerWindow, axis=1)
            self.correlationPerWindow /= max(self.correlationPerWindow)

        # TODO Convert scaler to MinMaxScaler instance
        # This should be applied to all techniques
        # Applying scale
        # scaler = MinMaxScaler()
        # self.correlationPerWindow= scaler.fit_transform(np.sum(scaler.fit_transform(self.correlationPerWindow), axis=1).reshape(-1, 1)).reshape(1, -1)[
        #     0] * -1 + 1

        self.process()

    def explain_all(self, methods):
        # Create a Pandas Excel writer using openpyxl as the engine
        with pd.ExcelWriter('output.xlsx', engine='openpyxl') as writer:
            # Write each dataframe to a different sheet
            for method in methods:
                print("Processing method {}".format(method))
                self.method = method
                self.explain()
                print("Exploring method {}".format(method))
                self.dataframes.append(deepcopy(self.analysisReport))
                self.dataframes[-1].to_excel(writer, sheet_name=str(method), index=False)

    def visualizeCorrelationPerWindow(self):
        plt.figure(figsize=(17, 7))
        plt.plot(self.correlationPerWindow)
        plt.show()

    def visualizeSmoothedCorrelation(self):
        plt.figure(figsize=(17, 7))
        plt.plot(self.smoothedCorrelation)
        plt.scatter(self.peakIndex, [self.smoothedCorrelation[peak] for peak in self.peakIndex])
        plt.scatter(self.valleyIndex, [self.smoothedCorrelation[valley] for valley in self.valleyIndex])
        plt.show()

    def visualizeBestCasesPerFeature(self, figsize, features_indices):
        fig, axs = plt.subplots(len(features_indices), figsize=figsize)

        # COMIENZA CÓDIGO ORIGINAL
        for n_component, feature_index in enumerate(features_indices):

            axs[n_component].set_title(self.inputNames[feature_index])
            axs[n_component].set_ylim((0, 115))
            for i, tupla in enumerate(self.bestSorted):
                axs[n_component].plot(self.windows[tupla[0]][:, feature_index], label="Case " + str(i + 1))

            axs[n_component].plot(self.targetWindow[:, feature_index], "--", label="Query case")
            axs[n_component].legend()
        plt.show()

    def visualizeWorstCases(self, figsize):
        fig, axs = plt.subplots(self.componentsLen, figsize=figsize)

        for n_component in range(self.componentsLen):
            axs[n_component].set_title(self.inputNames[n_component])
            axs[n_component].set_ylim((0, 115))
            for i, tupla in enumerate(self.worstSorted):
                axs[n_component].plot(self.windows[tupla[0]][:, n_component], label="Caso " + str(i))
            axs[n_component].plot(self.targetWindow[:, n_component], "--", label="Caso de estudio")
            axs[n_component].legend()
        plt.show()

    @deprecated(version='1.2.0', reason="If a selected window either best or worst, has an index position"
                                        "close to the end, there will not be further predictions to plot")
    def visualizeBestHistoryPredictions(self, figsize):
        fig, axs = plt.subplots(self.outputComponentsLen, figsize=figsize)

        for n_component in range(self.outputComponentsLen):

            axs[n_component].set_title(self.outputNames[n_component] + " PREDICTIONS")
            axs[n_component].set_ylim((0, 115))
            for i, tupla in enumerate(self.bestSorted):
                pass
                axs[n_component].plot(self.target[tupla[0]: tupla[0] + self.windowLen][:, n_component],
                                      label="Caso " + str(i))
                axs[n_component].scatter(self.windowLen, self.target[tupla[0] + self.windowLen + 1][n_component],
                                         label="Caso " + str(i) + " predicción")
            axs[n_component].scatter(self.windowLen, self.predictionTargetWindow[:, n_component], marker="d",
                                     label="Predicción del caso de estudio")
            axs[n_component].legend()
        plt.show()

    def compareBestCases(self, figsize):
        fig, axs = plt.subplots(len(self.inputNames), figsize=figsize)

        for feature in range(len(self.inputNames)):
            # Processing all case values per feature
            axs[feature].set_title(self.inputNames[feature])
            axs[feature].set_ylim((0, 115))
            # Process all feature values per each column/feature
            # Values represent the case index, in this case, the best case's indices
            counter = 1
            for values in self.bestSorted:

                axs[feature].plot(self.windows[values[0]][:,feature], label="Case " + str(counter))
                counter += 1
            axs[feature].plot(self.targetWindow[:,feature], label="Real case")
            axs[feature].legend()

        plt.show()

    # def compareBestCasesPredictions(self, figsize):
    #     fig, axs = plt.subplots(self.outputComponentsLen, figsize=figsize)
    #
    #     for feature in range(self.outputComponentsLen):
    #         # Processing all case values per feature
    #         axs[feature].set_title(self.outputNames[feature])
    #         axs[feature].set_ylim((0, 115))
    #         # Process all feature values per each column/feature
    #         # Values represent the case index, in this case, the best case's indices
    #         counter = 1
    #         for values in self.bestSorted:
    #
    #             axs[feature].plot(self.windows[values[0]][:,feature], label="Case " + str(counter))
    #             counter += 1
    #         axs[feature].plot(self.targetWindow[:,feature], label="Real case")
    #         axs[feature].legend()
    #
    #     plt.show()

    @deprecated(version='1.2.0', reason="If a selected window either best or worst, has an index position"
                                        "close to the end, there will not be further predictions to plot")
    def visualizeWorstHistoryPredictions(self, figsize):
        fig, axs = plt.subplots(self.outputComponentsLen, figsize=figsize)

        for n_component in range(self.outputComponentsLen):

            axs[n_component].set_title(self.outputNames[n_component] + " PREDICTIONS")
            axs[n_component].set_ylim((0, 115))
            for i, tupla in enumerate(self.worstSorted):
                axs[n_component].plot(self.target[tupla[0]: tupla[0] + self.windowLen][:, n_component],
                                      label="Caso " + str(i))
                axs[n_component].scatter(self.windowLen, self.target[tupla[0] + self.windowLen + 1][n_component],
                                         label="Caso " + str(i) + " predicción")
            axs[n_component].scatter(self.windowLen, self.predictionTargetWindow[:, n_component], marker="d",
                                     label="Predicción del caso de estudio")
            axs[n_component].legend()
        plt.show()

    def visualizeBestCasePredictions(self):
        plt.figure(figsize=(20, 5))
        plt.ylim((0, 100))
        plt.plot(self.outputNames, self.target[-1])
        for i, tupla in enumerate(self.bestSorted):
            plt.plot(self.target[tupla[0]])
            print(mean_absolute_error(self.target[tupla[0]], self.predictionTargetWindow.reshape(-1, 1)))
        plt.plot(self.predictionTargetWindow.reshape(-1, 1), "--", label="Resultados de predicción")
        plt.show()

    def visualizeWorstCasePredictions(self):
        plt.figure(figsize=(20, 5))
        plt.ylim((0, 100))
        plt.plot(self.outputNames, self.target[-1])
        for i, tupla in enumerate(self.worstSorted):
            plt.plot(self.target[tupla[0]])
        plt.plot(self.predictionTargetWindow.reshape(-1, 1), "--", label="Resultados de predicción")
        plt.show()

    def getAnalysisReport(self):
        return self.analysisReport


# Method that receives a list of techniques and creates an instance for each one and returns a kind of dictionary
def explain_methods(windows, targetWindow, target, prediction, num_cases, smoothnessFactor, inputNames, outputNames,
                    punishedSumFactor, methods):
    dictionary = {}
    for method in methods:
        instance = sktime_distance_comparison(windows=windows, targetWindow=targetWindow, target=target,
                                              prediction=prediction,
                                              num_cases=num_cases, smoothnessFactor=smoothnessFactor,
                                              inputNames=inputNames, outputNames=outputNames,
                                              punishedSumFactor=punishedSumFactor, method=method)
        instance.explain()
        dictionary[method] = instance
    return dictionary
