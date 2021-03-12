#!/usr/bin/env python
# coding: utf-8

import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from collections import Counter

from itertools import combinations

from pytorchUtility import *
import numpy as np

predictionDir = './cifar10/prediction'
models = ['densenet-L190-k40', 'densenetbc-100-12', 'resnext8x64d', 'wrn-28-10-drop', 'vgg19_bn', 
          'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110']
suffix = '.pt'

labelVectorsList = list()
predictionVectorsList = list()
tmpAccList = list()
for m in models:
    predictionPath = os.path.join(predictionDir, m+suffix)
    prediction = torch.load(predictionPath)
    predictionVectors = prediction['predictionVectors']
    predictionVectorsList.append(nn.functional.softmax(predictionVectors, dim=-1).cpu())
    labelVectors = prediction['labelVectors']
    labelVectorsList.append(labelVectors.cpu())
    tmpAccList.append(calAccuracy(predictionVectors, labelVectors))
    print(tmpAccList[-1])


minAcc = np.min(tmpAccList)
avgAcc = np.mean(tmpAccList)
maxAcc = np.max(tmpAccList)


trainPredictionDir = './cifar10/train'
trainLabelVectorsList = list()
trainPredictionVectorsList = list()
for m in models:
    trainPredictionPath = os.path.join(trainPredictionDir, m+suffix)
    trainPrediction = torch.load(trainPredictionPath)
    trainPredictionVectors = trainPrediction['predictionVectors']
    trainPredictionVectorsList.append(nn.functional.softmax(trainPredictionVectors, dim=-1).cpu())
    trainLabelVectors = trainPrediction['labelVectors']
    trainLabelVectorsList.append(labelVectors.cpu())

# obtain:
# team -> accuracy map
# model -> team
import timeit
teamAccuracyDict = dict()
modelTeamDict = dict()
teamNameDict = dict()
startTime = timeit.default_timer()
for n in range(2, len(models)+1):
    comb = combinations(list(range(len(models))), n)
    for selectedModels in list(comb):
        # accuracy
        tmpAccuracy = calAveragePredictionVectorAccuracy(predictionVectorsList, labelVectorsList[0], modelsList=selectedModels)[0].cpu().item()
        #print(selectedModels)
        teamName = "".join(map(str, selectedModels))
        teamNameDict[teamName] = selectedModels
        teamAccuracyDict[teamName] = tmpAccuracy
        for m in teamName:
            if m in modelTeamDict:
                modelTeamDict[m].add(teamName)
            else:
                modelTeamDict[m] = set([teamName,])
endTime = timeit.default_timer()
print("Time: ", endTime-startTime)


# calculate the diversity measures for all configurations
import numpy as np
from EnsembleBench.groupMetrics import *
np.random.seed(0)
nRandomSamples = 100
crossValidation = True
crossValidationTimes = 3

teamDiversityMetricMap = dict()
negAccuracyDict = dict()
diversityMetricsList = ['CK', 'QS', 'BD', 'FK', 'KW', 'GD']
startTime = timeit.default_timer()
for oneTargetModel in range(len(models)):
    sampleID, sampleTarget, predictions, predVectors = calDisagreementSamplesOneTargetNegative(trainPredictionVectorsList, trainLabelVectorsList[0], oneTargetModel)
    if len(predictions) == 0:
        print("negative sample not found")
        continue
    sampleID = np.array(sampleID)
    sampleTarget = np.array(sampleTarget)
    predictions = np.array(predictions)
    predVectors = np.array([np.array([np.array(pp) for pp in p]) for p in predVectors])
    for teamName in modelTeamDict[str(oneTargetModel)]:
        selectedModels = teamNameDict[teamName]
        teamSampleID, teamSampleTarget, teamPredictions, teamPredVectors = filterModelsFixed(sampleID, sampleTarget, predictions, predVectors, selectedModels) 
        if crossValidation:
            tmpMetrics = list()
            for _ in range(crossValidationTimes):
                randomIdx = np.random.choice(np.arange(teamPredictions.shape[0]), nRandomSamples)        
                tmpMetrics.append(calAllDiversityMetrics(teamPredictions[randomIdx], teamSampleTarget[randomIdx], diversityMetricsList))
            tmpMetrics = np.mean(np.array(tmpMetrics), axis=0)
        else:
            tmpMetrics = np.array(calAllDiversityMetrics(teamPredictions, teamSampleTarget, diversityMetricsList))
        #print(tmpMetrics)
        diversityMetricDict = {diversityMetricsList[i]:tmpMetrics[i].item()  for i in range(len(tmpMetrics))}
        targetDiversity = teamDiversityMetricMap.get(teamName, dict())
        targetDiversity[str(oneTargetModel)] = diversityMetricDict
        teamDiversityMetricMap[teamName] = targetDiversity
        
        tmpNegAccuracy = calAccuracy(torch.tensor(np.mean(np.transpose(teamPredVectors, (1, 0, 2)), axis=0)), torch.tensor(teamSampleTarget))[0].cpu().item()
        targetNegAccuracy = negAccuracyDict.get(teamName, dict())
        targetNegAccuracy[str(oneTargetModel)] = tmpNegAccuracy
        negAccuracyDict[teamName] = targetNegAccuracy

endTime = timeit.default_timer()
print("Time: ", endTime-startTime)


# calculate the targetTeamSizeDict
startTime = timeit.default_timer()
targetTeamSizeDict = dict()
for oneTargetModel in range(len(models)):
    for teamName in modelTeamDict[str(oneTargetModel)]:
        teamSize = len(teamName)
        teamSizeDict = targetTeamSizeDict.get(str(oneTargetModel), dict())
        fixedTeamDict = teamSizeDict.get(str(teamSize), dict())
        
        teamList = fixedTeamDict.get('TeamList', list())
        teamList.append(teamName)
        fixedTeamDict['TeamList'] = teamList
        
        # diversity measures
        diversityVector = np.expand_dims(np.array([teamDiversityMetricMap[teamName][str(oneTargetModel)][dm]
                                    for dm in diversityMetricsList]), axis=0)
        
        diversityMatrix = fixedTeamDict.get('DiversityMatrix', None)
        if diversityMatrix is None:
            diversityMatrix = diversityVector
        else:
            diversityMatrix = np.append(diversityMatrix, diversityVector, axis=0)
        #print(diversityMatrix, diversityMatrix.shape)
        fixedTeamDict['DiversityMatrix'] = diversityMatrix
        
        teamSizeDict[str(teamSize)] = fixedTeamDict
        targetTeamSizeDict[str(oneTargetModel)] = teamSizeDict 
endTime = timeit.default_timer()
print("Time: ", endTime-startTime)


teamSelectedFQDict = dict()
from EnsembleBench.teamSelection import *
for oneTargetModel in range(len(models)):
    targetFQDict = teamSelectedFQDict.get(str(oneTargetModel), dict())
    for teamSize in range(2, len(models)):
        targetTeamSizeFQDict = targetFQDict.get(str(teamSize), dict())
        fixedTeamDict = targetTeamSizeDict[str(oneTargetModel)][str(teamSize)]
        #print(len(fixedTeamDict['TeamList']))
        thresholds = list()
        kmeans = list()
        teamList = fixedTeamDict['TeamList']
        accuracyList = [np.mean(negAccuracyDict[teamName].values()) for teamName in teamList]
        diversityMatrix = fixedTeamDict['DiversityMatrix']
        #print(diversityMatrix[:, 0].shape)
        for i in range(len(diversityMetricsList)):
            tmpThreshold, tmpKMeans = getThresholdClusteringKMeans(accuracyList, diversityMatrix[:, i], kmeansInit='strategic')
            tmpThreshold = min(np.mean(diversityMatrix[:, i]), tmpThreshold)
            thresholds.append(tmpThreshold)
            kmeans.append(tmpKMeans)
        fixedTeamDict['Threshold'] = thresholds
        fixedTeamDict['KMeans'] = kmeans
        
        scaledDiversityMeasures = list()
        for i in range(len(diversityMetricsList)):
            #if max(diversityMatrix[:, i]) == min(diversityMatrix[:, i]):
                #print(diversityMetricsList[i], oneTargetModel, teamSize)
            scaledDiversityMeasures.append(normalize01(diversityMatrix[:, i]))
        scaledDiversityMeasures = np.stack(scaledDiversityMeasures, axis=1)
        #print(EQ.shape)
        fixedTeamDict['ScaledDiversityMatrix'] = scaledDiversityMeasures
        targetTeamSizeDict[str(oneTargetModel)][str(teamSize)] = fixedTeamDict
        
        # select team    
        # FQ
        for i, teamName in enumerate(fixedTeamDict['TeamList']):
            for j in range(len(diversityMetricsList)):
                targetTeamSizeFQDiversitySet = targetTeamSizeFQDict.get(diversityMetricsList[j], set())
                if diversityMatrix[i, j] < round(thresholds[j], 3):
                    targetTeamSizeFQDiversitySet.add(teamName)
                targetTeamSizeFQDict[diversityMetricsList[j]] = targetTeamSizeFQDiversitySet
        #print([len(value) for key, value in targetTeamSizeFQDict.items()])
        targetFQDict[str(teamSize)] = targetTeamSizeFQDict
        
    teamSelectedFQDict[str(oneTargetModel)] = targetFQDict

teamSelectedFQAllDict = dict()
#print(teamSelectedFQAllDict)
for j, dm in enumerate(diversityMetricsList):
    teamSelectedFQAllDiversitySet = teamSelectedFQAllDict.get(dm, set())
    for teamSize in range(2, len(models)):
        teamSizeSelectedTeamsSet = set()
        tmpTeamDict = dict() # teamName & Metric
        #print(teamSize, dm)
        for oneTargetModel in range(len(models)):
            for teamName in teamSelectedFQDict[str(oneTargetModel)][str(teamSize)][dm]:
                #print(teamName, teamSize, oneTargetModel)
                if teamName in tmpTeamDict:
                    continue
                tmpMetricList = list()
                teamModelIdx = map(int, [modelName for modelName in teamName])
                teamModelAcc = [tmpAccList[modelIdx][0].cpu().item() for modelIdx in teamModelIdx]
                teamModelWeights = np.argsort(teamModelAcc)
                #print(teamModelIdx, teamModelWeights)
                tmpModelWeights = list()
                for (k, modelName) in enumerate(teamName):
                    fixedTeamDict = targetTeamSizeDict[modelName][str(teamSize)]
                    for i, tmpTeamName in enumerate(fixedTeamDict['TeamList']):
                        if tmpTeamName == teamName:
                            tmpMetricList.append(fixedTeamDict['ScaledDiversityMatrix'][i, j])
                            tmpModelWeights.append(teamModelWeights[k])
                tmpTeamDict[teamName] = np.average(tmpMetricList, weights=tmpModelWeights)
        if len(tmpTeamDict) > 0:
            accuracyList = np.array([np.mean(negAccuracyDict[teamName].values()) for teamName in tmpTeamDict])
            metricList = np.array([tmpTeamDict[teamName] for teamName in tmpTeamDict])
            tmpThreshold, _ = getThresholdClusteringKMeansCenter(accuracyList, metricList, kmeansInit='strategic')
            for teamName in tmpTeamDict:
                if tmpTeamDict[teamName] < tmpThreshold:
                    teamSizeSelectedTeamsSet.add(teamName)
        teamSelectedFQAllDiversitySet.update(teamSizeSelectedTeamsSet)
    teamSelectedFQAllDict[dm] = teamSelectedFQAllDiversitySet

# teamSelectedFQAllDict contains the selected good ensemble teams
# following codes for statistics

from EnsembleBench.teamSelection import getNTeamStatistics

tmpSelectedAllMetricsSet = set()
for dm in diversityMetricsList:
    #print(len([teamName for teamName in teamSelectedFQAllDict[dm]]))
    tmpSelectedAllMetricsSet.update(teamSelectedFQAllDict[dm])
    accuracyTeamNameArray = [(teamAccuracyDict[teamName], teamName) for teamName in teamSelectedFQAllDict[dm]]
    accuracyArray = [aTNA[0] for aTNA in accuracyTeamNameArray]
    if len(accuracyArray) <= 0:
        print(dm, 'no team selected!')
        continue
    print(getNTeamStatistics(list(teamSelectedFQAllDict[dm]), teamAccuracyDict, minAcc, avgAcc, maxAcc, tmpAccList))
