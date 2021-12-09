import os
import time

import torch
import torch.nn as nn

import numpy as np

from collections import Counter

def calAccuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    #print(pred.type(), pred.size())
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    #print(target.type(), target.size())
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def calAveragePredictionVectorAccuracy(predictionVectorsList, target, modelsList=None, topk=(1,)):
    predictionVectorsStack = torch.stack(predictionVectorsList)
    if len(modelsList) > 0:
        predictionVectorsStack = predictionVectorsStack[modelsList,...]
    averagePrediction = torch.mean(predictionVectorsStack, dim=0)
    return calAccuracy(averagePrediction, target, topk)

def calNegativeSamplesSet(predictionVectorsList, target):
    """filter the disagreed samples, return an array of sets"""
    batchSize = target.size(0)
    predictionList = list()
    negativeSamplesSet = list()
    
    for pVL in predictionVectorsList:
        _, pred = pVL.max(dim=1)
        predictionList.append(pred)
        negativeSamplesSet.append(set())
        
    for i in xrange(batchSize):
        for j,_ in enumerate(predictionList):
            if predictionList[j][i] != target[i]:
                negativeSamplesSet[j].add(i)
    return negativeSamplesSet


def calDisagreementSamplesOneTargetNegative(predictionVectorsList, target, oneTargetIdx):
    """filter the disagreed samples"""
    batchSize = target.size(0)
    predictionList = list()
    
    for pVL in predictionVectorsList:
        _, pred = pVL.max(dim=1)
        predictionList.append(pred)
    
    # return sampleID, sampleTarget, predictions, predVectors
    sampleID = list()
    sampleTarget = list()
    predictions = list()
    predVectors = list()
    
    for i in xrange(batchSize):
        pred = []
        predVect = []
        for j, p in enumerate(predictionList):
            pred.append(p[i].item())
            predVect.append(predictionVectorsList[j][i])
        if predictionList[oneTargetIdx][i] != target[i]:
            sampleID.append(i)
            sampleTarget.append(target[i].item())
            predictions.append(pred)
            predVectors.append(predVect)
    return sampleID, sampleTarget, predictions, predVectors


def filterModelsFixed(sampleID, sampleTarget, predictions, predVectors, selectModels):
    filteredPredictions = predictions[:, selectModels]
    #print(filteredPredictions.shape)
    filteredPredVectors = predVectors[:, selectModels]
    return sampleID, sampleTarget, filteredPredictions, filteredPredVectors

