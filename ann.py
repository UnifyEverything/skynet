# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 15:24:03 2015

    This is a python implementation of an Artificial Neural Network
    We are using it to learn and solve math problems

@author: andrew
"""

from numpy import *
from pylab import *


class Neuron:
    """The Neuron class defines a neuron which exists within one layer of neurons"""
    
    # Each neuron has an integer numInputs
    numInputs = 0
    # Each input must have a weight, which are stored in a numpy ndarray
    # array of type double
    arrWeights = ndarray
    # Each neuron will have a firing threshhold.  This is treated as a weight
    threshhold = 1.0
    # Bias for neuron activation
    dBias = 1.0
    # activation response for neuron activation
    dActivationResponse = 1.0
    
    def __init__(self, numInput, vecWeight, threshhold):
        self.numInputs = numInput
        self.vecWeights = vecWeight


class NeuronLayer:
    """The NeuronLayer class defines a layer of neurons within our neural network"""
    
    # the integer number of neurons in this layer
    numNeurons = 0
    # the array of neurons
    # array of type Neuron
    arrNeurons = ndarray
    # number of inputs per neuron
    numInputsPerNeuron = 1
    
    def __init__(self, numNeuron, vecNeuron, numInputsPerNeuron):
        self.numNeurons = numNeuron
        self.vecNeurons = vecNeuron
        self.numInputsPerNeuron = numInputsPerNeuron

class PythonANN:
    """This is a simple Artificial Neural Network built in Python"""
    
    # global values for one neural network
    NumInputs = 0
    NumOutputs = 0
    NumHiddenLayers = 0
    NeuronsPerHiddenLyr = 0
    
    # ndarray of all layers in the neural network (including output layer)
    # array of type NeuronLayer
    arrLayers = ndarray
    
    def __init__(self, numIn, numOut, numLayer, numNeuron):
        self.NumInputs = numIn
        self.NumOutputs = numOut
        self.NumHiddenLayers = numLayer - 1
        self.NeuronsPerHiddenLyr = numNeuron
    
    def getWeights():
        pass
    
    def getNumWeights():
        pass
    
    def updateWeights(newWeights):
        pass
    
    def updateNN(inputs):
        # stores resulting outputs from each layer
        outputs = ndarray
        
        cWeight = 0
        
        # check that we have correct number of inputs
        if inputs.size != NumInputs:
            # if incorrect, just return empty array
            return outputs
        
        # for each layer...
        for i in range(NumHiddenLayers+1):
            if i >= 0:
                inputs = outputs
            
            outputs = zeros
            cWeight = 0
            
            # for each neuron sum the (inputs * weights)
            # throw total at sigmoid function to get output
            for j in arrLayers[i].numNeurons:
                netInput = 0.0
                NumInputs = arrLayers[i].arrNeurons[j].NumInputs
                
                # for each weight...
                for k in range(NumInputs-1):
                    # sum the weights * inputs
                    netInput += arrLayers[i].arrNeurons[j].arrWeight[k] * inputs[cWeight]
                    cWeight += 1
                
                # add in the bias
                netInput += arrLayers[i].arrNeurons[j].arrWeight[NumInputs-1] * dBias
                
                # store the outputs from each layer as we generate them
                # the combined activation is first filtered through the sigmoid function
                numpy.append(outputs, sigmoid(netInput, dActivationResponse))
                
                cWeight = 0
        return outputs
    
    def sigmoid(activationLevel, responseLevel):
        pass
    
    