import sys
import os

#Relative import .py files from the parent directory
sys.path.insert(0,os.getcwd())

import methionineTest as metTest
import readInput as ri
import fragmentAndSimulate as fas
import solveSystem as ss

import pandas as pd
import numpy as np
import sympy as sy

from datetime import date

today = date.today()

deltas = [-45,-35,-30,-25,-13,2.5,10,-250,-100,0,100,250,0]
fragSubset = ['full','133','104','102','61','56']
molecularDataFrame, expandedFrags, fragSubgeometryKeys, fragmentationDictionary = metTest.initializeMethionine(deltas, fragSubset)

#To generate "No fractionation" dataset, set calcFF = False in this box
#also rename sample and standard files. 
predictedMeasurement, MNDict, fractionationFactors = metTest.simulateMeasurement(molecularDataFrame, fragmentationDictionary, 
                                                                                 expandedFrags, 
                                                                                 fragSubgeometryKeys, 
                                                   abundanceThreshold = 0,
                                                   massThreshold = 4,
                                                   outputPath = str(today) + " No FM Sample Stochastic",
                                                   calcFF = True)

deltas = [-30,-30,-30,-30,0,0,0,0,0,0,0,0,0]
fragSubset = ['full','133','104','102','61','56']
molecularDataFrame, expandedFrags, fragSubgeometryKeys, fragmentationDictionary = metTest.initializeMethionine(deltas, fragSubset)

predictedMeasurement, MNDict, secondFF = metTest.simulateMeasurement(molecularDataFrame, fragmentationDictionary, expandedFrags, 
                                                                     fragSubgeometryKeys, 
                                                   abundanceThreshold = 0,
                                                   massThreshold = 4,
                                                   outputPath = str(today) + " No FM Standard Stochastic",
                                                   calcFF = False, fractionationFactors = fractionationFactors)

def stdToDf(perturbedStandard):
    '''
    Takes a perturbed standard from ss.perturbStandard and reorganizes it to be made into a dataframe.
    
    Inputs:
        perturbedStandard: The perturbed standard.

    Outputs:
        standardForDf: A dictionary in the proper format to be turned into a DataFrame.

    '''
    standardForDf = {}
    for MS, fragment in perturbedStandard.items():
        standardForDf[MS] = {}
        for fragKey, data in fragment.items():
            standardForDf[MS][fragKey] = {}
            for index, sub in enumerate(data['Subs']):
                standardForDf[MS][fragKey][sub] = data['Observed Abundance'][index]
                
    return standardForDf

def constructSolveMatrix(dataDf, Isotopologues, OValueCorrection, fragKeys, GJ = False, fullGJ = False):
    '''
    A matrix solving routine similar to that implemented in the monte carlo routines in solveSystem. In this case, the procedure is run for both of sample and standard individually, rather than after standardizing.

    Inputs:
        dataDf: A dataFrame containing the sample or standard observations.
        Isotopologues: A dataFrame containing isotopologues and information about their fragmentation.
        OValueCorrection: The O Value correction factors to correct for low abundance.
        fragKeys: A list of strings, where each string indicates a fragment (e.g. '133')
        GJ: Whether to use the Gauss Jordan elimination method or not. 
        fullGJ: if True, return the full output of ss.GJElim. Useful for debugging. 

    Outputs: 
        numpy: The numpy.linalg.lstsq solution to the matrix. 

    '''
    CMatrix = []
    MeasurementVector = []
    CMatrix.append([1]*len(Isotopologues.index))
    MeasurementVector.append(1)
    for frag in fragKeys:
        IsotopologueFragments = Isotopologues[frag + '_01 Subs']
        for sub, v in dataDf[frag].items():
            #If the observed intensity of a fragment is 0, we do not include it
            if v != 0:
                c = list(IsotopologueFragments.isin([sub]) * 1)
                CMatrix.append(c)
                MeasurementVector.append(v * OValueCorrection['M1'][frag])
    comp = np.array(CMatrix,dtype=float)
    meas = np.array(MeasurementVector,dtype = float)
    
    if GJ:
        AugMatrix = np.column_stack((comp, meas))

        solve = ss.GJElim(AugMatrix, augMatrix = True, store = True)
        
        rank = solve[1]
        
        if fullGJ:
            return solve
        
        return solve[0][:,-1][:rank]

    numpy = np.linalg.lstsq(comp, meas, rcond = -1)[0]
    
    return numpy

def numpyToUValuesM1(numpy, UValues, isotopologuesDict, molecularDataFrame, MNKey = 'M1'):
    '''
    Takes the solved matrix and applies a UM+N value to move to U value space. Compare with solveSystem.processM1MCResults

    Inputs:
        numpy: The solution from constructSolveMatrix
        UValues: A dictionary of molecular average U values
        isotopologuesDict: A dictionary, where keys are MNKeys ("M1", "M2") and values are dataframes containing the isotopologues associated with that mass selection.
        molecularDataFrame: A dataframe containing basic information about the molecule. 
        MNKey: The MNKey to use for the procedure. 

    Outputs:
        normMN.values: The U Values following application of UM+N.
    '''
    out = isotopologuesDict[MNKey][['Number','Stochastic','Composition','Stochastic U','Precise Identity']].copy()

    UMNCalc = UValues['13C']['Observed'] / numpy[:4].sum()

    out['U' + MNKey] = UMNCalc
    out['Calc U Values'] = numpy * out['U' + MNKey]

    #The Isotopologues Dataframe has the substitutions in a different order than the site-specific dataframe. 
    #This section reassigns the solutions of the isotopologues dataframe to the right order for the 
    #site-specific dataframe
    M1 = [0] * len(out.index)
    UM1 = [0] * len(out.index)
    U = [0] * len(out.index)
    for i, v in out.iterrows():
        identity = v['Precise Identity'].split(' ')[1]
        index = list(molecularDataFrame.index).index(identity)

        UM1[index] = v['U' + MNKey]
        U[index] = v['Calc U Values']

        #calculate relevant information
        normMN = U / molecularDataFrame['Number']
    
    return normMN.values

#Approximation for standard
#The deltas don't matter; we are just using this to track isotopologues, not their concentrations
deltas = [-30,-30,-30,-30,0,0,0,0,0,0,0,0,0]
fragSubset = ['full','133','104','102','61','56']
molecularDataFrame, expandedFrags, fragSubgeometryKeys, fragmentationDictionary = metTest.initializeMethionine(deltas, fragSubset)

#We only want MNDict here! We are not actually using the forward model we caclulate. 
predictedMeasurement, MNDict, FF = metTest.simulateMeasurement(molecularDataFrame, fragmentationDictionary, expandedFrags, fragSubgeometryKeys, 
                                                   abundanceThreshold = 0,
                                                   massThreshold = 4)

#Read in the computed data
standardJSON = ri.readJSON(str(today) + " No FM Standard Stochastic.json")
processStandard = ri.readComputedData(standardJSON, error = 0, theory = predictedMeasurement)
UValuesStd = ri.readComputedUValues(standardJSON, error = 0)

sampleJSON = ri.readJSON(str(today) + " No FM Sample Stochastic.json")
processSample = ri.readComputedData(sampleJSON, error = 0)
UValuesSmp = ri.readComputedUValues(sampleJSON, error = 0)

isotopologuesDict = fas.isotopologueDataFrame(MNDict, molecularDataFrame)
OCorrection = ss.OValueCorrectTheoretical(predictedMeasurement, processSample)

#Solve M1 System
MNKey = "M1"
Isotopologues = isotopologuesDict[MNKey]
perturbedStandard = ss.perturbStandard(processStandard, theory = True)
smp = ss.perturbSample(processSample, perturbedStandard, OCorrection, correctionFactors = False, abundanceCorrect = False)[MNKey]
stdForDf = stdToDf(perturbedStandard)
std = pd.DataFrame.from_dict(stdForDf[MNKey])
std.fillna(0, inplace = True)

shortFragKeys = [x.split('_')[0] for x in fragSubgeometryKeys]

#Solve sample and standard individually
numpySmp = constructSolveMatrix(smp, Isotopologues, OCorrection, shortFragKeys, GJ = False, fullGJ = False)
numpyStd = constructSolveMatrix(std, Isotopologues, OCorrection, shortFragKeys, GJ = False, fullGJ = False)

#compute U Values individually.
numpySmpU = numpyToUValuesM1(numpySmp, UValuesSmp, isotopologuesDict, molecularDataFrame)
numpyStdU = numpyToUValuesM1(numpyStd, UValuesStd, isotopologuesDict, molecularDataFrame)

outputData = {}

x = 1000*(numpySmpU /  numpyStdU - 1)
print(x)

#Repeat the above for each MNKey. We abandoned this approach as inferior to forward model standardization.
MNKey = "M2"
Isotopologues = isotopologuesDict[MNKey]
perturbedStandard = ss.perturbStandard(processStandard, theory = True)
smp = ss.perturbSample(processSample, perturbedStandard, OCorrection, correctionFactors = False, abundanceCorrect = False)[MNKey]
stdForDf = stdToDf(perturbedStandard)
std = pd.DataFrame.from_dict(stdForDf[MNKey])
std.fillna(0, inplace = True)

numpySmp = constructSolveMatrix(smp, Isotopologues, OCorrection, shortFragKeys, GJ = True, fullGJ = False)
numpyStd = constructSolveMatrix(std, Isotopologues, OCorrection, shortFragKeys, GJ = True, fullGJ = False)

GJSolution = constructSolveMatrix(smp, Isotopologues, OCorrection, shortFragKeys, GJ = True, fullGJ = True)
solutions = ss.checkSolutionIsotopologues(GJSolution, Isotopologues, MNKey, numerical = False)
Sindex = list(solutions.index).index('34S Ssulfur')

UMNCalcSmp = UValuesSmp['34S']['Observed'] / numpySmp[Sindex]
numpySmpU = numpySmp * UMNCalcSmp

UMNCalcStd = UValuesStd['34S']['Observed'] / numpyStd[Sindex]
numpyStdU = numpyStd * UMNCalcStd

x = 1000*(numpySmpU /  numpyStdU - 1)

solutions['Sample Standard'] = x
print(solutions.T[['13C Cmethyl   |   13C Cgamma','13C Ccarboxyl   |   15N Namine', '18O Ocarboxyl', '34S Ssulfur']].T)

MNKey = "M3"
Isotopologues = isotopologuesDict[MNKey]
perturbedStandard = ss.perturbStandard(processStandard, theory = True)
smp = ss.perturbSample(processSample, perturbedStandard, OCorrection, correctionFactors = False, abundanceCorrect = False)[MNKey]
stdForDf = stdToDf(perturbedStandard)
std = pd.DataFrame.from_dict(stdForDf[MNKey])
std.fillna(0, inplace = True)

numpySmp = constructSolveMatrix(smp, Isotopologues, OCorrection, shortFragKeys, GJ = True, fullGJ = False)
numpyStd = constructSolveMatrix(std, Isotopologues, OCorrection, shortFragKeys, GJ = True, fullGJ = False)

GJSolution = constructSolveMatrix(smp, Isotopologues, OCorrection, shortFragKeys, GJ = True, fullGJ = True)
solutions = ss.checkSolutionIsotopologues(GJSolution, Isotopologues, MNKey, numerical = False)
index = list(solutions.index).index('18O Ocarboxyl   |   15N Namine')

UMNCalcSmp = UValuesSmp['18O15N']['Observed'] / numpySmp[index]
numpySmpU = numpySmp * UMNCalcSmp

UMNCalcStd = UValuesStd['18O15N']['Observed'] / numpyStd[index]
numpyStdU = numpyStd * UMNCalcStd

x = 1000*(numpySmpU /  numpyStdU - 1)

solutions['Sample Standard'] = x
print(solutions.T[['13C Cmethyl   |   13C Cgamma   |   33S Ssulfur','18O Ocarboxyl   |   15N Namine']].T)

MNKey = "M4"
Isotopologues = isotopologuesDict[MNKey]
perturbedStandard = ss.perturbStandard(processStandard, theory = True)
smp = ss.perturbSample(processSample, perturbedStandard, OCorrection, correctionFactors = False, abundanceCorrect = False)[MNKey]
stdForDf = stdToDf(perturbedStandard)
std = pd.DataFrame.from_dict(stdForDf[MNKey])
std.fillna(0, inplace = True)

numpySmp = constructSolveMatrix(smp, Isotopologues, OCorrection, shortFragKeys, GJ = True, fullGJ = False)
numpyStd = constructSolveMatrix(std, Isotopologues, OCorrection, shortFragKeys, GJ = True, fullGJ = False)

GJSolution = constructSolveMatrix(smp, Isotopologues, OCorrection, shortFragKeys, GJ = True, fullGJ = True)
solutions = ss.checkSolutionIsotopologues(GJSolution, Isotopologues, MNKey, numerical = False)
index = list(solutions.index).index('36S Ssulfur')

UMNCalcSmp = UValuesSmp['36S']['Observed'] / numpySmp[index]
numpySmpU = numpySmp * UMNCalcSmp

UMNCalcStd = UValuesStd['36S']['Observed'] / numpyStd[index]
numpyStdU = numpyStd * UMNCalcStd

x = 1000*(numpySmpU /  numpyStdU - 1)

solutions['Sample Standard'] = x
print(solutions.T[['13C Cmethyl   |   13C Cgamma   |   13C Calphabeta   |   13C Calphabeta',
             '13C Cmethyl   |   17O Ocarboxyl   |   18O Ocarboxyl',
            '36S Ssulfur']].T)