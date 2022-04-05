import sys
import os

#Relative import .py files from the parent directory
sys.path.insert(0,os.getcwd())

from datetime import date
import copy
import time

from scipy import optimize
import numpy as np
import pandas as pd

import basicDeltaOperations as op
import methionineTest as metTest
import calcIsotopologues as ci
import fragmentAndSimulate as fas
import readInput as ri
import solveSystem as ss

#DETERMINE CLUMPING AMOUNTS FOR SOME INPUT CLUMPED DELTAS
#Initialize stochastic
deltas = [-45,-35,-30,-25,-13,2.5,10,-250,-100,0,100,250,0]
fragSubset = ['full','133','104','102','61','56']
molecularDataFrame, expandedFrags, fragSubgeometryKeys, fragmentationDictionary = metTest.initializeMethionine(deltas, fragSubset)

#calculate "byAtom" dict.
print("Calculating Isotopologue Concentrations")
siteElements = ci.strSiteElements(molecularDataFrame)
siteIsotopes, multinomialCoeff = ci.calculateSetsOfSiteIsotopes(molecularDataFrame)
bigA, SN = ci.calcAllIsotopologues(siteIsotopes, multinomialCoeff)
concentrationArray = ci.siteSpecificConcentrations(molecularDataFrame)
d = ci.calculateIsotopologueConcentrations(bigA, SN, concentrationArray)

print("Compiling Isotopologue Dictionary")
stochD = ci.calcAtomDictionary(d, molecularDataFrame)

###Example of how to find correct clumping values for some cap Deltas
target = np.array([250,-250,100])
ninit = {'N':0}

#Get stochastic concentrations
double13CStoch = stochD['110000000000000000000']['Conc']
double15NDStoch = stochD['000000001000000000100']['Conc']
double33SDStoch = stochD['000000010000010000000']['Conc']
UnsubStoch = stochD['000000000000000000000']['Conc']

stoch = {'d13C' :double13CStoch, 'd15N':double15NDStoch, 'd33S':double33SDStoch, 'Unsub':UnsubStoch}

#Construct an objective function to minimize
def objective(x):
    ninit['N'] += 1
    if ninit['N'] % 10 == 0:
        print(ninit['N'])
        
    double13CStoch = stoch['d13C']
    double15NDStoch = stoch['d15N']
    double33SDStoch = stoch['d33S']
    UnsubStoch = stoch['Unsub']
    
    #Add some clumped anamoly to each.
    double13CClump = double13CStoch + x[0]
    double15NDClump = double15NDStoch + 2 * x[1]
    double33SDClump = double33SDStoch + 2 * x[2]
    
    UnsubClump = UnsubStoch + x.sum()
    
    #Calculate Cap Delta for each clump of interest
    a = double13CStoch / UnsubStoch
    b = double13CClump / UnsubClump
    capDelta13C13C = 1000 * (b/a -1)
    
    a = double15NDStoch / UnsubStoch
    b = double15NDClump / UnsubClump
    capDelta15ND = 1000 * (b/a -1)
    
    a = double33SDStoch / UnsubStoch
    b = double33SDClump / UnsubClump
    capDelta33SD = 1000 * (b/a -1)
    
    capDeltas = np.array([capDelta13C13C, capDelta15ND, capDelta33SD])
    if ninit['N'] % 10 == 0:
        print(capDeltas)
    
    return 10**10 * ((capDeltas - target)**2).sum()

#Minimize objective function.
minimization = optimize.minimize(objective, np.array([0,0,0]),method = 'Nelder-Mead',options={'xtol': 0.001, 'ftol': 0.0001,
'maxiter': None, 'maxfev': None})

#Construct clumped dataset
today = date.today()

#Construct sample
deltas = [-45,-35,-30,-25,-13,2.5,10,-250,-100,0,100,250,0]
fragSubset = ['full','133','104','102','61','56']
molecularDataFrame, expandedFrags, fragSubgeometryKeys, fragmentationDictionary = metTest.initializeMethionine(deltas, fragSubset)

#Values from optimization, above
clumpD =  {'01':{'Sites':['Cmethyl','Cgamma'],'Amount':minimization['x'][0]}, 
           '02':{'Sites':['Namine','Hamine'],'Amount':minimization['x'][1]},
           '03':{'Sites':['Ssulfur','Hgamma'],'Amount':minimization['x'][2]}}

#clumpD = clumpD
predictedMeasurement, MNDict, FF = metTest.simulateMeasurement(molecularDataFrame, fragmentationDictionary, expandedFrags,
                                                               fragSubgeometryKeys,
                                                   abundanceThreshold = 0,
                                                   massThreshold = 4,
                                                   clumpD = clumpD,
                                                   outputPath = str(today) + " Sample Clumped")

#Standard, no clump
deltas = [-30,-30,-30,-30,0,0,0,0,0,0,0,0,0]
fragSubset = ['full','133','104','102','61','56']
molecularDataFrame, expandedFrags, fragSubgeometryKeys, fragmentationDictionary = metTest.initializeMethionine(deltas, fragSubset)

predictedMeasurement, MNDict, secondFF = metTest.simulateMeasurement(molecularDataFrame, fragmentationDictionary, expandedFrags,
                                                                     fragSubgeometryKeys,
                                                   abundanceThreshold = 0,
                                                   massThreshold = 4,
                                                   outputPath = str(today) + " Standard Stochastic")

###SOLVE SYSTEM
###Forward model of standard
deltas = [-30,-30,-30,-30,0,0,0,0,0,0,0,0,0]
fragSubset = ['full','133','104','102','61','56']
molecularDataFrame, expandedFrags, fragSubgeometryKeys, fragmentationDictionary = metTest.initializeMethionine(deltas, fragSubset)

predictedMeasurement, MNDictStd, FF = metTest.simulateMeasurement(molecularDataFrame, fragmentationDictionary, expandedFrags, 
                                                               fragSubgeometryKeys, 
                                                   abundanceThreshold = 0,
                                                   massThreshold = 4)

standardJSON = ri.readJSON(str(today) + " Standard Stochastic.json")
processStandard = ri.readComputedData(standardJSON, error = 0, theory = predictedMeasurement)

sampleJSON = ri.readJSON(str(today) + " Sample Clumped.json")
processSample = ri.readComputedData(sampleJSON, error = 0)
UValuesSmp = ri.readComputedUValues(sampleJSON, error = 0)

isotopologuesDict = fas.isotopologueDataFrame(MNDictStd, molecularDataFrame)
OCorrection = ss.OValueCorrectTheoretical(predictedMeasurement, processSample)

M1Results = ss.M1MonteCarlo(processStandard, processSample, OCorrection, isotopologuesDict,
                            fragmentationDictionary, N = 1)

processedResults = ss.processM1MCResults(M1Results, UValuesSmp, isotopologuesDict, molecularDataFrame)
ss.updateSiteSpecificDfM1MC(processedResults, molecularDataFrame)

M1Df = molecularDataFrame.copy()
#recalculate to inform stochastic reference frame for MN
recalcPrediction, MNDictFM, FFSecond = metTest.simulateMeasurement(M1Df, fragmentationDictionary, expandedFrags,
                                                                 fragSubgeometryKeys,
                                                   abundanceThreshold = 0,
                                                   massThreshold = 4)

isotopologuesDict = fas.isotopologueDataFrame(MNDictFM, molecularDataFrame)
OCorrection = ss.OValueCorrectTheoretical(predictedMeasurement, processSample)

#Solve MN
MNSol = {'M2':0,'M3':0,'M4':0}
UMNSubs = {'M2':['34S','18O'],'M3':['34S15N'],'M4':['36S']}

for MNKey in MNSol.keys():
    print("Solving " + MNKey)
    
    Isotopologues = isotopologuesDict[MNKey]

    results, comp, GJSol, meas = ss.MonteCarloMN(MNKey, Isotopologues, processStandard, processSample, 
                                        OCorrection, fragmentationDictionary, N = 1)
              
        
    dfOutput = ss.checkSolutionIsotopologues(GJSol, Isotopologues, MNKey, numerical = False)
    nullSpaceCycles = ss.findNullSpaceCycles(comp, Isotopologues)
    actuallyConstrained = ss.findFullyConstrained(nullSpaceCycles)
    processedResults = ss.processMNMonteCarloResults(MNKey, results, UValuesSmp, dfOutput, molecularDataFrame, MNDictStd,
                                                    UMNSub = UMNSubs[MNKey])
    dfOutput = ss.updateMNMonteCarloResults(dfOutput, processedResults)
    MNSol[MNKey] = dfOutput.loc[dfOutput.index.isin(actuallyConstrained)].copy()
    
    
#Iterator to Systematically improve precision.
#The main downside now is that it is very expensive. This would be improved in a situation with less perfect knowledge, i.e. fewer measurements and smaller matrices. 

#Typically, the errors induced by natural abundance clumps are small relative to measurement error. So I suspect this will only be necessary in boutique situations. 
N = 1
t = time.time()

for i in range(N):
    told = t
    t = time.time()
    if i > 0:
        tdiff = t-told
        print("Iteration " + str(i+1) + " of " + str(N))
        print("Previous iteration took " + str(tdiff) + " seconds")
    
    M2Solution = MNSol['M2']
    UnsubConc = MNDictFM['M0']['000000000000000000000']['Conc']
    largeClumps = {}
    for i, v in M2Solution[M2Solution['Number'] < 2].iterrows():
        if type(v['Clumped Deltas Stochastic']) != str:
            sites = i.split('   |   ')
            if np.abs(v['Clumped Deltas Stochastic']) > 50:
                for site in sites:
                    shortSite = site.split(' ')[1]
                    if shortSite not in largeClumps:
                        largeClumps[shortSite] = 0
                    largeClumps[shortSite] += (v['U Values'] - v['Stochastic U']) * UnsubConc

    UnsubConcAdj = UnsubConc
    for i, v in largeClumps.items():
        UnsubConcAdj += v

    UnsubChange = UnsubConcAdj/UnsubConc

    clumpAdjustedU = M1Df['Calc U Values'].copy().values
    for i, v in M1Df.iterrows():
        j = list(M1Df.index).index(i)

        clumpAdjustedU[j] *= UnsubChange

        if i in largeClumps:
            clumpAdjustedU[j] += largeClumps[i] / UnsubConc

    M1Df['Clump Adjusted U Values'] = clumpAdjustedU

    norm = [x / y for x, y in zip(M1Df['Clump Adjusted U Values'], molecularDataFrame['Number'])]
    deltas = [op.ratioToDelta(x,y) for x, y in zip(M1Df.IDS,norm)]
    M1Df['deltas'] = deltas
    M1Df['VPDB etc. Deltas'] = deltas

    #Recalculate to inform our stochastic reference frame 
    molecularDataFrame, expandedFrags, fragKeys, fragmentationDictionary = metTest.initializeMethionine(deltas,fragSubset = fragSubset,printHeavy = False)

    predictedMeasurement, MNDictFM, FF = metTest.simulateMeasurement(molecularDataFrame, fragmentationDictionary, expandedFrags, fragKeys,abundanceThreshold = 0, massThreshold = 4, disableProgress = True)

    isotopologuesDict = fas.isotopologueDataFrame(MNDictFM, molecularDataFrame)
    OCorrection = ss.OValueCorrectTheoretical(predictedMeasurement, processSample)
    
    MNSol = {'M2':0,'M3':0,'M4':0}

    for MNKey in MNSol.keys():
        Isotopologues = isotopologuesDict[MNKey]

        results, comp, GJSol, meas = ss.MonteCarloMN(MNKey, Isotopologues, processStandard, processSample, 
                                            OCorrection, fragmentationDictionary, N = 1,
                                              disableProgress = True)

        dfOutput = ss.checkSolutionIsotopologues(GJSol, Isotopologues, MNKey, numerical = False)
        nullSpaceCycles = ss.findNullSpaceCycles(comp, Isotopologues)
        actuallyConstrained = ss.findFullyConstrained(nullSpaceCycles)
        processedResults = ss.processMNMonteCarloResults(MNKey, results, UValuesSmp, dfOutput, molecularDataFrame, MNDictStd,
                                                        UMNSub = UMNSubs[MNKey], disableProgress = True)
        dfOutput = ss.updateMNMonteCarloResults(dfOutput, processedResults)
        MNSol[MNKey] = dfOutput.loc[dfOutput.index.isin(actuallyConstrained)].copy()

print("Finished with iteration")
#We did not update and so do not report the relative M1 Deltas.
M1Df = M1Df.drop(['Relative Deltas', 'Relative Deltas Error'], axis=1)
M1Df.to_csv("ClumpedM1Results.csv")
MNSol["M2"].to_csv("ClumpedM2Results.csv")
MNSol["M3"].to_csv("ClumpedM3Results.csv")
MNSol["M4"].to_csv("ClumpedM4Results.csv")