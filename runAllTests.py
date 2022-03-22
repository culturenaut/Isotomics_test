from datetime import date
import methionineTest as metTest
import fragmentAndSimulate as fas
import solveSystem as ss
import readInput as ri
import os
import json
import numpy as np
import copy

today = date.today()

#Only run the tests within this list. Tests 2 (Massive clumps) and 5 (alternate standardization) use substantially 
#different algorithms, and so are dealt with separately. 
toRun = ['PerfectMeasurement']
#toRun = ['PerfectMeasurement','ExpError','InstFrac','LowAbund','Unresolved','AllIssue']

#Use this for experimentalPACorrection
sixtyOneCorrect = {'MNKey':'M1','fragToCorrect':'61','subToCorrect':'33S','fragsToBenchmarkFrom':['104']}
oneThreeThreeCorrect = {'MNKey':'M1','fragToCorrect':'133','subToCorrect':'33S','fragsToBenchmarkFrom':['104']}
fullCorrect = {'MNKey':'M1','fragToCorrect':'full','subToCorrect':'33S','fragsToBenchmarkFrom':['104']}

#Use this for omitted Peaks. This is necessary because some ion beams are <1% for standard but not sample, or vice versa. 
#We determined which beams that applied to, then filled in this dictionary, so that they are omitted in both cases. 
forbiddenPeaks = {'M1':{'61':['D']},
                  'M2':{'104':['13C']},
                  'M3':{'full':['18O-33S'],'61':['34S-D']},
                  'M4':{'133':['13C-34S-D']}}

#Set composition for known standard
knownStd = [-30,-30,-30,-30,0,0,0,0,0,0,0,0,0]
#Set composition for unknown standard
unknownStd = [-48,-18,-38,-8,25,0,0,-100,100,100,-100,50,-50]

store61 = {}

with open('Unresolved_120k.json') as f:
    unresolvedDict = json.load(f)
              
conditions = {'PerfectMeasurement':{'abundanceThreshold':0,'massThreshold':4, 'saveInput' : True, 'saveOutput' : True,
                        'calcFF':False,'forwardModelDeltas':knownStd,
                        'forbiddenPeaks':{},'unresolvedDict':{},
                        'experimentalPACorrectList':[], 
                        'MN Error':0,'Molecular Average U Error':0,'N':1,'perturbTheoryPAAmtM1':0,
                        'perturbTheoryPAAmtMN':0, 'explicitOACorrect':{},
                        'extremeVals':{},'NUpdates':0,'abundanceCorrect':False},
              
              'ExpError':{'abundanceThreshold':0,'massThreshold':4, 'saveInput' : True, 'saveOutput' : True,
                        'calcFF':False,'forwardModelDeltas':knownStd,
                        'forbiddenPeaks':{},'unresolvedDict':{},
                        'experimentalPACorrectList':[], 
                        'MN Error':0.001,'Molecular Average U Error':0.0001,'N':1000,'perturbTheoryPAAmtM1':0,
                        'perturbTheoryPAAmtMN':0, 'explicitOACorrect':{},
                        'extremeVals':{},'NUpdates':0,'abundanceCorrect':False},
              
              'InstFrac':{'abundanceThreshold':0,'massThreshold':4, 'saveInput' : True, 'saveOutput' : True,
                        'calcFF':True,'forwardModelDeltas':unknownStd,
                        'forbiddenPeaks':{},'unresolvedDict':{},
                        'experimentalPACorrectList':[], 
                        'MN Error':0.001,'Molecular Average U Error':0.0001,'N':1000,'perturbTheoryPAAmtM1':0,
                        'perturbTheoryPAAmtMN':0, 'explicitOACorrect':{},
                        'extremeVals':{},'NUpdates':0,'abundanceCorrect':False},
              
              'LowAbund':{'abundanceThreshold':0.01,'massThreshold':4, 'saveInput' : True, 'saveOutput' : True,
                        'calcFF':False,'forwardModelDeltas':knownStd,
                        'forbiddenPeaks':forbiddenPeaks,'unresolvedDict':{},
                        'experimentalPACorrectList':[], 
                        'MN Error':0.001,'Molecular Average U Error':0.0001,'N':1000,'perturbTheoryPAAmtM1':0,
                        'perturbTheoryPAAmtMN':0.001, 'explicitOACorrect':{},
                        'extremeVals':{},'NUpdates':50,'abundanceCorrect':True},
              
              'Unresolved':{'abundanceThreshold':0,'massThreshold':4, 'saveInput' : True, 'saveOutput' : True,
                        'calcFF':False,'forwardModelDeltas':knownStd,
                        'forbiddenPeaks':{},'unresolvedDict':unresolvedDict,
                        'experimentalPACorrectList':[], 
                        'MN Error':0.001,'Molecular Average U Error':0.0001,'N':1000,'perturbTheoryPAAmtM1':0,
                        'perturbTheoryPAAmtMN':0.001, 'explicitOACorrect':{},
                        'extremeVals':{},'NUpdates':50,'abundanceCorrect':True},
              
              'AllIssue':{'abundanceThreshold':0.01,'massThreshold':4, 'saveInput' : True, 'saveOutput' : True,
                        'calcFF':True,'forwardModelDeltas':unknownStd,
                        'forbiddenPeaks':forbiddenPeaks,'unresolvedDict':unresolvedDict,
                        'experimentalPACorrectList':[], 
                        'MN Error':0.001,'Molecular Average U Error':0.0001,'N':1000,'perturbTheoryPAAmtM1':0,
                        'perturbTheoryPAAmtMN':0.001, 'explicitOACorrect':{},
                        'extremeVals':{},'NUpdates':50,'abundanceCorrect':True}}

conditions = {key:value for key, value in conditions.items() if key in toRun}

resultsAbbr = {}
perturbedSamples = {}
Halphabetagamma = {}
Calphabetagamma = {}

for testKey, testData in conditions.items():
    #Generate Synthetic Dataset
    print("Running " + testKey)
    
    #Sample
    print("Calculating Sample Data")
    deltasSmp = [-45,-35,-30,-25,-13,2.5,10,-250,-100,0,100,250,0]
    fragSubset = ['full','133','104','102','61','56']
    df, expandedFrags, fragSubgeometryKeys, fragmentationDictionary = metTest.initializeMethionine(deltasSmp, fragSubset,
                                                                                                  printHeavy = False)


    predictedMeasurementSmp, MNDictSmp, FFSmp = metTest.simulateMeasurement(df, fragmentationDictionary, expandedFrags,
                                                                   fragSubgeometryKeys,
                                                       abundanceThreshold = testData['abundanceThreshold'],
                                                       massThreshold = testData['massThreshold'],
                                                       outputPath = str(today) + ' ' + testKey + ' Sample',
                                                       disableProgress = True,
                                                       calcFF = testData['calcFF'],
                                                       omitMeasurements = testData['forbiddenPeaks'],
                                                       unresolvedDict = testData['unresolvedDict'])
    
    #Standard
    print("Calculating Standard Data")
    deltasStd = [-30,-30,-30,-30,0,0,0,0,0,0,0,0,0]
    df, expandedFrags, fragSubgeometryKeys, fragmentationDictionary = metTest.initializeMethionine(deltasStd, fragSubset,
                                                                                                  printHeavy = False)

    predictedMeasurementStd, MNDictStd, FFStd = metTest.simulateMeasurement(df, fragmentationDictionary, expandedFrags,
                                                                            fragSubgeometryKeys,
                                                       abundanceThreshold = testData['abundanceThreshold'],
                                                       massThreshold = testData['massThreshold'],
                                                       outputPath = str(today) + ' ' + testKey + ' Standard',
                                                       disableProgress = True,
                                                       fractionationFactors = FFSmp,
                                                       omitMeasurements = testData['forbiddenPeaks'],
                                                       unresolvedDict = testData['unresolvedDict'])
    
    #Forward Model of Standard
    print("Calculating Forward Model")
    df, expandedFrags, fragSubgeometryKeys, fragmentationDictionary = metTest.initializeMethionine(testData['forwardModelDeltas'], fragSubset, printHeavy = False)
    
    predictedMeasurementFMStd, MNDictFMStd, FFFM = metTest.simulateMeasurement(df, fragmentationDictionary, expandedFrags,
                                                                  fragSubgeometryKeys, 
                                                   abundanceThreshold = 0,
                                                   massThreshold = testData['massThreshold'],
                                                    disableProgress = True)
    
    #Read in  data 
    standardJSON = ri.readJSON(str(today) + ' ' + testKey + ' Standard.json')
    processStandard = ri.readComputedData(standardJSON, error = testData['MN Error'], theory = predictedMeasurementFMStd)

    sampleJSON = ri.readJSON(str(today) + ' ' + testKey + ' Sample.json')
    processSample = ri.readComputedData(sampleJSON, error = testData['MN Error'])
    UValuesSmp = ri.readComputedUValues(sampleJSON, error = testData['Molecular Average U Error'])

    #solve M+1
    print("Solving M+1")
    isotopologuesDict = fas.isotopologueDataFrame(MNDictFMStd, df)
    pACorrection = ss.percentAbundanceCorrectTheoretical(predictedMeasurementFMStd, processSample, massThreshold = testData['massThreshold'])

    M1Results = ss.M1MonteCarlo(processStandard, processSample, pACorrection, isotopologuesDict,
                                fragmentationDictionary, perturbTheoryPAAmt = testData['perturbTheoryPAAmtM1'],
                                experimentalPACorrectList = testData['experimentalPACorrectList'],
                                N = testData['N'], GJ = False, debugMatrix = False, disableProgress = True,
                               storePerturbedSamples = True, storepACorrect = True, 
                               explicitOACorrect = testData['explicitOACorrect'],
                               abundanceCorrect = testData['abundanceCorrect'])
    
    perturbedSamples[testKey] = M1Results['Extra Info']['Perturbed Samples']

    processedResults = ss.processM1MCResults(M1Results, UValuesSmp, isotopologuesDict, df, disableProgress = True,
                                            UMNSub = ['13C'])
    
    ss.updateSiteSpecificDfM1MC(processedResults, df)
    
    M1Df = df.copy()
    M1Df['deltas'] = M1Df['PDB etc. Deltas']
    
    
    oldDeltas = list(M1Df['deltas'])
    
    pADict = {}
    if testData['NUpdates'] > 0:
        iterateStandard = {key: value for (key, value) in processStandard.items() if key == 'M1'}
        iterateSample = {key: value for (key, value) in processSample.items() if key == 'M1'}
        
        M1Results, thispADict  = metTest.updateAbundanceCorrection(oldDeltas, fragSubset, fragmentationDictionary, expandedFrags, fragSubgeometryKeys, iterateStandard, iterateSample, isotopologuesDict, UValuesSmp, df, deviation = 4, 
                              perturbTheoryPAAmt = testData['perturbTheoryPAAmtM1'], 
                              NUpdates = testData['NUpdates'], 
                              breakCondition = 1e-3,
                              experimentalPACorrectList = testData['experimentalPACorrectList'],
                              extremeVals = testData['extremeVals'],
                              abundanceThreshold = testData['abundanceThreshold'], 
                              massThreshold = 1, 
                              omitMeasurements = testData['forbiddenPeaks'], 
                              unresolvedDict = testData['unresolvedDict'],
                              UMNSub = ['13C'],
                              N = testData['N'])
        
        pADict[testKey] = copy.deepcopy(thispADict)
        
    Halphabetagamma[testKey] = []
    for res in processedResults['Relative Deltas']:
        Halphabetagamma[testKey].append(res[8] + res[9])
        
    Calphabetagamma[testKey] = []
    for res in processedResults['Relative Deltas']:
        Calphabetagamma[testKey].append(res[1] + res[2])
        
        
    processedResults = ss.processM1MCResults(M1Results, UValuesSmp, isotopologuesDict, df, disableProgress = True,
                                    UMNSub = ['13C'])
    
    ss.updateSiteSpecificDfM1MC(processedResults, df)
    
    #Recalculate forward model to inform stochastic reference frame for M+2, M+3, M+4 
    M1Df = df.copy()
    M1Df['deltas'] = M1Df['PDB etc. Deltas']
    
    #recalculate to inform stochastic reference frame for MN
    print("Calculating Sample forward model")
    predictionFMSmp, MNDictFMSmp, secondFFFM = metTest.simulateMeasurement(M1Df, fragmentationDictionary, expandedFrags,
                                                                     fragSubgeometryKeys,
                                                       abundanceThreshold = 0,
                                                       massThreshold = testData['massThreshold'], disableProgress = True)

    isotopologuesDict = fas.isotopologueDataFrame(MNDictFMSmp, M1Df)
    pACorrection = ss.percentAbundanceCorrectTheoretical(predictionFMSmp, processSample, massThreshold = testData['massThreshold'])
    
    #Initialize dictionary for M+N solutions
    MNSol = {'M2':0,'M3':0,'M4':0}
    #Determine which substitutions are used for U Value scaling
    UMNSubs = {'M2':['34S'],'M3':['34S15N'],'M4':['36S']}

    for MNKey in MNSol.keys():
        print("Solving " + MNKey)

        Isotopologues = isotopologuesDict[MNKey]

        results, comp, GJSol, meas = ss.MonteCarloMN(MNKey, Isotopologues, processStandard, processSample, 
                                            pACorrection, fragmentationDictionary, N = testData['N'], disableProgress = True, perturbTheoryPAAmt = testData['perturbTheoryPAAmtMN'])


        dfOutput = ss.checkSolutionIsotopologues(GJSol, Isotopologues, MNKey, numerical = False)
        nullSpaceCycles = ss.findNullSpaceCycles(comp, Isotopologues)
        actuallyConstrained = ss.findFullyConstrained(nullSpaceCycles)
        processedResults = ss.processMNMonteCarloResults(MNKey, results, UValuesSmp, dfOutput, df, MNDictFMStd,
                                                        UMNSub = UMNSubs[MNKey], disableProgress = True)
        dfOutput = ss.updateMNMonteCarloResults(dfOutput, processedResults)
        MNSol[MNKey] = dfOutput.loc[dfOutput.index.isin(actuallyConstrained)].copy()
        
    if testData['saveInput'] == False: 
        if os.path.exists(str(today) + ' ' + testKey + ' Sample.json'):
            os.remove(str(today) + ' ' + testKey + ' Sample.json')
        if os.path.exists(str(today) + ' ' + testKey + ' Standard.json'):
            os.remove(str(today) + ' ' + testKey + ' Standard.json')
            
    if testData['saveOutput']:
        f = str(today) + '_' + testKey + '_' + 'Results.csv'
        M1Df.to_csv(f, mode = 'a',header=True)
        MNSol["M2"].to_csv(f, mode = 'a', header = True)
        MNSol["M3"].to_csv(f, mode = 'a', header = True)
        MNSol["M4"].to_csv(f, mode = 'a', header = True)
        
    resultsAbbr[testKey] = {'M1':{},'M2':{},'M3':{},'M4':{}}
    resultsAbbr[testKey]['M1']["Values"] = list(M1Df['Relative Deltas'])
    resultsAbbr[testKey]['M1']["Errors"] = list(M1Df['Relative Deltas Error'])
    
    resultsAbbr[testKey]['M2']['Identity'] = list(MNSol['M2'].index)
    resultsAbbr[testKey]['M2']['Values'] = list(MNSol['M2']['Clumped Deltas Relative'])
    resultsAbbr[testKey]['M2']['Errors'] = list(MNSol['M2']['Clumped Deltas Relative Error'])
    
    resultsAbbr[testKey]['M3']['Identity'] = list(MNSol['M3'].index)
    resultsAbbr[testKey]['M3']['Values'] = list(MNSol['M3']['Clumped Deltas Relative'])
    resultsAbbr[testKey]['M3']['Errors'] = list(MNSol['M3']['Clumped Deltas Relative Error'])
    
    resultsAbbr[testKey]['M4']['Identity'] = list(MNSol['M4'].index)
    resultsAbbr[testKey]['M4']['Values'] = list(MNSol['M4']['Clumped Deltas Relative'])
    resultsAbbr[testKey]['M4']['Errors'] = list(MNSol['M4']['Clumped Deltas Relative Error'])
    
with open(str(today) + 'Test_OW_Results_Abbr.json', 'w', encoding='utf-8') as f:
    json.dump(resultsAbbr, f, ensure_ascii=False, indent=4)
    
with open(str(today) + 'Test_OW_Perturbed_Samples.json', 'w', encoding='utf-8') as f:
    json.dump(perturbedSamples, f, ensure_ascii=False, indent=4)