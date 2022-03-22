import copy
import json

import numpy as np
import pandas as pd

import basicDeltaOperations as op
import calcIsotopologues as ci
import fragmentAndSimulate as fas
import solveSystem as ss

def initializeAlanine(deltas, fragSubset = ['full','44'], printHeavy = True):
    ##### INITIALIZE SITES #####
    IDList = ['Calphabeta','Ccarboxyl','Ocarboxyl','Namine','Hretained','Hlost']
    elIDs = ['C','C','O','N','H','H']
    numberAtSite = [2,1,2,1,6,2]

    l = [elIDs, numberAtSite, deltas]
    cols = ['IDS','Number','deltas']

    allFragments = {'full':{'01':{'subgeometry':[1,1,1,1,1,1],'relCont':1}},
                  '44':{'01':{'subgeometry':[1,'x','x',1,1,'x'],'relCont':1}}}

    fragmentationDictionary = {key: value for key, value in allFragments.items() if key in fragSubset}

    condensedFrags =[]
    fragSubgeometryKeys = []
    for fragKey, subFragDict in fragmentationDictionary.items():
        for subFragNum, subFragInfo in subFragDict.items():
            l.append(subFragInfo['subgeometry'])
            cols.append(fragKey + '_' + subFragNum)
            condensedFrags.append(subFragInfo['subgeometry'])
            fragSubgeometryKeys.append(fragKey + '_' + subFragNum)
    
    expandedFrags = [fas.expandFrag(x, numberAtSite) for x in condensedFrags]
    
    df = pd.DataFrame(l, columns = IDList)
    df = df.transpose()
    df.columns = cols
    
    if printHeavy:
        OConc = op.deltaToConcentration('O',deltas[2])
        del18 = op.ratioToDelta('18O',OConc[2]/OConc[0])

        print("Delta 18O")
        print(del18)
    
    return df, expandedFrags, fragSubgeometryKeys, fragmentationDictionary

def simulateMeasurement(df, fragmentationDictionary, expandedFrags, fragSubgeometryKeys, abundanceThreshold = 0, UValueList = [],
                        massThreshold = 1, clumpD = {}, outputPath = None, disableProgress = False, calcFF = False, fractionationFactors = {}, omitMeasurements = {}, ffstd = 0.05, unresolvedDict = {}, outputFull = False):
    '''
    clumpD currently only works for mass 1 substitutions
    '''
    
    byAtom = ci.inputToAtomDict(df, disable = disableProgress)
    
    #Introduce any clumps of interest with clumps
    if clumpD == {}:
        bySub = ci.calcSubDictionary(byAtom, df, atomInput = True)
    else:
        print("Adding clumps")
        stochD = copy.deepcopy(byCondensed)
        
        for clumpNumber, clumpInfo in clumpD.items():
            byAtom = ci.introduceClump(byAtom, clumpInfo['Sites'], clumpInfo['Amount'], df)
            
        for clumpNumber, clumpInfo in clumpD.items():
            ci.checkClumpDelta(clumpInfo['Sites'], df, byAtom, stochD)
            
        bySub = ci.calcSubDictionary(byAtom, df, byAtom = True)
    
    #Initialize Measurement output
    print("Simulating Measurement")
    allMeasurementInfo = {}
    allMeasurementInfo = fas.UValueMeasurement(bySub, allMeasurementInfo, massThreshold = massThreshold,
                                              subList = UValueList)

    MN = ci.massSelections(byAtom, massThreshold = massThreshold)
    MN = fas.trackMNFragments(MN, expandedFrags, fragSubgeometryKeys, df, unresolvedDict = unresolvedDict)
        
    predictedMeasurement, fractionationFactors = fas.predictMNFragmentExpt(allMeasurementInfo, MN, expandedFrags, 
                                                                           fragSubgeometryKeys, df, 
                                                 fragmentationDictionary, calcFF = calcFF, ffstd = ffstd,
                                                 abundanceThreshold = abundanceThreshold, fractionationFactors = fractionationFactors, omitMeasurements = omitMeasurements, unresolvedDict = unresolvedDict, outputFull = outputFull)
    
    if outputPath != None:
        output = json.dumps(predictedMeasurement)

        f = open(outputPath + ".json","w")
        f.write(output)
        f.close()
        
    return predictedMeasurement, MN, fractionationFactors

def updateAbundanceCorrection(latestDeltas, fragSubset, fragmentationDictionary, expandedFrags, fragSubgeometryKeys, processStandard, processSample, isotopologuesDict, UValuesSmp, df,
                              deviation = 6, NUpdates = 30, breakCondition = 1, perturbTheoryPAAmt = 0.002,
                              experimentalPACorrectList = [],
                              extremeVals = {},
                              abundanceThreshold = 0, 
                              massThreshold = 1, 
                              omitMeasurements = {}, 
                              unresolvedDict = {},
                              UMNSub = ['13C'],
                              N = 100,
                             setSpreadByExtreme = False,
                             oACorrectBounds = False):
    
    thispADict = {'residual':[],
                  'delta':[],
                  'pA':[],
                  'relDelta':[],
                  'relDeltaErr':[],
                  'Histogram':[]}
    
    for i in range(NUpdates):
        oldDeltas = latestDeltas
        M1Df, expandedFrags, fragSubgeometryKeys, fragmentationDictionary = initializeAlanine(latestDeltas, fragSubset,
                                                                                                  printHeavy = False)

        predictedMeasurementUpdate, MNDictUpdate, FFUpdate = simulateMeasurement(M1Df, fragmentationDictionary, 
                                                           expandedFrags,
                                                           fragSubgeometryKeys,
                                                           abundanceThreshold = abundanceThreshold,
                                                           massThreshold = massThreshold,
                                                           calcFF = False, 
                                                           outputPath = None,
                                                           disableProgress = True,
                                                           fractionationFactors = {},
                                                           omitMeasurements = omitMeasurements,
                                                           unresolvedDict = unresolvedDict)


        pACorrectionUpdate = ss.percentAbundanceCorrectTheoretical(predictedMeasurementUpdate, processSample, 
                                                         massThreshold = massThreshold)

        #Use results to inform later OA corrections        
        explicitOACorrect = {}

        for MNKey, MNData in pACorrectionUpdate.items():
            if MNKey not in explicitOACorrect:
                explicitOACorrect[MNKey] = {}
            for fragKey, fragData in MNData.items():
                if fragKey not in explicitOACorrect[MNKey]:
                    explicitOACorrect[MNKey][fragKey] = {}
                    
                if fragKey in extremeVals:
                    #use to set bounds
                    explicitOACorrect[MNKey][fragKey]['Bounds'] = extremeVals[fragKey]
                    
                    #use bounds to pick standard deviation intelligently
                    vals = (fragData, extremeVals[fragKey][0], extremeVals[fragKey][1])
                    spread = max(vals) - min(vals) 
                    onesigma = spread / deviation

                    explicitOACorrect[MNKey][fragKey]['Mu,Sigma'] = (fragData, onesigma)
                    
                else:
                    explicitOACorrect[MNKey][fragKey]['Mu,Sigma'] = (fragData, fragData * perturbTheoryPAAmt)
                
                

        M1Results = ss.M1MonteCarlo(processStandard, processSample, pACorrectionUpdate, isotopologuesDict,
                                    fragmentationDictionary, perturbTheoryPAAmt = perturbTheoryPAAmt,
                                    experimentalPACorrectList = experimentalPACorrectList,
                                    N = N, GJ = False, debugMatrix = False, disableProgress = True,
                                   storePerturbedSamples = False, storepACorrect = True, 
                                   explicitOACorrect = explicitOACorrect, perturbOverrideList = ['M1'])
        
        processedResults = ss.processM1MCResults(M1Results, UValuesSmp, isotopologuesDict, df, disableProgress = True,
                                        UMNSub = UMNSub)
    
        ss.updateSiteSpecificDfM1MC(processedResults, df)
        
        M1Df = df.copy()
        M1Df['deltas'] = M1Df['PDB etc. Deltas']
        
        thispADict['pA'].append(copy.deepcopy(pACorrectionUpdate['M1']))

        thispADict['delta'].append(list(M1Df['deltas']))
        
        residual = ((np.array(M1Df['deltas']) - np.array(oldDeltas))**2).sum()
        thispADict['residual'].append(residual)
        latestDeltas = M1Df['deltas'].values
        
        thispADict['relDelta'].append(M1Df['Relative Deltas'].values)
        thispADict['relDeltaErr'].append(M1Df['Relative Deltas Error'].values)
        print(residual)
        
        if i % 10 == 0 or residual <= breakCondition:
            correctVals = {'full':[],
                       '44':[]}
        
            for res in M1Results['Extra Info']['pA Correct']:
                correctVals['full'].append(res['full'])
                correctVals['44'].append(res['44'])

            thispADict['Histogram'].append(copy.deepcopy(correctVals))
        
        if residual <= breakCondition:
            break
            
    return M1Results, thispADict