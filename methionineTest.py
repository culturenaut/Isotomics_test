import copy
import json

import numpy as np
import pandas as pd

import basicDeltaOperations as op
import calcIsotopologues as ci
import fragmentAndSimulate as fas
import solveSystem as ss

def initializeMethionine(deltas, fragSubset = ['full','133','104','102','88','74High','74Low','61','56'], printHeavy = True):
    ##### INITIALIZE SITES #####
    IDList = ['Cmethyl','Cgamma','Calphabeta','Ccarboxyl','Ocarboxyl','Ssulfur','Namine','Hmethyl','Hgamma',
             'Halphabeta','Hamine','Hhydroxyl','Hprotonated']
    elIDs = ['C','C','C','C','O','S','N','H','H','H','H','H','H']
    numberAtSite = [1,1,2,1,2,1,1,3,2,3,2,1,1]

    l = [elIDs, numberAtSite, deltas]
    cols = ['IDS','Number','deltas']
    condensedFrags =[]
    fragKeys = []
    
    #88 and both 74 are conjecture. 74 High has only one oxygen, so we generally do not use it. 
    allFragments = {'full':{'01':{'subgeometry':[1,1,1,1,1,1,1,1,1,1,1,1,1],'relCont':1}},
              '133':{'01':{'subgeometry':[1,1,1,1,1,1,'x',1,1,1,'x',1,'x'],'relCont':1}},
              '104':{'01':{'subgeometry':[1,1,1,'x','x',1,1,1,1,1,1,'x','x'],'relCont':1}},
              '102':{'01':{'subgeometry':['x',1,1,1,1,'x',1,'x',1,1,1,1,'x'],'relCont':1}},
              '88':{'01':{'subgeometry':[1,1,1,'x','x',1,'x',1,1,'x',1,'x','x'],'relCont':1}},
              '74High':{'01':{'subgeometry':[1,'x',1,'x',1,'x',1,1,1,1,'x','x','x'],'relCont':1}},
              '74Low':{'01':{'subgeometry':[1,1,'x','x',1,'x',1,'x',1,'x',1,'x','x'],'relCont':1}},
              '61':{'01':{'subgeometry':[1,1,'x','x','x',1,'x',1,1,'x','x','x','x'],'relCont':1}},
              '56':{'01':{'subgeometry':['x',1,1,'x','x','x',1,'x',1,1,'x',1,'x'],'relCont':1}}}

    fragmentationDictionary = {key: value for key, value in allFragments.items() if key in fragSubset}
    
    for fragKey, subFragDict in fragmentationDictionary.items():
        for subFragNum, subFragInfo in subFragDict.items():
            l.append(subFragInfo['subgeometry'])
            cols.append(fragKey + '_' + subFragNum)
            condensedFrags.append(subFragInfo['subgeometry'])
            fragKeys.append(fragKey + '_' + subFragNum)

    df = pd.DataFrame(l, columns = IDList)
    df = df.transpose()
    df.columns = cols

    expandedFrags = [fas.expandFrag(x, numberAtSite) for x in condensedFrags]

    if printHeavy:
        SConc = op.deltaToConcentration('S',deltas[5])
        del34 = op.ratioToDelta('34S',SConc[2]/SConc[0])
        del36 = op.ratioToDelta('36S',SConc[3]/SConc[0])

        OConc = op.deltaToConcentration('O',deltas[4])
        del18 = op.ratioToDelta('18O',OConc[2]/OConc[0])
        print("Delta 34S")
        print(del34)
        print("Delta 36S")
        print(del36)

        print("Delta 18O")
        print(del18)
    
    return df, fragmentationDictionary, expandedFrags, fragKeys, 

def simulateMeasurement(df, fragmentationDictionary, expandedFrags, fragKeys, abundanceThreshold = 0, UValueList = [],
                        massThreshold = 4, clumpD = {}, outputPath = None, disableProgress = False, calcFF = False, fractionationFactors = {}, omitMeasurements = {}, ffstd = 0.05, unresolvedDict = {}, outputFull = False):
    
    M1Only = False
    if massThreshold == 1:
        M1Only = True
        
    byAtom = ci.inputToAtomDict(df, disable = disableProgress, M1Only = M1Only)
    
    #Introduce any clumps of interest with clumps
    if clumpD == {}:
        bySub = ci.calcSubDictionary(byAtom, df, atomInput = True)
    else:
        print("Adding clumps")
        stochD = copy.deepcopy(byAtom)
        
        for clumpNumber, clumpInfo in clumpD.items():
            byAtom = ci.introduceClump(byAtom, clumpInfo['Sites'], clumpInfo['Amount'], df)
            
        for clumpNumber, clumpInfo in clumpD.items():
            ci.checkClumpDelta(clumpInfo['Sites'], df, byAtom, stochD)
            
        bySub = ci.calcSubDictionary(byAtom, df, atomInput = True)
    
    #Initialize Measurement output
    if disableProgress == False:
        print("Simulating Measurement")
    allMeasurementInfo = {}
    allMeasurementInfo = fas.UValueMeasurement(bySub, allMeasurementInfo, massThreshold = massThreshold,
                                              subList = UValueList)

    MN = ci.massSelections(byAtom, massThreshold = massThreshold)
    MN = fas.trackMNFragments(MN, expandedFrags, fragKeys, df, unresolvedDict = unresolvedDict)
        
    predictedMeasurement, FF = fas.predictMNFragmentExpt(allMeasurementInfo, MN, expandedFrags, fragKeys, df, 
                                                 fragmentationDictionary,
                                                 abundanceThreshold = abundanceThreshold, calcFF = calcFF, ffstd = ffstd, fractionationFactors = fractionationFactors, omitMeasurements = omitMeasurements, unresolvedDict = unresolvedDict, outputFull = outputFull)
    
    if outputPath != None:
        output = json.dumps(predictedMeasurement)

        f = open(outputPath + ".json","w")
        f.write(output)
        f.close()
        
    return predictedMeasurement, MN, FF

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
        M1Df, expandedFrags, fragSubgeometryKeys, fragmentationDictionary = initializeMethionine(latestDeltas, fragSubset,
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
            correctVals = {'61':[],
                       '133':[],
                       'full':[]}
        
            for res in M1Results['Extra Info']['pA Correct']:
                correctVals['full'].append(res['full'])
                correctVals['133'].append(res['133'])
                correctVals['61'].append(res['61'])

            thispADict['Histogram'].append(copy.deepcopy(correctVals))
        
        if residual <= breakCondition:
            break
            
    return M1Results, thispADict