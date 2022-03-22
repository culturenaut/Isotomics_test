import copy
import json

import numpy as np
import pandas as pd

import basicDeltaOperations as op
import calcIsotopologues as ci
import fragmentAndSimulate as fas

def initializeAlanine(deltas, fragSubset = ['full','44'], printHeavy = True):
    '''
    This is a new comment.
    '''
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
                        molecularAvgMassThreshold = 4, MNMassThreshold = 4, clumpD = {}, outputPath = None, disableProgress = False, calcFF = False, fractionationFactors = {}, omitMeasurements = {}, ffstd = 0.05, unresolvedDict = {}, outputFull = False):
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
    allMeasurementInfo = fas.UValueMeasurement(bySub, allMeasurementInfo, massThreshold = molecularAvgMassThreshold,
                                              subList = UValueList)

    MN = ci.massSelections(byAtom, massThreshold = MNMassThreshold)
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