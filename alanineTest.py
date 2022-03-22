import copy
import json

import numpy as np
import pandas as pd

import basicDeltaOperations as op
import calcIsotopologues as ci
import fragmentAndSimulate as fas
import solveSystem as ss

'''
This is a set of functions to quickly initalize alanine molecules based on input delta values and to simulate its fragmentation.
'''

def initializeAlanine(deltas, fragSubset = ['full','44'], printHeavy = True):
    '''
    Initializes alanine, returning a dataframe with basic information about the molecule as well as information about fragmentation.

    Inputs:
        deltas: A list of 6 M1 delta values, giving the delta values by site for the 13C, 17O, 15N, and 2H isotopes. The sites are defined in the IDList variable, below.
        fragSubset: A list giving the subset of fragments to observe. If you are not observing all fragments, you may input only those you do observe. 
        printHeavy: The user manually specifies delta 17O, and delta 18O is set via mass scaling (see basicDeltaOperations). If True, this will print out delta 18O.

    Outputs:
        df: A dataframe containing basic information about the molecule. 
        expandedFrags: An ATOM depiction of each fragment, where an ATOM depiction has one entry for each atom (rather than for each site). See fragmentAndSimulate for details.
        fragSubgeometryKeys: A list of strings, e.g. 44_01, 44_02, corresponding to each subgeometry of each fragment. A fragment will have multiple subgeometries if there are multiple fragmentation pathways to form it.
        fragmentationDictionary: A dictionary like the allFragments variable, but only including the subset of fragments selected by fragSubset.
    '''
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
    Simulates M+N measurements of an alanine molecule with input deltas specified by the input dataframe df. 

    Inputs:
        df: A dataframe containing basic information about the molecule. 
        expandedFrags: An ATOM depiction of each fragment, where an ATOM depiction has one entry for each atom (rather than for each site). See fragmentAndSimulate for details.
        fragSubgeometryKeys: A list of strings, e.g. 44_01, 44_02, corresponding to each subgeometry of each fragment. A fragment will have multiple subgeometries if there are multiple fragmentation pathways to form it.
        fragmentationDictionary: A dictionary like the allFragments variable, but only including the subset of fragments selected by fragSubset.
        abundanceThreshold: A float; Does not include measurements below this M+N relative abundance, i.e. assuming they will not be  measured due to low abundance. 
        UValueList: A list giving specific substitutions to calculate molecular average U values for ('13C', '15N', etc.)
        massThreshold: An integer; will calculate M+N relative abundances for N <= massThreshold
        clumpD: Specifies information about clumps to add; otherwise the isotome follows the stochastic assumption. Currently works only for mass 1 substitutions (e.g. 1717, 1317, etc.) See ci.introduceClump for details.
        outputPath: A string, e.g. 'output', or None. If it is a string, outputs the simulated spectrum as a json. 
        disableProgress: Disables tqdm progress bars when True.
        calcFF: When True, computes a new set of fractionation factors for this measurement.
        fractionationFactors: A dictionary, specifying a fractionation factor to apply to each ion beam. This is used to apply fractionation factors calculated previously to this predicted measurement (e.g. for a sample/standard comparison with the same experimental fractionation)
        omitMeasurements: omitMeasurements: A dictionary, {}, specifying measurements which I will not observed. For example, omitMeasurements = {'M1':{'61':'D'}} would mean I do not observe the D ion beam of the 61 fragment of the M+1 experiment, regardless of its abundance. 
        ffstd: A float; if new fractionation factors are calculated, they are pulled from a normal distribution centered around 1, with this standard deviation.
        unresolvedDict: A dictionary, specifying which unresolved ion beams add to each other.
        outputFull: A boolean. Typically False, in which case beams that are not observed are culled from the dictionary. If True, includes this information; this should only be used for debugging, and will likely break the solver routine. 
        
    Outputs:
        predictedMeasurement: A dictionary giving information from the M+N measurements. 
        MN: A dictionary where keys are mass selections ("M1", "M2") and values are dictionaries giving information about the isotopologues of each mass selection.
        fractionationFactors: The calculated fractionation factors for this measurement (empty unless calcFF == True)

    '''
    
    byAtom = ci.inputToAtomDict(df, disable = disableProgress)
    
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