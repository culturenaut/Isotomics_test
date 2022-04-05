import sys
import os

#Relative import .py files from the parent directory
sys.path.insert(0,os.getcwd())

#You can also try, but in some softwares it was not working for me
#sys.path.insert(0, '..')

import methionineTest as metTest
import calcIsotopologues as ci
import fragmentAndSimulate as fas
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

from datetime import date
import copy
import json

today = date.today()

'''
Calculates unresolved peaks at a target resolution; outputs these as a .json.
'''

#Basic information about the molecule. 
deltas = [-45,-35,-30,-25,-13,2.5,10,-250,-100,0,100,250,0]
fragSubset = ['full','133','104','102','61','56']
df, expandedFrags, fragSubgeometryKeys, fragmentationDictionary = metTest.initializeMethionine(deltas, fragSubset)

#Calculating and mass selecting isotopologues. 
byAtom = ci.inputToAtomDict(df)
MN = ci.massSelections(byAtom, massThreshold = 4)
MN = fas.trackMNFragments(MN, expandedFrags, fragSubgeometryKeys, df, unresolvedDict = {})

unresolvedDict = {}
MNKeys = ['M1','M2','M3','M4']
#Goes through by mass selection
for MNKey in MNKeys:
    unresolvedMS = {}
    selectedIsotopologues = MN[MNKey]
    fragKeys = ['full_01','133_01','104_01','102_01','61_01','56_01']
    siteElements = ci.strSiteElements(df)
    #Input Resolution
    MS_RES = 120000

    #Goes by fragment, computing the mass of each isotopologues.
    for fragIdx, frag in enumerate(fragKeys):
        subByMass = {}
        for key, item in selectedIsotopologues.items():
            mass = fas.computeMass(item[frag + ' Identity'], siteElements)
            subs = item[frag + ' Subs']

            if subs not in subByMass:
                subByMass[subs] = {'Mass':mass,'Conc':0}

            subByMass[subs]['Conc'] += item['Conc']

        #Threshold mass difference for peaks to be resolved; if they are beyond this, they are.
        # 
        if fragIdx == 0:
            #"Full" fragment is at mass 150. For other fragments, pull their mass from the fragKey. 
            threshold = 1.5 * 150 / MS_RES
        else:
            threshold = 1.5 * float(frag.split('_')[0]) / MS_RES

            
        tupleList = []
        #Sort information into list.
        for subKey, subInfo in subByMass.items():
            tupleList.append((subInfo['Conc'],subInfo['Mass'],subKey))
        tupleList.sort(reverse = True)
        
        #For each pair of isotopologues, see if their masses are sufficiently different (beyond threshold).
        #If they are not, add them to a dictionary. 
        outputDict = {}
        for i, t in enumerate(tupleList):
            conc, mass, sub = t[0],t[1],t[2]
            for j, t2 in enumerate(tupleList):
                if j > i:
                    conc2, mass2, sub2 = t2[0],t2[1],t2[2]
                    if np.abs(mass2 - mass) <= threshold:
                        if sub2 not in outputDict:
                            outputDict[sub2] = {'Sub':[],'RelConc':[]}
                        outputDict[sub2]['Sub'].append(sub)
                        outputDict[sub2]['RelConc'].append(conc)
                        
        #Process the unresolved peaks into a dictionary. The key is the lower abundance substitution, while the value is the higher abundance one. 
        processedOutput = {}
        for lostSub, subInfo in outputDict.items():
            if len(subInfo['Sub']) == 1:
                processedOutput[lostSub] = subInfo['Sub'][0]
            else:
                maxConcIdx = subInfo['RelConc'].index(max(subInfo['RelConc']))
                processedOutput[lostSub] = subInfo['Sub'][maxConcIdx]

        shortKey = frag.split('_')[0]
        unresolvedMS[shortKey] = processedOutput
    unresolvedDict[MNKey] = copy.deepcopy(unresolvedMS)
    
#Output as a .json.
with open(str(today) + ' Unresolved_120k.json', 'w', encoding='utf-8') as f:
    json.dump(unresolvedDict, f, ensure_ascii=False, indent=4)