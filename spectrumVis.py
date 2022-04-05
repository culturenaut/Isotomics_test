import calcIsotopologues as ci
import fragmentAndSimulate as fas
from tqdm import tqdm
import matplotlib.pyplot as plt

import pandas as pd

'''
A set of functions for visualizing predicted spectra. These are useful when correlating observations with experimental data. 
'''

def fullSpectrumVis(molecularDataFrame, byAtom, figsize = (10,4), lowAbundanceCutOff = 0, massError = 0, 
xlim =(), ylim = ()):
    '''
    Visualizes the full spectrum (i.e. without fragmentation) based on the abundances of all isotopologues.

    Inputs:
        molecularDataFrame: A dataframe containing basic information about the molecule. 
        byAtom: A dictionary containing the isotopolouges of the molecule and their abundances. See calcIsotopologues for details. 
        figsize: The output figure size. 
        lowAbundanceCutOff: Do not show peaks below this relative abundance.
        massError: In amu, shifts all observed peaks by this amount. 
        xlim: Set an xlim for the plot, as (xlow, xhigh)
        ylim: as xlim. 

    Outputs:
        None. Displays plot. 
    '''
    selectedIsotopologues = byAtom
    lowAbundanceCutOff = 0.00
    massError = 0.000

    siteElements = ci.strSiteElements(molecularDataFrame)
    predictSpectrum = {}
    
    #calculates the mass of each isotopologue as well as its substitution. Adds its absolute concentration to the predicted 
    #spectrum
    for key, item in tqdm(selectedIsotopologues.items()):
        mass = fas.computeMass(key, siteElements)
        correctedMass = mass + massError
        subs = fas.computeSubs(key, siteElements)

        if correctedMass not in predictSpectrum:
            predictSpectrum[correctedMass] = {'Abs. Abundance':0}

            if 'Sub' not in predictSpectrum[correctedMass]:
                predictSpectrum[correctedMass]['Sub'] = subs

        predictSpectrum[correctedMass]['Abs. Abundance'] += item['Conc']
    
    #Calculates the relative abundances, and places these, the masses, and substitutions into lists to plot. 
    totalAbundance = 0
    for key, item in predictSpectrum.items():
        totalAbundance += item['Abs. Abundance']

    massPlot = []
    relAbundPlot = []
    subPlot = []
    for key, item in predictSpectrum.items():
        item['Rel. Abundance'] = item['Abs. Abundance'] / totalAbundance
        massPlot.append(key)
        relAbundPlot.append(item['Rel. Abundance'])
        subPlot.append(item['Sub'])
    
    #Constructs a figure; does not plot peaks below the relative abundance cut off. 
    fig, ax = plt.subplots(figsize = figsize)
    massPlotcutOff = []
    subPlotcutOff = []
    for i in range(len(massPlot)):
        if relAbundPlot[i] > lowAbundanceCutOff:
            ax.vlines(massPlot[i], 0, relAbundPlot[i])
            massPlotcutOff.append(massPlot[i])
            subPlotcutOff.append(subPlot[i])
    ax.set_xticks(massPlotcutOff)
    labels = [str(round(x,5)) +'\n' + y for x,y in zip(massPlotcutOff,subPlotcutOff)]
    ax.set_xticklabels(labels,rotation = 45);
    if xlim != ():
        ax.set_xlim(xlim[0],xlim[1]);
    if ylim != ():
        ax.set_ylim(ylim[0],ylim[1]);
    ax.set_ylabel("Relative Abundance")
        
    plt.show()
    
def MNSpectrumVis(molecularDataFrame, fragKey, predictedMeasurement, MNKey, MNDict, lowAbundanceCutOff = 0, 
massError = 0, xlim = (), ylim = ()):
    '''
    Visualizes the fragmented spectrum of an M+N experiment based on the abundances of fragment peaks.

    Inputs:
        molecularDataFrame: A dataframe containing basic information about the molecule. 
        fragKey: A string identifying the fragment, e.g. '133', '44'.
        predictedMeasurement: A dictionary giving information about the abundance of isotopic peaks in the fragment. See fragmentAndSimulate.predictMNFragmentExpt
        MNKey: A string identifying the mass selection to visualize; e.g. 'M1', 'M2'. This population of isotopologues is selected prior to fragmentation.
        MNDict: A dictionary; the keys are MNKeys ("M1", "M2") and the values are dictionaries containing the isotopologues present in each mass selection. See calcIsotopologues.massSelections
        lowAbundanceCutOff: Do not show peaks below this relative abundance.
        massError: In amu, shifts all observed peaks by this amount. 
        xlim: Set an xlim for the plot, as (xlow, xhigh)
        ylim: as xlim. 

    Outputs:
        None. Displays plot. 
    '''
    toShow = predictedMeasurement[MNKey][fragKey]
    siteElements = ci.strSiteElements(molecularDataFrame)
    massPlot = []
    relAbundPlot = []
    subPlot = []
    for subKey, observation in toShow.items():
        #This section (until correctedMass) is an elaborate routine to get the mass of the isotopologues with that
        #substitution. It may break in weird circumstances. We should try to improve this. 
        Isotopologues = pd.DataFrame.from_dict(MNDict[MNKey])
        Isotopologues = Isotopologues.T
        IsotopologuesWithSub = Isotopologues[Isotopologues[fragKey + '_01 Subs'] == subKey]

        IsotopologuesStr = IsotopologuesWithSub[fragKey + '_01 Identity'][0]

        mass = fas.computeMass(IsotopologuesStr, siteElements)
        correctedMass = mass + massError

        massPlot.append(correctedMass)
        relAbundPlot.append(observation['Rel. Abundance'])
        subPlot.append(subKey)

    fig, ax = plt.subplots(figsize = (10,4))
    massPlotcutOff = []
    subPlotcutOff = []
    for i in range(len(massPlot)):
        if relAbundPlot[i] > lowAbundanceCutOff:
            ax.vlines(massPlot[i], 0, relAbundPlot[i])
            #ax.vlines(massPlot[i]+ 0.001, 0, relAbundPlotNoFF[i], color = 'b', linestyle = '--')
            massPlotcutOff.append(massPlot[i])
            subPlotcutOff.append(subPlot[i])
    ax.set_xticks(massPlotcutOff)
    labels = [str(round(x,5)) +'\n' + y for x,y in zip(massPlotcutOff,subPlotcutOff)]
    ax.set_xticklabels(labels,rotation = 45);
    if xlim != ():
        ax.set_xlim(xlim[0],xlim[1]);
    if ylim != ():
        ax.set_ylim(ylim[0],ylim[1]);
    ax.set_ylabel("Relative Abundance")
    
    plt.show()
