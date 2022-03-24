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
'''
This is our central .py file for running all of our simulations. It allows the user to specify different sets of parameters for each simulation, then runs all of these.
'''
#See 'conditions' dictionary, below. This list specifies a subset of our tests that we will actually run this time. 
toRun = ['PerfectMeasurement']
#toRun = ['PerfectMeasurement','ExpError','InstFrac','LowAbund','Unresolved','AllIssue']

#Use this for experimentalOCorrection; you can define a list experimentalOCorrectList containing any of these dictionaries, if you wish to use the experimental correction.
sixtyOneCorrect = {'MNKey':'M1','fragToCorrect':'61','subToCorrect':'33S','fragsToBenchmarkFrom':['104']}
oneThreeThreeCorrect = {'MNKey':'M1','fragToCorrect':'133','subToCorrect':'33S','fragsToBenchmarkFrom':['104']}
fullCorrect = {'MNKey':'M1','fragToCorrect':'full','subToCorrect':'33S','fragsToBenchmarkFrom':['104']}

#Use this for omitted Peaks. This is necessary because some ion beams are <1% for standard but not sample, or vice versa. 
#We determined which beams that applied to, then filled in this dictionary, so that they are omitted in both cases. 
forbiddenPeaks = {'M1':{'61':['D']},
                  'M2':{'104':['13C']},
                  'M3':{'full':['18O-33S'],'61':['34S-D']},
                  'M4':{'133':['13C-34S-D']}}

#Set composition for standard
knownStd = [-30,-30,-30,-30,0,0,0,0,0,0,0,0,0]
#Set hypothesized composition for unknown standard. If we specify that we do not know our standard, we use this to generate our forward model, while the actual data comes from "knownStd"
unknownStd = [-48,-18,-38,-8,25,0,0,-100,100,100,-100,50,-50]

#Calculate the unresolved peaks separately and load them here. 
with open('Unresolved_120k.json') as f:
    unresolvedDict = json.load(f)
              
'''
This is a dictionary of simulations we can run. Each gets a name, e.g. PerfectMeasurement, and then a numnber of parameters to specify the run. 
These are:
abundanceThreshold: Do not observe peaks with M+N relative abundances below this threshold.
massThreshold: Simulate M+N datasets through N = massThreshold
saveInput: If true, saves .json files for sample and standard observations. 
saveOutput: Saves a .csv giving the output of the simulation.
calcFF: If true, then instrument fractionation occurs. Otherwise, we assume no fractionation. 
forwardModelDeltas: Specify the delta values for the standard used to calculate a forward model. If this is knownStd, we know the standard composition perfectly. If it is unknownStd, we are just guessing. 
forbiddenPeaks: Specify explicit peaks which we do not observe. This is useful for individual peaks that would be observed in one of sample/standard but not the other (e.g. because it is above the abundance threshold in only one); in this case, we add that peak to this dictionary to make sure we have the same set of peaks across both.
unresolvedDict: A dictionary specifying which peaks are unresolved, e.g. combine with each other. Currently, we suppose they combine linearly. 
experimentalOCorrectList: If we wish to use an experimental O Correction (see paper), specify here.
MNError: We assume the same relative error on all observations of M+N relative abundances, given here. 
Molecular Average U Error: We assume the same relative error on all molecular average U values, given here. 
N: The number of iterations for the monte carlo procedure to propagate error. N = 1000 is a reasonable choice. 
perturbTheoryOAmtM1: When using the Monte Carlo routine, we may perturb the O Correction factors each iteration. This specifies the relative error we use for this perturbation (i.e. we draw from N(O_factor,perturbTheoryOAmt*O_factor))
perturbTheoryOAmtMN: Allow a different O factor perturbation for M+2 through M+4 (or higher). 
explicitOCorrect: Currently not used; allows us to specify explicit bounds or other information for the O correction procedure.
NUpdates: If we run an iterated O Correction procedure, this specifies the number of iterations used. NUpdates = 30-50 has worked effectively for methionine. 
abundanceCorrect: A boolean, specifying whether to perform an O correction at all. 
'''
conditions = {'PerfectMeasurement':{'abundanceThreshold':0,'massThreshold':4, 'saveInput' : True, 'saveOutput' : True,
                        'calcFF':False,'forwardModelDeltas':knownStd,
                        'forbiddenPeaks':{},'unresolvedDict':{},
                        'experimentalOCorrectList':[], 
                        'MN Error':0,'Molecular Average U Error':0,'N':1,'perturbTheoryOAmtM1':0,
                        'perturbTheoryOAmtMN':0, 'explicitOCorrect':{},
                        'NUpdates':0,'abundanceCorrect':False},
              
              'ExpError':{'abundanceThreshold':0,'massThreshold':4, 'saveInput' : True, 'saveOutput' : True,
                        'calcFF':False,'forwardModelDeltas':knownStd,
                        'forbiddenPeaks':{},'unresolvedDict':{},
                        'experimentalOCorrectList':[], 
                        'MN Error':0.001,'Molecular Average U Error':0.0001,'N':1000,'perturbTheoryOAmtM1':0,
                        'perturbTheoryOAmtMN':0, 'explicitOCorrect':{},
                        'NUpdates':0,'abundanceCorrect':False},
              
              'InstFrac':{'abundanceThreshold':0,'massThreshold':4, 'saveInput' : True, 'saveOutput' : True,
                        'calcFF':True,'forwardModelDeltas':unknownStd,
                        'forbiddenPeaks':{},'unresolvedDict':{},
                        'experimentalOCorrectList':[], 
                        'MN Error':0.001,'Molecular Average U Error':0.0001,'N':1000,'perturbTheoryOAmtM1':0,
                        'perturbTheoryOAmtMN':0, 'explicitOCorrect':{},
                        'NUpdates':0,'abundanceCorrect':False},
              
              'LowAbund':{'abundanceThreshold':0.01,'massThreshold':4, 'saveInput' : True, 'saveOutput' : True,
                        'calcFF':False,'forwardModelDeltas':knownStd,
                        'forbiddenPeaks':forbiddenPeaks,'unresolvedDict':{},
                        'experimentalOCorrectList':[], 
                        'MN Error':0.001,'Molecular Average U Error':0.0001,'N':1000,'perturbTheoryOAmtM1':0,
                        'perturbTheoryOAmtMN':0.001, 'explicitOCorrect':{},
                        'NUpdates':50,'abundanceCorrect':True},
              
              'Unresolved':{'abundanceThreshold':0,'massThreshold':4, 'saveInput' : True, 'saveOutput' : True,
                        'calcFF':False,'forwardModelDeltas':knownStd,
                        'forbiddenPeaks':{},'unresolvedDict':unresolvedDict,
                        'experimentalOCorrectList':[], 
                        'MN Error':0.001,'Molecular Average U Error':0.0001,'N':1000,'perturbTheoryOAmtM1':0,
                        'perturbTheoryOAmtMN':0.001, 'explicitOCorrect':{},
                        'NUpdates':50,'abundanceCorrect':True},
              
              'AllIssue':{'abundanceThreshold':0.01,'massThreshold':4, 'saveInput' : True, 'saveOutput' : True,
                        'calcFF':True,'forwardModelDeltas':unknownStd,
                        'forbiddenPeaks':forbiddenPeaks,'unresolvedDict':unresolvedDict,
                        'experimentalOCorrectList':[], 
                        'MN Error':0.001,'Molecular Average U Error':0.0001,'N':1000,'perturbTheoryOAmtM1':0,
                        'perturbTheoryOAmtMN':0.001, 'explicitOCorrect':{},
                        'NUpdates':50,'abundanceCorrect':True}}

#Run only a specified subset of conditions
conditions = {key:value for key, value in conditions.items() if key in toRun}

#Initialize dictionaries to hold output
resultsAbbr = {}
perturbedSamples = {}

#Run each test
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
    
    #Read in simulated sample and standard data, generated above.  
    standardJSON = ri.readJSON(str(today) + ' ' + testKey + ' Standard.json')
    processStandard = ri.readComputedData(standardJSON, error = testData['MN Error'], theory = predictedMeasurementFMStd)

    sampleJSON = ri.readJSON(str(today) + ' ' + testKey + ' Sample.json')
    processSample = ri.readComputedData(sampleJSON, error = testData['MN Error'])
    UValuesSmp = ri.readComputedUValues(sampleJSON, error = testData['Molecular Average U Error'])

    #solve M+1
    print("Solving M+1")
    isotopologuesDict = fas.isotopologueDataFrame(MNDictFMStd, df)
    OCorrection = ss.OValueCorrectTheoretical(predictedMeasurementFMStd, processSample, massThreshold = testData['massThreshold'])

    M1Results = ss.M1MonteCarlo(processStandard, processSample, OCorrection, isotopologuesDict,
                                fragmentationDictionary, perturbTheoryOAmt = testData['perturbTheoryOAmtM1'],
                                experimentalOCorrectList = testData['experimentalOCorrectList'],
                                N = testData['N'], GJ = False, debugMatrix = False, disableProgress = True,
                               storePerturbedSamples = True, storeOCorrect = True, 
                               explicitOCorrect = testData['explicitOCorrect'],
                               abundanceCorrect = testData['abundanceCorrect'])
    
    #Track stored perturbed samples from each iteration. Can disable by setting storePerturbedSamples = False in M1MonteCarlo, above.
    perturbedSamples[testKey] = M1Results['Extra Info']['Perturbed Samples']

    #Process results from M+N relative abundance space to U value space, using the UM+1 value calculated via the sub(s) in UMNSub. 
    processedResults = ss.processM1MCResults(M1Results, UValuesSmp, isotopologuesDict, df, disableProgress = True,
                                            UMNSub = ['13C'])
    
    #Update the dataframe with results. 
    ss.updateSiteSpecificDfM1MC(processedResults, df)
    
    M1Df = df.copy()
    M1Df['deltas'] = M1Df['PDB etc. Deltas']

    oldDeltas = list(M1Df['deltas'])
    
    #Run the iterated correction scheme, if applicable. 
    ODict = {}
    if testData['NUpdates'] > 0:
        #Currently we only implement an iterated correction for M1; pull out only this information.
        iterateStandard = {key: value for (key, value) in processStandard.items() if key == 'M1'}
        iterateSample = {key: value for (key, value) in processSample.items() if key == 'M1'}
        
        M1Results, thisODict  = metTest.updateAbundanceCorrection(oldDeltas, fragSubset, fragmentationDictionary, expandedFrags, fragSubgeometryKeys, iterateStandard, iterateSample, isotopologuesDict, UValuesSmp, df, deviation = 4, 
                              perturbTheoryOAmt = testData['perturbTheoryOAmtM1'], 
                              NUpdates = testData['NUpdates'], 
                              breakCondition = 1e-3,
                              experimentalOCorrectList = testData['experimentalOCorrectList'],
                              abundanceThreshold = testData['abundanceThreshold'], 
                              massThreshold = 1, 
                              omitMeasurements = testData['forbiddenPeaks'], 
                              unresolvedDict = testData['unresolvedDict'],
                              UMNSub = ['13C'],
                              N = testData['N'])
        
        #Store results
        ODict[testKey] = copy.deepcopy(thisODict)
        
    #Take the final result of the iterated correction scheme
    processedResults = ss.processM1MCResults(M1Results, UValuesSmp, isotopologuesDict, df, disableProgress = True,
                                    UMNSub = ['13C'])
    
    #update dataframe with M1 data
    ss.updateSiteSpecificDfM1MC(processedResults, df)
    
    #Recalculate forward model using optimized M+1 data to inform stochastic reference frame for M+2, M+3, M+4 
    M1Df = df.copy()
    M1Df['deltas'] = M1Df['PDB etc. Deltas']
    
    #recalculate to inform stochastic reference frame for MN
    print("Calculating Sample forward model")
    predictionFMSmp, MNDictFMSmp, secondFFFM = metTest.simulateMeasurement(M1Df, fragmentationDictionary, expandedFrags,
                                                                     fragSubgeometryKeys,
                                                       abundanceThreshold = 0,
                                                       massThreshold = testData['massThreshold'], disableProgress = True)

    isotopologuesDict = fas.isotopologueDataFrame(MNDictFMSmp, M1Df)
    OCorrection = ss.OValueCorrectTheoretical(predictionFMSmp, processSample, massThreshold = testData['massThreshold'])
    
    #Initialize dictionary for M+N solutions
    MNSol = {'M2':0,'M3':0,'M4':0}
    #Determine which substitutions are used for U Value scaling
    UMNSubs = {'M2':['34S'],'M3':['34S15N'],'M4':['36S']}

    for MNKey in MNSol.keys():
        print("Solving " + MNKey)

        Isotopologues = isotopologuesDict[MNKey]

        #Solve for specific M+N isotopologues
        results, comp, GJSol, meas = ss.MonteCarloMN(MNKey, Isotopologues, processStandard, processSample, 
                                            OCorrection, fragmentationDictionary, N = testData['N'], disableProgress = True, perturbTheoryOAmt = testData['perturbTheoryOAmtMN'])


        dfOutput = ss.checkSolutionIsotopologues(GJSol, Isotopologues, MNKey, numerical = False)
        #Many will be codependent, not individually constarined. Determine which isotopologues have actually been solved for.
        nullSpaceCycles = ss.findNullSpaceCycles(comp, Isotopologues)
        actuallyConstrained = ss.findFullyConstrained(nullSpaceCycles)
        
        #Scale results to U values, update the dataframe.
        processedResults = ss.processMNMonteCarloResults(MNKey, results, UValuesSmp, dfOutput, df, MNDictFMStd,
                                                        UMNSub = UMNSubs[MNKey], disableProgress = True)
        dfOutput = ss.updateMNMonteCarloResults(dfOutput, processedResults)
        MNSol[MNKey] = dfOutput.loc[dfOutput.index.isin(actuallyConstrained)].copy()
        
    #Remove .json files for the simulated datasets
    if testData['saveInput'] == False: 
        if os.path.exists(str(today) + ' ' + testKey + ' Sample.json'):
            os.remove(str(today) + ' ' + testKey + ' Sample.json')
        if os.path.exists(str(today) + ' ' + testKey + ' Standard.json'):
            os.remove(str(today) + ' ' + testKey + ' Standard.json')
            
    #Output solution as csv
    if testData['saveOutput']:
        f = str(today) + '_' + testKey + '_' + 'Results.csv'
        M1Df.to_csv(f, mode = 'a',header=True)
        MNSol["M2"].to_csv(f, mode = 'a', header = True)
        MNSol["M3"].to_csv(f, mode = 'a', header = True)
        MNSol["M4"].to_csv(f, mode = 'a', header = True)
        
    #Generate an abbreviated results file for easy use.
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
    
#Output the abbreviated results file and the tracked perturbed samples (from the M+1 iterated solution).
with open(str(today) + 'Results_Abbr.json', 'w', encoding='utf-8') as f:
    json.dump(resultsAbbr, f, ensure_ascii=False, indent=4)
    
with open(str(today) + 'Perturbed_Samples.json', 'w', encoding='utf-8') as f:
    json.dump(perturbedSamples, f, ensure_ascii=False, indent=4)