# Isotomics

This repository contains an implementation of the isotopologue calculations and M+N algorithm presented in High-Dimensional Isotomics, Part 1: A Mathematical Framework for Isotomics. It contains .py files which house the functions used, a single .py script (runAllTests.py) which can be run to reproduce the major results of the paper, a .ipynb tutorial applied to alanine, a few miscellaneous additional tests of the algorithm from the Appendix (as .py files), and .json files including the synthetic data generated and used in the paper. We recommend that users interested in implementing their own M+N scripts begin with the tutorial, then modify a version of runAllTests.py that is appropriate for their application. 

# Python Files

The .py files, which contain all the functions needed to run the algorithm. The core files are:

basicDeltaOperations.py -- Contains basic manipulations between concentration, ratio, and delta space

calcIsotopologues.py -- Calculates all isotopologues of a molecule and their concentrations, for some input deltas

fragmentAndSimulate.py -- Fragments the isotopologues of a molecule, simulates measurements (which is used both for standardization and for computed datasets)

readInput.py -- Reads in computed or experimental datasets, processes them to prepare for algorithm

solveSystem.py -- The M+N algorithm itself; split into different versions for M+1 and general M+N. 

### Additional Python Files ###

There are some additional files which may be helpful:

spectrumVis.py -- Allows one to predict the spectra resulting from different M+N measurements; useful in assigning peaks to newly observed data

alanineTest/methionineTest.py -- Define wrapper functions which take care of most of the details in defining and simulating alanine and methionine datasets.
