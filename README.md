# macroscopic-roughness-correction
Public Release v1.0.4

This repository contains Python code and laboratory data used in the publication "An Alternative to Hapke's
Macroscopic Roughness Correction" by Dylan J. Shiltz and Charles M. Bachmann, which is accepted for publication by
the journal Icarus.

A Python implementation for the following models is included:
* Hapke's original macroscopic roughness correction, published in (Hapke, B., 1984, "Bidirectional Spectroscopy: 3. Correction for Macroscopic Roughness", Icarus, 59, 41-59)
* Hapke's modification for multi-facet scattering, published in Ch. 12 of (Hapke, B., 2012, "Theory of Reflectance and Emittance Spectroscopy", Cambridge University Press)
* The roughness correction proposed by Shiltz and Bachmann
* The single-facet Monte Carlo model used to validate the single-facet portion of Shiltz and Bachmann's model

The code for these models is included in the ``roughness_models`` directory.  Raw and processed data is included in the
``data`` directory.  The code used to run the models on the raw data and produce the processed data is included
in the ``processing`` directory.  The figures showing the model results, as well as the code
used to produce the figures, is included in the ``results`` directory.

This code was developed using Python version 3.9, with the required packages identified in ``requirements.txt``.
