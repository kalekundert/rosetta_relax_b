Rosetta Relax B
===============

A pipeline that relaxes protein crystal structures in the Rosetta force field, 
using the B-factors in the structure to determine how much movement to allow.

Installation
------------
- Prerequisites:

  - nextflow
  - conda

- Need to enable the Gray Lab conda channel:

  - Get license
  - https://{user}:{pass}@conda.graylab.jhu.edu

Usage
-----
- Not necessary to manually download this repository, just 

- Expect that it might take a long time to create the conda env.  Installing 
  PyRosetta requires a >1 GB download.

Gotchas
-------
- Be careful about using this pipeline to relax structures with different 
  resolutions, and then making comparisons between them.  Lower resolution 
  structures have higher B-factors, which will allow more aggressive 
  relaxation, ultimately biasing scores *towards* low-resolutions structures.
