Rosetta Relax B
===============

A pipeline that relaxes protein crystal structures in the Rosetta force field, 
using the B-factors in the structure to determine how much movement to allow.

Installation
------------
- Install nextflow:

  - May already be installed on your cluster
  - If not: https://www.nextflow.io/docs/latest/getstarted.html

    - module load java
    - curl -s https://get.nextflow.io | bash

- Install conda:

  - May already be installed on your cluster


- Enable the Gray Lab conda channel:

  - Get PyRosetta license
  - Add channel to `~/.condarc`:
    ```
    channels:
      - https://{user}:{pass}@conda.graylab.jhu.edu
      - conda-forge
      - defaults
    pip_interop_enabled: true
    auto_activate_base: false
    ```

Usage
-----
- Prepare the inputs:

  - Download PDB file
  - Remove any molecules you don't want in the final model, e.g. water, 
    glycerol, etc.  I do this manually in PyMol (because I want to be in 
    complete control of the process), but there are also scripts that will do 
    basic cleaning automatically.

- Adapt pipeline to your HPC environment:

  - Several common executors require that each process specify a queue.  No 
    queue names are specified by default, because different names will mean 
    different things in different HPC environments.  The recommended way to do 
    this is to create the following config file in your HPC environment.  
    Customize the queue names and resource requirements to your environment.  
    Then use `-profile mycluster` for all nextflow jobs on your cluster:
    ```
    # ~/.nextflow/config
    profiles {
      mycluster {
        process.queue = {
          task.memory >= '200 Gb' ? 'highmem' :
          task.time > '120h' ? 'long' :
          task.time > '12h' ? : 'medium' :
          'short'
        }
      }
    }
    ```
    See https://github.com/nf-core/configs for more details.

- Run the pipeline:
  ```
  $ nextflow run kalekundert/rosetta_relax_b -profile conda,mycluster
  ```

  - Note that it's not necessary to manually download the repository hosting 
    the pipeline.  Nextflow will automatically download and cache the most 
    recent version.

  - Right now, only the conda profile is supported.

  - Expect that it might take a long time to create the conda env.  Installing 
    PyRosetta requires a >1 GB download.

  - You might need to adapt this command to run on your HPC framework.  See: 
    https://www.nextflow.io/docs/latest/executor.html

Gotchas
-------
- Be careful about using this pipeline to relax structures with different 
  resolutions, and then making comparisons between them.  Lower resolution 
  structures have higher B-factors, which will allow more aggressive 
  relaxation, ultimately biasing scores *towards* low-resolutions structures.
