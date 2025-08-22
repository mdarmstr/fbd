# Fusion by Design (FBD)
This repository includes the supporting data and code for the article: __Fusion of common factor spaces for designed, heteromodal datasets by Michael Sorochan Armstrong and Jos\'e Camacho__. Please note that MATLAB scripts require the use of the Statistics and Machine Learning Toolbox.

## Directories
### data_cleaning 
This directory contains the scripts for extracting the data from the JSON format acquired from Metabolomics Workbench. The Python repositories can be installed from the ``requirements.txt`` file. The data extraction outputs 2 datasets and their associated metadata. You can create a Virtual Environment in Python through the following steps from the Linux CLI:
```bs
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Subsequent steps to ensure the fidelity of the meta data are performed in the MATLAB environment, with the ``clean_metadata.m`` code. If you do not have access to a MATLAB license, it is possible to use Octave, but the usability is not guaranteed. The authors bear no responsibility if you choose to use a Conda environment.

### MEDA
This directory contains a copy of the MEDA Toolbox (Camacho, 2017) v1.4. You must add all associated folders and subfolders to the path prior to executing any scripts.

### scripts
This directory contains all scripts used in the manuscript. Prior to running any, please make sure MEDA has been added to your root directory in MATLAB. FBD is a helper function in ``heteromodal_unidist.m``, ``BC_analysis.m`` and ``simulation_results.m``. You can load the pre-cleaned data from ``bcdata.mat`` for analysis through the ``BC_analysis.m`` script. The ``simulation_results.m`` generate and analyse the data internally, and the results can be summarized using the ``cdf_plot.m`` script. ``hetereomodal_unidist.m`` is a standalone script to generate one specific example of the FBD routine.

## Acknowledgements
The experimental data used in this study is available at the NIH Common Fund's National Metabolomics Data Repository (NMDR) website, the Metabolomics Workbench, https://www.metabolomicsworkbench.org where it has been assigned Project ID PR000284. The data can be accessed directly via it's Project DOI: 10.21228/M86K6W. This work is supported by Metabolomics Workbench/National Metabolomics Data Repository (NMDR) (grant\# U2C-DK119886), Common Fund Data Ecosystem (CFDE) (grant\# 3OT2OD030544) and Metabolomics Consortium Coordinating Center (M3C) (grant\# 1U2C-DK119889).

Michael Sorochan Armstrong has received funding from the European Union's Horizon Europe Research and Innovation Program under the Marie Skłodowska- Curie grant agreement Meta Analyses of Heterogeneous Omics Data (MAHOD) no. 101106986. This work was supported by grant no. PID2023-1523010B-IOO (MuSTARD), funded by the Agencia Estatal de Investigación in Spain, call no. MICIU/AEI/10.13039/501100011033, and by the European Regional Development Fund, with MICIU, that comes from Ministerio de Ciencia, Innovación y Universidades.

## Copyright
Copyright (C) 2025  Universidad de Granada
 
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.
