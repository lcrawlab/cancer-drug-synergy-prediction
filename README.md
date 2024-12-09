# cancer-drug-synergy-prediction
Github repo for cancer drug synergy prediction work that is available at <Link to Add>

## Introduction ##

TBD

## Installing Requirements ##
To install the requirements stored in requirements.txt, make sure you have a compatible python version with the needed packages. We recommend python 3.11. Then run:
- sh create_venv.sh

You also may need to enforce a numpy version below 2:
- pip install "numpy<2"

## Data Downloads ##
Download the NCI-ALMANAC dataset with the drug combination data
- https://wiki.nci.nih.gov/display/NCIDTPdata/NCI-ALMANAC

Download the CellMiner data
- DNA exome sequencing: https://discover.nci.nih.gov/cellminer/download/processeddataset/nci60_DNA__Exome_Seq_none.zip
- RNA expression: https://discover.nci.nih.gov/cellminer/download/processeddataset/nci60_RNA__5_Platform_Gene_Transcript_Average_z_scores.zip
- Protein expression: https://discover.nci.nih.gov/cellminer/download/processeddataset/nci60_Protein__SWATH_(Mass_spectrometry)_Protein.zip

While it is not necessary, you can also ensure the genes present in the dataset belong to the STRING protein-protein interaction network for additional biological analyses. If so, feel free to download the STRING database from https://string-db.org/cgi/download?sessionId=b22Ezc67moU2
- Save the 9606.protein.links.detailed.v11.5.txt file
- Save the 9606.protein.info.v11.5.txt file

## Pre-processing the data - preprocessing_files ##
1. If filtering by STRING:
    - Run the main function in string_preprocessing.py. For the first time, you should use the --from_original flag
    - Run the preprocessing.ipynb
2. Run the filter_data.ipynb
    - If not using STRING to filter, modify to exclude the STRING filtration
3. Run the nci_almanac_therapy_classification.ipynb
    - Make sure to have the manual retrieval of drug to therapy classes mapping

## Generating the dataset CSV files - dataset_creation ##
1. Create the morgan fingerprint only CSV files by running create_mfp_csv.ipynb
2. Create the -omics identifiers and PCNNGL mask CSV files by running create_omics_csv_identifiers_masks.ipynb
3. Create the tissue type and drug class type indices files and the MFP+Omics CSV dataset files by running create_omics_csv.ipynb

## Run the Models - models ##
- Model parameter implementations are present in the models/src directory
- Running training and evaluation code is present in the models/run directory
- Test jupyter notebooks for original model implementations are present in the models/test directory

## Example Case ##
