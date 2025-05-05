# cancer-drug-synergy-prediction
Github repo for cancer drug synergy prediction work by Alexandra M. Wong and Lorin Crawford.

## Introduction ##

Drug resistance poses a significant challenge to cancer treatment, often caused by intratumor heterogeneity. Combination therapies have shown to be an effective strategy to prevent resistant cancer cells from escaping single-drug treatments. However, discovering new drug combinations through traditional molecular assays can be costly and time-consuming. _In silico_ approaches offer the opportunity to overcome this limitation by enabling the exploration of many candidate combinations at scale. This study systematically evaluates the effectiveness of various machine learning algorithms and drug synergy prediction tasks. Our findings challenge the assumption that multi-modal data and complex model architectures automatically yield the best predictive performance.

## Installing Requirements ##
To install the requirements stored in requirements.txt, make sure you have a compatible python version with the needed packages. We recommend python 3.11. Then run:
```sh create_venv.sh```

You also may need to enforce a numpy version below 2:
```pip install "numpy<2"```

## Data Downloads ##
Download the [NCI-ALMANAC dataset](https://wiki.nci.nih.gov/display/NCIDTPdata/NCI-ALMANAC) with the drug combination data

Download the CellMiner data
- [DNA exome sequencing](https://discover.nci.nih.gov/cellminer/download/processeddataset/nci60_DNA__Exome_Seq_none.zip)
- [RNA expression](https://discover.nci.nih.gov/cellminer/download/processeddataset/nci60_RNA__5_Platform_Gene_Transcript_Average_z_scores.zip)
- [Protein expression](https://discover.nci.nih.gov/cellminer/download/processeddataset/nci60_Protein__SWATH_(Mass_spectrometry)_Protein.zip)

We also ensured the genes present in the dataset belong to the STRING protein-protein interaction network for additional biological analyses. If so, feel free to download the [STRING database](https://string-db.org/cgi/download?sessionId=b22Ezc67moU2)
- Save the `9606.protein.links.detailed.v11.5.txt` file
- Save the `9606.protein.info.v11.5.txt` file

## Pre-processing the data - `preprocessing_files/` ##
1. Run the main function in `string_preprocessing.py`. For the first time, you should use the `--from_original` flag
2. Run the `preprocessing.ipynb`
3. Run the `filter_data.ipynb`
4. Run the `nci_almanac_therapy_classification.ipynb`
    - Make sure to have the manual retrieval of drug to therapy classes mapping

## Generating the dataset CSV files - `dataset_creation/` ##
1. Create the morgan fingerprint only CSV files by running `create_mfp_csv.ipynb`
2. Create the -omics identifiers and PCNNGL mask CSV files by running `create_omics_csv_identifiers_masks.ipynb`
3. Create the tissue type and drug class type indices files and the MFP+Omics H5 dataset files by running `create_omics_csv.ipynb`

## Run the Models - `models/` ##
1. Model parameter implementations are present in the `models/src` directory
2. Running training and evaluation code is present in the `models/run` directory

## Example / Tutorial ##
1. Unzip the compressed `example_data/all_cancer_256_mfp_bc0_comboscore.zip` file
2. There is an example dataset stored in `example_data` directory along with an example mask file for the PCNNGL model. Run the `example_models_run.ipynb` jupyter notebook, which will create output files in the `example_output` directory. This jupyter notebook includes cases of training different parameters of all models used in the study and shows test performance for the best of the example parameter models. Note that example parameters have been chosen for ease of locally running the code on a standard personal computer to demonstrate functionality. The full dataset and training analyses will require GPU-enabled and larger memory computing clusters.

## Relevant Citations ##
A.M. Wong and L. Crawford. Rethinking cancer drug synergy prediction: a call for standardization in machine learning applications. bioRxiv.
[https://doi.org/10.1101/2024.12.24.630216](https://doi.org/10.1101/2024.12.24.630216)

## Questions and Feedback ##
For questions or concerns with this work, please contact [Alexandra M. Wong](mailto:alexandra_wong@brown.edu) or [Lorin Crawford](mailto:lcrawford@microsoft.com). Feedback and questions on the software, paper, and tutorial is appreciated!
