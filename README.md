Codes for a potential implementation to new regions of the paper ["AGBD: A Global-scale Biomass Dataset"][agbd] by Sjalelli et al. (2025) using Google Earth Engine as the main data source. Still some misalignment in the predictions, probably caused by a mistake made by yours truly, so use with caution.
All credit goes to the original authors, so for licensing and citation refer to the main paper [repository][repo].

## Execution using Colab

- Open the notebook in colab (already available and tested at this link: https://colab.research.google.com/drive/1zcMDL9hucOLRxU4Hl7Dv4Z1XfaeS9chu?usp=sharing)
- Select a machine equipped with a GPU
- You'll need to authenticate to your own gee account and project where required.

Otherwise the notebook has been tested and runs without issues.

## Local Execution

So far, the alternative .py files have only been executed on a Windows machine equipped with a rtx 5080 GPU. They are provided here only as an additional reference (and to enable the notebook to retrieve and use utils.py) with no guarantee they'll work elsewhere. 

If you want to attempt to execute them locally, start by creating a conda environment using the provided environment .yml file. you'll also need to download the nico_net.py and models.py scripts from the main repository of the paper (https://github.com/ghjuliasialelli/AGBD)

[agbd]: https://agbdataset.github.io/
[repo]: https://github.com/ghjuliasialelli/AGBD