Codes for a potential implementation of the paper "AGBD: A Global-scale Biomass Dataset" using Google Earth Engine as the main data source.

## Execution using Colab

- Open the notebook in colab (already available and tested at this link: https://colab.research.google.com/drive/1zcMDL9hucOLRxU4Hl7Dv4Z1XfaeS9chu?usp=sharing)
- Select a machine equipped with a GPU
- You'll need to authenticate to your own gee account and project where required.

Otherwise the notebook has been tested and runs without issues.

## Local Execution

So far, the alternative .py files have only been executed on a Windows machine equipped with a rtx 5080 GPU. They are provided here only as an additional reference (and to enable the notebook to retrieve and use utils.py) with no guarantee they'll work elsewhere. 

If you want to attempt to execute them locally, start by creating a conda environment using the provided environment .yml file. you'll also need to download the nico_net.py and models.py scripts from the main paper repository (https://github.com/ghjuliasialelli/AGBD)

## License

The same [Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc] used by the main AGBD repository applies here too for consistency.

## Citing

If you use the codes provided in this repository, please still remember to cite the main reference to the paper:

```
@article{Sialelli2025,
  title = {AGBD: A Global-scale Biomass Dataset},
  volume = {X-G-2025},
  ISSN = {2194-9050},
  url = {http://dx.doi.org/10.5194/isprs-annals-X-G-2025-829-2025},
  DOI = {10.5194/isprs-annals-x-g-2025-829-2025},
  journal = {ISPRS Annals of the Photogrammetry,  Remote Sensing and Spatial Information Sciences},
  publisher = {Copernicus GmbH},
  author = {Sialelli,  Ghjulia and Peters,  Torben and Wegner,  Jan D. and Schindler,  Konrad},
  year = {2025},
  month = jul,
  pages = {829–838}
}
```
[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
