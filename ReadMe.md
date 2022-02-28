### Contrastive spectra matching 
This repository provides the implementation for our paper [Raman spectra matching with contrastive representation learning](https://arxiv.org/abs/2202.12549). We experimentally show that we significantly outperform or is on par with the existing approaches for Raman spectra identification on three publically available datasets. 

![](concept.png)

#### Requirement 
```bash
git clone https://github.com/lyn1874/raman_spectra_matching_with_contrastive_learning.git
cd contrastive_spectra_matching
conda env create -f spectra_matching.yaml
conda activate torch_dl
```

#### Testing
The top-1 matching process and conformal prediction process per dataset is shown in the jupyter file `test_experiment.ipynb`

#### Training
To train a spectra matching model for each dataset, run the following script:
```bash
./run_rruff.sh raw '0 1 2 3'
./run_rruff.sh excellent_unoriented '0 1 2 3'
./run_organic.sh raw '0 1 2 3'
./run_organic.sh preprocess '0 1 2 3'
./run_bacteria.sh bacteria_random_reference_finetune '0 1 2 3'
```
The `repeat_g` in each script represents the number of models that are used for the ensemble calculation

#### Reproduce figures

```python
python paper_figures.py --index figure_augmentation_example --save False --pdf_pgf pdf
```

#### Todo
- [ ] Upload datasets to DTU data
- [ ] Upload checkpoints to DTU data

#### Citation
If you use this code, please cite:
```
@misc{li2022raman,
      title={Raman Spectrum Matching with Contrastive Representation Learning}, 
      author={Bo Li and Mikkel N. Schmidt and Tommy S. Alstrøm},
      year={2022},
      eprint={2202.12549},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

