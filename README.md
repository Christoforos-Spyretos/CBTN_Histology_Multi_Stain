# Multi-Stain Fusion of Histopathology Images Using Deep Learning for Pediatric Brain Tumor Classification

This repository contains the code for the pre-processing, model training and evaluation for the classification of
pediatric WSI brain tumors from the [Children's Brain Tumor Network](https://cbtn.org) dataset. 

[BioRxiv](https://www.biorxiv.org/content/10.64898/2026.04.10.717785v1.abstract) | [Cite](#reference)

## Abstract   
The classification of pediatric brain tumors is investigated using deep learning on hematoxylin and eosin (H&E) and
antigen Ki-67 (Ki-67) whole slide images (WSIs) from the Children’s Brain Tumor Network (CBTN) dataset. A total of 1,662
unregistered WSIs (1,047 H&E and 615 Ki-67 images) were analyzed, including low-grade glioma/astrocytoma (grades 1, 2)
(LGG), high-grade glioma/astrocytoma (grades 3, 4) (HGG), medulloblastoma (MB), ependymoma (EP) and ganglioglioma. The
The aim of this study was to effectively classify pediatric brain tumors using H&E and Ki-67 WSIs individually, and to
investigate whether early, intermediate, and late fusion could improve the predictive performance. From each WSI, 224×
224 pixel patches were extracted, and the instance (patch)-level features were obtained using the histology foundation
model CONCHv1_5. The instances were aggregated using clustering-constrained attention multiple instance learning (CLAM)
for patient-level classification. Model interpretability and explainability was assessed through attention heatmaps,
cell density and Ki-67 labelling index (LI) maps. In the binary grade classification between LGG and HGG, the
intermediate concatenation fusion achieved the best performance with a balanced accuracy of 0.88 ± 0.05, (p < 0.005)
compared to the single-stain models (H&E: 0.84 ± 0.05, Ki-67: 0.86 ± 0.05). For the 5-class tumor type classification,
the one-hidden layer late fusion learning model achieved the highest balanced accuracy of 0.83 ± 0.04 (p < 0.005),
outperforming the single-stain models (H&E: 0.77 ± 0.05, Ki-67: 0.74 ± 0.05). Overall, most of the fusion approaches
outperformed the single-stain models in both classification tasks (p < 0.005). The Ki-67 attention maps demonstrated
moderate to strong Spearman correlation (ρ = 0.576 − 0.823) with the cell density and Ki-67 LI maps, suggesting that
these features are associated with the model’s predictions, although additional features may contribute. The results
show that H&E and Ki-67 images provide complementary information, and most of the multi-stain fusion approaches using
deep learning improve pediatric brain tumor diagnosis.

## Key Highlights

![Workflow](Figures/methodology_version_2.jpg)

<!-- The weights for the CONCHv1\_5 pre-trained model are available at
\url{https://github.com/mahmoodlab/CONCH?tab=readme-ov-file} and the original code for CLAM is available at
\url{https://github.com/mahmoodlab/CLAM}. -->

## Reference

```
@article{spyretos2026multi,
  title={Multi-Stain Fusion of Histopathology Images Using Deep Learning for Pediatric Brain Tumor Classification},
  author={Spyretos, Christoforos and Tampu, Iulian Emil and Lindblad, Joakim and Haj-Hosseini, Neda},
  journal={bioRxiv},
  pages={2026--04},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}
```