# Forward Model Code and Papers

This directory contains runnable interfaces for several renal forward models and the paper PDFs they correspond to. The parameters directory is intentionally not described here.

## Code-Paper Mapping

| Code file | Model / purpose | Corresponding paper PDF |
| --- | --- | --- |
| xu_forward_model_44.ipynb | Xu 44-parameter forward model interface for OAT/GSA, returning total_GFR_nl_min and diagnostics. | paper/xu-et-al-2025-toward-a-full-scale-model-of-renal-hemodynamics-using-a-reconstructed-vascular-tree.pdf |
| moss_forward_model_143.ipynb | MOSS 143-parameter forward model interface for OAT/GSA. | paper/moss-thomas-2014-hormonal-regulation-of-salt-and-water-excretion-a-mathematical-model-of-whole-kidney-function-and.pdf |
| mahato_forward_model_120.ipynb | Mahato/Haire 120-parameter forward model interface for OAT/GSA, mainly exposing run_forward_model() and mahato_forward_gfr(). | paper/s41540-018-0077-9.pdf |
| hallow_2017_gfr_gsa_86d.ipynb | Hallow 2017 GFR GSA model with an 86-dimensional input vector and GFR_ml_min as the primary output. | paper/hallow-et-al-2017-primary-proximal-tubule-hyperreabsorption-and-impaired-tubular-transport-counterregulation-determine.pdf |
| renal_qsp_model_fixed.py | Python wrapper for the Hallow renal QSP model, preserving the original model core and adding DataFrame/UACR/GSA interfaces. | paper/A_Quantitative_Systems_Physiology_Model_of_Renal_F.pdf |
| richfield_model.py | Richfield-style steady-state GFR and UACR forward model with replaceable glomerular/tubule kernel interfaces. | paper/fphys-15-1410764.pdf |

## Paper Files

- paper/xu-et-al-2025-toward-a-full-scale-model-of-renal-hemodynamics-using-a-reconstructed-vascular-tree.pdf
- paper/moss-thomas-2014-hormonal-regulation-of-salt-and-water-excretion-a-mathematical-model-of-whole-kidney-function-and.pdf
- paper/s41540-018-0077-9.pdf
- paper/fphys-15-1410764.pdf
- paper/hallow-et-al-2017-primary-proximal-tubule-hyperreabsorption-and-impaired-tubular-transport-counterregulation-determine.pdf
- paper/A_Quantitative_Systems_Physiology_Model_of_Renal_F.pdf

## Notes

- The notebooks are forward-model interfaces prepared mainly for OAT/GSA batch calls.
- The Python files are importable model/wrapper files for scripts or sensitivity analysis workflows.

