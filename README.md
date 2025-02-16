
# OICD_for_VQA

This repository contains the implementation for the paper:
**Optimal Interpolation-based Coordinate Descent Method for Variational Quantum Algorithms**

The experiments focus on the **MaxCut**, **TFIM** and **XXZ** models as described in the paper.

## Running the Code

To execute the code, please run the following commands in a Git Bash terminal:

```bash
bash run_tfim_HVA_Wiersema.sh
bash run_XXZ_HVA_Wiersema.sh
```

These scripts perform numerical simulations of the **TFIM** and **XXZ** models using the proposed **Optimal Interpolation-based Coordinate Descent (OICD)** method.

`NotOptInterp_maxcut_HEA.ipynb` This Jupyter notebook demonstrates the importance of optimal interpolation nodes in the OICD method. Here, we use the **MaxCut** problem as an example.
