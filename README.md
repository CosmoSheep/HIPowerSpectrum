##### This is a repository of public available data and codes we use in [Long et al. 2021](https://arxiv.org/pdf/2107.07615.pdf). 

---
- **fmass_mean.csv & fmass_mean-nv.csv** - filtering mass data with and without streaming velocity effects, $3.5\leq z_{obs}\leq 5.5$, $6\leq z_{re}\leq12$. 
- **analyze_fscale.py** - extracting filtering scales and masses from Gadget2 simulations.
- **bias.py** - calculation of bias coefficients $b_1, b_2, b_{s^2}$ and $b_v$.
- **Makefile.v1 & Makefile.v0** - modified Gadget2 initial condition generation configure files, v1 adds streaming velocity effects.
- **save.c** - modified files in Gadget2 N-GenIC file to enable streaming velocity effect.

---
> Copyright:You may freely distribute and copy the data and codes in the repository. You may also modify it as you wish, and distribute these modified versions as long as you indicate prominently any changes you made in the original code, and as long as you leave the copyright notices, and the no-warranty notice intact.

> See COPYRIGHT and COPYING for Gadget2 license information.
