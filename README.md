# **MP2-F12(3C)**

The explicitly correlated approach is one of the most important breakthroughs in ab initio electronic
structure theory and broadly used to speed up convergence of electron correlation energy with respect to the basis set size.

The MP2-F12 correlation energy is implemented as the sum of the MP2 correlation energy in the orbital basis set (OBS) and the F12 correction:

<p align="center"> 
<img src="http://latex.codecogs.com/gif.latex?E%24_%7BMP2-F12%7D%24%20%3D%20E%24_%7BMP2%7D%24%20&plus;%20E%24_%7BF12%7D%24">
</p>

### Included Reference Implementations
- `mp2f12FixedAnsatz.py`: Explicitly Correlated Second-Order M{\o}ller-Plesset Perturbation Theory (MP2-F12(3C)) with Ten-no's diagonal fixed-amplitude Ansatz.
- `mp2f12SpinOrbital   `: Explicitly Correlated Second-Order M{\o}ller-Plesset Perturbation Theory (MP2-F12(3C)) with fully optimized amplitudes.

The first code calculates the F12 correction expressed as:

<p align="center"> 
<img src="http://latex.codecogs.com/gif.latex?E_%7BF12%7D%20%3D%202%5Csum_%7Bi%2Cj%2Ck%2Cl%7D%5E%7Bocc%7D%20V%5E%7Bij%7D_%7Bkl%7D%282t_%7Bij%7D%5E%7Bkl%7D%20-%20t_%7Bji%7D%5E%7Bkl%7D%29%20&plus;%20%5Csum_%7B%20%5Csubstack%7Bi%2Cj%2Ck%2Cl%2C%20%5C%5C%20m%2Cn%7D%7D%5E%7Bocc%7Dt_%7Bkl%7D%5E%7Bmn%7DB%5E%7Bij%7D_%7Bmn%7D%282t_%7Bij%7D%5E%7Bkl%7D-t_%7Bji%7D%5E%7Bkl%7D%29%20-">
</p>
<p align="center"> 
<img src="http://latex.codecogs.com/gif.latex?%5Csum_%7B%5Csubstack%7Bi%2Cj%2Ck%2Cl%2C%20%5C%5C%20m%2Cn%7D%7D%5E%7Bocc%7D%28%5Cepsilon_k%20&plus;%20%5Cepsilon_l%29%20t_%7Bkl%7D%5E%7Bmn%7DX%5E%7Bij%7D_%7Bmn%7D%282t_%7Bij%7D%5E%7Bkl%7D-t_%7Bji%7D%5E%7Bkl%7D%29">
</p>

with the intermediate matrix elements defined as:

<p align="center"> 
<img src="http://latex.codecogs.com/gif.latex?V%5E%7Bij%7D_%7Bkl%7D%20%3D%20%5Clangle%20ij%7C%20r_%7B12%7D%5E%7B-1%7D%20%5Cwidehat%7BQ%7D_%7B12%7D%20%5Cwidehat%7BF%7D_%7B12%7D%20%7C%20kl%20%5Crangle">
</p>

<p align="center"> 
<img src="http://latex.codecogs.com/gif.latex?X%5E%7Bij%7D_%7Bmn%7D%20%26%3D%20%5Clangle%20ij%7C%20%5Cwidehat%7BF%7D_%7B12%7D%20%5Cwidehat%7BQ%7D_%7B12%7D%20%5Cwidehat%7BF%7D_%7B12%7D%20%7C%20mn%20%5Crangle">
</p>

<p align="center"> 
<img src="http://latex.codecogs.com/gif.latex?B%5E%7Bij%7D_%7Bmn%7D%20%26%3D%20%5Clangle%20ij%7C%20%5Cwidehat%7BF%7D_%7B12%7D%20%5Cwidehat%7BQ%7D_%7B12%7D%20%28%5Cwidehat%7Bf%7D_1%20&plus;%20%5Cwidehat%7Bf%7D_2%20%29%5Cwidehat%7BQ%7D_%7B12%7D%20%5Cwidehat%7BF%7D_%7B12%7D%20%7C%20mn%20%5Crangle">
</p>


and amplitudes:

<p align ="center">
<img src = "http://latex.codecogs.com/gif.latex?t_%7Bij%7D%5E%7Bkl%7D%20%3D%20%5Cfrac%7B3%7D%7B8%7D%20%5Cdelta_%7Bki%7D%20%5Cdelta_%7Blj%7D%20&plus;%20%5Cfrac%7B1%7D%7B8%7D%20%5Cdelta_%7Bkj%7D%20%5Cdelta_%7Bli%7D"
</p>

In the second code the correlation energy is obtained for each electron pair (i,j) in the iterative approach:

  
Indices used:
* i, j, ... - occupied orbitals
* a, b, ... - virtual orbitals
* p, q, ... - all molecular orbitals (OBS)
* x, y, ... - CABS orbitals



### REFERENCES
1) <a href="http://aip.scitation.org/doi/abs/10.1063/1.2712434 "> H.-J. Werner, T. B. Adler, and F. R. Manby, J. Chem. Phys. 126, 164102 (2007) </a> 
2) <a href="http://aip.scitation.org/doi/abs/10.1063/1.4862255 "> S. Yoo Willow, J. Zhang, E. F. Valeev, and S. Hirata, J. Chem. Phys. 140, 031101 (2014) </a>  

