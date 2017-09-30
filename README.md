**MP2-F12(3*C)**

The code calculates explicitly correlated second-order Moller-Plesset energies (MP2-F12(*C)) with Ten-no's diagonal fixed-amplitude Ansatz.

The MP2-F12 correlation energy is implemented as the sum of the MP2 correlation energy in the orbital basis set (OBS) and the F12 correction:

<p align="center"> 
<img src="http://latex.codecogs.com/gif.latex?E%24_%7BMP2-F12%7D%24%20%3D%20E%24_%7BMP2%7D%24%20&plus;%20E%24_%7BF12%7D%24">
</p>

where the last term is expressed as:

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

<p align ="center">
<img src = "http://latex.codecogs.com/gif.latex?X%5E%7Bij%7D_%7Bmn%7D%20%26%3D%20%5Clangle%20ij%7C%20%5Cwidehat%7BF%7D_%7B12%7D%20%5Cwidehat%7BQ%7D_%7B12%7D%20%5Cwidehat%7BF%7D_%7B12%7D%20%7C%20mn%20%5Crangle"
</p>

and amplitudes:

<p align ="center">
<img src = "http://latex.codecogs.com/gif.latex?t_%7Bij%7D%5E%7Bkl%7D%20%3D%20%5Cfrac%7B3%7D%7B8%7D%20%5Cdelta_%7Bki%7D%20%5Cdelta_%7Blj%7D%20&plus;%20%5Cfrac%7B1%7D%7B8%7D%20%5Cdelta_%7Bkj%7D%20%5Cdelta_%7Bli%7D"
</p>


Indices used:
* i, j, ... - occupied orbitals
* a, b, ... - virtual orbitals
* p, q, ... - all molecular orbitals (OBS)
* x, y, ... - CABS orbitals

