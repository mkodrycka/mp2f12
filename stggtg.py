"""
Hard-coded fit parameters for 6-GTG to STG fit
"""

from psi4 import Vector

def stggtg(correlation_factor):
	"""
    	Returns a psi4 Vector of coefficients and exponents for the fit.
    	Parameters were produced by Molpro using default weight function.
	"""
    
	coeff = Vector(6)
	exp = Vector(6)
	
	if abs(correlation_factor - 0.3) < 1E-6:
       		coeff[0] =  -0.366770;  exp[0] =   0.020000;
        	coeff[1] =  -0.249990;  exp[1] =   1.910630;
        	coeff[2] =  -0.411230;  exp[2] =   0.164920;
        	coeff[3] =   0.144690;  exp[3] =   3.317210;
        	coeff[4] =  -0.101790;  exp[4] =  10.456340;
        	coeff[5] =   0.010510;  exp[5] =  59.343800;
    	elif abs(correlation_factor - 0.4) < 1E-6:
        	coeff[0] =  -0.238260;  exp[0] =   0.020000;
        	coeff[1] =  -0.480870;  exp[1] =   0.183200;
        	coeff[2] =  -0.226240;  exp[2] =   1.829960;
        	coeff[3] =   0.061120;  exp[3] =   3.715420;
        	coeff[4] =  -0.095680;  exp[4] =  14.289840;
        	coeff[5] =   0.007750;  exp[5] =  79.117180;
    	elif abs(correlation_factor - 0.5) < 1E-6:
        	coeff[0] =  -0.406480;  exp[0] =   0.071110;
        	coeff[1] =  -0.288210;  exp[1] =   0.385600;
        	coeff[2] =  -0.138570;  exp[2] =   1.481200;
        	coeff[3] =  -0.077240;  exp[3] =   5.070960;
        	coeff[4] =  -0.046720;  exp[4] =  19.235850;
        	coeff[5] =  -0.028770;  exp[5] = 106.869320;
    	elif abs(correlation_factor - 0.6) < 1E-6:
        	coeff[0] =  -0.369300;  exp[0] =   0.092470;
        	coeff[1] =  -0.296110;  exp[1] =   0.466250;
        	coeff[2] =  -0.150170;  exp[2] =   1.747530;
        	coeff[3] =  -0.085140;  exp[3] =   5.939820;
        	coeff[4] =  -0.051780;  exp[4] =  22.481490;
       		coeff[5] =  -0.031930;  exp[5] = 124.820720;
    	elif abs(correlation_factor - 0.7) < 1E-6:
        	coeff[0] =  -0.338470;  exp[0] =   0.115650;
        	coeff[1] =  -0.301030;  exp[1] =   0.549910;
        	coeff[2] =  -0.160120;  exp[2] =   2.017790;
        	coeff[3] =  -0.092220;  exp[3] =   6.814240;
        	coeff[4] =  -0.056370;  exp[4] =  25.738680;
        	coeff[5] =  -0.034820;  exp[5] = 142.820870;
    	elif abs(correlation_factor - 0.8) < 1E-6:
        	coeff[0] =  -0.312410;  exp[0] =   0.140560;
        	coeff[1] =  -0.303870;  exp[1] =   0.636620;
        	coeff[2] =  -0.168750;  exp[2] =   2.292810;
        	coeff[3] =  -0.098630;  exp[3] =   7.697710;
        	coeff[4] =  -0.060580;  exp[4] =  29.021470;
        	coeff[5] =  -0.037480;  exp[5] = 160.949090;
    	elif abs(correlation_factor - 0.9) < 1E-6:
        	coeff[0] =  -0.290080;  exp[0] =   0.167130;
        	coeff[1] =  -0.305230;  exp[1] =   0.726380;
        	coeff[2] =  -0.176300;  exp[2] =   2.573150;
        	coeff[3] =  -0.104480;  exp[3] =   8.592630;
        	coeff[4] =  -0.064480;  exp[4] =  32.339550;
        	coeff[5] =  -0.039940;  exp[5] = 179.260070;
    	elif abs(correlation_factor - 1.0) < 1E-6:
        	coeff[0] =  -0.270700;  exp[0] =   0.195320;
        	coeff[1] =  -0.305520;  exp[1] =   0.819200;
        	coeff[2] =  -0.182970;  exp[2] =   2.859170;
        	coeff[3] =  -0.109860;  exp[3] =   9.500730;
        	coeff[4] =  -0.068100;  exp[4] =  35.699890;
        	coeff[5] =  -0.042240;  exp[5] = 197.793280;
    	elif abs(correlation_factor - 1.1) < 1E-6:
        	coeff[0] =  -0.253730;  exp[0] =   0.225070;
        	coeff[1] =  -0.305020;  exp[1] =   0.915070;
        	coeff[2] =  -0.188880;  exp[2] =   3.151190;
        	coeff[3] =  -0.114830;  exp[3] =  10.423280;
        	coeff[4] =  -0.071480;  exp[4] =  39.107710;
        	coeff[5] =  -0.044390;  exp[5] = 216.578300;
    	elif abs(correlation_factor - 1.2) < 1E-6:
        	coeff[0] =  -0.238740;  exp[0] =   0.256340;
        	coeff[1] =  -0.303930;  exp[1] =   1.013980;
        	coeff[2] =  -0.194170;  exp[2] =   3.449400;
        	coeff[3] =  -0.119440;  exp[3] =  11.361280;
        	coeff[4] =  -0.074650;  exp[4] =  42.567040;
        	coeff[5] =  -0.046420;  exp[5] = 235.637920;
    	elif abs(correlation_factor - 1.3) < 1E-6:
        	coeff[0] =  -0.225400;  exp[0] =   0.289110;
        	coeff[1] =  -0.302410;  exp[1] =   1.115930;
        	coeff[2] =  -0.198900;  exp[2] =   3.753970;
        	coeff[3] =  -0.123730;  exp[3] =  12.315490;
        	coeff[4] =  -0.077630;  exp[4] =  46.081060;
        	coeff[5] =  -0.048330;  exp[5] = 254.990170;
    	elif abs(correlation_factor - 1.4) < 1E-6:
        	coeff[0] =  -0.213450;  exp[0] =   0.323350;
        	coeff[1] =  -0.300570;  exp[1] =   1.220900;
        	coeff[2] =  -0.203160;  exp[2] =   4.065030;
        	coeff[3] =  -0.127740;  exp[3] =  13.286540;
        	coeff[4] =  -0.080450;  exp[4] =  49.652310;
        	coeff[5] =  -0.050140;  exp[5] = 274.649570;
    	elif abs(correlation_factor - 1.5) < 1E-6:
        	coeff[0] =  -0.202700;  exp[0] =   0.359030;
        	coeff[1] =  -0.298500;  exp[1] =   1.328890;
        	coeff[2] =  -0.207010;  exp[2] =   4.382690;
        	coeff[3] =  -0.131500;  exp[3] =  14.274940;
        	coeff[4] =  -0.083110;  exp[4] =  53.282910;
        	coeff[5] =  -0.051860;  exp[5] = 294.628060;
    	elif abs(correlation_factor - 1.6) < 1E-6:
        	coeff[0] =  -0.192960;  exp[0] =   0.396140;
        	coeff[1] =  -0.296250;  exp[1] =   1.439880;
        	coeff[2] =  -0.210490;  exp[2] =   4.707040;
        	coeff[3] =  -0.135020;  exp[3] =  15.281110;
        	coeff[4] =  -0.085640;  exp[4] =  56.974610;
        	coeff[5] =  -0.053490;  exp[5] = 314.935560;
    	elif abs(correlation_factor - 1.7) < 1E-6:
        	coeff[0] =  -0.184120;  exp[0] =   0.434650;
        	coeff[1] =  -0.293880;  exp[1] =   1.553870;
        	coeff[2] =  -0.213660;  exp[2] =   5.038150;
        	coeff[3] =  -0.138350;  exp[3] =  16.305400;
        	coeff[4] =  -0.088040;  exp[4] =  60.728860;
        	coeff[5] =  -0.055050;  exp[5] = 335.580490;
    	elif abs(correlation_factor - 1.8) < 1E-6:
        	coeff[0] =  -0.176040;  exp[0] =   0.474560;
        	coeff[1] =  -0.291440;  exp[1] =   1.670850;
        	coeff[2] =  -0.216540;  exp[2] =   5.376070;
        	coeff[3] =  -0.141480;  exp[3] =  17.348120;
        	coeff[4] =  -0.090330;  exp[4] =  64.546950;
        	coeff[5] =  -0.056530;  exp[5] = 356.570020;
    	elif abs(correlation_factor - 1.9) < 1E-6:
        	coeff[0] =  -0.168640;  exp[0] =   0.515840;
        	coeff[1] =  -0.288940;  exp[1] =   1.790810;
        	coeff[2] =  -0.219170;  exp[2] =   5.720860;
       		coeff[3] =  -0.144450;  exp[3] =  18.409540;
        	coeff[4] =  -0.092510;  exp[4] =  68.429960;
       		coeff[5] =  -0.057950;  exp[5] = 377.910370;
    	elif abs(correlation_factor - 2.0) < 1E-6:
        	coeff[0] =  -0.161840;  exp[0] =   0.558490;
        	coeff[1] =  -0.286410;  exp[1] =   1.913760;
        	coeff[2] =  -0.221570;  exp[2] =   6.072570;
        	coeff[3] =  -0.147250;  exp[3] =  19.489870;
        	coeff[4] =  -0.094590;  exp[4] =  72.378830;
        	coeff[5] =  -0.059310;  exp[5] = 399.606950;
    	elif abs(correlation_factor - 2.1) < 1E-6:
        	coeff[0] =  -0.155580;  exp[0] =   0.602500;
        	coeff[1] =  -0.283870;  exp[1] =   2.039670;
        	coeff[2] =  -0.223780;  exp[2] =   6.431240;
        	coeff[3] =  -0.149920;  exp[3] =  20.589320;
        	coeff[4] =  -0.096590;  exp[4] =  76.394420;
       		coeff[5] =  -0.060620;  exp[5] = 421.664550;
    	elif abs(correlation_factor - 2.2) < 1E-6:
        	coeff[0] =  -0.149780;  exp[0] =   0.647850;
        	coeff[1] =  -0.281340;  exp[1] =   2.168560;
        	coeff[2] =  -0.225800;  exp[2] =   6.796890;
        	coeff[3] =  -0.152450;  exp[3] =  21.708080;
        	coeff[4] =  -0.098490;  exp[4] =  80.477460;
        	coeff[5] =  -0.061870;  exp[5] = 444.087390;
    	elif abs(correlation_factor - 2.3) < 1E-6:
        	coeff[0] =  -0.144410;  exp[0] =   0.694540;
        	coeff[1] =  -0.278820;  exp[1] =   2.300410;
        	coeff[2] =  -0.227650;  exp[2] =   7.169570;
        	coeff[3] =  -0.154850;  exp[3] =  22.846290;
        	coeff[4] =  -0.100320;  exp[4] =  84.628620;
       		coeff[5] =  -0.063070;  exp[5] = 466.879280;
    	elif abs(correlation_factor - 2.4) < 1E-6:
        	coeff[0] =  -0.139410;  exp[0] =   0.742560;
        	coeff[1] =  -0.276330;  exp[1] =   2.435210;
        	coeff[2] =  -0.229360;  exp[2] =   7.549300;
        	coeff[3] =  -0.157140;  exp[3] =  24.004100;
        	coeff[4] =  -0.102080;  exp[4] =  88.848500;
        	coeff[5] =  -0.064230;  exp[5] = 490.043630;
    	elif abs(correlation_factor - 2.5) < 1E-6:
        	coeff[0] =  -0.134760;  exp[0] =   0.791910;
        	coeff[1] =  -0.273870;  exp[1] =   2.572980;
       		coeff[2] =  -0.230930;  exp[2] =   7.936110;
        	coeff[3] =  -0.159330;  exp[3] =  25.181640;
        	coeff[4] =  -0.103770;  exp[4] =  93.137640;
        	coeff[5] =  -0.065350;  exp[5] = 513.583530;

    	return coeff, exp
