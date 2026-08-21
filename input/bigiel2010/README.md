# Bigiel et al. (2010) outer-disk radial profiles

`RadialProfiles.csv` contains the galaxy-by-galaxy points at
`R/R25 >= 1` from Figure 2 of Bigiel et al. (2010), *Extremely Inefficient
Star Formation in the Outer Disks of Nearby Galaxies* (AJ 140, 1194;
doi:10.1088/0004-6256/140/5/1194; arXiv:1007.3498).

The paper's electronic supplement provides pixel-distribution data, not the
radial-profile table plotted in Figure 2. The values here were extracted from
the vector EPS files in the original arXiv source. The original EPS files and
the one-time extraction utility are not distributed in this repository.

The H I and FUV error bars are the plotted 1-sigma uncertainties in the mean
within each annulus. `e_SigmaSFR_stat_Msun_yr_kpc2` is the plotted statistical
uncertainty. `e_SigmaSFR_Msun_yr_kpc2` additionally includes the paper's
approximately 50% FUV-to-SFR conversion uncertainty in quadrature. The loader
then cross-calibrates each galaxy's FUV-only Bigiel SFR scale to the
Leroy et al. (2008) FUV+24-micron scale using their common coverage near R25
and propagates the cross-calibration uncertainty. Both raw and calibrated
values are retained in each exported `bigiel_profiles_used.csv`. The H I
surface density is used as the outer-disk gas surface density because Bigiel
et al. treat molecular gas as negligible in this radial regime.
