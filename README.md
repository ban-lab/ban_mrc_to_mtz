
 ban_mrc_to_mtz.py v1.0: Generate .mtz reflection file from .mrc and .star files with FOM blurring - GNU licensed 16-02-2016 - __chsa__ 
 <Reference Greber BJ, Boehringer D, Leibundgut M, Bieri P, Leitner A, Schmitz N, Aebersold R, Ban N. Nature. 515: 283-6 (2014)>
 
 Required inputs: ./ban_mrc_to_mtz.py  final_mrc_map.mrc  relion_postprocess.star [--rfree (r_free_percentage)]
 FSC is fitted and the resolution dependent curve used to calculate radial FOM and SIGF values for reciprocal space refinement
 the script requires numpy and scipy and can operate directly from a relion postprocess .star file or a copy of the text below

 --angpix             1.00                                          // Angstrom per voxel value in the final .mrc map provided

 _rlnFinalResolution  3.00                                          // Final real space resolution requested for the .mtz file

 _rlnResolution                       #1                            // FSC -reciprocal space resolution in Angstroms per voxel
 _rlnFourierShellCorrelationCorrected #2                            // FSC -Fourier shell correlation between independent maps
 0.001001 1.000000
 0.002248 0.999999
 ...
 ...
 ...

 FSC table terminating in a blank line, columns specified by the numbers above, there must be no blank lines before the start
