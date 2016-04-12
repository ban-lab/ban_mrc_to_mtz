#!/usr/bin/python
'''
/*
 * Copyright 2014-2016 - Dr. Christopher H. S. Aylett
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of version 2 of the GNU General Public License as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details - YOU HAVE BEEN WARNED!
 *
 * BAN_MRC_TO_MTZ V1.0: Generate MTZ output file from input MRC and STAR files with FOM blurring
 *
 * Credit: Chris Aylett, Daniel Boehringer, Marc Leibundgut, Nikolaus Schmitz, Nenad Ban
 *
 * Author: Chris Aylett { __chsa__ }
 *
 * Date: 16/02/2016
 *
 */
'''

# GENERATES MTZ REFLECTION FILE FROM AND MRC FILE WITH FOM BLURRING

DEBUG = False

# IMPORTS

import sys, os
while not 'scipy' in sys.modules or not 'numpy' in sys.modules or not 'matplotlib' in sys.modules:
    try:
        import scipy
        import numpy as np
        import matplotlib
        from scipy.optimize import curve_fit
        from scipy.stats    import bernoulli
        from matplotlib     import pyplot as plt
        import matplotlib.colors as colors
        import matplotlib.cm     as cmx
    except:
        print(" This script requires the SCIPY stack, which unfortunately could not be imported from your environment...")
        print(" Operating system dependent instructions for install are available at: https://www.scipy.org/install.html")
        additional_path = raw_input(" Alternatively, if the stack is available, but not linked to python on your machine, provide the path now")
        if os.path.isdir(additional_path.strip()):
            sys.path.insert[0](os.path.abspath(additional_path))

# FUNCTION DEFINITIONS

def mrc_to_numpy(map_file):
    '''Read mrc file to numpy array'''
    f = open(map_file, 'rb')
    header = f.read(1024)
    f.seek(0)
    map_size = np.fromstring(f.read(12), dtype = np.int32)
    map_mode = np.fromstring(f.read(4), dtype = np.int32)
    f.seek(64)
    map_order = np.fromstring(f.read(12), dtype = np.int32) - 1
    map_order = map_order.astype(dtype=np.int32)
    f.seek(40)
    map_shape = np.fromstring(f.read(12), dtype = np.float32)
    map_shape = map_shape[map_order]
    f.seek(196)
    map_origin = np.fromstring(f.read(12), dtype = np.float32)
    map_origin = map_origin[map_order]
    map_origin = np.rint(map_origin * (map_size.astype(np.float32) / map_shape)).astype(np.int32)
    f.seek(1024)
    if map_mode == 0:
        mrc_map = np.fromstring(f.read(), dtype = np.int8)
    elif map_mode == 1:
        mrc_map = np.fromstring(f.read(), dtype = np.int16)
    elif map_mode == 2:
        mrc_map = np.fromstring(f.read(), dtype = np.float32)
    elif map_mode == 6:
        mrc_map = np.fromstring(f.read(), dtype = np.uint16)
    else:
        print(" Map mode must be either 8 or 16 bit integers or 32 bit floats / reals - file mode was not recognised...")
        exit(1)
    f.close()
    mrc_map = mrc_map.astype(np.float32)
    mrc_map = mrc_map.reshape(map_size)
    mrc_map = np.transpose(mrc_map, map_order)
    mrc_map = np.roll(mrc_map, map_origin[2], axis = 0)
    mrc_map = np.roll(mrc_map, map_origin[1], axis = 1)
    mrc_map = np.roll(mrc_map, map_origin[0], axis = 2)
    if DEBUG:
        f = open(map_file.replace(".mrc", "_rolled.mrc"), 'w')
        f.write(header)
        f.write(mrc_map.tostring())
        f.close()
    mrc_map = np.swapaxes(mrc_map, 0, 2)
    return mrc_map

def read_star_column(star_file):
    '''Extract fsc, resolution and angpix value from relion postprocess star file'''
    f = open(star_file, 'r')
    fsc_curve = [[],[]]
    res_col = None
    res     = False
    fsc_col = None
    fsc     = False

    for line in f:

        if '--angpix' in line:
            try:
                angpix = float(line.split('--angpix')[1].split()[0])
                print(' ' + str(angpix) + ' Angstroms per pixel')
            except:
                print(" Error reading angpix value from star file...")
                exit(1)
            continue

        line = line.strip().split()

        if line and '_rlnFinalResolution' in line[0]:
            try:
                resolution = float(line[1])
                print(' ' + str(resolution) + ' final resolution')                
            except:
                print(" Error extracting final resolution from star file...")
                exit(1)
            continue

        if line and '_rlnResolution' in line[0]:
            try:
                res_col = int(line[1][1:]) - 1
                res     = True
                print(' Resolution in column ' + str(res_col + 1) + ' of the table')
            except:
                print(" Error extracting resolution column from star file...")
                exit(1)
            continue

        if line and '_rlnFourierShellCorrelationCorrected' in line[0]:
            try:
                fsc_col = int(line[1][1:]) - 1
                fsc     = True
                print(' Corrected fsc in column ' + str(fsc_col + 1) + ' of the table')
            except:
                print(" Error extracting correctedfsc column from star file...")
                exit(1)
            continue

        if not line:
            res = False
            fsc = False
            continue

        if line and res and fsc:
            try:
                fsc_curve[0].append(float(line[res_col]))
                fsc_curve[1].append(float(line[fsc_col]))
            except:
                continue
            continue

    if not fsc_curve[0]:
        print(" Error reading the star file...")
        exit(1)

    return fsc_curve, resolution, angpix

def curve_function(x, a, b, c, d, e):
    '''Double sigmoidal curve for fsc fitting'''
    y = a * (1 - (1 / (1 + np.exp(-x * b + c)))) + (1 - a) * (1 - (1 / (1 + np.exp(-x * d + e))))
    if np.all(np.isfinite(y)):
        return y
    else:
        return 0

def fsc_to_sigf(amp, fsc):
    '''Estimate a proportional SIGF from amplitude and Cref using SSNR - Pawel Penczek, Methods 
       Enzymol. 2010 - weighting to prevent software from complaining about v. low sigfs'''
    cref = fsc_to_fom(fsc)
    sigf = amp / ((0.99 * cref) / (1 - 0.99 * cref))
    if np.isfinite(sigf):
        return sigf
    else:
        return amp

def fsc_to_fom(fsc):
    '''Convert fsc value to FOM - Peter Rosenthal & Richard Henderson, J. Mol. Biol., Oct. 2003'''
    fom = np.sqrt((2 * fsc) / (1 + fsc))
    if np.isfinite(fom):
        return fom
    else:
        return 0.5

def fom_to_hl(fom, phi):
    '''Convert FOMs to HLA and HLB - Kevin Cowtan - www.ysbl.york.ac.uk/~cowtan/clipper'''
    x0 = np.abs(fom)
    a0 = -7.107935 * x0
    a1 = 3.553967 - 3.524142 * x0
    a2 = 1.639294 - 2.228716 * x0
    a3 = 1.0 - x0
    w = a2 / (3.0 * a3)
    p = a1 / (3.0 * a3) - w * w
    q = -w * w * w + 0.5 * (a1 * w - a0) / a3
    d = np.sqrt(q * q + p * p * p)
    q1 = q + d
    q2 = q - d
    r1 = np.power(np.abs(q1), 1.0 / 3.0)
    r2 = np.power(np.abs(q2), 1.0 / 3.0)
    if q1 <= 0.0:
        r1 = -r1
    if q2 <= 0.0:
        r2 = -r2
    x = r1 + r2 - w
    HLA = x * np.cos(phi)
    HLB = x * np.sin(phi)
    if np.isfinite(HLA) and np.isfinite(HLB):
        return HLA, HLB
    else:
        print(" Error determining HL coefficients for FOM = "+str(fom)+' and phase = '+str(phi))
        exit(1)

def fit_fsc(fsc_curve):
    '''Fit curve to fsc'''
    input   = 'N'
    while input[0] != 'y':
        curve_x = np.asarray(fsc_curve[0])
        curve_y = np.asarray(fsc_curve[1])
        try:
            fsc_coeffs, var = curve_fit(curve_function, curve_x, curve_y, [0.5, 50.0, 5.0, 50.0, 15.0])
        except:
            print(" Error fitting fsc curve...")
            exit(1)
        residuals = []
        for i, val in enumerate(fsc_curve[0]):
            calc = curve_function(val, *fsc_coeffs)
            residuals.append(np.abs(calc - fsc_curve[1][i]))

        print(' Residuals for the fitted curve - mean: ' + str(np.mean(np.asarray(residuals))) + ' - max: ' + str(np.max(np.asarray(residuals)))+' - these should be in the region - mean: <= 0.01 - max: <= 0.05 - to proceed')
        print(" Close graph window and type Y to Accept current curve fit or N to remove the value with the largest residual (typically due to filter abberation) and refit the curve")

        # PLOT GRAPHS
        cm = plt.get_cmap('cool')
        max = np.max(np.asarray(residuals))
        min = np.min(np.asarray(residuals))
        c_norm     = colors.Normalize(vmin = min, vmax = max)
        scalar_map = cmx.ScalarMappable(norm = c_norm, cmap = cm)
        c_val = [0 for i in fsc_curve[0]]

        fig = plt.figure()
        ax = plt.subplot(111)
        
        x = np.arange(fsc_curve[0][0], fsc_curve[0][-1], 0.01)
        ax.plot(x, curve_function(x, *fsc_coeffs), linewidth=2, linestyle=':', color='black')

        for i in range(len(fsc_curve[0])):
            c_val[i] = scalar_map.to_rgba(residuals[i])
            x = fsc_curve[0][i]
            y = fsc_curve[1][i]
            ax.plot(x, y, 'o', markersize = 5, color = c_val[i], markeredgewidth = 0.5)

        plt.xlabel('Resolution')
        plt.ylabel('fsc')
        plt.title('Observed and calculated fsc curves plotted against resolution')
        plt.show()

        fsc_curve[0].pop(residuals.index(max))
        fsc_curve[1].pop(residuals.index(max))
        
        input = raw_input(' -- required action: ').lower()
        if not input:
            input = 'N'

    return fsc_coeffs

def fft_to_hkl(h, k, l, val, fsc_coeffs, resolution, full_size, flag_frac):
    '''Reformat fft record as hkl record'''
    if h or k or l:
        res = full_size / (np.linalg.norm(np.asarray([h, k, l])))
    else:
        res = 0.0

    if res < resolution or not np.isfinite(res):
        return None, None

    mag = np.abs(val)
    angle = np.angle(val, deg = True)

    if angle < 0:
        angle += 360.0

    fsc = curve_function((1. / res), *fsc_coeffs)
    sig = fsc_to_sigf(mag, fsc)
    fom = fsc_to_fom(fsc)
    hla, hlb = fom_to_hl(fom, np.angle(val))
    rf = bernoulli.rvs(flag_frac)
    record = np.array([h, k, l, mag, sig, angle, fom, hla, hlb, 0.0, 0.0, rf], dtype = np.float32)
    
    if not np.all(np.isfinite(record))
        return None, None

    return record, res

def write_mtz_file(map_file, full_size, res_min, res_max, hkl_fp):
    '''Output hkl records to mtz file format'''
    if '.mrc' in map_file:
        f = open(map_file.replace('.mrc', '.mtz'), 'wb')
    else:
        f = open(map_file+'.mtz', 'wb')
    blank = np.array([0. for i in range(20)], dtype = np.float32)
    f.write(blank.tostring())

    for line in hkl_fp:
        f.write(line.tostring())

    head_loc = np.array([f.tell() / 4 + 1], dtype = np.int32)
    f.write('VERS MTZ:V1.1                                                                   ')
    f.write('TITLE CHSAYLETT DBOEHRINGER MLEIBUNDGUT NSCHMITZ NBAN BAN MRC TO MTZ V1.0 OUTPUT')
    f.write('NCOL %8i %12i        0                                             '%(hkl_fp.shape[1], hkl_fp.shape[0]))
    f.write('CELL  %9.4f %9.4f %9.4f   90.0000   90.0000   90.0000               '%(full_size, full_size, full_size))
    f.write('SORT    0   0   0   0   0                                                       ')
    f.write("SYMINF   1  1 P     1                   'P1'     1                              ")
    f.write('SYMM X,  Y,  Z                                                                  ')
    f.write('RESO %18.16f   %18.16f                                    '%(1/res_max**2, 1/res_min**2))
    f.write('VALM NAN                                                                        ')
    f.write('COLUMN H                              H%18f%18f    1'%(np.min(hkl_fp[:,0]), np.max(hkl_fp[:,0])))
    f.write('COLUMN K                              H%18f%18f    1'%(np.min(hkl_fp[:,1]), np.max(hkl_fp[:,1])))
    f.write('COLUMN L                              H%18f%18f    1'%(np.min(hkl_fp[:,2]), np.max(hkl_fp[:,2])))
    f.write('COLUMN FOBS                           F%18f%18f    1'%(np.min(hkl_fp[:,3]), np.max(hkl_fp[:,3])))
    f.write('COLUMN SIGF                           Q%18f%18f    1'%(np.min(hkl_fp[:,4]), np.max(hkl_fp[:,4])))
    f.write('COLUMN PHIB                           P%18f%18f    1'%(np.min(hkl_fp[:,5]), np.max(hkl_fp[:,5])))
    f.write('COLUMN FOM                            W%18f%18f    1'%(np.min(hkl_fp[:,6]), np.max(hkl_fp[:,6])))
    f.write('COLUMN HLA                            A%18f%18f    1'%(np.min(hkl_fp[:,7]), np.max(hkl_fp[:,7])))
    f.write('COLUMN HLB                            A%18f%18f    1'%(np.min(hkl_fp[:,8]), np.max(hkl_fp[:,8])))
    f.write('COLUMN HLC                            A                 0                 0    1')
    f.write('COLUMN HLD                            A                 0                 0    1')
    f.write('COLUMN RF                             I                 0                 1    1')
    f.write('NDIF        1                                                                   ')
    f.write('PROJECT       1 project                                                         ')
    f.write('CRYSTAL       1 crystal                                                         ')
    f.write('DATASET       1 dataset                                                         ')
    f.write('DCELL         1  %9.4f %9.4f %9.4f   90.0000   90.0000   90.0000    '%(full_size, full_size, full_size))
    f.write('DWAVEL        1    1.00000                                                      ')
    f.write('END                                                                             ')
    f.write('MTZENDOFHEADERS                                                                 ')
    f.seek(0)
    f.write('MTZ ')
    f.write(head_loc.tostring())
    f.write('DA')
    f.close()
    return

# MAIN LOOP

def main():

    print('\n ban_mrc_to_mtz.py v1.0: Generate .mtz reflection file from .mrc and .star files with FOM blurring - GNU licensed 16-02-2016 - __chsa__')
    print(' <Please reference Greber BJ, Boehringer D, Leibundgut M, Bieri P, Leitner A, Schmitz N, Aebersold R, Ban N. Nature. 515: 283-6 (2014)>\n')

    if len(sys.argv) < 3:
        print(' Required inputs: '+sys.argv[0]+'  final_mrc_map.mrc  relion_postprocess.star [--rfree (r_free_percentage)]')
        print(' FSC is fitted and the resolution dependent curve used to calculate radial FOM and SIGF values for reciprocal space refinement')
        print(' the script requires numpy and scipy and can operate directly from a relion postprocess .star file or a copy of the text below\n')
        print(' --angpix             1.00                                          // Angstrom per voxel value in the final .mrc map provided')
        print
        print(' _rlnFinalResolution  3.00                                          // Final real space resolution requested for the .mtz file')
        print
        print(' _rlnResolution                       #1                            // FSC -reciprocal space resolution in Angstroms per voxel')
        print(' _rlnFourierShellCorrelationCorrected #2                            // FSC -Fourier shell correlation between independent maps')
        print(' 0.001001 1.000000')
        print(' 0.002248 0.999999')
        print(' ...')
        print(' ...')
        print(' ...')
        print
        print(' FSC table terminating in a blank line, columns specified by the numbers above, there must be no blank lines before the start\n')
        exit(1)

    # Set by default to our normal parameter - can be changed by flag in input
    r_free_percentage = 0.025
    for i, val in enumerate(sys.argv):
        if "--rfree" in val:
            try:
                r_free_percentage = float(sys.argv[i+1]) / 100
                print(" R free fraction set to "+str(r_free_percentage))
            except:
                print(" R free percentage could not be set!")

    map_file, star_file = sys.argv[1], sys.argv[2]

    if not os.path.isfile(map_file) or not os.path.isfile(star_file):
        print(' Inputs were not valid files')
        exit(1)

    print(" Reading mrc map")
    map = mrc_to_numpy(map_file)
    map_size = map.shape[0]

    print(" Reading star file")
    fsc_curve, resolution, ANGPIX = read_star_column(star_file)

    print(" Fitting fsc curve")
    fsc_coeff = fit_fsc(fsc_curve)

    print(" Converting to reciprocal space")
    fft = np.fft.fftn(map)
    hkl_fp = []
    full_size = map_size * ANGPIX
    res_min = 999.
    res_max = -999.

    # This formulation avoids duplicated reflection records
    print(" Converting to reflection format")
    i = 0
    while i <= int(full_size / resolution):
        if i == 0:
            j = 0
        else:
            j= -int(full_size / resolution)
        while j <= int(full_size / resolution):
            if i == 0 and j == 0:
                k = 0
            else:
                k = -int(full_size / resolution)
            while k <= int(full_size / resolution):

                # Negative values for the h k l reflections given the inverted convention for FFT hand in crystallography
                RECORD, res = fft_to_hkl(-i, -j, -k, fft[i,j,k], fsc_coeff, resolution, full_size, r_free_percentage)
                if res:
                    if res < res_min:
                        res_min = res
                    if res > res_max:
                        res_max = res
                if res:
                    if np.any(RECORD):
                        hkl_fp.append(RECORD)
                k += 1
            j += 1
        i += 1

    hkl_fp = np.array(hkl_fp)

    print(" Writing MTZ file")
    write_mtz_file(map_file, full_size, res_min, res_max, hkl_fp)
    print("++++ That's all folks! ++++")
    return 0

if __name__ == "__main__":
    sys.exit(main())
