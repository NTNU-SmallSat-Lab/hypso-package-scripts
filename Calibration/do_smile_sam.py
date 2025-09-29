'''This script uses the integer smile coefficients to correct the smile effect in HYPSO-1 and HYPSO-2 data.
It uses the frame average of the fullframe Libya Desert site captures to demonstrate the smile correction.'''
# %% imports 
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# import the Hypso package, can also use the one installed via pip
#sys.path.insert(0, '/home/ariaa/smallSatLab/hypso-package-new/hypso/')
from hypso import Hypso1, Hypso2

# %% definitions
h1_w_center = 600
h2_w_center = 588

# to remove the dark bar on the start of the image
h1_x_start = 31 
h2_x_start = 44 

dir_path = '/home/cameron/Nedlastinger/'

# plot options
plot_scene = True
plot_frame = True
plot_corrections =  True
plot_binned_corrections = True

# %% load and plot

# HYPSO-1 Capture
h1_l1a_nc_file = os.path.join(dir_path, 'libya_2025-05-06T08-42-10Z-l1a.nc')
satobj_h1 = Hypso1(path=h1_l1a_nc_file, verbose=True)
# remove the dark bar on the start
h1_l1a_cube = np.array(satobj_h1.l1a_cube)
h1_l1a_cube_cut = h1_l1a_cube[:, h1_x_start:, :] # remove the dark bar

# HYPSO-2 Capture
h2_l1a_nc_file = os.path.join(dir_path, 'libya_2025-04-27T09-22-39Z-l1a.nc')
satobj_h2 = Hypso2(path=h2_l1a_nc_file, verbose=True)
# remove the dark bar on the start
h2_l1a_cube = np.array(satobj_h2.l1a_cube)
h2_l1a_cube_cut = h2_l1a_cube[:, h2_x_start:, :] # remove the dark bar

# plot
best_rgb_fit_h1 = []
best_rgb_fit_h2 = []
for c in [630, 532, 465]: # find bands closest to wavelengths of red, green, blue
    best_rgb_fit_h1.append(np.argmin(np.abs( c - satobj_h1.wavelengths)) )
    best_rgb_fit_h2.append(np.argmin(np.abs( c - satobj_h2.wavelengths)) )

# %% plot the scene and frame

if plot_scene:
    # Plotting the L1a data
    fig, ax = plt.subplots(2, 1, figsize=(20, 3))
    # Plotting the L1a data for HYPSO-1
    plt.suptitle('Libya Desert Site', fontsize=20)
    ax[0].set_title('HYPSO-1')
    ax[0].imshow( h1_l1a_cube_cut[:,:,best_rgb_fit_h1] / np.max(h1_l1a_cube_cut[:,:,best_rgb_fit_h1]) * 0.8) # *0.8 just to look good
    ax[0].axis('off')
    # Plotting the L1a data for HYPSO-2
    ax[1].set_title('HYPSO-2')
    ax[1].imshow( h2_l1a_cube_cut[:,:,best_rgb_fit_h2] / np.max(h2_l1a_cube_cut[:,:,best_rgb_fit_h2]) * 0.8) # *0.8 just to look good
    ax[1].axis('off')
    # Adjusting the layout
    plt.tight_layout()
    plt.show()

# get frame average
h1_l1a_frame_avg = np.mean(h1_l1a_cube_cut, axis=0)
h2_l1a_frame_avg = np.mean(h2_l1a_cube_cut, axis=0)

if plot_frame:
    # Plotting the L1a data
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    plt.suptitle('Libya Fullframe (frame avg.)', fontsize=16)
    # Plotting the L1a data for HYPSO-1
    ax[0].set_title('HYPSO-1')
    ax[0].imshow(h1_l1a_frame_avg)
    # Plotting the L1a data for HYPSO-2
    ax[1].set_title('HYPSO-2')
    ax[1].imshow(h2_l1a_frame_avg)
    # Adjusting the layout
    plt.tight_layout()
    plt.show()


# %% functions for smile correction

def smile_correction_one_frame_int(frame, int_smile_coeffs):
    ''' Run smile correction on each row in a frame, using 
    integer smile coefficients.
    '''
    image_height, image_width = frame.shape
    frame_smile_corrected = np.zeros([image_height, image_width])
    for i in range(image_height):
        for j in range(image_width):
            new_idx = j - int_smile_coeffs[i]
            # make sure the index is within bounds
            if new_idx < 0:
                new_idx = 0
            elif (new_idx >= image_width):
                new_idx = image_width - 1

            frame_smile_corrected[i,j] = frame[i, new_idx]
    
    return frame_smile_corrected


# %% apply smile correction

# Read smile coeff from file
h1_poly_round = np.load(os.path.join(dir_path, 'h1_int_in-orbit_smile.npy'))
h2_poly_round = np.load(os.path.join(dir_path, 'h2_int_in-orbit_smile.npy'))

h1_corrected_frame_autocorr = smile_correction_one_frame_int(h1_l1a_frame_avg, h1_poly_round)[:,:-8] # remove last 8 pixels to avoid edge effects, these pixels are never used in the image
h2_corrected_frame_autocorr = smile_correction_one_frame_int(h2_l1a_frame_avg, h2_poly_round)[:,:-8]

# %% plot corrections

if plot_corrections:
    plt.figure(figsize=(15,8))
    plt.suptitle('HYPSO-1 Smile Correction', fontsize=16)
    plt.subplot(1, 2, 1)
    plt.title('Original frame')
    plt.imshow(h1_l1a_frame_avg)
    plt.axvline(x=1429, color='r', linestyle=':', linewidth=1)
    # plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.title('Corrected using autocorrelation method') 
    plt.imshow(h1_corrected_frame_autocorr)
    plt.axvline(x=1429, color='r', linestyle=':', linewidth=1)
    # plt.colorbar()

    plt.figure(figsize=(15,8))
    plt.suptitle('HYPSO-2 Smile Correction', fontsize=16)
    plt.subplot(1, 2, 1)
    plt.title('Original frame')
    plt.imshow(h2_l1a_frame_avg)
    plt.axvline(x=1429, color='r', linestyle=':', linewidth=1)
    # plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.title('Corrected using autocorrelation method') 
    plt.imshow(h2_corrected_frame_autocorr)
    plt.axvline(x=1429, color='r', linestyle=':', linewidth=1)
    # plt.colorbar()

# %% bin data by 9

# removed frames, to make divisible by 9
h1_l1a_frame_avg_binned = h1_l1a_frame_avg[:,1:].reshape(h1_l1a_frame_avg.shape[0], -1, 9).mean(axis=2)
h2_l1a_frame_avg_binned = h2_l1a_frame_avg[:,1:].reshape(h2_l1a_frame_avg.shape[0], -1, 9).mean(axis=2)

h1_corrected_frame_autocorr_binned = h1_corrected_frame_autocorr[:,2:].reshape(h1_corrected_frame_autocorr.shape[0], -1, 9).mean(axis=2)
h2_corrected_frame_autocorr_binned = h2_corrected_frame_autocorr[:,2:].reshape(h2_corrected_frame_autocorr.shape[0], -1, 9).mean(axis=2)

# %% plot binned data


if plot_binned_corrections:
    plt.figure(figsize=(8, 10))
    plt.suptitle('HYPSO-1 Smile Correction (binned)', fontsize=16)
    
    plt.subplot(1,2, 1)
    plt.title('Original frame')
    plt.imshow(h1_l1a_frame_avg_binned)
    # plt.colorbar()

    plt.subplot(1,2, 2)
    plt.title('Corrected using autocorrelation method') 
    plt.imshow(h1_corrected_frame_autocorr_binned)
    # plt.colorbar()
    plt.show()

    plt.figure(figsize=(8, 10))
    plt.suptitle('HYPSO-2 Smile Correction (binned)', fontsize=16)

    plt.subplot(1, 2, 1)
    plt.title('Original frame')
    plt.imshow(h2_l1a_frame_avg_binned)
    # plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title('Corrected using autocorrelation method') 
    plt.imshow(h2_corrected_frame_autocorr_binned)
    # plt.colorbar()
    plt.show()

