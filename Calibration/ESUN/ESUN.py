
from importlib.resources import files
from dateutil import parser

import numpy as np
import pandas as pd
import xarray as xr



#read_coeffs_from_file(self.spectral_coeff_file, 'spectral', self.x_start, self.x_stop, self.y_start, self.y_stop, self.bin_factor)




fwhm = np.array([5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46,
                              5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46,
                              5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 3.34,
                              3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34,
                              3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34,
                              3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34,
                              3.34, 3.34, 3.34, 3.34, 3.29, 3.29, 3.29, 3.29, 3.29, 3.29, 3.29,
                              3.29, 3.29, 3.29, 3.29, 3.29, 3.29, 3.29, 3.29, 3.29, 3.29, 3.29,
                              3.29, 3.29, 3.29, 3.29, 3.29, 3.32, 3.32, 3.32, 3.32, 3.32, 3.32,
                              3.42, 3.42, 3.42, 3.42, 3.42, 3.42, 3.42, 3.54, 3.54, 3.54, 3.54,
                              3.58, 3.58, 3.58, 3.59, 3.59, 3.59, 3.59, 3.59, 3.59, 3.59])



def get_spectral_response_function(wavelengths, fwhm: np.array) -> None:
    """
    Get Spectral Response Functions (SRF) from HYPSO for each of the 120 bands. Theoretical FWHM of 3.33nm is
    used to estimate Sigma for an assumed gaussian distribution of each SRF per band.

    :return: None.
    """

    fwhm_nm = fwhm
    sigma_nm = fwhm_nm / (2 * np.sqrt(2 * np.log(2)))

    srf = []
    for i, band in enumerate(wavelengths):
        center_lambda_nm = band
        start_lambda_nm = np.round(center_lambda_nm - (3 * sigma_nm[i]), 4)
        soft_end_lambda_nm = np.round(center_lambda_nm + (3 * sigma_nm[i]), 4)

        srf_wl = [center_lambda_nm]
        lower_wl = []
        upper_wl = []
        for ele in wavelengths:
            if start_lambda_nm < ele < center_lambda_nm:
                lower_wl.append(ele)
            elif center_lambda_nm < ele < soft_end_lambda_nm:
                upper_wl.append(ele)

        # Make symmetric
        while len(lower_wl) > len(upper_wl):
            lower_wl.pop(0)
        while len(upper_wl) > len(lower_wl):
            upper_wl.pop(-1)

        srf_wl = lower_wl + srf_wl + upper_wl

        good_idx = [(True if ele in srf_wl else False) for ele in wavelengths]

        # Delta based on Hypso Sampling (Wavelengths)
        gx = None
        if len(srf_wl) == 1:
            gx = [0]
        else:
            gx = np.linspace(-3 * sigma_nm[i], 3 * sigma_nm[i], len(srf_wl))
        gaussian_srf = np.exp(
            -(gx / sigma_nm[i]) ** 2 / 2)  # Not divided by the sum, because we want peak to 1.0

        # Get final wavelength and SRF
        srf_wl_single = wavelengths
        srf_single = np.zeros_like(srf_wl_single)
        srf_single[good_idx] = gaussian_srf

        srf.append([srf_wl_single, srf_single])

    return srf


x_start = 428
x_stop = 1508
bin_factor = 9

#read_coeffs_from_file(self.spectral_coeff_file, 'spectral', self.x_start, self.x_stop, self.y_start, self.y_stop, self.bin_factor)

coefficients = np.load("/home/cameron/Nedlastinger/h2_spectral_calibration_wavelengths_center_row.npz")
key = list(coefficients.keys())[0]

coefficients = coefficients[key][x_start:x_stop].reshape(-1, bin_factor).mean(axis=1).reshape(-1)


spectral_coeffs = coefficients

wavelengths = spectral_coeffs


srf = get_spectral_response_function(wavelengths=wavelengths, fwhm=fwhm)


# Read Solar Data
solar_data_path = "/home/cameron/Nedlastinger/Solar_irradiance_Thuillier_2002.csv"
solar_df = pd.read_csv(solar_data_path)

# Create new solar X with a new delta
solar_array = np.array(solar_df)
current_num = solar_array[0, 0]
delta = 0.01
new_solar_x = [solar_array[0, 0]]
while current_num <= solar_array[-1, 0]:
    current_num = current_num + delta
    new_solar_x.append(current_num)

# Interpolate for Y with original solar data
new_solar_y = np.interp(new_solar_x, solar_array[:, 0], solar_array[:, 1])

# Replace solar Dataframe
solar_df = pd.DataFrame(np.column_stack((new_solar_x, new_solar_y)), columns=solar_df.columns)

# Estimation of TOA Reflectance
band_number = 0

ESUN_values = np.zeros([120])


#print(srf)

import csv 

with open('HYPSO2_ESUN.csv', 'w', newline='') as csvfile:

    writer = csv.writer(csvfile, delimiter=",")

    writer.writerow(['nm', 'mW/m2/nm'])

    for single_wl, single_srf in srf:
        # Resample HYPSO SRF to new solar wavelength
        resamp_srf = np.interp(new_solar_x, single_wl, single_srf)

        #print(resamp_srf)

        weights_srf = resamp_srf / np.sum(resamp_srf)
        ESUN = np.sum(solar_df['mW/m2/nm'].values * weights_srf)  # units matche HYPSO from device.py

        #ESUN_values[band_number] = ESUN
        
        wl = round(single_wl[band_number], 2)

        writer.writerow([wl, ESUN])

        #print(single_wl[band_number])

        band_number = band_number + 1

        #print(single_srf)
