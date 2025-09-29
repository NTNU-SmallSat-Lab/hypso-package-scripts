from PIL import Image, ImageOps

def auto_adjust_img(img: np.ndarray) -> np.ndarray:
    """
    Automatically adjust the image contracts using histogram equalization.

    :param img: Image to adjust as a numpy array.

    :return: Adjusted image as a numpy array.
    """

    img = Image.fromarray(np.uint8(img * 255 / np.max(img)))

    # Convert image to grayscale
    gray_img = ImageOps.grayscale(img)

    # Compute histogram
    hist = gray_img.histogram()

    # Compute cumulative distribution function (CDF)
    cdf = [sum(hist[:i + 1]) for i in range(len(hist))]

    # Normalize CDF
    cdf_normalized = [
        int((x - min(cdf)) * 255 / (max(cdf) - min(cdf))) for x in cdf]

    # Create lookup table
    lookup_table = dict(zip(range(256), cdf_normalized))

    # Apply lookup table to each channel
    channels = img.split()
    adjusted_channels = []
    for channel in channels:
        adjusted_channels.append(channel.point(lookup_table))

    # Merge channels and return result
    pil_image = Image.merge(img.mode, tuple(adjusted_channels))

    return np.array(pil_image) / 255.0


R_wl=650, 
G_wl=550, 
B_wl=450

# Option 1) Use specific bands -------------------------------------
R = np.argmin(abs(satobj.wavelengths - R_wl))
G = np.argmin(abs(satobj.wavelengths - G_wl))
B = np.argmin(abs(satobj.wavelengths - B_wl))

# get the rgb image
rgb = satobj.l1a_cube[:, :, [R, G, B]]
rgb_img = auto_adjust_img(rgb)


from matplotlib.pyplot import figure

figure(figsize=(15, 20), dpi=80)

plt.imshow(rgb_img)

