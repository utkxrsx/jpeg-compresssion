



import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
A = cv2.imread('C:\\Users\\utkar\\OneDrive\\Desktop\\cs\\jpeg\\IMG_20240312_011026.jpg')

# Convert to grayscale
B = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)

# Display the original image
plt.figure(figsize=(12, 6))
plt.subplot(3, 2, 1)
plt.imshow(256 - A)
plt.title("Original Image")

# Compute the FFT
Bt = np.fft.fft2(B)
Blog = np.log(np.abs(np.fft.fftshift(Bt)) + 1)
plt.subplot(3, 2, 2)
plt.imshow(256 - Blog, cmap='gray')
plt.title("FFT of Grayscale Image")

# Thresholding and visualization
Btsort = np.sort(np.abs(Bt).flatten())
for i, keep in enumerate([0.99, 0.05, 0.01, 0.002]):
    thresh = Btsort[int((1 - keep) * len(Btsort))]
    ind = np.abs(Bt) > thresh
    Atlow = Bt * ind
    Alow = np.uint8(np.fft.ifft2(Atlow))
    plt.subplot(3, 2, i + 3)
    plt.imshow(256 - Alow, cmap='gray')
    plt.title(f"{keep * 100:.2f}% Threshold")


plt.tight_layout()
plt.show()

