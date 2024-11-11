import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu
#from osum import tanimoto

#loading grayscale image
img = cv2.imread(r"D:\Desktop\FAKS\OBRADA SLIKE\MR_segment\mr_03.png", 0)  # Load as grayscale

# multi-thresholding (Otsu)
# Calculate the thresholds for 3 regions ??? (background, tissue 1, tissue 2, tissue 3) ako stavim 3 klase, dobijem 2
thresholds = threshold_multiotsu(img, classes=4)  # 4 classes (3 tissues + background>)

#segmentacija pragom
regions = np.digitize(img, bins=thresholds)

#binarne maske
tissue1_mask = (regions == 1).astype(np.uint8) * 255 
tissue2_mask = (regions == 2).astype(np.uint8) * 255  
tissue3_mask = (regions == 3).astype(np.uint8) * 255  

# jet colormap - visualization
# nije dobro - hocu da mi plava bude crna a crvena bela, neka je greska
colored_img = cv2.applyColorMap(img, cv2.COLORMAP_JET)


#grayscale to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

colored_img[regions == 1] = [255, 0, 0]  # Red for tissue 1
colored_img[regions == 2] = [0, 255, 0]  # Green for tissue 2
colored_img[regions == 3] = [0, 0, 255]  # Blue for tissue 3

# histograms 
tissue1_pixels = img[regions == 1]  
tissue2_pixels = img[regions == 2]  
tissue3_pixels = img[regions == 3] 


hist_tissue0, bins_tissue0 = np.histogram(img, bins=256, range=(0,255))
hist_tissue1, bins_tissue1 = np.histogram(tissue1_pixels, bins=256, range=(0, 255))
hist_tissue2, bins_tissue2 = np.histogram(tissue2_pixels, bins=256, range=(0, 255))
hist_tissue3, bins_tissue3 = np.histogram(tissue3_pixels, bins=256, range=(0, 255))


#plotting
plt.figure(figsize=(12, 12))

# original grayscale MRI
plt.subplot(2, 4, 1)
plt.title("Original Grayscale MRI")
plt.imshow(img, cmap='gray')
plt.axis('off')

# tissue masks subplot
plt.subplot(2, 4, 2)
plt.title("Tissue 1")
plt.imshow(tissue1_mask, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.title("Tissue 2")
plt.imshow(tissue2_mask, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.title("Tissue 3")
plt.imshow(tissue3_mask, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 5)
plt.title("Original Histogram")
plt.plot(bins_tissue0[:-1], hist_tissue0, color='m')
plt.xlim([-10, 255])

# Display the histograms
plt.subplot(2, 4, 6)
plt.title("Tissue 1 Histogram")
plt.plot(bins_tissue1[:-1], hist_tissue1, color='r')
plt.xlim([0, 255])
plt.ylim([0, 500])

plt.subplot(2, 4, 7)
plt.title("Tissue 2 Histogram")
plt.plot(bins_tissue2[:-1], hist_tissue2, color='g')
plt.xlim([0, 255])
plt.ylim([0, 500])

plt.subplot(2, 4, 8)
plt.title("Tissue 3 Histogram")
plt.plot(bins_tissue3[:-1], hist_tissue3, color='b')
plt.xlim([0, 255])
plt.ylim([0, 500])

# Display the colored segmented overlay
plt.figure(figsize=(8, 8))
plt.title("Overlay of Segmented Tissues")
plt.imshow(colored_img)
plt.axis('off')

plt.tight_layout()
plt.show()

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage.filters import threshold_multiotsu

# # Step 1: Load the grayscale MRI image
# img = cv2.imread(r"D:\Desktop\FAKS\OBRADA SLIKE\MR_segment\mr_03.png", 0)  # Load as grayscale

# # Step 2: Apply multi-thresholding (Otsu)
# # Calculate the thresholds for 3 regions (background, tissue 1, tissue 2, tissue 3)
# thresholds = threshold_multiotsu(img, classes=4)  # Use 4 classes for 3 tissues + background

# # Step 3: Segment the image based on these thresholds
# # 0 -> background, 1 -> tissue 1, 2 -> tissue 2, 3 -> tissue 3
# regions = np.digitize(img, bins=thresholds)

# # Step 4: Create binary masks for each tissue class
# tissue1_mask = (regions == 1).astype(np.uint8) * 255  # Tissue 1 mask
# tissue2_mask = (regions == 2).astype(np.uint8) * 255  # Tissue 2 mask
# tissue3_mask = (regions == 3).astype(np.uint8) * 255  # Tissue 3 mask

# # Step 5: Apply a colormap for visualization (jet colormap)
# # The jet colormap will map dark values to blue and bright values to red
# colored_img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

# # Step 6: Overlay the segmented regions on the original grayscale image
# # Convert grayscale image to RGB for overlay
# img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

# # Overlay each segmented mask with different colors
# colored_img[regions == 1] = [255, 0, 0]  # Red for tissue 1
# colored_img[regions == 2] = [0, 255, 0]  # Green for tissue 2
# colored_img[regions == 3] = [0, 0, 255]  # Blue for tissue 3

# # Step 7: Display the results
# plt.figure(figsize=(12, 12))

# plt.subplot(2, 2, 1)
# plt.title("Original Grayscale MRI Image")
# plt.imshow(img, cmap='gray')
# plt.axis('off')

# plt.subplot(2, 2, 2)
# plt.title("Tissue 1 Binary Mask")
# plt.imshow(tissue1_mask, cmap='gray')
# plt.axis('off')

# plt.subplot(2, 2, 3)
# plt.title("Tissue 2 Binary Mask")
# plt.imshow(tissue2_mask, cmap='gray')
# plt.axis('off')

# plt.subplot(2, 2, 4)
# plt.title("Tissue 3 Binary Mask")
# plt.imshow(tissue3_mask, cmap='gray')
# plt.axis('off')

# plt.figure(figsize=(10, 10))
# plt.title("Overlay of Segmented Tissues")
# plt.imshow(colored_img)
# plt.axis('off')

# plt.show()


# import cv2
# import numpy as np
# from sklearn.mixture import GaussianMixture
# import matplotlib.pyplot as plt

# # Load the grayscale MRI image
# img = cv2.imread(r"D:\Desktop\FAKS\OBRADA SLIKE\MR_segment\mr_03.png", 0)  # Load as grayscale

# # Step 0: Apply noise filtration
# # Uncomment and adjust the filtering method if needed
# # filtered_img = cv2.medianBlur(img, 5)  # Kernel size 5
# # filtered_img = cv2.bilateralFilter(img, d=5, sigmaColor=60, sigmaSpace=60)
# # filtered_img = cv2.GaussianBlur(img, (5, 5), 0)  # Kernel size 5x5, sigma = 0
# filtered_img = img

# # Step 1: Reshape filtered image into a 2D array of pixel intensities (required for GMM)
# pixels = filtered_img.reshape(-1, 1)

# # Step 2: Fit Gaussian Mixture Model (GMM) to pixel intensities
# # gmm = GaussianMixture(n_components=3, covariance_type='full', reg_covar=1e-4, init_params='kmeans', max_iter=300, random_state=42)
# gmm = GaussianMixture(n_components=3, random_state=42)
# gmm.fit(pixels)

# # Step 3: Predict the tissue class for each pixel
# gmm_labels = gmm.predict(pixels)

# # Reshape the predicted labels to the original image shape
# segmented_img = gmm_labels.reshape(img.shape)

# # Step 4: Form binary images (masks) for each tissue class
# tissue1_mask = (segmented_img == 0)
# tissue2_mask = (segmented_img == 1)
# tissue3_mask = (segmented_img == 2)

# # Convert the boolean masks to binary images (0 and 255)
# tissue1_binary = np.uint8(tissue1_mask) * 255
# tissue2_binary = np.uint8(tissue2_mask) * 255
# tissue3_binary = np.uint8(tissue3_mask) * 255

# # Step 5: Create a color overlay
# # Create an empty color image (3 channels for RGB)
# segmented_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

# # Assign colors to each tissue type (RGB values)
# segmented_image[tissue1_mask] = [255, 0, 0]    # Red for tissue 1
# segmented_image[tissue2_mask] = [0, 255, 0]    # Green for tissue 2
# segmented_image[tissue3_mask] = [0, 0, 255]    # Blue for tissue 3

# # Convert the original grayscale image to RGB format
# img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

# # Create a transparent overlay
# # Ensure both images are of the same data type (uint8)
# img_rgb = img_rgb.astype(np.uint8)
# segmented_image = segmented_image.astype(np.uint8)

# # Blend the images
# overlay = cv2.addWeighted(img_rgb, 0.7, segmented_image, 0.3, 0)

# # Display the results
# plt.figure(figsize=(12, 12))

# plt.subplot(2, 2, 1)
# plt.title("Original Grayscale MRI Image")
# plt.imshow(img, cmap='gray')
# plt.axis('off')

# plt.subplot(2, 2, 2)
# plt.title("Tissue 1 Binary Mask")
# plt.imshow(tissue1_binary, cmap='gray')
# plt.axis('off')

# plt.subplot(2, 2, 3)
# plt.title("Tissue 2 Binary Mask")
# plt.imshow(tissue2_binary, cmap='gray')
# plt.axis('off')

# plt.subplot(2, 2, 4)
# plt.title("Tissue 3 Binary Mask")
# plt.imshow(tissue3_binary, cmap='gray')
# plt.axis('off')

# plt.figure(figsize=(10, 10))
# plt.title("Overlay Segmentation on Original Image")
# plt.imshow(overlay)
# plt.axis('off')

# plt.show()
