import cv2

Isport numpy as np isport matplotlib.pyplot as pit

#Load the leage

ing cv2.imread("/content/image for 4.jpg, 0)

Apply Canny edge detection

edges canny cv2.Canny (ing, 100, 200)

Apply Sobel edge detection

sobelx cv2.5obel(ing, cv2.CV 64F, 1, 8, ksize-5) sobely cv2.Sobel(ing, cv2.CV 64F, 8, 1, ksize-5)

edges sobel np.sqrt(np.square(sobelk) np.square(sobely)) edges sabel cv2.normalize(edges sobel, None, alpha-e, beta-255, nonn type-cv2.NORM MINMAX) edges sobel edges sobel.astype(np.uint8)

#Apply Laplacian edge detection

edges laplacian cv2.normalize(edges laplacian, None, alpha-e, beta-255, norm type-cv2.NORM MINNAX) edges laplacian edges laplacian.astype(np.uints)

edges laplacian cv2.Laplacian(img, cv2.CV_64F)

Display the Images and corpare the results

plt.subplot(2, 2, 1)
plt.imshow(ing, cпap "gray")

plt.title('Original Image') 
plt.sticks([]), plt.yticks([])

plt.subplot(2,2,2) 
plt.imshow(edges canny, cmap-gray") 
plt.title("Canny Edge Detection')

plt.xticks([]), plt.yticks(11)

plt.subplot(2,2,3) 
pit.imshow(edges_sobel, cnap-gray")

plt.title('Sobel Edge Detection')
plt.xticks([]), plt.yticks(())

plt.subplot(2,2, 6)
plt.imshow(edges laplacian, cmap-gray") 
plt.title('Laplacian Edge Detection') 
plt.xticks([]), plt.yticks([])

plt.tight layout() 
plt.show()
