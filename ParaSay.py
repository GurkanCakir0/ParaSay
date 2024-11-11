import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = 'para.jpg'
image = cv2.imread(img_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gaus = cv2.GaussianBlur(gray, (3, 3), 0)
_, thresh = cv2.threshold(gaus, 150, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((3, 3), np.uint8)
morph = cv2.dilate(thresh, kernel, iterations=5)
morph = cv2.erode(morph, kernel, iterations=5)

contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

min_contour_area = 500
coin_count = 0

for cnt in contours:
    if cv2.contourArea(cnt) > min_contour_area:
        coin_count += 1
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

fig, axes = plt.subplots(1, 5, figsize=(20, 10))

axes[0].imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
axes[0].set_title("Gray Image")
axes[0].axis("off")

axes[1].imshow(cv2.cvtColor(gaus, cv2.COLOR_BGR2RGB))
axes[1].set_title("Gaus Image")
axes[1].axis("off")

axes[2].imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
axes[2].set_title("Thresh Image")
axes[2].axis("off")

axes[3].imshow(cv2.cvtColor(morph, cv2.COLOR_BGR2RGB))
axes[3].set_title("Morphological Image")
axes[3].axis("off")

axes[4].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[4].set_title(f'Para Miktarı: {coin_count}')
axes[4].axis("off")

plt.show()
