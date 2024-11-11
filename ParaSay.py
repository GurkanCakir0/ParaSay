import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = 'para.png'
image = cv2.imread(img_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gaus = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(gaus, 200, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((5, 5), np.uint8)
morph = cv2.dilate(thresh, kernel, iterations=7)
morph = cv2.erode(morph, kernel, iterations=5)

contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
coin = { "1 TL":59206.0, "50 Kuruş":46804.5, "25 Kuruş":31886.0,"10 Kuruş":19778.0, "5 Kuruş":10094.5 }

say = 0
min_contour_area = 500
coin_count = 0

for cnt in contours:
    #print(f"Kontur Alanı: {cv2.contourArea(cnt)}")
    if cv2.contourArea(cnt) > min_contour_area:
        coin_count += 1
        x, y, w, h = cv2.boundingRect(cnt)

        if cv2.contourArea(cnt) >= coin["1 TL"]:
            value = 1.0
            label = "1 TL"
        elif cv2.contourArea(cnt) >= coin["50 Kuruş"]:
            value = 0.5
            label = "50 Kuruş"
        elif cv2.contourArea(cnt) >= coin["25 Kuruş"]:
            value = 0.25
            label = "25 Kuruş"
        elif cv2.contourArea(cnt) >= coin["10 Kuruş"]:
            value = 0.1
            label = "10 Kuruş"
        elif cv2.contourArea(cnt) >= coin["5 Kuruş"]:
            value = 0.05
            label = "5 Kuruş"
        else:
            continue

        say += value
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


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
axes[4].set_title(f'Para Miktarı: {say}')
axes[4].axis("off")

plt.show()
