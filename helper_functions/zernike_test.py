img = cv2.imread('test.png',1)
gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

a = MultiHarrisZernike()
kp, des = a.detectAndCompute(gr)

outImage	 = cv2.drawKeypoints(gr, kp, gr,color=[255,255,0],
                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
fig, ax= plt.subplots(dpi=200)
plt.title('Multiscale Harris with Zernike Angles')
plt.axis("off")
plt.imshow(outImage)
plt.show()
# -*- coding: utf-8 -*-

