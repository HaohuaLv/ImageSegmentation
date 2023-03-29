A simple implementation of Image Segmentation.

1. Run `!python ImageSegmentation.py`.
2. Press `~`(cv2.GC_BGD, dark green), `1`(cv2.GC_FGD, dark red), `2`(cv2.GC_PR_BGD, light green) and `3`(cv2.GC_PR_FGD, light red) to modify the mask brush and brush ROI. (Brush size can be adjusted with the mouse wheel.)
3. Press `Table` to implement cv2.grabCut().
4. Repeat step2~3 until output is satisfied.
5. Press `Enter` to save output.png and masked_img.png. 
![Image text](https://github.com/HaohuaLv/ImageSegmentation/blob/master/input.png)
![Image text](https://github.com/HaohuaLv/ImageSegmentation/blob/master/masked_img.png)
![Image text](https://github.com/HaohuaLv/ImageSegmentation/blob/master/output.png)
