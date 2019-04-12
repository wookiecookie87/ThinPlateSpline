import os
import cv2
import numpy as np


idx = 160

while 1:

    texture_dir = 'sample/textures'
    texture_names = os.listdir('sample/textures')

    texture = cv2.imread(os.path.join(texture_dir, texture_names[idx]), 1) # bgr
    height, width = texture.shape[:2]
    tps = cv2.createThinPlateSplineShapeTransformer()

    x = width - 1
    x_middle = int(width/2)

    dx_middle = 20

    y = height - 1
    y_middle = int(height/2)

    dx = 20
    dy = 70

    # sourceShape = np.array ([[0, 0],  [120, 0],  [135, 0], [150, 0], [270, 0],d
    #                          [0, 521], [120, 521],  [135, 521], [150, 521], [270, 521]], np. float32)
    #
    # targetShape = np.array ([[-30, -30],  [120, 0],  [135, 0], [150, 0],   [300, -30],
    #                         [-30, 551],  [120, 521], [135, 521], [150, 521],   [300, 551]], np. float32)

    sourceShape = np.array ([[0, 0], [x_middle-(dx_middle+50), 0], [x_middle-dx_middle, 0],  [x_middle, 0], [x_middle+dx_middle, 0], [x_middle+(dx_middle+50), 0], [x, 0],

                             [0, y_middle+30], [0, y_middle-30], [x, y_middle-30], [x, y_middle+30],
                             [0, y_middle + 100], [0, y_middle - 100], [x, y_middle - 100], [x, y_middle + 100],
                             [0, y_middle + 150], [0, y_middle - 150], [x, y_middle - 150], [x, y_middle + 150],
                             [0, y], [x_middle-(dx_middle+50), y], [x_middle-dx_middle, y],  [x_middle, y], [x_middle+dx_middle, y], [x_middle+(dx_middle+50), y], [x, y]], np. float32)

    targetShape = np.array ([[0-dx, 0-dy], [x_middle-(dx_middle+50), -10],  [x_middle-dx_middle, 0],  [x_middle, 0], [x_middle+dx_middle, 0],   [x_middle+(dx_middle+50), -10],  [x+dx, -dy],

                             [0, y_middle + 30], [0, y_middle - 30], [x, y_middle - 30], [x, y_middle + 30],
                             [-3, y_middle + 100], [-3, y_middle - 100], [x+3, y_middle - 100], [x+3, y_middle + 100],
                             [-6, y_middle + 150], [-6, y_middle - 150], [x+6, y_middle - 150], [x+6, y_middle + 150],
                            [0-dx, y+dy],  [x_middle-(dx_middle+50), y+10],  [x_middle-dx_middle, y], [x_middle, y], [x_middle+dx_middle, y], [x_middle+(dx_middle+50), y+10],  [x+dx, y+dy]], np. float32)

    sourceShape = sourceShape.reshape(1, -1, 2)
    targetShape = targetShape.reshape(1, -1, 2)
    matches = list ()
    for i in range(sourceShape.size):
        matches.append(cv2.DMatch (i,i,0))
    tps.estimateTransformation (sourceShape, targetShape, matches)
    tps.applyTransformation (sourceShape)
    out_img = tps.warpImage(texture)

    cv2.imshow("img", texture);
    cv2.imshow("out_img", out_img);

    cv2.imwrite(texture_names[idx], out_img)

    # grayscale = cv2.cvtColor(out_img, cv2.COLOR_BGR2GRAY)
    #
    # thresholded = cv2.threshold(grayscale, 0, 255, cv2.THRESH_OTSU)
    #
    # bbox = cv2.boundingRect(thresholded)
    # x, y, w, h = bbox
    # print(bbox)
    # foreground = out_img[y:y + h, x:x + w]
    # cv2.imshow("out_img", out_img);

    k = cv2.waitKey(0)
    if k == 27:
        break
    elif k == ord('v'):
        idx += 1
    elif k == ord('c'):
        idx -= 1

    if idx < 0:
        idx = len(texture_names) - 1
    if idx >= len(texture_names):
        idx = 0

cv2.destroyAllWindows()
