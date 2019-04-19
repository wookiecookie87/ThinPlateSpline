import os
import cv2
import numpy as np

def overlapMask(output_fg):
    output_fg_mask = output_fg.copy()
    output_fg_mask[np.where((output_fg_mask != [0, 0, 0]).all(axis=2))] = (255, 255, 255)
    output_fg_mask = cv2.cvtColor(output_fg_mask, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((5, 5), np.uint8)
    output_fg_mask = cv2.erode(output_fg_mask, kernel, iterations=1)
    output_fg = cv2.bitwise_and(output_fg, output_fg, mask=output_fg_mask)

    output_bg_mask = cv2.bitwise_not(output_fg_mask)
    output_bg = cv2.bitwise_and(hand, hand, mask=output_bg_mask)

    return output_fg + output_bg





idx = 0

hand = cv2.imread('sample/hand/0000.jpg', 1)
nail_texture = cv2.imread('resource/designs/d_28.png', 1)

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
    dy = 60

    # sourceShape = np.array ([[0, 0],  [120, 0],  [135, 0], [150, 0], [270, 0],d
    #                          [0, 521], [120, 521],  [135, 521], [150, 521], [270, 521]], np. float32)
    #
    # targetShape = np.array ([[-30, -30],  [120, 0],  [135, 0], [150, 0],   [300, -30],
    #                         [-30, 551],  [120, 521], [135, 521], [150, 521],   [300, 551]], np. float32)

    sourceShape = np.array ([[0, 0], [x_middle-(dx_middle+50), 0], [x_middle-dx_middle, 0],  [x_middle, 0], [x_middle+dx_middle, 0], [x_middle+(dx_middle+50), 0], [x, 0],

                             [0, y_middle+30], [0, y_middle-30], [x, y_middle-30], [x, y_middle+30],

                             [0, y_middle + 100], [0, y_middle - 100], [x, y_middle - 100], [x, y_middle + 100],

                             # [0, y_middle + 150], [0, y_middle - 150], [x, y_middle - 150], [x, y_middle + 150],

                             [0, y], [x_middle-(dx_middle+50), y], [x_middle-dx_middle, y],  [x_middle, y], [x_middle+dx_middle, y], [x_middle+(dx_middle+50), y], [x, y]], np. float32)

    targetShape = np.array ([[0-dx, 0-dy], [x_middle-(dx_middle+50), -17],  [x_middle-dx_middle, 0],
                             [x_middle, 0],
                             [x_middle+dx_middle, 0],   [x_middle+(dx_middle+50), -17],  [x+dx, -dy],

                             [-3, y_middle + 30], [-3, y_middle - 30], [x+3, y_middle - 30], [x+3, y_middle + 30],
                             [-6, y_middle + 100], [-6, y_middle - 100], [x+6, y_middle - 100], [x+6, y_middle + 100],
                             # [-6, y_middle + 150], [-6, y_middle - 150], [x+6, y_middle - 150], [x+6, y_middle + 150],

                            [0-dx, y+dy],  [x_middle-(dx_middle+50), y+17],  [x_middle-dx_middle, y],
                             [x_middle, y],
                             [x_middle+dx_middle, y], [x_middle+(dx_middle+50), y+17],  [x+dx, y+dy]], np. float32)


    sourceShape = sourceShape.reshape(1, -1, 2)
    targetShape = targetShape.reshape(1, -1, 2)
    matches = list()
    for i in range(sourceShape.size):
        matches.append(cv2.DMatch (i,i,0))
    tps.estimateTransformation (sourceShape, targetShape, matches)
    tps.applyTransformation (sourceShape)
    out_img = tps.warpImage(texture)

    # cv2.circle(hand, (330, 160), 3, (0, 0, 255), -1)
    # cv2.circle(hand, (298, 190), 3, (0, 0, 255), -1)
    # cv2.circle(hand, (319, 224), 3, (0, 0, 255), -1)
    # cv2.circle(hand, (368, 220), 3, (0, 0, 255), -1)
    # cv2.circle(hand, (364, 183), 3, (0, 0, 255), -1)
    #
    # cv2.circle(hand, (130, 285), 3, (0, 0, 255), -1)
    # cv2.circle(hand, (108, 320), 3, (0, 0, 255), -1)
    # cv2.circle(hand, (123, 360), 3, (0, 0, 255), -1)
    # cv2.circle(hand, (179, 350), 3, (0, 0, 255), -1)
    # cv2.circle(hand, (169, 298), 3, (0, 0, 255), -1)


    t_height, t_width = nail_texture.shape[:2]

    t_x = t_width - 1
    t_y = t_height - 1

    cv2.circle(nail_texture, (int(t_x/2), 0), 3, (0, 0, 255), -1)
    cv2.circle(nail_texture, (3, 80), 3, (0, 0, 255), -1)
    cv2.circle(nail_texture, (18, t_y-1-18), 3, (0, 0, 255), -1)
    cv2.circle(nail_texture, (t_x-1-18, t_y-1-18), 3, (0, 0, 255), -1)
    cv2.circle(nail_texture, (t_x-3, 80), 3, (0, 0, 255), -1)

    cv2.imshow("nail_texture", nail_texture)

    src_pts = np.float32([[int(t_x/2), 0], [27, 27], [18, t_y-1-18], [t_x-1-18, t_y-1-18], [t_x-27, 27]])
    dst_pts = np.float32([[330, 160], [298, 190], [319, 224], [368, 220], [364, 183]])
    dst_pts_2 = np.float32([[130, 285], [108, 320], [123, 360], [179, 350], [169, 298]])


    M, mask = cv2.findHomography(src_pts, dst_pts)
    M_2, mask_2 = cv2.findHomography(src_pts, dst_pts_2)

    im_out = cv2.warpPerspective(nail_texture, M, (hand.shape[1], hand.shape[0]))
    im_out_2 = cv2.warpPerspective(nail_texture, M_2, (hand.shape[1], hand.shape[0]))
    homograophy_output_fg = im_out + im_out_2


    M_affine = cv2.getAffineTransform(src_pts[1:4], dst_pts[1:4])
    M_affine_2 = cv2.getAffineTransform(src_pts[1:4], dst_pts_2[1:4])

    affine_im_out = cv2.warpAffine(nail_texture, M_affine, (hand.shape[1], hand.shape[0]))
    affine_im_out_2 = cv2.warpAffine(nail_texture, M_affine_2, (hand.shape[1], hand.shape[0]))
    affine_output_fg = affine_im_out + affine_im_out_2


    homography_output = overlapMask(homograophy_output_fg)
    affine_output = overlapMask(affine_output_fg)


    # pt_right = np.float32([[100, 100], [x, 0], [100, y-100], [x, y]])
    # pt_left= np.float32([[0, 0], [x-100, 100], [0, y], [x-100, y-100]])
    # pt_top= np.float32([[0,0 ], [x, 0], [30, y-200], [x-30, y-200]])
    # pt_bottom = np.float32([[30, 200], [x-30, 200], [0, y], [x, y]])


    # M_right = cv2.getPerspectiveTransform(pt1, pt_right)
    # M_left = cv2.getPerspectiveTransform(pt1, pt_left)
    # M_top = cv2.getPerspectiveTransform(pt1, pt_top)
    # M_bottom = cv2.getPerspectiveTransform(pt1, pt_bottom)
    #
    #
    # result_right = cv2.warpPerspective(out_img, M_right, (width, height))
    # result_left = cv2.warpPerspective(out_img, M_left, (width, height))
    # result_top = cv2.warpPerspective(out_img, M_top, (width, height))
    # result_bottom = cv2.warpPerspective(out_img, M_bottom, (width, height))
    #
    # t_result_right = cv2.warpPerspective(texture, M_right, (width, height))
    # t_result_left = cv2.warpPerspective(texture, M_left, (width, height))
    # t_result_top = cv2.warpPerspective(texture, M_top, (width, height))
    # t_result_bottom = cv2.warpPerspective(texture, M_bottom, (width, height))

    # cv2.imshow("img", texture)
    # cv2.imshow("out_img", out_img)

    cv2.imshow("hand", hand)

    cv2.imshow("homography_output", homography_output)
    cv2.imshow("affine_output", affine_output)
    #cv2.imshow("homography", im_out)
    #cv2.imshow("homography_2", im_out_2)
    #cv2.imshow("homography", im_out_2+im_out)
    #cv2.imshow("affine", affine_im_out + affine_im_out_2)

    #
    # cv2.imshow("t_result_right", t_result_right)
    # cv2.imshow("t_result_left", t_result_left)
    # cv2.imshow("t_result_top", t_result_top)
    # cv2.imshow("t_result_bottom", t_result_bottom)

    #cv2.imwrite(texture_names[idx], out_img)

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
