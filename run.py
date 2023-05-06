import cv2
import sys
from mBEST import mBEST
from mBEST.color_masks import detect_pink_green_teal

dlo_seg = mBEST()

img = cv2.imread(sys.argv[1])

mask = detect_pink_green_teal(img)

dlo_seg.set_image(img)
paths, path_img = dlo_seg.run(mask, plot=True)

# dlo_seg.run(mask, intersection_color=[255, 0, 0], plot=True)
# dlo_seg.run(mask, save_fig=True, save_path="", save_id=0)
