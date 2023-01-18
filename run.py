import sys
from mBEST import mBEST
from mBEST.color_masks import *

dlo_seg = mBEST(epsilon=40, delta=25)

img = cv2.imread(sys.argv[1])

mask = detect_color_pink_and_green(img, hw=75)

mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
kernel = np.ones((3, 3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

dlo_seg.set_image(img)
# paths = dlo_seg.run(mask)
dlo_seg.run(mask, plot=True)
# dlo_seg.run(mask, intersection_color=[255, 0, 0], plot=True)
