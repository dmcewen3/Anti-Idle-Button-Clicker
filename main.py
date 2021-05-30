from PIL import ImageGrab, ImageOps
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from pyautogui import click
from time import sleep

# this code is based on the OpenCV template matching tutorial
#https://docs.opencv.org/master/d4/dc6/tutorial_py_template_matching.html

#I had the best results with SQDIFF and SQDIFF_NORMED but it works with any method
def main(visualize = False, num_clicks = 100, method = cv.TM_SQDIFF_NORMED, template = 'button_template.jpg'):

    #specify which image to use for the button template
    button_template = cv.imread(template, flags = cv.IMREAD_GRAYSCALE)
    h, w = button_template.shape

    for x in range(0,num_clicks):
        img = screen_shot().copy()

        #match the template to the image
        #todo - add a way to average different methods for better results
        #todo circle matching
        res = cv.matchTemplate(img, button_template, method)
        
        #get the results of the match
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right, center, upper_center, lower_center = get_values(w, h, top_left)
        
        if visualize:
            visualization(res, img, method, top_left, bottom_right, upper_center, lower_center)
            break
        else:
            do_click(center, 0.03)
        

#get a screenshot, convert to np.array, make a copy
#todo grab anti Idle screen and take screen shots from within there to make more efficient
def screen_shot() -> np.array:
    screen_grab_image = ImageOps.grayscale(ImageGrab.grab())
    screen_grab = np.array(screen_grab_image, dtype = np.uint8)

    return screen_grab

#get values needed for drawing rectangles and clicking the center
def get_values(w, h, top_left) -> int:
    bottom_right = (top_left[0] + w, top_left[1] + h)
    center = (int((top_left[0] + bottom_right[0]) / 2), int((top_left[1] + bottom_right[1]) / 2))
    upper_center = (center[0] - 1, center[1] - 1)
    lower_center = (center[0] + 1, center[1] + 1)
    
    return bottom_right, center, upper_center, lower_center

#click the center of the match, adjust delay to click when button is not moving.
#pull mouse to corner of screen to force quit clicks
def do_click(center, delay):
    click(x = center[0], y = center[1])
    sleep(delay)

#display results of matching
def visualization(res, img, method, top_left, bottom_right, upper_center, lower_center):
    cv.rectangle(img, top_left, bottom_right, 255, 2)
    cv.rectangle(img, upper_center, lower_center, 255, 2)
    plt.subplot(121),plt.imshow(res, cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img, cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(method)
    print(type(method))
    plt.show()

if __name__ == '__main__':
    main()