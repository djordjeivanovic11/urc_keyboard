import cv2
import sys
import pytesseract
import numpy as np
import math
# https://www.youtube.com/watch?v=oXlwWbU8l2o OPENCV TUTORIAL, until the Advanced part

# setting up pytesseract, check the tesseract installation path on your system and update for your OS
# WINDOWS_TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if sys.platform == "darwin":
    pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# this function connects to a webcam or external camera which can be set up using the index
def connect_to_webcam(camera_index=0, resolution=(1920, 1080), fps=60):
    cap = cv2.VideoCapture(camera_index)

    # we check if the camera opens correctly
    if not cap.isOpened():
        sys.exit(f'error: could not open webcam at index {camera_index}.')

    # set the camera's resolution and frame rate
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap

# this function captures a frame from the webcam at a certain point in time
def capture_frame(cap):
    ret, frame = cap.read()
    # check if the frame was captured correctly
    if not ret:
        sys.exit('error: could not capture frame.')
    return frame

"""
 This function attempts to detect the keyboard in the captured frame. It is not perfect
    and may need some tweaking depending on the lighting conditions and camera angle. This maybe because
    cameras on the laptop are generally not very good and may not be able to detect the keyboard properly. My keyboard is also 
    quite reflective. This will be great for testing.
"""
"""https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html for more information on thresholding"""
def detect_keyboard(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale for easier processing
    # maybe counter inturtive but it seems that this is the best way to detect the keyboard as seen in docs
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # reduce noise with a blur
    # apply adaptive thresholding to detect edges and shapes
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    # find contours in the thresholded image to locate the keyboard shape
    """https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html for more information on contours"""
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keyboard_contour = None
    max_area = 0  # initialize maximum area found

    # loop through contours to find the largest one that resembles a rectangle
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 5000:  # skip small areas
            continue
        # approximate the contour to a polygon
        peri = cv2.arcLength(contour, True)
        # approxPolyDP is used to approximate the polygonal curves with a specific precision
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        # check if the contour has 4 corners and a larger area than the previous becuase we are looking for a rectangle
        # but there might be rectangles in the backgorund. It might even pick up a specific key on the keyboard as the entire keyboard
        # in case where the keyboard is very close
        if len(approx) == 4 and area > max_area:  # check for rectangular shape
            keyboard_contour = approx
            max_area = area
    return keyboard_contour # first step in finding specific letters and translating that to movement

# this function performs a perspective transform to get a top-down view of the keyboard
def four_point_transform(image, pts):
    """
    The goal here is to convert an image of the keyboard taken at an angle
    to a flat, top-down view which will make character detection easier and more accurate,
    since it removes the distortions from the camera angle.
    """

    # create an empty array to hold the reordered points
    rect = np.zeros((4, 2), dtype="float32")
    
    # sum of the coordinates identifies the top-left and bottom-right corners
    # the smallest sum corresponds to the top-left, and the largest sum is the bottom-right
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left corner
    rect[2] = pts[np.argmax(s)]  # bottom-right corner

    # the difference between coordinates helps identify the top-right and bottom-left corners
    # the smallest difference corresponds to the top-right, and the largest difference is the bottom-left
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right corner
    rect[3] = pts[np.argmax(diff)]  # bottom-left corner

    """
    now that we have identified the corners in the correct order (top-left, top-right, 
    bottom-right, bottom-left), we can compute the width and height of the new image.
    this helps determine the dimensions of the "warped" (flattened) image.
    """

    # extract the reordered corners for easier calculations
    (tl, tr, br, bl) = rect

    # calculate the width of the new image
    widthA = np.linalg.norm(br - bl)  # bottom width
    widthB = np.linalg.norm(tr - tl)  # top width
    maxWidth = int(max(widthA, widthB))  # use the maximum width to avoid distortion

    # calculate the height of the new image
    heightA = np.linalg.norm(tr - br)  # right height
    heightB = np.linalg.norm(tl - bl)  # left height
    maxHeight = int(max(heightA, heightB))  # use the maximum height to avoid distortion

    """
    we can then define the destination points for perspective transformation, which represent the
    "ideal" top-down view of the keyboard. these points create a new image that is perfectly
    rectangular.
    """

    dst = np.array([
        [0, 0],  # top-left corner
        [maxWidth - 1, 0],  # top-right corner
        [maxWidth - 1, maxHeight - 1],  # bottom-right corner
        [0, maxHeight - 1]  # bottom-left corner
    ], dtype="float32")

    """
    the next step is to compute the perspective transformation matrix 'M' that maps
    the original points (rect) to the desired points (dst). this matrix allows us
    to 'warp' the image into the new perspective.
    https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#gae66ba39ba2e47dd0750555c7e986ab85
    """

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    """
    finally, we resize the warped image to make it clearer and sharper. this step doubles
    the dimensions of the flattened image, making the details (like letters) more visible
    for the next stage of character recognition.
    """

    warped = cv2.resize(warped, (warped.shape[1] * 2, warped.shape[0] * 2), interpolation=cv2.INTER_CUBIC)

    return warped


# this function tries to detect the letters on the keyboard and map their coordinates
def build_coordinate_map(warped_image, passcode_letters, max_retries=10):
    # convert to hls color space and use the lightness channel for better detection
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2HLS_FULL)[:, :, 1]

    """
    The HLS color focuses on lightness channel and helps make the letters stand out, regardless of color.
    this step might be helpful because letters may have different colors, but the lightness channel
    captures intensity. This might be useful since our keyboard for the challenge seems to be very colorful
    """

    # use morphological closing to reduce noise and fill gaps
    morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    """
    this step is important because some letters might appear 
    broken due to noise or irregular lighting, and closing helps merge these parts to make
    the letters more solid.
    """

    # apply thresholding to create a binary image for tesseract
    _, thresh = cv2.threshold(morph, 150, 255, cv2.THRESH_BINARY_INV)

    """
    thresholding converts the image into a binary form (black-and-white). this will 
    make letters stand out clearly against the background. the threshold value may need
    to be adjusted based on the image quality and lighting conditions.
    """

    # configure tesseract to detect only specified letters in the passcode. why detect all letters when we only need a few
    # this should be extended to non letter characters as well
    whitelist = ''.join(sorted(passcode_letters))
    custom_config = f'--oem 3 --psm 6 -c tessedit_char_whitelist={whitelist} -c preserve_interword_spaces=1 '

    coordinate_dict = {}  # this dictionary will store detected letter coordinates
    missing_letters = set(passcode_letters)  # keep track of letters that are still missing since one screening 
    # might not be enough to capture all the letters
    retries = 0  # start the retry counter

    # retry loop to improve detection accuracy
    while missing_letters and retries < max_retries:
        """
        we try multiple times to improve letter detection, as OCR may miss some letters 
        on the first attempt due to noise, lighting, or other factors.
        """

        # run tesseract OCR on the preprocessed image
        data = pytesseract.image_to_data(thresh, config=custom_config, output_type=pytesseract.Output.DICT)
        n_boxes = len(data['level'])  # get the number of detected boxes (regions of text)

        # loop through detected boxes to extract characters and their coordinates
        for i in range(n_boxes):
            char = data['text'][i].strip().upper()
            
            # check if the character is empty or not in the passcode set
            if char == '' or char not in passcode_letters:
                continue

            try:
                conf = int(data['conf'][i])  # get the confidence level of the detected character
            except ValueError:
                continue  # if confidence is not an integer, skip this character

            if conf < 30:  # skip very-low-confidence results to avoid false positives, # this can be adjusted
                # but it is important to note that the higher the confidence the more accurate the detection
                # i put it so that we can at least get some detection, but as seen in annotated_keyboard 
                continue

            """
            for each detected character, tesseract provides its bounding box (x, y, width, height).
            we use this information to calculate the center of the letter and create a bounding box 
            around it.
            """
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            x_center = x + w // 2
            y_center = y + h // 2

            # update coordinate_dict with highest confidence detection for each letter
            if char in coordinate_dict:
                if conf > coordinate_dict[char]['confidence']:
                    coordinate_dict[char] = {
                        'center': (x_center, y_center),  # center coordinates of the letter
                        'box': (x, y, x + w, y + h),  # bounding box of the letter
                        'confidence': conf  # confidence level of detection
                    }
            else:
                coordinate_dict[char] = {
                    'center': (x_center, y_center),
                    'box': (x, y, x + w, y + h),
                    'confidence': conf
                }
                missing_letters.discard(char)  # mark the letter as found

        retries += 1  # increment the retry counter

        # display the current progress of detection
        print(f"attempt {retries}/{max_retries}: found {len(passcode_letters) - len(missing_letters)} out of {len(passcode_letters)} letters")

    # print final status after all retries
    if missing_letters:
        print(f"detection incomplete after {max_retries} attempts. missing letters: {missing_letters}")
    else:
        print("all passcode letters detected successfully.")

    return coordinate_dict



# this function converts 2d pixel coordinates to 3d real-world coordinates,
# aiding the rover in determining the spatial position of detected letters for URC missions.
def map_to_3d(x_pixel, y_pixel, frame_width, frame_height, fov_horizontal=60, fov_vertical=None, reference_distance=12):
    
    # calculate vertical FOV if not provided, but we will somehow provide this from the electrical
    # using the aspect ratio of the frame
    if fov_vertical is None:
        aspect_ratio = frame_height / frame_width
        fov_vertical = 2 * math.degrees(math.atan(math.tan(math.radians(fov_horizontal) / 2) * aspect_ratio))

    # convert horizontal and vertical FOV to radians for further calculations, docs for more information
    fov_h_rad = math.radians(fov_horizontal)
    fov_v_rad = math.radians(fov_vertical)

    # calculate the real-world width and height of the camera's FOV at the reference distance
    real_width = 2 * reference_distance * math.tan(fov_h_rad / 2)
    real_height = 2 * reference_distance * math.tan(fov_v_rad / 2)

    # determine scaling factors to convert pixels to real-world units
    scale_x = real_width / frame_width
    scale_y = real_height / frame_height

    # calculate the real-world x-coordinate, centering it around the image's midpoint
    x_centered = (x_pixel - frame_width / 2) * scale_x

    # calculate the real-world y-coordinate, centering it and inverting the y-axis
    y_centered = (frame_height / 2 - y_pixel) * scale_y

    # set the z-coordinate to the reference distance, representing the depth
    z = reference_distance # very much simplified but this is the general idea i guess

    return (x_centered, y_centered, z)


# main function that runs the program and coordinates all the steps
def main():
    cap = connect_to_webcam()  # connect to the webcam
    passcode = 'URCSOFTWARE'  # define the passcode to detect
    passcode_letters = set(passcode)

    # define colors for marking each detected letter
    colors = [
        (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
        (255, 0, 255), (255, 255, 0), (128, 0, 128), (0, 128, 128),
        (128, 128, 0), (128, 128, 128), (0, 0, 128)
    ]

    # continuously process frames until all letters are detected
    while True:
        frame = capture_frame(cap)
        original_frame = frame.copy()
        cv2.imshow('webcam', frame)  # display the original webcam feed

        keyboard_contour = detect_keyboard(frame)  # try to detect the keyboard
        if keyboard_contour is not None:
            # draw the detected contour on the original frame
            cv2.drawContours(original_frame, [keyboard_contour], -1, (255, 0, 0), 2)
            cv2.imshow('keyboard detection', original_frame)

            # apply perspective transform for a top-down view of the keyboard
            warped_image = four_point_transform(original_frame, keyboard_contour.reshape(4, 2))
            if warped_image is None:
                print("perspective transform failed. retrying...")
                continue

            # build coordinate map of detected passcode letters
            coordinate_dict = build_coordinate_map(warped_image, passcode_letters)

            # if letters are detected, annotate the image
            if coordinate_dict:
                annotated_image = warped_image.copy()
                passcode_coords_3d = []  # list to store 3d coordinates

                frame_height, frame_width = warped_image.shape[:2]

                # loop through each passcode letter and annotate it on the image
                # i wanted to 
                for idx, char in enumerate(passcode):
                    if char in coordinate_dict:
                        coord_info = coordinate_dict[char]
                        x_center, y_center = coord_info['center']
                        x1, y1, x2, y2 = coord_info['box']

                        # assign a unique color for each letter
                        color = colors[idx % len(colors)]

                        # draw a rectangle around the letter
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                        # label the letter and its position number
                        cv2.putText(annotated_image, f'{char} ({idx+1})', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                        # calculate 3d coordinates
                        x_3d, y_3d, z_3d = map_to_3d(x_center, y_center, frame_width, frame_height)
                        passcode_coords_3d.append((char, x_3d, y_3d, z_3d))

                        # optionally mark the center point
                        cv2.circle(annotated_image, (x_center, y_center), 5, color, -1)
                    else:
                        print(f"letter '{char}' not found.")

                # print the 3d coordinates of detected letters
                print("3d coordinates of passcode letters:")
                for coord in passcode_coords_3d:
                    char, x_3d, y_3d, z_3d = coord
                    print(f"letter '{char}': ({x_3d:.2f}, {y_3d:.2f}, {z_3d:.2f}) inches")

                # save the annotated image
                cv2.imwrite('annotated_keyboard.png', annotated_image)
                print("annotated image saved as 'annotated_keyboard.png'. exiting...")
                break  # exit after successful detection
            else:
                print("no passcode letters found yet. retrying...")
        else:
            print("keyboard not detected. retrying...")

        # press 'q' to exit manually
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()  # close all open windows

if __name__ == "__main__":
    main()
