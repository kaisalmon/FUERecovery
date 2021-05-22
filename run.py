import cv2
import glob
import numpy as np
from mtcnn import MTCNN
from skimage.transform import match_histograms
import imageio
import datetime
from datetime import timedelta


model = MTCNN()

imageDir = "./images/*.*"
cascPath = "./cascade.xml"
ref_image_path = "./images/day4.jpg"
target_face_size = 600
fps = 6

faceCascade = cv2.CascadeClassifier(cascPath)


def process_image(imagePath, number):
    image = cv2.imread(imagePath)
    image = first_step(image)
    image = color_match(image, reference_image, 0)
    image = crop_image(image, 80)
    image = add_text(image, number)
    return image


def first_step(image):
    image = detect_face(image)
    image = scale_image(image, 50)
    image = whitepatch_balancing(image, 200, 100, 40, 40, 0.7)
    return image


def add_text(image, number):
    th = 60
    cv2.rectangle(image, (0,0), (image.shape[1], th), (0, 0, 0), -1)
    # fontScale
    fontScale = 1

    org = (8, th-16)

    # Red color in BGR
    color = (1.,1.,1.)

    # Line thickness of 2 px
    thickness = 2

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "Day "+f'{number:03}', org, font, fontScale,
                        color, thickness, cv2.LINE_AA, False)

    date = datetime.datetime.strptime('2021-05-16', '%Y-%m-%d') + timedelta(days=number)
    org = (370, th-16)
    cv2.putText(image, str(date.date()), org, font, fontScale,
                        color, thickness, cv2.LINE_AA, False)

    return image

def whitepatch_balancing(image, from_row, from_column, row_width, column_width, strength):

    image_patch = image[from_row:from_row+row_width,
                        from_column:from_column+column_width]


    whitepoint = image_patch.max(axis=0).mean(axis=0)

    whitepoint_image = np.copy(image)
    whitepoint_image[:] = whitepoint


    white_balanced = cv2.divide(image.astype(np.float32), whitepoint_image.astype(np.float32))
    return (white_balanced * strength) + (1 - strength) / 255 * image.astype(np.float32)


def crop_image(image, pixels):
    y = pixels
    x = pixels
    h = image.shape[0] - pixels*2
    w = image.shape[1] - pixels*2
    return image[y:y + h, x:x + w]

def color_match(image, reference, strength):
    color_matched = match_histograms(image, reference, multichannel=True)
    beta = 1.0 - strength
    image = cv2.addWeighted(color_matched, strength, image, beta, 0.0)
    return image

def scale_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def detect_face(image):
    faces = model.detect_faces(image)
    print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    if len(faces) == 0:
        return image
        #raise Exception("Couldn't detect face")
    face = faces[0]['box']
    (x, y, w, h) = face
    #print(x,y,w,h)

    #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    avg_face_dim = (h+w)/2;
    scale = target_face_size/avg_face_dim
    print("scale", scale)

    M2 = np.float32([
        [scale, 0, 0],
        [0, scale, 0],
    ])

    scaled = cv2.warpAffine(image, M2, (image.shape[1], image.shape[0]));

    dx = (-x-w/2) * scale + image.shape[1]/2
    dy = (-y-h/2) * scale + image.shape[0]/2

    M = np.float32([
        [1, 0, dx],
        [0, 1, dy]
    ])

    scaled_and_centered = cv2.warpAffine(scaled, M, (image.shape[1], image.shape[0]));


    return scaled_and_centered

reference_image = first_step(cv2.imread(ref_image_path))
images = glob.glob(imageDir)

if True:
    with imageio.get_writer("output/input.gif", mode="I", fps=fps) as writer:
        idx = 0
        for path in images:
            frame_length = 1
            if idx == 0 or idx == len(images) - 1:
                frame_length = 4
            image = scale_image(cv2.imread(path), 50)
            for i in range(0, frame_length):
                writer.append_data(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            idx += 1

with imageio.get_writer("output/output.gif", mode="I",  fps=fps) as writer:
    idx = 0
    for path in images:
        frame_length = 1
        if idx == 0 or idx == len(images) - 1:
            frame_length = 4
        image = process_image(path, idx+1) * 255
        cv2.imwrite(path.replace("images", "output"), image)
        for i in range(0, frame_length):
            writer.append_data(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        idx += 1
