import cv2
import glob
import numpy as np
from skimage.io import imread, imsave
from skimage import exposure
from skimage.transform import match_histograms

imageDir = "./images/*.*"
cascPath = "./cascade.xml"
ref_image_path = "./images/day4.jpg"
reference_image = cv2.imread(ref_image_path)
target_face_size = 360


faceCascade = cv2.CascadeClassifier(cascPath)


def process_image(imagePath):
    image = cv2.imread(imagePath)
    image = scale_image(image, 50)
    image = detect_face(image)
    image = color_match(image, reference_image, 0.3)
    image = crop_image(image, 80)
    return image

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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        minNeighbors=1,
        minSize=(300, 300),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    if len(faces) == 0:
        raise Exception("Couldn't detect face")
    face = faces[0]
    (x, y, w, h) = face
    #print(x,y,w,h)

    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

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

images = glob.glob(imageDir)
print(images)
for path in images:
    image = process_image(path)
    cv2.imshow("Faces found", image)
    cv2.waitKey(1)

cv2.waitKey(100)