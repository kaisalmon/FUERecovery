import cv2
import glob
import numpy as np
from mtcnn import MTCNN
from skimage.transform import match_histograms
import imageio
import datetime
from datetime import timedelta
import hashlib
from scipy import interpolate

model = MTCNN()

imageDir = "./images/day*.*"
cascPath = "./cascade.xml"
target_face_size = 600
fps = 6
skip_first_n = 0

faceCascade = cv2.CascadeClassifier(cascPath)


def get_digest(file_path='./run.py'):
    h = hashlib.sha256()

    with open(file_path, 'rb') as file:
        while True:
            # Reading is buffered, so we can read smaller chunks.
            chunk = file.read(h.block_size)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def process_image(imagePath, number):
    meta_path = imagePath.replace('.jpg', '.hash').replace('images', 'output')
    hash = read_metafile(meta_path)

    if hash == get_digest():
        print("Using cache", meta_path)
        output_file = imagePath.replace("images", "output").replace('jpg', 'png')
        return cv2.imread(output_file)

    image = cv2.imread(imagePath)
    image = first_step(image)
    image = crop_image(image, 80)
    image = add_text(image, number)

    with open(meta_path, "w") as metafile:
        metafile.write(get_digest())

    return image


def read_metafile(meta_path):
    try:
        with open(meta_path, 'r') as file:
            hash = file.read()
        return hash
    except FileNotFoundError:
        return None


def first_step(image):
    image = detect_face(image)
    image = scale_image(image, 50)
    # image = whitepatch_balancing(image.astype(np.float32), (375, 390), (170,170,220), 1)
    image = whitepatch_balancing(image.astype(np.float32), (150, 390), (200, 200, 200), 1)
    return image


def add_text(image, number):
    th = 60
    cv2.rectangle(image, (0, 0), (image.shape[1], th), (0, 0, 0), -1)
    # fontScale
    fontScale = 1

    org = (8, th - 16)
    color = (255, 255, 255)
    thickness = 2

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "Day " + f'{number:03}', org, font, fontScale,
                color, thickness, cv2.LINE_AA, False)

    date = datetime.datetime.strptime('2021-05-16', '%Y-%m-%d') + timedelta(days=number)
    org = (370, th - 16)
    cv2.putText(image, str(date.date()), org, font, fontScale,
                color, thickness, cv2.LINE_AA, False)

    return image


def whitepatch_balancing(image, input_coords, target_color, strength):
    patch_size = 5
    image = image.clip(0, 255)

    image_patch = image[input_coords[0] - patch_size:input_coords[0] + patch_size,
                  input_coords[1] - patch_size:input_coords[1] + patch_size]

    print("input min:", image_patch.min(axis=0).min(axis=0))
    print("input mean:", image_patch.mean(axis=0).mean(axis=0))
    print("input max:", image_patch.max(axis=0).max(axis=0))


    input_color = image_patch.mean(axis=0).mean(axis=0)

    multiplier = target_color / input_color

    print("input_color", input_color)
    print("multiplier", multiplier)
    multiplier_image = np.copy(image).astype(np.float32)
    multiplier_image[::] = multiplier

    multiplied = cv2.multiply(image.astype(np.float32), multiplier_image)
    result = (multiplied * strength) + (1 - strength) * image
    result = result.clip(0, 255)


    return result.astype(np.uint8)


def crop_image(image, pixels):
    y = pixels
    x = pixels
    h = image.shape[0] - pixels * 2
    w = image.shape[1] - pixels * 2
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
        raise Exception("Couldn't detect face")
    face = faces[0]['box']
    (x, y, w, h) = face
    # print(x,y,w,h)

    if False:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for key in faces[0]['keypoints']:
            print(key)
            (fx, fy) = faces[0]['keypoints'][key]
            cv2.rectangle(image, (fx - 3, fy - 3), (fx + 3, fy + 3), (0, 255, 0), 2)

    avg_face_dim = (h + w) / 2;
    scale = target_face_size / avg_face_dim
    dx = (-x - w / 2) * scale + image.shape[1] / 2
    dy = (-y - h / 2) * scale + image.shape[0] / 2

    M = np.float32([
        [scale, 0, dx],
        [0, scale, dy]
    ])

    scaled_and_centered = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]));

    return scaled_and_centered


images = glob.glob(imageDir)[skip_first_n:]

if False:
    with imageio.get_writer("output/input.gif", mode="I", fps=fps) as writer:
        idx = skip_first_n
        for path in images:
            frame_length = 1
            if idx == 0 or idx == len(images) - 1:
                frame_length = 4
            image = scale_image(cv2.imread(path), 50)
            for i in range(0, frame_length):
                writer.append_data(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            idx += 1

with imageio.get_writer("output/output.gif", mode="I", fps=fps) as writer:
    idx = skip_first_n
    for path in images:
        print(path)
        frame_length = 1
        if idx == skip_first_n or idx == len(images) + skip_first_n - 1:
            frame_length = 4
        image = process_image(path, idx + 1)
        cv2.imwrite(path.replace("images", "output").replace('jpg', 'png'), image)
        for i in range(0, frame_length):
            writer.append_data(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        idx += 1
