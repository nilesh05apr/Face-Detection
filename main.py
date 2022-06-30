import cv2 as cv


def detect(img):
    gry_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    face = face_cascade.detectMultiScale(gry_img, 1.05, 5)

    for x, y, w, h in face:
        img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    resized_img = cv.resize(img, (int(img.shape[0] / 2), int(img.shape[1] / 2)))
    return resized_img


face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
img_path = 'images/<image_path>'
img = cv.imread(img_path, 1)
faces = detect(img)
output_path = 'output/'
cv.imwrite(output_path + 'sample output.jpg', faces)