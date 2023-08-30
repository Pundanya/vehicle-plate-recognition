import matplotlib.pyplot as plt
import pytesseract
import cv2


def open_img(img_path):
    car_plate_img = cv2.imread(img_path)
    car_plate_img = cv2.cvtColor(car_plate_img, cv2.COLOR_BGR2RGB)

    return car_plate_img


def car_plate_extract(image, car_plate_haar_cascade):
    car_plate_rects = car_plate_haar_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)

    for x, y, w, h in car_plate_rects:
        car_plate_img = image[y+15:y+h-10, x+15:x+w-20]

    return car_plate_img


def enlarge_img(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    return resized_image


def main():
    car_plate_img_rgb = open_img(img_path='src/3.jpg')
    car_plate_haar_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

    car_plate_extract_img = car_plate_extract(car_plate_img_rgb, car_plate_haar_cascade)
    car_plate_extract_img = enlarge_img(car_plate_extract_img, 150)

    car_plate_extract_img_gray = cv2.cvtColor(car_plate_extract_img, cv2.COLOR_RGB2GRAY)
    plt.axis('off')
    plt.imshow(car_plate_extract_img_gray, cmap='gray')
    plt.show()

    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    print('Номер автомобиля: ', pytesseract.image_to_string(
        car_plate_extract_img_gray,
        config='--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
          )


if __name__ == '__main__':
    main()
