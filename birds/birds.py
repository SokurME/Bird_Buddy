import tensorflow as tf
import os
from PIL import Image
import numpy as np
import cv2
# подключение библиотеки для работы с контактами ввода/вывода
import RPi.GPIO as GPIO 
# подключение библиотеки для работы с задержками
import time 
# импорт модуля для телеграмм канала
import telepot

def convert_to_opencv(image):
    # RGB -> BGR conversion is performed as well.
    image = image.convert('RGB')
    r,g,b = np.array(image).T
    opencv_image = np.array([b,g,r]).transpose()
    return opencv_image

def crop_center(img,cropx,cropy):
    h, w = img.shape[:2]
    startx = w//2-(cropx//2)
    starty = h//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]

def resize_down_to_1600_max_dim(image):
    h, w = image.shape[:2]
    if (h < 1600 and w < 1600):
        return image

    new_size = (1600 * w // h, 1600) if (h > w) else (1600, 1600 * h // w)
    return cv2.resize(image, new_size, interpolation = cv2.INTER_LINEAR)

def resize_to_256_square(image):
    h, w = image.shape[:2]
    return cv2.resize(image, (256, 256), interpolation = cv2.INTER_LINEAR)

def update_orientation(image):
    exif_orientation_tag = 0x0112
    if hasattr(image, '_getexif'):
        exif = image._getexif()
        if (exif != None and exif_orientation_tag in exif):
            orientation = exif.get(exif_orientation_tag, 1)
            # orientation is 1 based, shift to zero based and flip/transpose based on 0-based values
            orientation -= 1
            if orientation >= 4:
                image = image.transpose(Image.TRANSPOSE)
            if orientation == 2 or orientation == 3 or orientation == 6 or orientation == 7:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            if orientation == 1 or orientation == 2 or orientation == 5 or orientation == 6:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image
    
def send_bot(obj):
    bot = telepot.Bot('5866226591:AAFHoLsDmMU8siud04rxskTkZCPKllgydcY')
    # Адрес телеграм-канала
    CHANNEL_NAME = '@birds_794'
    img = open("/home/pi/Desktop/Birds/test.png", 'rb')
    bot.sendPhoto(CHANNEL_NAME, img)
    bot.sendMessage(CHANNEL_NAME, obj)
    
f = False
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
TRIGGER=19
ECHO=26
GPIO.setup(TRIGGER, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

def ultra():
    global f
    GPIO.output(TRIGGER, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(TRIGGER, GPIO.LOW)
    while GPIO.input(ECHO) == 0:
        start = time.time()
    while GPIO.input(ECHO) == 1:
        end = time.time()
    timepassed = end - start
    distance = round(timepassed * 17150, 2)
    print("The distance from object is ",distance,"cm")
    if distance < 15 and not f:
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2);
        _, frame = cap.read()
        cv2.imwrite("/home/pi/Desktop/Birds/test.png", frame)
        obj = recognition()
        if obj == "Синица" or obj == "Воробей" or obj == "Соловей":
            send_bot(obj)
            f = True  
    elif distance >= 15 and f:
        f = False

def recognition():
    graph_def = tf.compat.v1.GraphDef()
    labels = []

    # These are set to the default names from exported models, update as needed.
    filename = "/home/pi/Desktop/Birds/model.pb"
    labels_filename = "/home/pi/Desktop/Birds/labels.txt"

    # Import the TF graph
    with tf.io.gfile.GFile(filename, 'rb') as f:
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    # Create a list of labels.
    with open(labels_filename, 'rt') as lf:
        for l in lf:
            labels.append(l.strip())

    # Load from a file
    imageFile = "/home/pi/Desktop/Birds/test.png" # тут ссылка на фото
    image = Image.open(imageFile)

    # Update orientation based on EXIF tags, if the file has orientation info.
    image = update_orientation(image)

    # Convert to OpenCV format
    image = convert_to_opencv(image)

    # If the image has either w or h greater than 1600 we resize it down respecting
    # aspect ratio such that the largest dimension is 1600
    image = resize_down_to_1600_max_dim(image)

    # We next get the largest center square
    h, w = image.shape[:2]
    min_dim = min(w,h)
    max_square_image = crop_center(image, min_dim, min_dim)

    # Resize that square down to 256x256
    augmented_image = resize_to_256_square(max_square_image)

    # Get the input size of the model
    with tf.compat.v1.Session() as sess:
        input_tensor_shape = sess.graph.get_tensor_by_name('Placeholder:0').shape.as_list()
    network_input_size = input_tensor_shape[1]

    # Crop the center for the specified network_input_Size
    augmented_image = crop_center(augmented_image, network_input_size, network_input_size)

    # These names are part of the model and cannot be changed.
    output_layer = 'loss:0'
    input_node = 'Placeholder:0'

    with tf.compat.v1.Session() as sess:
        try:
            prob_tensor = sess.graph.get_tensor_by_name(output_layer)
            predictions = sess.run(prob_tensor, {input_node: [augmented_image] })
        except KeyError:
            print ("Couldn't find classification output layer: " + output_layer + ".")
            print ("Verify this a model exported from an Object Detection project.")
            exit(-1)

    # Print the highest probability label
        highest_probability_index = np.argmax(predictions)
       # print('Classified as: ' + labels[highest_probability_index])
       # print()
       # print(highest_probability_index)
       # if highest_probability_index < 50:
       #     labels[highest_probability_index] = "None"
        # Or you can print out all of the results mapping labels to probabilities.
        label_index = 0
        for p in predictions:
            truncated_probablity = np.float64(np.round(p,8))
            #print (labels[label_index], truncated_probablity)
            label_index += 1
        print(labels[highest_probability_index])
    return (labels[highest_probability_index])



while True:
    ultra()
    time.sleep(1)

 
IO.setwarnings(False)
# отключаем показ любых предупреждений
IO.setmode (IO.BCM)
# мы будем программировать контакты GPIO по их функциональным номерам 
#(BCM), то есть мы будем обращаться к PIN29 как ‘GPIO5’
IO.setup(17,IO.OUT)
# инициализируем GPIO17 в качестве цифрового выхода
p = IO.PWM(17,50)
# инициализируем GPIO17 как контакт для формирования ШИМ сигнала с
#частотой 50 Гц
IO.setup(4,IO.OUT)
# инициализируем GPIO4 в качестве цифрового выхода
q = IO.PWM(4,50)
# инициализируем GPIO4 как контакт для формирования ШИМ сигнала с
#частотой 50 Гц
p.start(7.5)
# генерируем ШИМ сигнал с коэффициентом заполнения 7.5%
q.start(7.5)
# генерируем ШИМ сигнал с коэффициентом заполнения 7.5%
while 1:
    # бесконечный цикл
    p.ChangeDutyCycle(6)
    # изменяем коэффициент заполнения чтобы повернуть сервомотор в
    # положение 90º
    q.ChangeDutyCycle(6)
    time.sleep(3)
    p.ChangeDutyCycle(1)
    # изменяем коэффициент заполнения чтобы повернуть сервомотор в
    #положение 0º
    q.ChangeDutyCycle(1)
    time.sleep(5)
    # задержка на 1 секунду

