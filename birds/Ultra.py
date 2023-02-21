import RPi.GPIO as GPIO
import time
import cv2
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
        cap = cv2.VideoCapture(0);
        _, frame = cap.read()
        cv2.imwrite("/home/pi/Documents/Birds/Bird.png", frame)
        f = True
    elif distance >= 15 and f:
        f = False
while True:
    ultra()
    time.sleep(1)

