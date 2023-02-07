import cv2 # импорт модуля cv2
import telebot # импорт модуля  для телеграмм канала
bot = telebot.TeleBot('5866226591:AAFHoLsDmMU8siud04rxskTkZCPKllgydcY')
CHANNEL_NAME = '@birds_794' # Адрес телеграм-канала
#cv2.VideoCapture("видеофайл.mp4"); вывод кадров из видео файла
cap = cv2.VideoCapture(0); # видео поток с веб камеры

cap.set(3,1280) # установка размера окна
cap.set(4,700)

ret, frame1 = cap.read()
ret, frame2 = cap.read()

k=0

while cap.isOpened(): # метод isOpened() выводит статус видеопотока
  xmin=0
  ymin=0
  xmax=1
  ymax=1
 
  diff = cv2.absdiff(frame1, frame2) # нахождение разницы двух кадров, которая проявляется лишь при изменении одного из них, т.е. с этого момента наша программа реагирует на любое движение.
 
  gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) # перевод кадров в черно-белую градацию
 
  blur = cv2.GaussianBlur(gray, (5, 5), 0) # фильтрация лишних контуров
 
  _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY) # метод для выделения кромки объекта белым цветом
 
  dilated = cv2.dilate(thresh, None, iterations = 3) # данный метод противоположен методу erosion(), т.е. эрозии объекта, и расширяет выделенную на предыдущем этапе область
  
  сontours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # нахождение массива контурных точек
 
   
  for contour in сontours:
    (x, y, w, h) = cv2.boundingRect(contour) # преобразование массива из предыдущего этапа в кортеж из четырех координат
   
   
    if cv2.contourArea(contour) < 700: # условие при котором площадь выделенного объекта меньше 700 px
      continue
      h=True
    if xmin==0 or x<xmin:
        xmin=x
    if ymin==0 or y<ymin:
        ymin=y
    if x+w>xmax:
        xmax=x+w
    if y+h>ymax:
        ymax=y+h

  frame3=frame1[ymin:ymax,xmin:xmax]
  #cv2.drawContours(frame1, сontours, -1, (0, 255, 0), 2) также можно было просто нарисовать контур объекта
 
  cv2.imshow("frame1", frame1)
  cv2.imshow("frame2", frame3)
  f=(xmax-xmin)*(ymax-ymin)
  print(f)
  if k==1:
      cv2.imwrite("Photo1.png", frame1)
      k=2
  elif k==2:
      cv2.imwrite("Photo2.png", frame1)
      k=3
  elif k==3:
      cv2.imwrite("Photo3.png", frame1)
      break
  if f==1:
      k=1
  cv2.imwrite("Bird.png", frame3)
   
  frame1 = frame2  #
  ret, frame2 = cap.read() #
  h=False
  if cv2.waitKey(40) == 27:
    break
 
 
 

cap.release()
cv2.destroyAllWindows()
img = open('Bird.png', 'rb')
bot.send_photo(CHANNEL_NAME, img)

