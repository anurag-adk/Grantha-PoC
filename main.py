from matplotlib import pyplot as plt 
import cv2
import easyocr
from pylab import rcParams
from IPython.display import Image

rcParams['figure.figsize']=8,16
reader =easyocr.Reader(['en','ne'])

file_name="img.png"
# Image(file_name)

output=reader.readtext(file_name)
print(output)

cord=output[1][0]
x_min,y_min=[int(min(idx)) for idx in zip(*cord)]
x_max,y_max=[int(max(idx)) for idx in zip(*cord)]

image = cv2.imread(file_name)
cv2.rectangle(image,(x_min,y_min),(x_max,y_max),(255,0,0),2)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
