import cv2
import os
file = open('ShelfImages/annotation_train.txt', 'r')
line = file.readline()
print(line)

target = {'filename_id': (line.split(' ')[0])}
# for line in file:

indict = {}
image = cv2.imread(os.path.join('D:/projects/a-PyTorch-Tutorial-to-Object-Detection/ShelfImages/train',
                                    target['filename_id']))

for i in range(1, len(line.split(' '))):
    n = 2 + (i - 1) * 5
    x = int(line.split(' ')[n])
    y = int(line.split(' ')[n + 1])
    w = int(line.split(' ')[n + 2])
    h = int(line.split(' ')[n + 3])
    # bboxes.append([x, y, w, h])
    indict['x'] = x
    indict['y'] = y
    indict['w'] = w
    indict['h'] = h

    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    if n == (len(line.split(' ')) - 5):
        break
target[line.split(' ')[0]] = [indict]
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', image)
cv2.waitKey(0)
print(target)
file.close()
