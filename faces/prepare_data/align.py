from cv2 import cv
import cv2 
import numpy as np
from numpy import linalg as LA
import pdb 

def subimage(image, centre, theta, width, height):
   image = cv.fromarray(image)
   #output_image = cv.CreateImage((width, height), image.depth, image.nChannels)
   output_image = cv.CreateImage((width, height), 8, 3)
   mapping = np.array([[np.cos(theta), -np.sin(theta), centre[0]],
                       [np.sin(theta), np.cos(theta), centre[1]]])
   map_matrix_cv = cv.fromarray(mapping)
   cv.GetQuadrangleSubPix(image, output_image, map_matrix_cv)
   return output_image

def pad_image(src):
  h, w, _ = src.shape
  pl = (int)(max(h,w)*0.3)
  res = src.copy()
  top = np.flipud(res[0:pl,:].copy())
  res = np.concatenate([top, res],0)

  bottom = np.flipud(res[res.shape[0]-pl:,:].copy())
  res = np.concatenate([ res, bottom],0)

  left = np.fliplr(res[:,0:pl,:].copy())
  res = np.concatenate([ left,res],1)

  right = np.fliplr(res[:,res.shape[1]-pl:,:].copy())
  res = np.concatenate([ res, right],1)

  return res, pl


def align(path, landmarks):
  x0,y0,x1,y1,x2,y2,x3,y3 = landmarks
  src = cv2.imread(path, 1)
  src, pl = pad_image(src)
  e0 = np.array([x0+pl, y0+pl])
  e1 = np.array([x1+pl, y1+pl])
  m0 = np.array([x2+pl, y2+pl])
  m1 = np.array([x3+pl, y3+pl])
  
  xx = e1 - e0
  yy = (e0+e1)/2. - (m0+m1)/2.
  c = (e0+e1)/2. - 0.1*yy
  s = np.max((4.*LA.norm(xx), 3.6*LA.norm(yy)))
  s = int(s)
  yy90 = np.array([yy[1],-yy[0]])
  x = xx - yy90
  angle = np.arctan(x[1]/x[0])

  
  res = subimage(src, c, angle, s, s)
  mat=cv.GetMat(res)
  res = np.asarray(mat)
  return res

with open('list_landmarks_celeba.txt') as f:
  lines = f.readlines()
lines = map(lambda x: x.strip('\r\n'), lines)

lines = lines[2:]

for line in lines:
  try:
    for i in range(20):
      line = line.replace('  ', ' ')
    seg = line.replace('  ', ' ').split(' ')
    path = 'img_celeba/' + seg[0]
    landmarks = np.array(seg[1:]).astype(int)
    landmarks = [landmarks[i] for i in [0,1,2,3, 6,7,8,9]]
    img = align(path, landmarks)
    cv2.imwrite('img_self_align/'+seg[0], img)
  except:
    print line + 'ERROR'

#src = cv2.imread('000015.jpg', 1)
#e0 = np.array([151, 181])
#e1 = np.array([219, 167])
#m0 = np.array([178, 252])
#m1 = np.array([234, 240])


#cv2.imwrite('1.png', a)

#cv2.circle(src,(187,181), 10, (0,0,255), -1)
