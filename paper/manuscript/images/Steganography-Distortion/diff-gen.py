import PIL.Image
import numpy as np

imo = PIL.Image.open('goldfish2-original.png')
imd = PIL.Image.open('goldfish2-distortion.png')

imoarr = np.array(imo)
imdarr = np.array(imd)

imdiff = imoarr.astype(np.int32) - imdarr.astype(np.int32)

imdiff_r, imdiff_g, imdiff_b = [imdiff[:, :, i] for i in range(3)]

imdiff_image_r = np.repeat(imdiff_r[:, :, np.newaxis], 3, axis=2)
imdiff_image_g = np.repeat(imdiff_g[:, :, np.newaxis], 3, axis=2)
imdiff_image_b = np.repeat(imdiff_b[:, :, np.newaxis], 3, axis=2)

# R-Channel
color_zro = (255, 255, 255)
color_pos = (251, 116,  84)
color_neg = (191,  21,  27)

imdiff_image_r[imdiff_r ==  0] = color_zro
imdiff_image_r[imdiff_r == +1] = color_pos
imdiff_image_r[imdiff_r == -1] = color_neg

# G-Channel
color_zro = (255, 255, 255)
color_pos = (163, 217, 157)
color_neg = ( 22, 128,  60)

imdiff_image_g[imdiff_g ==  0] = color_zro
imdiff_image_g[imdiff_g == +1] = color_pos
imdiff_image_g[imdiff_g == -1] = color_neg

# B-Channel
color_zro = (255, 255, 255)
color_pos = (154, 200, 224)
color_neg = ( 38, 118, 184)

imdiff_image_b[imdiff_b ==  0] = color_zro
imdiff_image_b[imdiff_b == +1] = color_pos
imdiff_image_b[imdiff_b == -1] = color_neg

PIL.Image.fromarray(imdiff_image_r.astype(np.uint8)).save('goldfish2-diff-r.png')
PIL.Image.fromarray(imdiff_image_g.astype(np.uint8)).save('goldfish2-diff-g.png')
PIL.Image.fromarray(imdiff_image_b.astype(np.uint8)).save('goldfish2-diff-b.png')

