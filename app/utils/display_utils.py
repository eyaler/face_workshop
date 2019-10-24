
import matplotlib.pyplot as plt

#from app.utils import im_utils

# Plot images inline using Matplotlib
# def pltimg(im,title=None, mode='rgb',figsize=(8,12),dpi=160,output=None):
#   plt.figure(figsize=figsize)
#   plt.xticks([]),plt.yticks([])
#   if title is not None:
#     plt.title(title)
#   if mode.lower() == 'bgr':
#     im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
#   f = plt.gcf()
#   plt.imshow(im)
#   plt.show()
#   plt.draw()
#   if output is not None:
#     bbox_inches='tight'
#     ext=osp.splitext(output)[1].replace('.','')
#     f.savefig(output,dpi=dpi,format=ext)
#     print('Image saved to: {}'.format(output))