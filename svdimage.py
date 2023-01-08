import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

url = 'D:\\Video\\160677131_1713657132147314_1730400168674006607_n.jpg'
img = Image.open(url)
imggray = img.convert('LA')
imgmat = np.array(list(imggray.getdata(band=0)), float)
imgmat.shape = (imggray.size[1], imggray.size[0])
imgmat = np.matrix(imgmat)
U, sigma, Vt = np.linalg.svd(imgmat)
num = 200 

reconstruction = np.matrix(U[:, :num]) * np.diag(sigma[:num]) * np.matrix(Vt[:num, :])
_ = plt.imshow(reconstruction, cmap='gray')
plt.show()

