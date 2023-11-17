import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from keras.utils import load_img
from attention_unet_model import attention_unet

input_shape = (128, 128, 1)
model = attention_unet(input_shape)
model.load_weights('model-mri.h5')

def preprocessing(img_path):
    
    im_width = 128
    im_height = 128
    
    X = np.zeros((1, im_height, im_width, 1), dtype=np.float32)
    
    img = load_img(img_path)
    x_img = np.array(img)
    x_img = resize(x_img, (128, 128, 1), mode = 'constant', preserve_range = True)
    
    X[0,] = x_img/255.0
    
    return X

def predict_tumur(img):
    preds_val = model.predict(img, verbose=1)
    preds_val_t = (preds_val > 0.5).astype(np.uint8)
    tumour_image = preds_val_t
    plt.imshow(img.reshape(128,128,1), cmap='gray')
    plt.imshow(tumour_image.reshape(128,128,1), alpha=.6)
    target_path = 'pred.png'
    plt.savefig(target_path)

    plt.show()
    
    return target_path

img_path = "Dataset\LGG_Segmentation\TCGA_CS_4941_19960909\TCGA_CS_4941_19960909_11.tif"
img = preprocessing(img_path)
predict_tumur(img)