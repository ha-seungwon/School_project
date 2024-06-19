# Image augmentation example

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,              # 0~180
        width_shift_range=0.2,          
        height_shift_range=0.2,
        rescale=1./255,                 # 픽셀값을 0~1 로 변환
        shear_range=0.2,                # shearing transformations 
        zoom_range=0.2,                 # randomly zooming
        horizontal_flip=True,           # randomly flipping 
        fill_mode='nearest')            # filling in newly created pixels

img = load_img('d:/data/dog.jpg')  
x = img_to_array(img)  #      this is a Numpy array with shape (3, 331, 237)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 331, 237)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='d:/data/aug/', save_prefix='dog', save_format='jpeg'):
    i += 1
    if i > 30:
        break                 # or for working infinitely  

