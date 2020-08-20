from . import sol5_utils
from imageio import imread
from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve

from tensorflow.keras.layers import Input, Conv2D, Activation, Add
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam

import numpy as np


im_cache = {}
NORMALIZE = 255
MIN_SIG = 0
MAX_SIG = 0.2
TRAINING_SLICE = 0.8


def read_image(filename, representation):

    image = imread(filename)
    image_f = image.astype(np.float64)
    image_f /= 255
    if representation == 1 :
        if len(image_f.shape) == 3 : # Image is RGB and needs converting to grayscale
            image_f = rgb2gray(image_f)

    return image_f

def load_dataset(filenames, batch_size, corruption_func, crop_size):

    while True:

        corrupted_list = np.zeros((batch_size,crop_size[0],crop_size[1],1)) # Both lists have shape (batch_size, h, w, 1)
        clean_list = np.zeros((batch_size,crop_size[0],crop_size[1],1))
        rand_images = np.random.choice(filenames,batch_size)
        for i in range(batch_size):
            im = load_image(rand_images[i])
            clean_patch, corrupt_patch = generate_patches(im,crop_size,corruption_func)
            corrupted_list[i] = corrupt_patch
            clean_list[i] = clean_patch

        yield corrupted_list,clean_list

def load_image(path, repr=1):
    """
    helper for load_dataset
    :param path: path to query if exists or not in the im_cache
    :return: the image, either from im_cache or loaded directly and added there.
    """
    if path in im_cache:
        return im_cache[path]
    else:
        im = read_image(path,repr)
        im_cache[path] = im
        return im


def generate_patches(im, crop_size, corruption_func):

    larger_crop = tuple([k * 3 for k in crop_size]) # Create 3 * (crop_size) in order to apply corruption
    large_clean_patch = extract_patches([im],larger_crop)[0]
    large_corrupt_patch = corruption_func(large_clean_patch)
    clean_patch, corrupt_patch = extract_patches([large_clean_patch,large_corrupt_patch],crop_size)
    clean_patch -= 0.5
    corrupt_patch -= 0.5
    clean_patch = clean_patch[:,:,np.newaxis]
    corrupt_patch = corrupt_patch[:,:,np.newaxis]   #reshape to (h,w,1)
    return (clean_patch, corrupt_patch)



def extract_patches(images, crop_size):
    """
    Helper for load_dataset
    :param images: Array representing images to extract patches from. assumes same width and height
    :param crop_size: a tuple with (height, width) representing patch size
    :return: Randomly selected corresponding patches from images with dimensions of (height,width,1)
    """
    patches = []
    x_dim = images[0].shape[1]
    y_dim = images[0].shape[0]
    start_x = np.random.randint(0, x_dim - crop_size[1])
    # Randomly generates starting x,y position for crop. asserting that there is enough space for the window.
    start_y = np.random.randint(0, y_dim - crop_size[0])

    for im in images:
        patch = np.array(im[start_y:start_y + crop_size[0], start_x:start_x + crop_size[1]])
        patches.append(patch)

    return patches

def resblock(input_tensor, num_channels):


    b = Conv2D(filters=num_channels, kernel_size=(3,3),padding="same")(input_tensor)
    c = Activation('relu')(b)
    b = Conv2D(filters=num_channels, kernel_size=(3,3),padding="same")(c)
    c = Add()([input_tensor,b])
    return Activation('relu')(c)

def build_nn_model(height, width, num_channels, num_res_blocks):

    source = Input(shape=(height,width,1))
    b = Conv2D(num_channels,kernel_size=(3,3),padding="same")(source)
    c = Activation("relu")(b)
    b = resblock(c,num_channels)
    for i in range(num_res_blocks -1):
        b = resblock(b,num_channels)
    c = Conv2D(filters=1,kernel_size=(3,3),padding="same")(b)
    b = Add()([source,c])
    return Model(inputs=source, outputs=b)


def train_model(model, images, corruption_func, batch_size, steps_per_epoch, num_epochs, num_valid_samples):

    CROP_SIZE = (model.input_shape[1], model.input_shape[2])
    randomized_images = np.random.permutation(images)
    SLICE = int(TRAINING_SLICE * len(images))
    training_images = randomized_images[:SLICE]
    validation_images = randomized_images[SLICE:]
    training_generator = load_dataset(training_images,batch_size,corruption_func,CROP_SIZE)
    validation_generator = load_dataset(validation_images,batch_size,corruption_func,CROP_SIZE)
    adam = Adam(beta_2=0.9)
    model.compile(loss=mean_squared_error, optimizer=adam)
    model.fit_generator(training_generator,steps_per_epoch=steps_per_epoch, epochs= num_epochs, validation_data=validation_generator,
                        validation_steps=num_valid_samples)

def restore_image(corrupted_image, base_model):

    height, width = corrupted_image.shape
    a = Input(shape=(height,width,1))
    b = base_model(a)
    prediction_model = Model(inputs=a, outputs=b)
    # pred = new_model.predict((corrupted_image - 0.5).reshape(height, width, 1)[np.newaxis, ...])[0]
    # pred = pred.squeeze(axis=2) + 0.5
    the_restored_image = prediction_model.predict((corrupted_image - 0.5).reshape(height,width,1)[np.newaxis,...], batch_size=1)[0]  #black magic
    return np.clip(the_restored_image.squeeze(axis=2) + 0.5, a_min=0, a_max=1).astype(np.float64)


def corruption_func(im):
    """
    In order to conform to corruption_func for build_model
    :param im:
    :return:
    """
    return add_gaussian_noise(im,MIN_SIG,MAX_SIG)


def add_gaussian_noise(image, min_sigma, max_sigma):

    sigma = np.random.uniform(min_sigma, max_sigma)
    corrupted = (image + np.random.normal(loc=0, scale=sigma, size=image.shape)) * NORMALIZE
    corrupted.round()
    return np.clip(corrupted / NORMALIZE, 0, 1)


def learn_denoising_model(num_res_blocks= 5, quick_mode = False):

    TRAINING_PATCH_HEIGHT = 24
    TRAINING_PATCH_WIDTH = 24
    NUM_OF_CHANNELS = 48
    BATCH_SIZE_LOLS = 10 if quick_mode else 100
    STEPS_PER_EPOCH = 3 if quick_mode else 100
    NUM_EPOCHS = 2 if quick_mode else 5
    NUM_VALID_SAMPLES = 30 if quick_mode else 1000

    filenames = sol5_utils.images_for_denoising()
    model = build_nn_model(TRAINING_PATCH_HEIGHT, TRAINING_PATCH_WIDTH, NUM_OF_CHANNELS,num_res_blocks)
    train_model(model, images=filenames, corruption_func=corruption_func, batch_size=BATCH_SIZE_LOLS,
                steps_per_epoch= STEPS_PER_EPOCH, num_epochs=NUM_EPOCHS, num_valid_samples=NUM_VALID_SAMPLES)

    return model

def add_motion_blur(image, kernel_size, angle):

    if kernel_size % 2 == 0: raise ValueError("Even integer given as kernel size. Size should be odd integer")
    kernel = sol5_utils.motion_blur_kernel(kernel_size, angle)
    return convolve(image, kernel)  # scipy.ndimage.filters.convolve


def random_motion_blur(image, list_of_kernel_sizes):

    angle = np.random.uniform(0,np.pi)
    ker_size = np.random.choice(list_of_kernel_sizes)   #Randomly chooses kernel size from list
    corrupted_im = add_motion_blur(image,ker_size,angle) * NORMALIZE
    corrupted_im.round()
    return np.clip(corrupted_im / NORMALIZE, 0, 1)


def motion_blur(im):
    """
    in order to fit the protocol of corruption_func
    :param im:
    :return:
    """
    return random_motion_blur(im,[7])


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):

    PATCH_HEIGHT = 16
    PATCH_WIDTH = 16
    NUM_CHANNELS = 32
    NUM_OF_EPOCHS = 2 if quick_mode else 10
    BATCH_SIZE = 10 if quick_mode else 100
    STEPS_PER_EPOCH = 3 if quick_mode else 100
    VALIDATION_SAMPLE = 30 if quick_mode else 1000


    filenames = sol5_utils.images_for_deblurring()
    model = build_nn_model(PATCH_HEIGHT, PATCH_WIDTH, NUM_CHANNELS,num_res_blocks)
    train_model(model, images=filenames, corruption_func=motion_blur,batch_size=BATCH_SIZE,
                steps_per_epoch=STEPS_PER_EPOCH, num_epochs=NUM_OF_EPOCHS, num_valid_samples=VALIDATION_SAMPLE)
    return model
