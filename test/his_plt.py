# histogram plotting

from All_functions import *

def myreadfile_resample_pad(dirr, pad_size):

    #This version can import 3D array regardless of the size
    from nilearn.datasets import load_mni152_template
    from nilearn.image import resample_to_img
    template = load_mni152_template()

    os.chdir(dirr)
    number = 0

    flag = True
    imgs_array = np.array([])
    path_list=[f for f in os.listdir(dirr) if not f.startswith('.')]
    path_list.sort()
    for file in path_list:
        if file.endswith(".nii"):
            #print(os.path.join(dirr, file))
            img = nib.load(os.path.join(dirr, file))
            img_array = resample_to_img(img, template)
            img_array = img.get_fdata()
            #img_array = tf.keras.utils.normalize(img_array)
            img_array = padding_zeros(img_array, pad_size)
            img_array = img_array.reshape(-1,img_array.shape[0],img_array.shape[1],img_array.shape[2])
            number += 1
            if flag == True:
                imgs_array = img_array

            else:
                imgs_array = np.concatenate((imgs_array, img_array), axis=0)

            flag = False
    return number, imgs_array, path_list

save_path = '/Users/kaiyi/OneDrive/augs_nii'
os.chdir(save_path)
for key in mul_dir:
    i=0
    _, imgs, file_lists = myreadfile_resample_pad(save_path,128)
    for i in range(len(imgs)):
        img_flat = imgs[i].flat
        plt.hist(img_flat) # The array must be one dimension
        os.chdir(save_path)
        plt.savefig('hist_%s.png'%(file_lists[i]))
        plt.close()
        i+=1

    print('Total %s imgs were plot into histogram.')