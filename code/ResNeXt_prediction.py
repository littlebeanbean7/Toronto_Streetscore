import pandas as pd
import os
import glob
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator

## define to conduct augmentation on test data or not
aug = True



def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    img_width = 224
    img_height = 224
    img_folder = "boston_test/cropped_image/"
    img_name_col = "_file"
    n_aug_img = 5


    df = pd.DataFrame(data=os.listdir(img_folder), columns=[img_name_col])
    df.sort_values(img_name_col,inplace=True)

    print("generating predictions for %d images" % df.shape[0])

    gen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

    model_saved_dir = "saved_models/ResNeXt50/"
    model_files_list = list(glob.glob(model_saved_dir + "*hdf5*"))
    model_files_list.sort(reverse=True)


    for file in model_files_list:
        model_loaded = load_model(file)
        file = file.split("/")[-1]
        print("%s model loaded." % file)
        predict_all_data = []

        for i in tqdm(range(df.shape[0])):
            img = image.load_img(img_folder + df[img_name_col][i], target_size=(img_width, img_height))
            x = np.array(img)
            if aug == False:
                x = x / 255.
                x = np.expand_dims(x, axis=0)
                x = np.vstack([x])
                predict = model_loaded.predict(x).squeeze()
            else:
                predictitions_w_aug = []
                for j in range(n_aug_img):
                    seed = 1234
                    trans_img = gen.random_transform(x, seed + j + i)
                    trans_img = trans_img/255.
                    trans_img = np.expand_dims(trans_img, axis=0)
                    trans_img = np.vstack([trans_img])
                    predict = model_loaded.predict(trans_img).squeeze()
                    predictitions_w_aug.append(predict)
                predictitions_w_aug = np.array(predictitions_w_aug)
                predict = np.average(predictitions_w_aug, axis=0)
            predict_all_data.append(predict)
        predict_all_data = np.array(predict_all_data)
        print("\npredictions done with model %s" % file)
        df_predict = pd.DataFrame(predict_all_data, columns=['0','1'])
        prediction = pd.concat([df, df_predict], axis=1)
        prediction.to_csv("boston_test/prediction" + "boston.test_"+ file + ".prediction" + ".csv" , index=False)
    print("Done.")

if __name__ == '__main__':
    main()