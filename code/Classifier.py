## Reference: this script referenced a lot from
## https://github.com/laizheng/ML1020_GROUP_PROJECT/blob/master/codes/Classifier.py

import pandas as pd
import numpy as np
import os
import math
from keras.models import load_model
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import argparse
import pickle
from ResNeXtClassifierTemplate import ResNeXt101ClassifierTemplate
from ResNeXt50ClassifierTemplate import ResNeXt50ClassifierTemplate
## define arguments
parser = argparse.ArgumentParser()
parser.add_argument("-ds", "--downsample", default=0, type=bool,
                    help="Whether to downsample dataset")
parser.add_argument('-epochs', "--nbr_epochs", default=30, type=int)
parser.add_argument('-bs', "--batch_size", default=32, type=int)
args = parser.parse_args()
os.environ['KMP_DUPLICATE_LIB_OK']='True'

## define variables
imgs_folder = "boston_train/cropped_image/"
train_csv = "data/boston_train_fetched_with_target.csv"
target = "safety"
img_name_col = "_file"
classifier_template = ResNeXt50ClassifierTemplate()
save_model_path = "saved_models"

## define 'Classifier'
class Classifier:
    def __init__(self, train_imgs_csvfile= train_csv):
        self.df = pd.read_csv(train_imgs_csvfile)
        self.df[target] = self.df[target].astype(str)
        self.classnames = np.unique(self.df[target])
        if args.downsample is True:
            self.df = self.df.iloc[list(range(0, self.df.shape[0], 200))].reset_index(drop=True)
        self.nbr_epochs = args.nbr_epochs

    def fit(self, classifier_template, df_train, df_val, imgs_folder,
            batch_size=32, nbr_epochs=30,
            model_save_file=None,
            history_save_file=None):

        if model_save_file is None:
            print("It's mandatory to save model!")
            raise ValueError

        train_generator = ImageDataGenerator(
                rescale=1. / 255,
               # brightness_range=(1, 1.13),
                shear_range=0.1,
                zoom_range=0.1,
                rotation_range=10.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True).flow_from_dataframe(
                df_train,
                directory=imgs_folder, x_col=img_name_col, y_col=target,
                target_size=(classifier_template.img_width, classifier_template.img_height),
                batch_size=batch_size,
                shuffle=True,
                class_mode='categorical',
                classes={"0": 0, "1": 1})

        if df_val is not None:
            validation_steps = math.ceil(df_val.shape[0] / batch_size)

            validation_generator = ImageDataGenerator(rescale=1. / 255).flow_from_dataframe(
                df_val,
                directory=imgs_folder, x_col=img_name_col, y_col=target,
                target_size=(classifier_template.img_width, classifier_template.img_height),
                batch_size=batch_size,
                shuffle=True,
                class_mode='categorical',
                classes={"0": 0, "1": 1})

            #early = EarlyStopping(monitor="val_loss", mode="min", patience=6)
            #callbacks_list = [early]
            callbacks_list = []
            best_model_callback = ModelCheckpoint(model_save_file, monitor='val_loss',
                                                      verbose=1, save_best_only=True)
            callbacks_list.append(best_model_callback)
        else:
            validation_generator = None
            validation_steps = None
            callbacks_list = []

        model = classifier_template.create_model()

        history = model.fit_generator(
            train_generator,
            steps_per_epoch=math.ceil(df_train.shape[0] / batch_size),
            nb_epoch=nbr_epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks_list)

        model.save(model_save_file)
        if history_save_file is not None:
            with open(history_save_file, 'wb') as handle:
                pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return model

    def eval(self, model, df, imgs_folder, classifier_template, batch_size=32):

        validation_steps = math.ceil(df.shape[0] / batch_size)

        validation_generator = ImageDataGenerator(rescale=1. / 255).flow_from_dataframe(
            df,
            directory=imgs_folder, x_col = img_name_col, y_col = target,
            target_size=(classifier_template.img_width, classifier_template.img_height),
            batch_size=batch_size,
            shuffle=True,
            class_mode='categorical')

        eval_res = model.evaluate_generator(validation_generator, steps=validation_steps, verbose=1)
        res = {}
        res["loss"] = eval_res[0]
        res["acc"] = eval_res[1]
        return res

    def cross_validate_fit(self, classifier_template, saved_folder=None):
        # df_train must contains two columns: img and classname
        if saved_folder is None:
            print("Model saved path is not provided.")
            raise ValueError
        if os.path.isdir(saved_folder) is False:
            os.mkdir(saved_folder)
        if os.path.isdir(saved_folder+"/"+classifier_template.model_name) is False:
            os.mkdir(saved_folder+"/"+classifier_template.model_name)

        df = self.df[[img_name_col, target]]
        y = self.df[target]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for i, (train_index, val_index) in enumerate(skf.split(df, y)):
            if i < 3:
                continue
            print("CV round %d..." % i)
            df_train = df.iloc[train_index].reset_index(drop=True)
            df_val = df.iloc[val_index].reset_index(drop=True)
            best_model_file = saved_folder+"/"+classifier_template.model_name + "/bestmodel.hdf5.cv" + str(i)
            #save history
            history_file = saved_folder+"/"+classifier_template.model_name + "/history.cv" + str(i) + ".pickle"
            self.fit(classifier_template, df_train, df_val, imgs_folder,
                     batch_size=32, nbr_epochs=self.nbr_epochs, model_save_file=best_model_file,
                     history_save_file = history_file
                     )

    def fit_on_wholedataset(self, classifier_template, saved_folder=None):
        if saved_folder is None:
            print("Model saved path is not provided.")
            raise ValueError
        if os.path.isdir(saved_folder) is False:
            os.mkdir(saved_folder)
        if os.path.isdir(saved_folder+"/"+classifier_template.model_name) is False:
            os.mkdir(saved_folder+"/"+classifier_template.model_name)
        df = self.df[[img_name_col, target]]
        best_model_file = saved_folder + "/" + classifier_template.model_name + "/bestmodel.wholedata.hdf5"
        #save history
        history_file = saved_folder + "/" + classifier_template.model_name + "/history.wholedata.pickle"
        self.fit(classifier_template, df, df_val=None, imgs_folder=imgs_folder,
                 batch_size=32, nbr_epochs=self.nbr_epochs,
                 model_save_file=best_model_file,
                 history_save_file = history_file
                 )

    def cross_validate_eval(self, saved_folder, classifier_template):
        if saved_folder is None:
            print("Need to provide the folder to the saved models.")
            raise ValueError
        # use the models generated from cross validation
        scores = {}
        scores["train_loss"] = []
        scores["train_acc"] = []
        scores["val_loss"] = []
        scores["val_acc"] = []

        df = self.df[[img_name_col, target]]
        y = self.df[target]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for i, (train_index, val_index) in enumerate(skf.split(df, y)):
            #if i < 2:
            #    continue
            print("Evaluating CV round %d..." % i)
            df_train = df.iloc[train_index].reset_index(drop=True)
            df_val = df.iloc[val_index].reset_index(drop=True)
            model_save_file = saved_folder+"/"+ classifier_template.model_name + "/bestmodel.hdf5.cv" + str(i)
            model = load_model(model_save_file)
            res_train = self.eval(model, df_train, imgs_folder, classifier_template)
            res_val = self.eval(model, df_val, imgs_folder, classifier_template)
            scores["train_loss"].append(res_train["loss"])
            scores["train_acc"].append(res_train["acc"])
            scores["val_loss"].append(res_val["loss"])
            scores["val_acc"].append(res_val["acc"])
        df_scores = pd.DataFrame(scores)
        df_scores.index.name = "CV round"
        df_scores = df_scores.T
        df_scores["mean"] = df_scores.mean(axis=1)
        df_scores["std"] = df_scores.std(axis=1)
        with open(saved_folder + "/" + classifier_template.model_name + "/cv_score.pickle", 'wb') as handle:
            pickle.dump(df_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(saved_folder + "/" + classifier_template.model_name + "/cv_score.txt", 'w') as handle:
            handle.write(saved_folder + "\n")
            handle.write(str(df_scores))
            handle.write("\n")
        return df_scores

    def whole_dataset_eval(self, saved_folder, classifier_template):
        if saved_folder is None:
            print("Need to provide the folder to the saved models.")
            raise ValueError
        # use the models generated from cross validation
        scores = {}
        scores["loss"] = []
        scores["acc"] = []

        df = self.df[[img_name_col, target]]
        model_save_file = saved_folder + "/" + classifier_template.model_name + "/bestmodel.wholedata.hdf5"
        model = load_model(model_save_file)
        res = self.eval(model, df, imgs_folder, classifier_template)
        scores["loss"].append(res["loss"])
        scores["acc"].append(res["acc"])
        df_scores = pd.DataFrame(scores)
        with open(saved_folder + "/" + classifier_template.model_name + "/wholedata_score.pickle", 'wb') as handle:
            pickle.dump(df_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(saved_folder + "/" + classifier_template.model_name + "/wholedata_score.txt", 'w') as handle:
            handle.write(saved_folder + "\n")
            handle.write(str(df_scores))
            handle.write("\n")
        return df_scores

    #def generate_submission(self):
    #    pass



## call 'Classifier'
def main():
    clf = Classifier()
    clf.cross_validate_fit(classifier_template, save_model_path)
    clf.cross_validate_eval(save_model_path, classifier_template)
    #clf.fit_on_wholedataset(classifier_template, save_model_path)
    #clf.whole_dataset_eval(save_model_path, classifier_template)


if __name__ == '__main__':
    main()
