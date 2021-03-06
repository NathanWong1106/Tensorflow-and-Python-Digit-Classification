import tensorflow as tf
import cv2
import sys
import os
SIZE = 28

def main():
    if len(sys.argv) != 3:
        print('USAGE: python prediction.py [model] [directory or image]')
        sys.exit(0)

    model = tf.keras.models.load_model(sys.argv[1])
    path = sys.argv[2]

    # predict on one image
    if os.path.isfile(path):
        prediction = classification(path, model)
        if prediction is not None:
            print(f'The predictor thinks that the image at {path} is the number {prediction}')

    # search through entire directory
    elif os.path.isdir(path):
        print(f'---------------SEARCHING THROUGH {path}------------------\n')
        print("FILE NAME\tPREDICTION")
        searchdir(path, model)

def classification(path, model):
        # prepare image for prediction
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (SIZE, SIZE))
        #invert image (white to black and v.v.)
        img = cv2.bitwise_not(img)
        #normalize values
        img = img / 255.0
        # model expects dim=4
        img = img.reshape(1, SIZE, SIZE, 1)

        #select most probable number and return
        classification = model.predict([img]).argmax()
        return classification

def searchdir(path, model):
    for img in os.listdir(path):
        imgPath = os.path.join(path, img)
        if os.path.isfile(imgPath) and (imgPath.endswith('.jpg') or imgPath.endswith('.png')):
            prediction = classification(imgPath, model) 
            if prediction is not None:
                print(f'{img}\t\t{prediction}')
        elif os.path.isdir(imgPath):
            searchdir(imgPath, model)
main()