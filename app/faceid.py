# import kivy dependencies first

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
# import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
# import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# import other dependencies

import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np


# build app and layout

class CamApp(App):
    def build(self):
        self.webCam = Image(size_hint=(1,.8))
        self.button = Button(text='Verify',on_press = self.verify,size_hint=(1,.1))
        self.verificationLabel = Label(text='verification Uninitiated',size_hint=(1,.1))

        # Add item to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.webCam)
        layout.add_widget(self.button)
        layout.add_widget(self.verificationLabel)

        # load keras model
        self.model = tf.keras.models.load_model('siameseModel.h5',custom_objects={'L1Dist':L1Dist})

        # setup video capture device

        self.capture =cv2.VideoCapture(0)
        Clock.schedule_interval(self.update,1.0/33.0)

        return layout
    
    # Run continuously to get webcam feed
    def update(self,*args):

        # read frame from opencv
        ret,frame = self.capture.read()
        if not ret or frame is None:
            Logger.warning("Could not read frame from webcam.")
            return  # Skip processing this frame
        frame = frame[120:120+250,200:200+250,:]

        # flip horizontall and convert image to capture
        buf  = cv2.flip(frame,0).tostring()
        imgTexture = Texture.create(size=(frame.shape[1],frame.shape[0]),colorfmt='bgr')
        imgTexture.blit_buffer(buf,colorfmt='bgr',bufferfmt='ubyte')
        self.webCam.texture=imgTexture
    
    def preprocess(self,filePath):
        byteImg = tf.io.read_file(filePath)
        img = tf.io.decode_jpeg(byteImg)
        img  =tf.image.resize(img,(100,100))
        img = img/255.0
        return img
    
    # verification function to verify
    # verification function
    
    def verify(self,*args):
        # specify threshold
        detectionThreshold=0.99
        verificationThreshold=0.8

        #capture input image from our webcam
        savePath = os.path.join('applicationData','inputImages','inputImages.jpg')
        ret,frame = self.capture.read()
        frame=frame[120:120+250,200:200+250,:]
        cv2.imwrite(savePath,frame)

        # build results array
        results=[]
        for image in os.listdir(os.path.join('applicationData','verificationImages')):
            inputImg = self.preprocess(os.path.join('applicationData','inputImages','inputImages.jpg'))
            validationImg = self.preprocess(os.path.join('applicationData','verificationImages',image))

            # make prediction
            result=self.model.predict(list(np.expand_dims([inputImg,validationImg],axis=1)))
            results.append(result)

        # detectionThreshold -> metrics above which a prediction is considered positive
        detection = np.sum(np.array(results) > detectionThreshold)

    #   verificationThreshold -> proportion of positive prediction /total positive samples
        verification = detection/len(os.listdir(os.path.join('applicationData','verificationImages')))
        verified = verification>verificationThreshold

    #   set verification text
        self.verificationLabel.text = 'verified' if verified ==True else 'unverified'

        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)
        

        return results,verified

     


if __name__=='__main__':
    CamApp().run()