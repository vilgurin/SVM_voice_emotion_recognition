import soundfile # to read audio file
import numpy as np
import librosa # to extract speech features
import glob
import os
import pickle # to save model after training
from sklearn.model_selection import train_test_split # for splitting training and testing
from SVM import SupportVectorMachine

# mfcc corresponds to an existing MFCC method
# mfcc2 - custom MFCC, written from scratch   
def extractFeature(file_name, **kwargs):
    mfcc = kwargs.get("mfcc")
    mfcc2 = kwargs.get("mfcc2")

    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        
        if mfcc2:
            from MFCC import MFCC
            mfccs2 = MFCC(sample=X, sampleRate= sample_rate, filtersAmount= 40) 
            mfccs2 = np.mean(mfccs2.mfcc().T, axis = 0)
            result = np.hstack((result, mfccs2))
    return result


def dataLoader(emotionType, test_size=0.01):
    X, y = [], []
    int2emotion = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
    }

    AVAILABLE_EMOTIONS = {
        "angry": 147,
        "sad": 131,
        "happy": 151
    }
    
    emotionCounter = 0
    otherCounter = 0
    for file in glob.glob("data/Actor_*/*.wav"):

        basename = os.path.basename(file)
        emotion = int2emotion[basename.split("-")[2]]

        if emotion not in AVAILABLE_EMOTIONS:
            continue

        if emotionCounter < AVAILABLE_EMOTIONS[emotionType] and emotion == emotionType:
            features = extractFeature(file, mfcc2=True)
            X.append(features)
            y.append(1.0)
            emotionCounter += 1
        if otherCounter < AVAILABLE_EMOTIONS[emotionType] and emotion != emotionType:
            features = extractFeature(file, mfcc2=True)
            X.append(features)
            y.append(-1.0)
            otherCounter += 1
            
    return train_test_split(np.array(X), y, test_size=test_size, random_state=7)

# Choosing data for training the model. In this case
# this is the data that corresponds to sad feeling
X_train, X_test, y_train, y_test = dataLoader("sad", test_size=0.01)
y_train_new = []
y_test_new = []

X_train = np.array(X_train)
X_test = np.array(X_test)

y_train_new = np.array(y_train) 
y_test_new = np.array(y_test)

def kernel(x, y, sigma=1):
    if np.ndim(x) == 1 and np.ndim(y) == 1:
        return np.exp(-np.power(np.linalg.norm(x-y),2)/2*sigma**2)
    elif np.ndim(x) > 1 and np.ndim(y) > 1:
      result = []
      temp = []
      for i in range(len(x)):
        for j in range(len(y)):
          diff = x[i]-y[j]
          temp_result = np.exp(-np.power(np.linalg.norm(diff),2)/2*sigma**2)
          temp.append(temp_result)
        result.append(temp)
        temp = []
      return np.array(result)
    else:
        return np.exp(- (np.linalg.norm(x - y, 2, axis=1) ** 2) / (2 * sigma ** 2))

# to train the model for other emotions use the code below

# model = SupportVectorMachine(C = 1,kernel = kernel)
# model.fit(X_train,y_train_new)
# saved_model = pickle.dump(model, open("'Model name'.model", "wb"))

if __name__ == "__main__":
    from recordVoice import record
    record()
    X_test_voice = extractFeature("input.wav", mfcc2=True)
    X_test_voice = np.array([X_test_voice])
    
    loaded_modelHappy = pickle.load(open("modelHappy.model",'rb'))
    loaded_modelSad = pickle.load(open("modelSad.model",'rb'))
    loaded_modelAngry = pickle.load(open("modelAngry.model",'rb'))
    modelsArr = [loaded_modelHappy, loaded_modelSad, loaded_modelAngry]
    lst_emotions = []
    for i in range(3):
        lst_emotions.append(modelsArr[i].predict(X_test_voice))
    
    max_index = lst_emotions.index(max(lst_emotions))
    if max_index == 0:
        print("happy")
    elif max_index == 1:
        print("sad")
    elif max_index == 2:
        print("angry")
