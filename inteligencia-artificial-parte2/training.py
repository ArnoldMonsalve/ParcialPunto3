import random
import json
import pickle
import numpy as np
import nltk


#Para pasar las palabras a su raiz
from nltk.stem import WordNetLemmatizer

#Para crear la red neuronal
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

lemmatizer=WordNetLemmatizer()

intents=json.loads(open('archivo.json').read())

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

words=[]
classes=[]
documents=[]
ignore_letters=['?','!','¿','.',',']

#Clasifica los patrones y las categorias
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list=nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words=[lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words=sorted(set(words))

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

#Pasa la informacion a unos y ceros segun las palabras presentes en cada categoria para hacer el entrenamiento

training=[]
output_empty=[0]*len(classes)
for document in documents:
    bag=[]
    word_patterns=document[0]
    word_patterns=[lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row=list(output_empty)
    output_row[classes.index(document[1])]=1
    training.append([bag,output_row])
random.shuffle
training = np.array(training, dtype=object)
print(training)

#Reparte los datos para pasarlos a la red
train_x=list(training[:,0])
train_y=list(training[:,1])

#Creamos agentes ineligentes con apoyo de una red neuronal
model=Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),)))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))

#Creamos el optimizador y lo compilados

sgd = SGD(learning_rate=0.001, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
#Entrenamos el modelo y lo guardam
train_process=model.fit(np.array(train_x),np.array(train_y), epochs=100, batch_size=5,verbose=1)
model.save("chatbot_model.keras",train_process)