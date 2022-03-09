import numpy as np 
import pandas as pd 
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

num_classes = 7 
width, height = 48, 48

data = pd.read_csv('../data/fer2013.csv')

# Verimizin boyutu (Kaça kaç matris olduğu) öğrenilir.
# print("Matrix shape (column, row): ", data.shape)

# İlk 5 veri kontrol edilir.
# print(data.head(5))

# Kaç tane verinin ne için kullanıldığı öğrenilir.
# print(data.Usage.value_counts())

# Veri setinde kaç adet duygu olduğu ve her bir duyguda toplam kaç adet görsel olduğu öğrenilir.
emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
emotion_counts = data['emotion'].value_counts(sort=False).reset_index()
emotion_counts.columns = ['emotion', 'number']
emotion_counts['emotion'] = emotion_counts['emotion'].map(emotion_map)
# print(emotion_counts)

# Duygularla ilgili analiz tablosu çizdirilir.
# plt.figure(figsize=(8,6))
# sns.barplot(x= emotion_counts.emotion, y=emotion_counts.number)
# plt.title('Emotion Distribution')
# plt.ylabel('Number', fontsize=10)
# plt.xlabel('Emotions', fontsize=10)
# plt.show()

# Birkaç görsel incelenir.
# def rowToImage(row):
#     pixels, emotion = row['pixels'], emotion_map[row['emotion']]
#     img = np.array(pixels.split())
#     img = img.reshape(48,48)
#     image = np.zeros((48,48,3))
#     image[:,:,0] = img
#     image[:,:,1] = img
#     image[:,:,2] = img
#     return np.array([image.astype(np.uint8), emotion], dtype=object)

# plt.figure(0, figsize=(16,10))
# for i in range(1,8):
#     face = data[data['emotion'] == i-1].iloc[0]
#     img = rowToImage(face)
#     plt.subplot(2,4,i)
#     plt.imshow(img[0])
#     plt.title(img[1])

# plt.show()  

# Veri eğitim, test ve doğrulama olarak sınıflandırılır.
data_train = data[data['Usage']=='Training'].copy()
data_test  = data[data["Usage"] != 'Training'].copy()
# print("Train Shape: ",  data_train.shape, "\nValidation Shape: ", data_val.shape, "\nTest Shape: ",data_test.shape)

# Test, eğitim ve doğrulama verileriyle ilgili grafik çizdirilir.
# def setup_axe(axe,dataframe,title):
#     dataframe['emotion'].value_counts(sort=False).plot(ax=axe, kind='bar', rot=0, color=["red", "black", "cyan", "grey", "blue", "yellow", "brown"])
#     axe.set_xticklabels(emotion_counts.emotion)
#     axe.set_xlabel("Emotions")
#     axe.set_ylabel("Number")
#     axe.set_title(title)
    
#     # set individual bar lables using above list
#     for i in axe.patches:
#         # get_x pulls left or right; get_height pushes up or down
#         axe.text(i.get_x()-.06, i.get_height()+130, \
#                 str(round((i.get_height()), 2)), fontsize=10, color='black',
#                     rotation=0)
   
# fig, axes = plt.subplots(1,2, figsize=(15,5), sharey=True)
# setup_axe(axes[0],data_train,'train')
# setup_axe(axes[1],data_test,'test')
# plt.show()


# (i) String veriler integer listelere atılır.
# (ii) Resim yeniden boyutlandırılır ve normalize edilir.
# (iii) Sınıflar şifrelendirilerek eetiketlenir.

def convert_reshape_normalize_onehot(dataframe, dataName):
    dataframe['pixels'] = dataframe['pixels'].apply(lambda pixel_sequence: [int(pixel) for pixel in pixel_sequence.split()])
    data_X = np.array(dataframe['pixels'].tolist(), dtype='float32').reshape(-1,width, height,1)/255.0   
    data_Y = to_categorical(dataframe['emotion'], num_classes)  
    # print(dataName, "_X shape: {}, ", dataName, "_Y shape: {}".format(data_X.shape, data_Y.shape))
    return data_X, data_Y

# Training data    
train_X, train_Y = convert_reshape_normalize_onehot(data_train, "train") 
train_X = np.asarray(train_X)
train_Y = np.asarray(train_Y)

# Test data
test_X, test_Y = convert_reshape_normalize_onehot(data_test, "test") 
test_X = np.asarray(test_X)
test_Y = np.asarray(test_Y)

np.save('../data/train_X', train_X)
np.save('../data/train_Y', train_Y)
np.save('../data/test_X', test_X)
np.save('../data/test_Y', test_Y)