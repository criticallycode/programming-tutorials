### PCA On Mushrooms Dataset ###

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

m_data = pd.read_csv('mushrooms.csv')

# machine learning systems work with integers, we need to encode these
# string characters into ints

encoder = LabelEncoder()
# now apply the transformation to all the columns:
for col in m_data.columns:
    m_data[col] = encoder.fit_transform(m_data[col])

X_features = m_data.iloc[:,1:23]
y_label = m_data.iloc[:, 0]

#let's confirm we're slicing this right
print(X_features.head())
print(y_label.head())
print(X_features.describe)

# scale the features
scaler = StandardScaler()
X_features = scaler.fit_transform(X_features)

# visualize
pca = PCA()
pca.fit_transform(X_features)
pca_variance = pca.explained_variance_

plt.figure(figsize=(8, 6))
plt.bar(range(22), pca_variance, alpha=0.5, align='center', label='individual variance')
plt.legend()
plt.ylabel('Variance ratio')
plt.xlabel('Principal components')
plt.show()

pca2 = PCA(n_components=17)
pca2.fit(X_features)
x_3d = pca2.transform(X_features)

plt.figure(figsize = (8,6))
plt.scatter(x_3d[:,0], x_3d[:,5], c= m_data['class'])
plt.show()

pca3 = PCA(n_components=2)
pca3.fit(X_features)
x_3d = pca3.transform(X_features)

plt.figure(figsize = (8,6))
plt.scatter(x_3d[:,0], x_3d[:,1], c= m_data['class'])
plt.show()

### LDA Example On Titanic Dataset ###

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

training_data = pd.read_csv("C:/jupyter_notebooks/Titanic/train.csv")

# let's drop the cabin column, because it has a lot of missing values
# ticket numbers contain far too many categories as well, so let's drop that too

training_data.drop(labels=['Cabin', 'Ticket'], axis=1, inplace=True)

training_data["Age"].fillna(training_data["Age"].median(), inplace=True)
training_data["Embarked"].fillna("S", inplace=True)

encoder_1 = LabelEncoder()
# fit the encoder on the data
encoder_1.fit(training_data["Sex"])

# transform and replace the training data
training_sex_encoded = encoder_1.transform(training_data["Sex"])
training_data["Sex"] = training_sex_encoded

encoder_2 = LabelEncoder()
encoder_2.fit(training_data["Embarked"])

training_embarked_encoded = encoder_2.transform(training_data["Embarked"])
training_data["Embarked"] = training_embarked_encoded

# assume the name is going to be useless and drop it
training_data.drop("Name", axis = 1, inplace = True)

# remember that the scaler takes arrays, so any value we wish to reshape we need to turn into array to scale
ages_train = np.array(training_data["Age"]).reshape(-1, 1)
fares_train = np.array(training_data["Fare"]).reshape(-1, 1)

scaler = StandardScaler()

training_data["Age"] = scaler.fit_transform(ages_train)
training_data["Fare"] = scaler.fit_transform(fares_train)

# now to select our training and testing data
features = training_data.drop(labels=['PassengerId', 'Survived'], axis=1)
labels = training_data['Survived']

# do some training on the validation set
# make the train and test data from validation data

### LDA Classify ###

X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=27)

model = LDA()
model.fit(X_train, y_train)
preds = model.predict(X_val)
acc = accuracy_score(y_val, preds)
f1 = f1_score(y_val, preds)
print("Accuracy: {}".format(acc))
print("F1 Score: {}".format(f1))

### Prior To Transform ###

X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=27)

logreg_clf = LogisticRegression()
logreg_clf.fit(X_train, y_train)
preds = logreg_clf.predict(X_val)
acc = accuracy_score(y_val, preds)
f1 = f1_score(y_val, preds)
print("Accuracy: {}".format(acc))
print("F1 Score: {}".format(f1))

### LDA Transform ###

LDA_transform = LDA(n_components=1)
LDA_transform.fit(features, labels)
features_new = LDA_transform.transform(features)

# Print the number of features
print('Original feature #:', features.shape[1])
print('Reduced feature #:', features_new.shape[1])
## View the ratio of explained variance
print(LDA_transform.explained_variance_ratio_)

X_train, X_val, y_train, y_val = train_test_split(features_new, labels, test_size=0.2, random_state=27)

logreg_clf = LogisticRegression()
logreg_clf.fit(X_train, y_train)
preds = logreg_clf.predict(X_val)
acc = accuracy_score(y_val, preds)
f1 = f1_score(y_val, preds)
print("Accuracy: {}".format(acc))
print("F1 Score: {}".format(f1))

### SVD Example ###

import numpy
from PIL import Image

def load_image(image):
    image = Image.open(image)
    im_array = numpy.array(image)

    red = im_array[:, :, 0]
    green = im_array[:, :, 1]
    blue = im_array[:, :, 2]

    return red, green, blue

def channel_compress(color_channel, singular_value_limit):
    u, s, v = numpy.linalg.svd(color_channel)
    compressed = numpy.zeros((color_channel.shape[0], color_channel.shape[1]))
    n = singular_value_limit

    left_matrix = numpy.matmul(u[:, 0:n], numpy.diag(s)[0:n, 0:n])
    inner_compressed = numpy.matmul(left_matrix, v[0:n, :])
    compressed = inner_compressed.astype('uint8')
    return compressed

red, green, blue = load_image("dog3.jpg")
singular_val_lim = 350

def compress_image(red, green, blue, singular_val_lim):
    compressed_red = channel_compress(red, singular_val_lim)
    compressed_green = channel_compress(green, singular_val_lim)
    compressed_blue = channel_compress(blue, singular_val_lim)

    im_red = Image.fromarray(compressed_red)
    im_blue = Image.fromarray(compressed_blue)
    im_green = Image.fromarray(compressed_green)

    new_image = Image.merge("RGB", (im_red, im_green, im_blue))
    new_image.show()
    new_image.save("dog3-edited.jpg")

compress_image(red, green, blue, singular_val_lim)