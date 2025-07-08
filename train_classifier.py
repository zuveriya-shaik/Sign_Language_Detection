import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Convert labels (A-Z) to integers (0-25)
label_to_index = {chr(65 + i): i for i in range(26)}
labels = np.array([label_to_index[label] for label in labels])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save model and label mapping
f = open('model.p', 'wb')
pickle.dump({'model': model, 'label_to_index': label_to_index}, f)
f.close()
