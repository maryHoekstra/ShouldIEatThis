import numpy as np
import pandas as pd
from keras import models
from keras import layers
from sklearn.preprocessing import MultiLabelBinarizer
import pickle, pdb
import coremltools

def main():
	f = open('store.pckl', 'rb')
	eng_data = pickle.load(f)
	f.close()

	np.random.seed(1234)
	train, validate, test = np.split(eng_data.sample(frac=1, random_state=134),[int(.6*len(eng_data)), int(.8*len(eng_data))])

	mlb = MultiLabelBinarizer()
	X_train = mlb.fit_transform(train['new_ing']).astype(np.float32)
	y_train = train['nutrition-score-fr_100g'].values

	X_val = mlb.transform(validate['new_ing']).astype(np.float32)
	y_val = validate['nutrition-score-fr_100g'].values


	all_ing = len(X_train[0])
	print(all_ing)


	model = models.Sequential()
	model.add(layers.Dense(256,activation='relu',input_shape=(all_ing,)))
	model.add(layers.Dense(256,activation='relu'))
	model.add(layers.Dense(1))
	model.summary()

	model.compile(optimizer='adam',loss='mean_squared_error', metrics=['accuracy'])

	results = model.fit(X_train, y_train, epochs=1, batch_size=256, validation_data=(X_val,y_val))

	#print(np.mean(results.history["val_acc"]))

	test2 = test[0:1]
	X_test = mlb.transform(test2['new_ing']).astype(np.float32)
	pdb.set_trace()
	y_test = test2['nutrition-score-fr_100g'].values

	prediction = model.predict(X_test)
	print(prediction)
	print(y_test)

	coreml_model = coremltools.converters.keras.convert(model, input_names=['sparse_ing'], output_names=['score'])
	coreml_model.save('NutritionScore.mlmodel')



if __name__ == '__main__':
    main()
