import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import optimizers
from sklearn.preprocessing import MultiLabelBinarizer
import pickle, pdb
import coremltools

def main():

	# load the preprossed data
	f = open('store.pckl', 'rb')
	eng_data = pickle.load(f)
	f.close()

	# split the data in training, validation and test sets
	np.random.seed(1234)
	train, validate, test = np.split(eng_data.sample(frac=1, random_state=134),[int(.6*len(eng_data)), int(.8*len(eng_data))])

	# convert the ingredients into sparse vector
	mlb = MultiLabelBinarizer()
	X_train = mlb.fit_transform(train['new_ing']).astype(np.float32)
	y_train = train['nutrition-score-fr_100g'].values

	X_val = mlb.transform(validate['new_ing']).astype(np.float32)
	y_val = validate['nutrition-score-fr_100g'].values

	X_test = mlb.transform(test['new_ing']).astype(np.float32)
	y_test = test['nutrition-score-fr_100g'].values


	all_ing = len(X_train[0])

	# build the model
	model = models.Sequential()
	model.add(layers.Dense(256,activation='relu',input_shape=(all_ing,)))
	model.add(layers.Dense(256,activation='relu'))
	model.add(layers.Dense(1))
	model.summary()

	model.compile(optimizer='adam',loss='mean_squared_error', metrics=['accuracy'])

	results = model.fit(X_train, y_train, epochs=20, batch_size=200, validation_data=(X_val,y_val))

	# summarize history for accuracy

	fig = plt.figure()
	plt.plot(results.history['acc'])
	plt.plot(results.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.show()
	#fig.savefig('accuracy.png')


	# summarize history for loss
	fig = plt.figure()
	plt.plot(results.history['loss'])
	plt.plot(results.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.show()
#	plt.savefig('loss.png')


	prediction = model.predict(X_test)
	print(prediction)
	print(y_test)

	test_score, test_acc = model.evaluate(X_test, y_test, verbose=False)
	print('Test score: ', test_score)    #Loss on test
	print('Test accuracy: ', test_acc)

	coreml_model = coremltools.converters.keras.convert(model, input_names=['sparse_ing'], output_names=['score'])
	coreml_model.save('ShouldIEatThis?/ShouldIEatThis?/NutritionScore.mlmodel')



if __name__ == '__main__':
    main()
