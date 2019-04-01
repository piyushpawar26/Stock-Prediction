import numpy as np 
import pandas as pd 
import pickle
from collections import Counter
import os
from constants import DAYS, LEFT_REQUIREMENT, RIGHT_REQUIREMENT, TRAINING_SIZE
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM 
from keras.models import Sequential
from keras.models import model_from_json
from keras.utils import np_utils

def process_data_for_labels(ticker):
	if not os.path.exists('bse_joined_closes.csv'):
		get_data.compile_bse_data()
	df = pd.read_csv('bse_joined_closes.csv', index_col=0)
	tickers = df.columns.values.tolist()
	df.fillna(0, inplace=True)
	for day in range(1, DAYS+1):
		df[ticker + '_' + str(day) + 'd'] = (df[ticker].shift(-day) - df[ticker]) / df[ticker]
	df.fillna(0, inplace=True)
	return tickers, df 

def buy_sell_hold(data):
	votes = []
	final = []
	for row in data:
		vals = []
		for col in row:
			if float(col) > RIGHT_REQUIREMENT:
				vals += [2]
			elif float(col) < -LEFT_REQUIREMENT:
				vals += [0]
			else:
				vals += [1]
		votes += [vals]
	votes = list(map(list, zip(*votes)))
	for res in votes:
		final += [Counter(res).most_common(1)[0][0]]
	return np.asarray(final)   

def extract_feature_sets(ticker):
	tickers, df = process_data_for_labels(ticker)
	future_days = pd.DataFrame([df[ticker + '_' + str(day) + 'd'] for day in range(1, DAYS+1)])
	y = buy_sell_hold(future_days.values)
	df.fillna(0, inplace=True)
	df = df.replace([np.inf, -np.inf], 0)
	df_vals = df[ticker].pct_change()
	df_vals = df_vals.replace([np.inf, -np.inf], 0)
	df_vals.fillna(0, inplace=True)
	X = df_vals.values 
	return X, y

def prediction(ticker):
	model_file_name = 'lstm_' + ticker + '.json'
	weights_file_name = 'weights_' + ticker + '.h5'
	X, y = extract_feature_sets(ticker)
	limit = int(len(X) * TRAINING_SIZE)
	y = np_utils.to_categorical(y, num_classes=3)
	X_train, X_test, y_train, y_test = X[:limit], X[limit:], y[:limit], y[limit:]
	X_train = X_train.reshape(limit, 1, 1)
	X_test = X_test.reshape(len(X)-limit, 1, 1)
	y_train = y_train.reshape(limit, 1, 3)
	y_test = y_test.reshape(len(X)-limit, 1, 3)
	if not os.path.exists(model_file_name):
		# model = Sequential()
		# model.add(LSTM(100, input_shape=(1,1), return_sequences=True, activation='relu'))
		# model.add(Dropout(0.2))
		# model.add(LSTM(50, return_sequences=True, activation='relu'))
		# model.add(Dropout(0.2))
		# model.add(Dense(3))
		# model.add(Activation('softmax'))

		model = Sequential()

		model.add(Dense(128, input_shape=(1, 1)))
		model.add(Activation('relu'))
		model.add(Dropout(.2))

		model.add(Dense(64))
		model.add(Activation('relu'))
		model.add(Dropout(.25))

		model.add(Dense(32))
		model.add(Activation('relu'))
		model.add(Dropout(.5))

		model.add(Dense(3))
		model.add(Activation('softmax'))
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		model.fit(X_train, y_train, epochs=10, batch_size=128, verbose=2)
		# save_lstm_model(model, model_file_name, weights_file_name)
	# else:
	# 	model = read_lstm_model(model_file_name, weights_file_name)
	preds = model.predict(X_test)
	# preds = get_list_values(preds)
	# score = model.evaluate(X_test, y_test)
	# print('Score: ', score[1]*100)
	print(preds[:100])

# def get_list_values(preds):
# 	final = []
# 	for i in preds:
# 		for j in i:
# 			for k in j:
# 				final += [int(round(k))]
# 	return final 

# def save_lstm_model(model, model_file_name, weights_file_name):
# 	model_json = model.to_json()
# 	with open(model_file_name, "w") as json_file:
# 	    json_file.write(model_json)
# 	model.save_weights(weights_file_name)

# def read_lstm_model(model_file_name, weights_file_name):
# 	json_file = open(model_file_name, 'r')
# 	loaded_model_json = json_file.read()
# 	json_file.close()
# 	loaded_model = model_from_json(loaded_model_json)
# 	loaded_model.load_weights(weights_file_name)
# 	return loaded_model

if __name__ == '__main__':
	prediction('ONGC')


# X = X.reshape(-1, 1)
# y = y.reshape(-1, 1)
# clf = VotingClassifier([
# 						('lsvc', svm.LinearSVC()),
# 						('lr', LogisticRegression()),
# 						('rfc', RandomForestClassifier())
# 						])
# if not os.path.exists('voting_classifier.pickle'):
# 	clf.fit(X_train, y_train)
# 	with open('voting_classifier.pickle', 'wb') as f:
# 		pickle.dump(clf, f)
# else:
# 	with open('voting_classifier.pickle', 'rb') as f:
# 		clf = pickle.load(f)
# clf.fit(X_train, y_train)
# accuracy = clf.score(X_test, y_test)
# print("Accuracy: " + str(accuracy))