import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web
from pandas_datareader._utils import RemoteDataError
import requests
import pickle
import bs4

def save_bse_tickers():
	response = requests.get("http://www.bseindia.com/markets/Equity/EuqityDB.aspx")
	soup = bs4.BeautifulSoup(response.text, 'lxml')
	body = soup.body
	table = body.find_all('table')[2]
	tickers = []
	for row in table.find_all('tr')[2:]:
		tickers += [row.find_all('td')[1].text.strip()]
	with open('bse_tickers.pickle', 'wb') as f:
		pickle.dump(tickers, f)
	return tickers

def get_data_from_yahoo():
	if not os.path.exists('bse_tickers.pickle'):
		tickers = save_bse_tickers()
	with open('bse_tickers.pickle', 'rb') as f:
		tickers = pickle.load(f)
	if not os.path.exists('bse_stocks'):
		os.mkdir('bse_stocks')
	p, d, files = os.walk('bse_stocks/').__next__()
	file_count = len(files)
	print(file_count)
	while file_count != len(tickers):
		start = dt.datetime(2000, 1, 1)
		end = dt.datetime.now() - dt.timedelta(days=1)
		for ticker in tickers:
			path = 'bse_stocks/' + ticker + '.csv'
			try:
				df = web.DataReader(ticker + '.BO', 'yahoo', start, end)
				df.to_csv(path)
			except RemoteDataError:
				pass
	return tickers

def compile_bse_data():
	if not os.path.exists('bse_tickers.pickle') or not os.path.exists('bse_stocks'):
		tickers = get_data_from_yahoo()
	with open('bse_tickers.pickle', 'rb') as f:
		tickers = pickle.load(f)
	if os.path.exists('bse_joined_closes.csv'):
		return
	main_df = pd.DataFrame()
	for ticker in tickers:
		path = 'bse_stocks/' + ticker + '.csv'
		df = pd.read_csv(path)
		df.set_index('Date', inplace=True)
		df.rename(columns={'Adj Close': ticker}, inplace=True)
		df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)
		if main_df.empty:
			main_df = df
		else:
			main_df = main_df.join(df, how='outer')
	main_df.to_csv('bse_joined_closes.csv')


if __name__ == '__main__':
	compile_bse_data()
