import pandas
from pandas import Series
from datetime import datetime
import numpy as np
import datetime as dt
import os


class DataCombiner:
    def __init__(self, csv_stocks_filename, csv_news_filename, combine_data_directly=True, category=None, news_filter=None, normalized_whitelist='abcdefghijklmnopqrstuvwxyz 0123456789.,;\'-:?'):
        self.csv_stocks_filename = csv_stocks_filename
        self.csv_news_filename = csv_news_filename
        self.category = category
        self.news_filter = news_filter
        self.whitelist = normalized_whitelist
        if combine_data_directly:
            self.combine_data()

    def combine_data(self):
        print("load Stocks")
        df_stocks = pandas.read_csv(self.csv_stocks_filename)
        # Make the CSV smaller to filter to the date ranges of the available news
        df_stocks = df_stocks[(df_stocks['Date'] > '2014-03-09')
                              & (df_stocks['Date'] < '2014-08-29')]
        print("load News")
        df_news = pandas.read_csv(self.csv_news_filename,sep=',').fillna('')
        if self.category is not None:
            print("Filter Category")
            df_news_filtered = df_news[df_news['CATEGORY'] == self.category]

        if self.news_filter is not None:
            print("Filter News")
            df_news_filtered = df_news[df_news['TITLE'].str.contains(self.news_filter, case=False)]
        print("Normalizing")
        # Normalize headline
        df_news_filtered['normalized_headline'] = df_news['TITLE'].apply(
            self.__normalize_headline)

        print("Converting Timestamps")
        # Convert the timestamps to same time format as the stocks
        df_news_filtered['TIMESTAMP'] = df_news['TIMESTAMP'].apply(
            lambda x: datetime.fromtimestamp(int(int(x) / 1000)).strftime('%Y-%m-%d'))
        df_news_filtered=df_news_filtered.groupby('TIMESTAMP')['normalized_headline'].apply(' '.join).reset_index()
        df_news_filtered['price'] = df_news_filtered['TIMESTAMP'].apply(
            lambda row: self.__findStockPrice(row, df_stocks)).astype(np.int32)
        df_news_filtered['change_in_price'] = df_news_filtered['TIMESTAMP'].apply(
            lambda row: self.__findStockPriceChange(row, df_stocks)).astype(np.int32)
        print("Calculating today's change")
        df_news_filtered['today'] = df_news_filtered['TIMESTAMP'].apply(
            lambda row: self.__findStockChange(row, df_stocks)).astype(np.int32)
        print("Calculating tomorrow's change")
        df_news_filtered['tomorrow'] = df_news_filtered['TIMESTAMP'].apply(
            lambda row: self.__findStockChange(row, df_stocks, dateOffset=1)).astype(np.int32)
        print("Calculating day after tomorrows's change")
        df_news_filtered['day_after_tomorrow'] = df_news_filtered['TIMESTAMP'].apply(
            lambda row: self.__findStockChange(row, df_stocks, dateOffset=2)).astype(np.int32)

        

        print("Done")
        self.combined_data = df_news_filtered
        return self.combined_data
    def __findStockPrice(self, row, dataset):
        currentstockDay = None
        currentstockDay = dataset[dataset['Date'] == row]
        if (not currentstockDay.empty):
            return currentstockDay.iloc[0]['Open']
        else:
            targetstockDay=currentstockDay
            offset=1
            while(targetstockDay.empty and offset<=7):
                date = datetime.strptime(row, '%Y-%m-%d')
                targetdate = date - dt.timedelta(days=offset)
                tar_date=targetdate.strftime('%Y-%m-%d')
                targetstockDay=dataset[dataset['Date'] == tar_date]
                offset=offset+1
            if offset>7:
                return 0
            else:
                return targetstockDay.iloc[0]['Close']
    def __findStockPriceChange(self,row,dataset):
        date=datetime.strptime(row,'%Y-%m-%d')
        prevdate=date-dt.timedelta(days=1)
        prev_date=prevdate.strftime('%Y-%m-%d')
        prev_price=self.__findStockPrice(prev_date,dataset)
        if not (prev_price== 0):
            return self.__findStockPrice(row,dataset)-prev_price
        else:
            return 0
    def __findStockChange(self, row, dataset, dateOffset=0):
        currentstockDay = None
        date = datetime.strptime(row, '%Y-%m-%d')
        targetdate = date + dt.timedelta(days=dateOffset)
        row = date.strftime('%Y-%m-%d')
        target=targetdate.strftime('%Y-%m-%d')
        currentstockDay = dataset[dataset['Date'] == row]
        targetstockDay=dataset[dataset['Date']==target]
        if (not currentstockDay.empty) and (not targetstockDay.empty):
            return targetstockDay.iloc[0]['Close'] > currentstockDay.iloc[0]['Open']
        else:
            if not currentstockDay.empty:
                return self.__findStockChange(row,dataset,dateOffset-1)
            else:
                return False
    def __normalize_headline(self, row):
        result = row.lower()
        # Delete useless character strings
        result = result.replace('...', '')
        whitelist = set(self.whitelist)
        result = ''.join(filter(whitelist.__contains__, result))
        return result

    def to_pandas(self):
        if self.combined_data == None:
            raise TypeError("Run combine_data() first")

        return self.combined_data

    def to_csv(self, filename):
        if self.combined_data is None:
            raise TypeError("Run combine_data() first")
        self.combined_data.to_csv(filename, sep="\t")
        print("Saved To {}".format(filename))

if __name__ == '__main__':
    news = "./uci-news-aggregator.csv"
    DataCombiner("./MSFT.csv", news, category=None,news_filter='Microsoft').to_csv("./combined_MSFT_microsoft_news.csv")
    DataCombiner("./MSFT.csv", news, category='t',).to_csv("./combined_msft_tech_news.csv")
    DataCombiner("./MSFT.csv", news, category='t',news_filter='Microsoft').to_csv("./combined_MSFT_microsoft_tech_news.csv")

    DataCombiner("./GOOG.csv", news, category=None,news_filter='Google').to_csv("./combined_GOOG_google_news.csv")
    DataCombiner("./GOOG.csv", news, category='t',).to_csv("./combined_GOOG_tech_news.csv")
    DataCombiner("./GOOG.csv", news, category='t',news_filter='Google').to_csv("./combined_GOOG_google_tech_news.csv")

    DataCombiner("./IBM.csv", news, category=None,news_filter='IBM').to_csv("./combined_IBM_IBM_news.csv")
    DataCombiner("./IBM.csv", news, category='t',).to_csv("./combined_IBM_tech_news.csv")
    DataCombiner("./IBM.csv", news, category='t',news_filter='IBM').to_csv("./combined_IBM_IBM_tech_news.csv")

    DataCombiner("./AAPL.csv", news, category=None,news_filter='Apple').to_csv("./combined_AAPL_Apple_news.csv")
    DataCombiner("./AAPL.csv", news, category='t',).to_csv("./combined_AAPL_tech_news.csv")
    DataCombiner("./AAPL.csv", news, category='t',news_filter='Apple').to_csv("./combined_AAPL_Apple_tech_news.csv")
    

    


    

