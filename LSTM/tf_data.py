import pandas
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from unidecode import unidecode
import pandas_datareader.data as web
from datetime import datetime
import datetime as dt
import requests
import pandas as pd
from bs4 import BeautifulSoup
from keras.utils.np_utils import to_categorical
import numpy as np
class TF_Data:
    def __init__(self, data_file, validation_split=0.1, top_words=50):
        self.top_words = top_words
        self.validation_split = validation_split
        self.df = pandas.read_csv(data_file, sep='\t')
        self.headlines = self.df['normalized_headline'].as_matrix()
        self.prices=self.df['price'].as_matrix()
        self.tokenizer = Tokenizer(self.top_words)
        self.tokenizer.fit_on_texts(self.headlines)
        self.all_x = self.tokenizer.texts_to_sequences(self.headlines)
        self.curdate=datetime.strptime('2009-8-15','%Y-%m-%d')
    def generator(self,stock,batch_size=1):
        news=np.zeroes((batch_size,10000))
        stock=np.zeroes((batch_size,500,1))
        label=np.zeroes((batch_size,8))
        while True:
            for i in range(batch_size):
                news[i]=self.get_news(self.curdate)
                stock[i]=self.get_stock(stock,self.curdate)
                label[i]=self.get_label(self.curdate)
                self.curdate=self.curdate+dt.timedelta(days=1)
            yield [news,stock],label
    def get_news(self,date):
        page = requests.get('https://www.thehindu.com/archive/web/'+date.strftime("%Y/%m/%d")) 
        soup = BeautifulSoup(page.content, 'html.parser')
        weblinks = soup.find_all('a')
        pagelinks = []
        for link in weblinks:    
        #url = link.contents[0].find_all('a')[0]   
        href=link.get('href')
        if (href is not None):
          pagelinks.add(href)
        title = []
        thearticle = []
        datestamp=[]

        for link in pagelinks:    
        # store the text for each article
            paragraphtext = []    

            url = link
            # get page text
            try:
              page = requests.get(url)
            except:
              continue
            # parse with BFS
            soup = BeautifulSoup(page.text, 'html.parser')     
            #get date
            curdate=None
            meta=soup.find_all('meta')
            for entry in meta:
              name=entry.get('name')
              if name=="publish-date":
                curdate=entry.get('content')
                datestamp.append(curdate)
                break
            if curdate is not None:
              # get article title
                atitle = soup.find('title')
                thetitle = atitle.get_text()
                title.append(thetitle)

        data = {'Title':title,  
          'PageLink':finallinks,  
          'Date':datestamp}
        news = pd.DataFrame(data=data)
        # using this news data frame to preprocess(just news) like done in CombineData.py

    def get_stocks(self,stock,date):
        # import pandas as pd
        # pd.core.common.is_list_like = pd.api.types.is_list_like
        # period of analysis
        start = date - dt.timedelta(days=500)
        f = web.DataReader(stock, 'yahoo', start, date)

        # nice looking timeseries (DataFrame to panda Series)
        f = f.reset_index()
        f = np.array(f.Close.values,dtype=np.float64)
        return f
        #issue here is we dont get 500 entries using time offset as 500 because there are many holidays in between
    
    def load_data(self, day='tomorrow'):
        # np.random.seed(0)
        self.df['target']=self.df['today']*4+self.df['tomorrow']*2+self.df['day_after_tomorrow']*1
        idx = np.arange(len(self.all_x))
        # np.random.shuffle(idx)
        self.all_y=self.df['target'].as_matrix()
        print(self.all_y)
        price_list=[]
        for index in range(500,len(self.prices)):
            price_list.append(self.prices[index-500:index])
        self.all_price=np.array(price_list)
        self.all_x = np.array(self.all_x)[idx]
        self.all_y = np.array(self.all_y)[idx]
        self.all_y=to_categorical(self.all_y,num_classes=8)
        split = int(self.validation_split * len(self.all_x))
        training_price=self.all_price[:-2]
        training_price= np.reshape(training_price, (training_price.shape[0], training_price.shape[1], 1))
        training_x = self.all_x[500:-2]
        training_y = self.all_y[500:-2]
        validation_price=self.all_price[-2:]
        validation_price= np.reshape(validation_price, (validation_price.shape[0], validation_price.shape[1], 1))
        validation_x = self.all_x[-2:]       
        validation_y = self.all_y[-2:]
        print(validation_y)
        # np.random.seed(None)
        return (training_price,training_x, training_y), (validation_price,validation_x, validation_y)

    def test_sentence(self, text):
        return np.array(sequence.pad_sequences(self.tokenizer.texts_to_sequences([unidecode(text)]),maxlen=100))
        


