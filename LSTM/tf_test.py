from keras.models import load_model
from tf_data import TF_Data
import argparse

parser = argparse.ArgumentParser(
    prog='tf_test.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='Evaluating LSTM Model')

parser.add_argument("test_sentence")
args = parser.parse_args()
test_sentence_str = args.test_sentence
model = load_model('../data/models/combined_msft_tech_news_today_.hdf5')
filename = '../data/combined_msft_tech_news.csv'

data = TF_Data(filename,top_words=5000)
test_sentence = data.test_sentence(test_sentence_str)
print(model.predict(test_sentence)[0][0])