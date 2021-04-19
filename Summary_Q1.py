import steamreviews

#port royale 4 (mixed)
gameID1 = 1024650
#port royale 3 (mostly positive)
gameID2 = 205610
#port royale 2 (mixed)
gameID3 = 12470
gameIDs = [gameID1, gameID2, gameID3]
steamreviews.download_reviews_for_app_id_batch(gameIDs)


import json
import pandas as pd
import os
from io import StringIO

def extract_reviews_to_csv(gameID):
    json_path = 'data/review_' + str(gameID) +'.json'
    json_abspath = os.path.abspath(json_path)
    f = open(json_abspath, 'r')
    data = json.load(f)
    review_list = []
    #Below, we make use of the fact that
    #each review stored in 'data['reviews']' is stored as a tuple {user ID, details of the review},
    #and each 'details of the review' is itself a dictationary.
    #This can be converted into a nested dictinoary using data['reviews'].items(), so that user ID
    #become the key and the details of the review becomes the corresponding value.
    for user_id, review_info in data['reviews'].items():
        #For each item, we build our own disctionary, selecting info we care at the moment
        review_select = {'Steam ID': user_id,
                        'Review': review_info['review'],
                        'Language': review_info['language'],
                        'Recommended': review_info['voted_up'],
                        'Play time': review_info['author']['playtime_forever'],
                        'Purchase': review_info['steam_purchase']}
        review_list.append(review_select)

    review_deposit = json.dumps(review_list, indent=4)
    #convert json object to csv
    df = pd.read_json(StringIO(review_deposit))
    out_file_name = os.path.dirname(json_abspath) + '/review_' + str(gameID) + '.csv'
    df.to_csv(out_file_name, index=False, encoding='utf-8')
    print('Reviews being saved to:', out_file_name)

for gameID in gameIDs: extract_reviews_to_csv(gameID)

def build_pd(gameID):
    json_path = 'data/review_' + str(gameID) +'.csv'
    json_abspath = os.path.abspath(json_path)
    df = pd.read_csv(json_abspath)
    #We at this stage only analyze reviews in English.
    df = df[df['Language'] == 'english']
    df['Review'] = df['Review'].astype(str)
    #remove unnecessary key
    df = df.drop(['Steam ID'], axis=1)
    #add a key for word count in the review
    df['Length'] = df['Review'].apply(lambda x : len(x.split()))
    recommend_df = df[df['Recommended'] == True]
    recommend_df = recommend_df.reset_index(drop=True)
    not_recommend_df = df[df['Recommended'] == False]
    not_recommend_df = not_recommend_df.reset_index(drop=True)
    return recommend_df, not_recommend_df

recommend_df_L = []
not_recommend_df_L = []
for gameID in gameIDs:
    df1, df2 = build_pd(gameID)
    recommend_df_L.append(df1)
    not_recommend_df_L.append(df2)

import matplotlib.pyplot as plt

game_name_L =['Port royale 4', 'Port royale 3', 'Port ryoale 2']

def plot_word_count(recommend_df_L, not_recommend_df_L, game_name_L):

    num_row = len(game_name_L)
    plt.figure(figsize=(9,9))
    j = num_row
    for i in range(0,num_row):
        k = i * 2 + 1
        plt.subplot(j,2,k)
        plt.title(game_name_L[i], fontsize=14)
        plt.hist(recommend_df_L[i]['Length'].values, bins=6)
        plt.xlabel('Word count of \'Recommended\' reviews', fontsize=14)
        plt.ylabel('Counts', fontsize=14)
        plt.xlim(-0.5,1400)
        plt.ylim(0,250)
        plt.tight_layout(pad=3.0)

    for i in range(0,num_row):
        k = i * 2 + 2
        plt.subplot(j,2,k)
        plt.title(game_name_L[i], fontsize=14)
        plt.hist(not_recommend_df_L[i]['Length'].values, bins=6)
        plt.xlabel('Word count of \'Not Recommended\' reviews', fontsize=14)
        plt.ylabel('Counts', fontsize=14)
        plt.xlim(-0.5,1400)
        plt.ylim(0,250)
        plt.tight_layout(pad=3.0)

    plt.show()

plot_word_count(recommend_df_L, not_recommend_df_L, game_name_L)


import string
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def clean_text(df):
    max_iter = len(df)
    #build a table to translate each punctuation to empty text.
    table = str.maketrans('', '', string.punctuation)
    #build a list of stopwords based on an existing library
    stop_words = stopwords.words('english')
    #for transforming each word into its root
    porter = PorterStemmer()
    for i in range(0,max_iter):
        text_raw_str = df.iloc[i]['Review']
        tokens = word_tokenize(text_raw_str)
        tokens_lower = [w.lower() for w in tokens]
        tokens_no_punct = [w.translate(table) for w in tokens_lower]
        words = [word for word in tokens_no_punct if word.isalpha()]
        words = [w for w in words if not w in stop_words]
        words_root = [porter.stem(word) for word in words]
        separator = ' '
        text_clean_str = separator.join(words_root)
        df.at[i, 'Review'] = text_clean_str
    return df

rec_clean_df_L = []
not_rec_clean_df_L = []
for pd in recommend_df_L: rec_clean_df_L.append(clean_text(pd))
for pd in not_recommend_df_L: not_rec_clean_df_L.append(clean_text(pd))



from sklearn.feature_extraction.text import CountVectorizer

#Please be aware that many of the text cleaning above can be done within
#CountVectorizer, except I am not familiair enough to CountVectorizer yet.
def word_count_vect(df):
    count_vect = CountVectorizer(ngram_range=(1,2),
                                stop_words=['game','ca','nt']).fit(df['Review'])

    words = count_vect.transform(df['Review'])
    #the total count for each identified vocabulary
    sum_words = words.sum(axis=0)

    word_count = []
    for word, index in count_vect.vocabulary_.items():
        word_count.append((word, sum_words[0, index]))

    sorted_word_count = sorted(word_count, key=lambda x: x[1],
                                reverse=True)

    return sorted_word_count

rec_word_count_L = []
not_rec_word_count_L = []
for df in rec_clean_df_L: rec_word_count_L.append(word_count_vect(df))
for df in not_rec_clean_df_L: not_rec_word_count_L.append(word_count_vect(df))



def plot_common_words(rec_word_count_L, not_rec_word_count_L, game_name_L):

    num_row = len(game_name_L)
    plt.figure(figsize=(20,20))
    j = num_row
    for i in range(0,num_row):
        k = i * 2 + 1
        data_raw = rec_word_count_L[i]
        x = []  #vocabulary
        y = []  #count
        for data in data_raw:
            x.append(data[0])
            y.append(data[1])

        plt.subplot(j,2,k)
        plt.title(game_name_L[i], fontsize=14)
        plt.barh(x, y)
        plt.ylabel('Feature frequency of \'Recommended\' reviews', fontsize=14)
        plt.ylim(0,250)
        plt.gca().invert_yaxis()
        plt.tight_layout(pad=3.0)

    for i in range(0,num_row):
        k = i * 2 + 2
        data_raw = not_rec_word_count_L[i]
        x = []  #vocabulary
        y = []  #count
        for data in data_raw:
            x.append(data[0])
            y.append(data[1])

        plt.subplot(j,2,k)
        plt.title(game_name_L[i], fontsize=14)
        plt.barh(x, y)
        plt.ylabel('Feature frequency of \'Not Recommended\' reviews', fontsize=14)
        plt.ylim(0,250)
        plt.gca().invert_yaxis()
        plt.tight_layout(pad=3.0)

    plt.show()

plot_common_words(rec_word_count_L, not_rec_word_count_L, game_name_L)


from sklearn.feature_extraction.text import TfidfVectorizer

def apply_tfidf(df):
    tfvec = TfidfVectorizer(ngram_range=(1,2),
        stop_words=['game','ca','nt'],
        max_df=0.9,use_idf=True)

    #fit the model to the reviews
    s = tfvec.fit(df['Review'])
    score = tfvec.transform(df['Review'])
    score_mat = score.todense()
    score_L = score_mat.sum(axis=0)
    name_score_L = pd.DataFrame(score_L.T,
                index=tfvec.get_feature_names(),
                columns=['TD-IDF'])
    sorted_L = name_score_L.sort_values(by=['TD-IDF'],ascending=False)
    return sorted_L

rec_tfidf_L = []
not_rec_tfidf_L = []
for df in rec_clean_df_L: rec_tfidf_L.append(apply_tfidf(df))
for df in not_rec_clean_df_L: not_rec_tfidf_L.append(apply_tfidf(df))

for i in  range(0,3):
    print('Recommended features:')
    print(game_name_L[i])
    print(rec_tfidf_L[i][:6])

for i in  range(0,3):
    print('Not Recommended features:')
    print(game_name_L[i])
    print(not_rec_tfidf_L[i][:6])
