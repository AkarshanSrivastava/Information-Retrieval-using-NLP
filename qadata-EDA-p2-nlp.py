# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 15:35:40 2019

@author: A Nanditha
"""
import json
import pandas as pd
'''
open json file as input read file & write the parsed json data using outfile to
 qa_Electronics_formatted.json
'''
with open("qa_Electronics.json", 'r') as infile, open('qa_Electronics_formatted.json', 'w') as outfile:
    lines = infile.readlines() # read lines from input json file
    linecount = len(lines)     # get the total number of lines in the json file
#write the json formatted file starting with '['   
    newline = '[' + '\n'
    outfile.write(newline)

#write lines with ',' appended
    outfile.write('\n'.join([json.dumps(eval(line))+ ',' for line in lines[0:linecount-1]]))  
    line = lines[linecount-1]
    outfile.write(json.dumps(eval(line)))

#write the json formatted file and end with ']'   
    newline = '\n' + ']'
    outfile.write(newline)
 
#save the formatted json file
outfile.close()

#read the formatted json data
with open('qa_Electronics_formatted.json', 'r') as dfread:
    parseddata = dfread.read()

#convert the formatted json data into a dataframe   
dfdata =  pd.read_json(parseddata)

#Display's the first five lines of the dataframe
dfdata.head()

#Display's the last five lines of the dataframe
dfdata.tail()

#view the dataframe
print(dfdata)

#Convert 'dfdata'(dataframe) to csv file
#dfdata.to_csv(r"H:\\Project NLP-2\\qa_Electronics.csv",index=None,header=True)

import os
os.chdir("H:\\Project NLP-2")
#import the required libraries
import pandas as pd
import matplotlib.pyplot as plt
import re
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = stopwords.words('english')
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud, STOPWORDS
from nltk import word_tokenize, pos_tag, pos_tag_sents
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#Adding new stop words
newStopWords = set(STOPWORDS)
stop.extend(newStopWords)
stop.extend(["yes","one","use","bought","got","put","using","still","turn","kind",
             "really","take","","thank","work","well","better","make","see","going",
             "hold","though","either","two","look","good","look","without","please",
             "let","know","im","look","want","anyone","come","need","thank","use",
             "say","show","also","iv","shown","previously","large","result","via",
             "side","build","thrus","etc","getiing","detais","brooken","betr",
             "mptherboard","seem","brans","vis"])

#import re
contractions_expansions_dict = {
  "ain't": "am not",
  "aren't": "are not",
  "bcoz":"because",
  "b'coz":"because",
  "B'Coz":"because",
  "BCOZ":"because",
  "B'coz":"because",
  "can't": "cannot",
  "CAN'T":"cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "charge r": "charger",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "Doe the": "Does the",
  "dose it": "does it",
  "dose the": "does the",
  "Dose it": "does it",
  "Dost": "Does",
  ' "Genie." ': ' "Genie".' ,
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "hav": "have",
  "have": "have",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "How's": "How is",
  "I'd": "I would",
  'I"d': "I would",
  "I'd've": "I would have",
  "id\f" : "if",
  "I'll": "I will",
  "I'll've": "I will have",
  "I'm": "I am",
  'I"m': "I am",
  "i'm" :"i am",
  "Iwould": "I would",
  "iwould": "i would",
  "I've": "I have",
  'I"ve': "I have",
  "i've": "i have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  'it"s': "it is",
  "It's": "It is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "scraches": "scratches",
  "shalli" : "shall i",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "t his": "this",
  "to've": "to have",
  "w/ a": "with a",
  "wasn't": "was not",
  "WIl":"Will",
  "wil l": "will",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "Won't": "Would not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "What's": "What is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "work o a": "work on a",
  "work w/": "work with",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you will",
  "you'll've": "you will have",
  "you're": "you are",
  "you've": "you have",
  "100-400 L" : "100 to 400 L",
  "70-200mm f/2.8G ED VR II AF-S":"70 to 200mm f/2.8G ED VR II AF-S",
  "$5/mo." : "$5 per month",
  "120-140v" : "120 to 140 volts",
  "4.9inx7.7in ?":"4.9 inch x 7.7 inch ?",
  "7' ' ": "7 inch"
    
}

contractions_expansions_re = re.compile('(%s)' % '|'.join(contractions_expansions_dict.keys()))
def expand_contractions(s, contractions_expansions_dict=contractions_expansions_dict):
     def replace(match):
         return contractions_expansions_dict[match.group(0)]
     return contractions_expansions_re.sub(replace, s)
 
#import the csv file    
import pandas as pd
qa_data = pd.read_csv("H:\\Project NLP-2\\qa_Electronics.csv")

#df.drop(['A', 'B'], axis=1, inplace=True)
#drop the unnecessary columns assuming that they are not required for analysis
qa_data.drop(['unixTime','answerType','questionType','answerTime','asin'],axis=1, inplace=True)
#Sub dataframe is created with only 2 columns (question and answer)
qa_data_sub=qa_data[["question","answer"]]
#Coverting both columns to lowercase
qa_data_sub=qa_data_sub.applymap(lambda s:s.lower() if type(s) == str else s)
#Remove null values from the dataframe
qa_data_sub=qa_data_sub.dropna(how='any',axis=0)
#Remove duplicate questions and answers
qa_data_sub_mod=qa_data_sub.drop_duplicates(keep='first', inplace=False)
print(qa_data_sub_mod.head())
#Removing, punctuations,digits and converting the text to lowercase
def clean_text(text):
    text=text.lower()
    text=expand_contractions(text)#remove contractions
    text=re.sub(r"http\S+", "", text)#remove urls
    text=re.sub('\[.*?\]','',text)
    text=re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text=re.sub('\w*\d\w*','',text)
    text=re.sub('[''""]','',text)
    text=re.sub('\n','',text)
    text=re.sub(' x ','',text)
    return text
clean_txt= lambda x: clean_text(str(x))
#Implement the text cleaning on question and answer columns
qa_data_sub_mod_q=pd.DataFrame(qa_data_sub_mod.question.apply(clean_txt))
qa_data_sub_mod_a=pd.DataFrame(qa_data_sub_mod.answer.apply(clean_txt))
qa_data_sub_mod_q['question_withoutstopwords']=qa_data_sub_mod_q['question'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
qa_data_sub_mod_a['answer_withoutstopwords']=qa_data_sub_mod_a['answer'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#Lemmatization
#word_tokenizer = nltk.tokenize.WhitespaceTokenizer()
#lemmatizer = nltk.stem.WordNetLemmatizer()
#def lemmatize_text(text):
#    return [lemmatizer.lemmatize(w,'v') for w in word_tokenizer.tokenize(str(text))]
#qa_data_sub_mod_q['question_lemma'] = pd.DataFrame(qa_data_sub_mod_q['question_withoutstopwords'], columns=['question_withoutstopwords'])
#qa_data_sub_mod_a['answer_lemma'] = pd.DataFrame(qa_data_sub_mod_a['answer_withoutstopwords'], columns=['answer_withoutstopwords'])
#qa_data_sub_mod_q['question_lemma'] = qa_data_sub_mod_q.question_lemma.apply(lemmatize_text)
#qa_data_sub_mod_a['answer_lemma'] = qa_data_sub_mod_a.answer_lemma.apply(lemmatize_text)
#print(qa_data_sub_mod_q['question_lemma'].head())

#Lemmatization for question and answer
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
#Instantiate lemmatizer
lemmatizer = WordNetLemmatizer()
def word_lemmatizer(text):
    #lem_text = [lemmatizer.lemmatize(i) for i in text]
    lem_text = [lemmatizer.lemmatize(text)]
    return lem_text

#my_csv['lemmatizer']=my_csv['stopwords'].apply(lambda x: word_lemmatizer(x))
#my_csv4['new_ans_lemmatizer']=my_csv4['new_ans_remove_singlecharacters'].apply(lambda x: word_lemmatizer(x))
#print(my_csv4[['new_ans_remove_singlecharacters','new_ans_lemmatizer']].head())
qa_data_sub_mod_q['question_lemma'] = pd.DataFrame(qa_data_sub_mod_q['question_withoutstopwords'], columns=['question_withoutstopwords'])
qa_data_sub_mod_a['answer_lemma'] = pd.DataFrame(qa_data_sub_mod_a['answer_withoutstopwords'], columns=['answer_withoutstopwords'])
qa_data_sub_mod_q['question_lemma'] = qa_data_sub_mod_q.question_lemma.apply(word_lemmatizer)
qa_data_sub_mod_a['answer_lemma'] = qa_data_sub_mod_a.answer_lemma.apply(word_lemmatizer)
print(qa_data_sub_mod_q['question_lemma'].head())
##Stemming for question and answer
ps = PorterStemmer()
qa_data_sub_mod_q['question_stem'] = qa_data_sub_mod_q['question_lemma'].apply(lambda x: ' '.join([ps.stem(t) for t in x]))
qa_data_sub_mod_a['answer_stem'] = qa_data_sub_mod_a['answer_lemma'].apply(lambda x: ' '.join([ps.stem(t) for t in x]))
print(qa_data_sub_mod_q['question_stem'].head())
#Convert the dataframe to list
question_txt=qa_data_sub_mod_q['question_stem'].tolist()
answer_txt=qa_data_sub_mod_a['answer_stem'].tolist()
print(answer_txt[1:5])
#Question and Answer WordCloud
wordcloud_question = WordCloud(width=2800,height=2400,max_font_size=500,random_state=42,stopwords=stop).generate(str(question_txt))
plt.imshow(wordcloud_question)
wordcloud_answer = WordCloud(width=2800,height=2400,max_words=150,max_font_size=500,random_state=42,stopwords=stop).generate(str(answer_txt))
plt.imshow(wordcloud_answer)

#Get Positive words
with open("H:\\Project NLP-2\\positive words.txt","r") as pos:
  poswords = pos.read().split("\n")
#Get Negative Words
with open("H:\\Project NLP-2\\negative words.txt","r") as neg:
            negwords = neg.read().split("\n")
#Positive Wordclouds
question_text_pos = " ".join ([w for w in question_txt if w in poswords])
answer_text_pos = " ".join ([w for w in answer_txt if w in poswords])
#question
wordcloud_question_pos = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400,max_words=150,max_font_size=500,random_state=42,
                      stopwords=stop).generate((question_text_pos))          
plt.imshow(wordcloud_question_pos)
#answer
wordcloud_answer_pos = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400,max_words=150,max_font_size=500,random_state=42,
                      stopwords=stop).generate((answer_text_pos))
plt.imshow(wordcloud_answer_pos)
#Negative Wordclouds
question_text_neg = " ".join ([w for w in question_txt if w in negwords])
answer_text_neg = " ".join ([w for w in answer_txt if w in negwords])
#question
wordcloud_question_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400,max_words=150,max_font_size=500,random_state=42,
                      stopwords=stop).generate((question_text_neg))
plt.imshow(wordcloud_question_neg)
#answer
wordcloud_answer_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400,max_words=150,max_font_size=500,random_state=42,
                      stopwords=stop).generate((answer_text_neg))
plt.imshow(wordcloud_answer_neg)

#creating corpus for question and qnswer columns
#print(type(qa_data_sub_mod_q['question_lemma']))
#print(qa_data_sub_mod_q['question_lemma'].head())
#print((qa_data_sub_mod_q.dtypes))
qa_data_sub_mod_q['question_lemma']=qa_data_sub_mod_q['question_lemma'].astype(str)
#print((qa_data_sub_mod_q.dtypes))
#print(type(qa_data_sub_mod_q['question_lemma']))
qa_data_sub_mod_a['answer_lemma']=qa_data_sub_mod_a['answer_lemma'].astype(str)
#print((qa_data_sub_mod_a.dtypes))
corpusQ=qa_data_sub_mod_q['question_lemma']
corpusA=qa_data_sub_mod_a['answer_lemma']

#Barplots for question
#Unigram,Bi-Gram and Tri_Gram Analysis for question
#Most frequently occuring Uni-grams for question
def get_top_n_words(corpusQ, n=None):
    vec = CountVectorizer().fit(corpusQ)
    bag_of_words = vec.transform(corpusQ)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in      
                   vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                       reverse=True)
    return words_freq[:n]
#Convert most freq unigram words to dataframe for plotting bar plot for question
top_words_Q = get_top_n_words(corpusQ, n=25)
top_df_q = pd.DataFrame(top_words_Q)
top_df_q.columns=["Word", "Frequency"]
top_df_q.head(25)
#Barplot of most freq words for question
import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
g = sns.barplot(x="Word", y="Frequency", data=top_df_q)
g.set_xticklabels(g.get_xticklabels(), rotation=30)

#Most frequently occuring Bi-grams for question
def get_top_n2_words(corpusQ, n=None):
    vec1 = CountVectorizer(ngram_range=(2,2),  
            max_features=2000).fit(corpusQ)
    bag_of_words = vec1.transform(corpusQ)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]
#Convert most freq bi-gram words to dataframe for plotting bar plot for question
top2_words_Q = get_top_n2_words(corpusQ, n=20)
top2_df_q = pd.DataFrame(top2_words_Q)
top2_df_q.columns=["Bi-gram", "Frequency"]
print(top2_df_q)
#Barplot of most freq Bi-grams for question
import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
h=sns.barplot(x="Bi-gram", y="Frequency", data=top2_df_q)
h.set_xticklabels(h.get_xticklabels(), rotation=45)

#Most frequently occuring Tri-grams for question
def get_top_n3_words(corpusQ, n=None):
    vec1 = CountVectorizer(ngram_range=(3,3), 
           max_features=2000).fit(corpusQ)
    bag_of_words = vec1.transform(corpusQ)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]
#Convert most freq tri-gram words to dataframe for plotting bar plot for question
top3_words_Q = get_top_n3_words(corpusQ, n=20)
top3_df_q = pd.DataFrame(top3_words_Q)
top3_df_q.columns=["Tri-gram", "Frequency"]
print(top3_df_q)
#Barplot of most freq Tri-grams for question
import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
i=sns.barplot(x="Tri-gram", y="Frequency", data=top3_df_q)
i.set_xticklabels(i.get_xticklabels(), rotation=45)

#Barplots for answer
#Unigram,Bi-Gram and Tri_Gram Analysis for answer
#Most frequently occuring Uni-grams for answer
def get_top_n_words(corpusA, n=None):
    vec = CountVectorizer().fit(corpusA)
    bag_of_words = vec.transform(corpusA)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in      
                   vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                       reverse=True)
    return words_freq[:n]
#Convert most freq unigram words to dataframe for plotting bar plot for answer
top_words_A = get_top_n_words(corpusA, n=25)
top_df_a = pd.DataFrame(top_words_A)
top_df_a.columns=["Word", "Frequency"]
top_df_a.head(25)
#Barplot of most freq words for answer
import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
g = sns.barplot(x="Word", y="Frequency", data=top_df_a)
g.set_xticklabels(g.get_xticklabels(), rotation=30)

#Most frequently occuring Bi-grams for answer
def get_top_n2_words(corpusA, n=None):
    vec1 = CountVectorizer(ngram_range=(2,2),  
            max_features=2000).fit(corpusA)
    bag_of_words = vec1.transform(corpusA)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]
#Convert most freq bi-gram words to dataframe for plotting bar plot for answer
top2_words_A = get_top_n2_words(corpusA, n=20)
top2_df_a = pd.DataFrame(top2_words_A)
top2_df_a.columns=["Bi-gram", "Frequency"]
print(top2_df_a)
#Barplot of most freq Bi-grams for answer
import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
h=sns.barplot(x="Bi-gram", y="Frequency", data=top2_df_a)
h.set_xticklabels(h.get_xticklabels(), rotation=45)

#Most frequently occuring Tri-grams for answer
def get_top_n3_words(corpusA, n=None):
    vec1 = CountVectorizer(ngram_range=(3,3), 
           max_features=2000).fit(corpusA)
    bag_of_words = vec1.transform(corpusA)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]
#Convert most freq tri-gram words to dataframe for plotting bar plot for answer
top3_words_A = get_top_n3_words(corpusA, n=20)
top3_df_a = pd.DataFrame(top3_words_A)
top3_df_a.columns=["Tri-gram", "Frequency"]
print(top3_df_a)
#Barplot of most freq Tri-grams for answer
import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
j=sns.barplot(x="Tri-gram", y="Frequency", data=top3_df_a)
j.set_xticklabels(j.get_xticklabels(), rotation=45)


#2:Feature Engineering for question
#2:1:LDA-TOPIC EXTRACTION(QUESTION)
questext_corpus = qa_data_sub_mod_q['question_lemma'].values.tolist()
#print(corpus[1])
#my_csv2
#Tokenizing words
import gensim
import nltk
from nltk import pos_tag
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
 
def doc_to_words_ques(sentences):
  for sentence in sentences:
    yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
 
words_ques = list(doc_to_words_ques(questext_corpus))
 
print(words_ques[1:5])

#2.1.a:Lemmatization for Question(LDA)
import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def lemmatization(texts, allowed_postags_ques=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for idx, sent in enumerate(texts):
        if (idx) % 500 == 0:
            print(str(idx) + ' documents lemmatised')
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags_ques])
    return texts_out
 
data_lemmatised_ques = lemmatization(words_ques, allowed_postags_ques=['NOUN', 'ADJ', 'VERB', 'ADV'])
print(data_lemmatised_ques[0:5])

#2.1.b:create Dictionary for Question(LDA)
import gensim.corpora as corpora
id2word_ques = corpora.Dictionary(data_lemmatised_ques)
print(id2word_ques)

#2.1.c:Create Corpus for Question(LDA)
corpus_ques = [id2word_ques.doc2bow(text) for text in data_lemmatised_ques]
print(corpus_ques[:2])

#2.1.d:Building LDA Model(For QUESTION)
import gensim
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
lda_model_ques = gensim.models.ldamodel.LdaModel(corpus=corpus_ques,id2word=id2word_ques,num_topics=5,per_word_topics=True)
print(lda_model_ques)    

#If the number_topics = 5, then below is the o/p
#o/p:#LdaModel(num_terms=47362, num_topics=5, decay=0.5, chunksize=2000)

#If the number_topics = 10, then below is the o/p
#o/p:#LdaModel(num_terms=47362, num_topics=5, decay=0.5, chunksize=2000)

from pprint import pprint
pprint(lda_model_ques.print_topics())
doc_lda_ques = lda_model_ques[corpus_ques]
print(doc_lda_ques)

#2.1.e:LDA-Visualising The Topic Model(For QUESTION) for 10 Topics
import pyLDAvis
#import pyLDAvis.genism
from pyLDAvis import gensim
pyLDAvis.enable_notebook()
vis_ques = pyLDAvis.gensim.prepare(lda_model_ques,corpus_ques,id2word_ques)
vis_ques

#2.1.f:LDA-Practical Applications Of The Topic Model(For QUESTION)
def format_topics_sentences(ldamodel=lda_model_ques,corpus=corpus_ques,texts=questext_corpus):
  #Array of top 10 topics
  top10array = []
    
  for row in range(ldamodel.num_topics):
    wp = ldamodel.show_topic(row)
    topic_keywords = ", ".join([word for word,prop in wp])
    top10array.append((row+1,topic_keywords))
    
  top10dict = dict(top10array)
    
  sent_topics_df = pd.DataFrame(pd.DataFrame([sorted(topic[0],key=lambda x: (x[1]), reverse = True) for topic in ldamodel[corpus_ques]])[0])
  sent_topics_df.columns = ["Data_Question"]
  sent_topics_df['Dominant_Topic_Question'] = sent_topics_df.Data_Question.apply(lambda x: x[0]+1)
  sent_topics_df['Perc_Contribution_Question'] = sent_topics_df.Data_Question.apply(lambda x: round(x[1],4))
  sent_topics_df['Topic_Keywords_Question'] = sent_topics_df.Dominant_Topic_Question.apply(lambda x: top10dict[x])
    
  #Add original text to the end of the output
  contents = pd.Series(texts)
  sent_topics_df = pd.concat([sent_topics_df,contents.rename("Text_Question")], axis=1)
  sent_topics_df = sent_topics_df[['Dominant_Topic_Question','Perc_Contribution_Question','Topic_Keywords_Question','Text_Question']]
  return(sent_topics_df)
#df_topic_sents_keywords_Question = format_topics_sentences(ldamodel=lda_model_ques, corpus=corpus_ques, texts=questext_corpus)
df_topic_sents_keywords_Question = format_topics_sentences()
df_topic_sents_keywords_Question

#2.1.g:Group top 5 sentences under each topic for question
sent_topics_sorteddf_mallet_ques = pd.DataFrame()
 
sent_topics_outdf_grpd = df_topic_sents_keywords_Question.groupby('Dominant_Topic_Question')
 
for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet_ques = pd.concat([sent_topics_sorteddf_mallet_ques, 
                                             grp.sort_values(['Perc_Contribution_Question'], ascending=[0]).head(1)], 
                                            axis=0)
# Reset Index    
sent_topics_sorteddf_mallet_ques.reset_index(drop=True, inplace=True)
 
# Format
sent_topics_sorteddf_mallet_ques.columns = ['Topic_Num_Question', "Topic_Perc_Contribution_Question", "Topic_Keywords_Question", "Text_Question"]
 
# Show
#sent_topics_sorteddf_mallet_ques.head()
sent_topics_sorteddf_mallet_ques

#2.1.h:Perplexity and coherence scores for question
#Perplexity score for question
print('\nPerplexity For Question:',lda_model_ques.log_perplexity(corpus_ques))
#o/p:Perplexity For Question: -7.961109885348404

#Compute coherence score for question
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
coherence_model_lda_ques=CoherenceModel(model=lda_model_ques,texts=data_lemmatised_ques,dictionary=id2word_ques,coherence='c_v')
coherence_lda_ques=coherence_model_lda_ques.get_coherence()
print('\nCoherence Score For Question:',coherence_lda_ques)
#o/p:Coherence Score For Question: 0.23627604478045625a

#2.2:LDA-TOPIC EXTRACTION(ANSWER)
anstext_corpus = qa_data_sub_mod_a['answer_lemma'].values.tolist()
#print(corpus[1])
#my_csv2
#Tokenizing words
import gensim
import nltk
from nltk import pos_tag
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
 
def doc_to_words_ans(sentences):
  for sentence in sentences:
    yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
 
words_ans = list(doc_to_words_ans(anstext_corpus))
 
print(words_ans[1:5])

#2.2.a:Lemmatisation for Answer(LDA)
import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


def lemmatization(texts, allowed_postags_ans=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for idx, sent in enumerate(texts):
        if (idx) % 500 == 0:
            print(str(idx) + ' documents lemmatised')
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags_ans])
    return texts_out
 
data_lemmatised_ans = lemmatization(words_ans, allowed_postags_ans=['NOUN', 'ADJ', 'VERB', 'ADV'])
print(data_lemmatised_ans[0:5])

#2.2.b:create Dictionary for Answer(LDA)
import gensim.corpora as corpora
id2word_ans = corpora.Dictionary(data_lemmatised_ans)
print(id2word_ans)

#2.2.c:Create Corpus for Answer(LDA)
corpus_ans = [id2word_ans.doc2bow(text) for text in data_lemmatised_ans]
print(corpus_ans[:2])

#2.2.d:Building LDA Model(For ANSWER)
import gensim
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
lda_model_ans = gensim.models.ldamodel.LdaModel(corpus=corpus_ans,id2word=id2word_ans,num_topics=5,per_word_topics=True)
print(lda_model_ans)    

#If the number_topics = 5, then below is the o/p
#o/p:#LdaModel(num_terms=77924, num_topics=5, decay=0.5, chunksize=2000)

#If the number_topics = 10, then below is the o/p
#o/p:#LdaModel(num_terms=77924, num_topics=5, decay=0.5, chunksize=2000)

from pprint import pprint
pprint(lda_model_ans.print_topics())
doc_lda_ans = lda_model_ans[corpus_ans]
print(doc_lda_ans)

#2.2.e:LDA-Visualising The Topic Model(For ANSWER) for 10 Topics
import pyLDAvis
#import pyLDAvis.genism
from pyLDAvis import gensim
pyLDAvis.enable_notebook()
vis_ans = pyLDAvis.gensim.prepare(lda_model_ans,corpus_ans,id2word_ans)
vis_ans

#2.2.f:LDA-Practical Applications Of The Topic Model(For ANSWER)
def format_topics_sentences(ldamodel=lda_model_ans,corpus=corpus_ans,texts=anstext_corpus):
  #Array of top 10 topics
  top10array = []
    
  for row in range(ldamodel.num_topics):
    wp = ldamodel.show_topic(row)
    topic_keywords = ", ".join([word for word,prop in wp])
    top10array.append((row+1,topic_keywords))
    
  top10dict = dict(top10array)
    
  sent_topics_df = pd.DataFrame(pd.DataFrame([sorted(topic[0],key=lambda x: (x[1]), reverse = True) for topic in ldamodel[corpus_ans]])[0])
  sent_topics_df.columns = ["Data_Answer"]
  sent_topics_df['Dominant_Topic_Answer'] = sent_topics_df.Data_Answer.apply(lambda x: x[0]+1)
  sent_topics_df['Perc_Contribution_Answer'] = sent_topics_df.Data_Answer.apply(lambda x: round(x[1],4))
  sent_topics_df['Topic_Keywords_Answer'] = sent_topics_df.Dominant_Topic_Answer.apply(lambda x: top10dict[x])
    
  #Add original text to the end of the output
  contents = pd.Series(texts)
  sent_topics_df = pd.concat([sent_topics_df,contents.rename("Text_Answer")], axis=1)
  sent_topics_df = sent_topics_df[['Dominant_Topic_Answer','Perc_Contribution_Answer','Topic_Keywords_Answer','Text_Answer']]
  return(sent_topics_df)
#df_topic_sents_keywords_Answer = format_topics_sentences(ldamodel=lda_model_ans, corpus=corpus_ans, texts=anstext_corpus)
df_topic_sents_keywords_Answer = format_topics_sentences()
df_topic_sents_keywords_Answer

#2.2.g:Group top 5 sentences under each topic for answer
sent_topics_sorteddf_mallet_ans = pd.DataFrame()
 
sent_topics_outdf_grpd = df_topic_sents_keywords_Answer.groupby('Dominant_Topic_Answer')
 
for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet_ans = pd.concat([sent_topics_sorteddf_mallet_ans, 
                                             grp.sort_values(['Perc_Contribution_Answer'], ascending=[0]).head(1)], 
                                            axis=0)
 
# Reset Index    
sent_topics_sorteddf_mallet_ans.reset_index(drop=True, inplace=True)
 
# Format
sent_topics_sorteddf_mallet_ans.columns = ['Topic_Num_Answer', "Topic_Perc_Contribution_Answer", "Topic_Keywords_Answer", "Text_Answer"]
 
# Show
#sent_topics_sorteddf_mallet_ans.head()
sent_topics_sorteddf_mallet_ans

#Perplexity and coherence scores for answer
#Perplexity score for answer
print('\nPerplexity For Answer:',lda_model_ans.log_perplexity(corpus_ans))
#o/p:Perplexity For Answer:-7.691656166846952

#Compute coherence score for answer
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
coherence_model_lda_ans=CoherenceModel(model=lda_model_ans,texts=data_lemmatised_ans,dictionary=id2word_ans,coherence='c_v')
coherence_lda_ans=coherence_model_lda_ans.get_coherence()
print('\nCoherence Score For Answer:',coherence_lda_ans)
#o/p:Coherence Score For Answer: 0.6068922394830516

#3.Advanced text processing
#Sentiment Analysis for question and answer
#sentiment polarity which lies in the range of [-1,1]
#where '1' means positive sentiment and '-1' means negative sentiment
#from textblob import TextBlob
#3.1.a:sentiment polarity for question
qa_data['question_polarity']=qa_data['question'].map(lambda text:TextBlob(text).sentiment.polarity)
qa_data['question_polarity'].describe()
#o/p:count    314263.000000
#mean          0.075658
#std           0.196443
#min          -1.000000
#25%           0.000000
#50%           0.000000
375%           0.125000
#max           1.000000
#Name: question_polarity, dtype: float64

#3.1.a:sentiment polarity for answer
qa_data['answer_polarity']=qa_data['answer'].map(lambda text:TextBlob(str(text)).sentiment.polarity)
qa_data['answer_polarity'].describe()
#o/p:count    314263.000000
#mean          0.110605
#std           0.238930
#min          -1.000000
#25%           0.000000
#50%           0.000000
#75%           0.220238
#max           1.000000
#Name: answer_polarity, dtype: float64
#print(qa_data.dtypes)

#3.2.a:sentences with positive polarity for question
print('5 random sentences with the highest positive sentiment polarity for question:\n')
question_pos_pol=qa_data.loc[qa_data['question_polarity']==1,['question']].sample(5).values
for pos_sentences_q in question_pos_pol:
  print(pos_sentences_q[0])
#3.2.b:sentences with positive polarity for answer
print('5 random sentences with the highest positive sentiment polarity for answer:\n')
answer_pos_pol=qa_data.loc[qa_data['answer_polarity']==1,['answer']].sample(5).values
for pos_sentences_a in answer_pos_pol:
  print(pos_sentences_a[0])

#3.3.a:sentences with negative polarity for question
print('5 random sentences with the highest negative sentiment polarity for question:\n')
question_neg_pol=qa_data.loc[qa_data['question_polarity']==-1,['question']].sample(5).values
for neg_sentences_q in question_neg_pol:
  print(neg_sentences_q[0])
#3.3.b:sentences with negative polarity for answer
print('5 random sentences with the highest negative sentiment polarity for answer:\n')
answer_neg_pol=qa_data.loc[qa_data['answer_polarity']==-1,['answer']].sample(5).values
for neg_sentences_a in answer_neg_pol:
  print(neg_sentences_a[0])
  
#3.4.a:sentences with neutral polarity for question
print('5 random sentences with the neutral sentiment polarity for question:\n')
question_neu_pol=qa_data.loc[qa_data['question_polarity']==0,['question']].sample(5).values
for neu_sentences_q in question_neu_pol:
  print(neu_sentences_q[0])
#3.4.b:sentences with neutral polarity for answer
print('5 random sentences with the neutral sentiment polarity for answer:\n')
answer_neu_pol=qa_data.loc[qa_data['answer_polarity']==0,['answer']].sample(5).values
for neu_sentences_a in answer_neu_pol:
  print(neu_sentences_a[0])

#3.5:polarity percentages for question and answer
#3.5.a:positive percentage for question
question_percentage_positive=qa_data.loc[qa_data.question_polarity>0,['question']].values
len(question_percentage_positive)
print("Positive Percentage In Question:{} %".format(100*len(question_percentage_positive)/len(qa_data.question_polarity)))
#o/p:Positive Percentage In Question:30.20050085437993 %
#3.5.b:positive percentage for answer
answer_percentage_positive=qa_data.loc[qa_data.answer_polarity>0,['answer']].values
len(answer_percentage_positive)
print("Positive Percentage In Answer:{} %".format(100*len(answer_percentage_positive)/len(qa_data.answer_polarity)))
#o/p:Positive Percentage In Answer:47.44115597445452 %

#3.6.a:negative percentage for question
question_percentage_negative=qa_data.loc[qa_data.question_polarity<0,['question']].values
len(question_percentage_negative)
print("Negative Percentage In Question:{} %".format(100*len(question_percentage_negative)/len(qa_data.question_polarity)))
#o/p:Negative Percentage In Question:10.57967371278197 %
#3.6.b:negative percentage for answer
answer_percentage_negative=qa_data.loc[qa_data.answer_polarity<0,['answer']].values
len(answer_percentage_negative)
print("Negative Percentage In Answer:{} %".format(100*len(answer_percentage_negative)/len(qa_data.answer_polarity)))
#o/p:Negative Percentage In Answer:14.428997368446174 %

#3.6.a:neutral percentage for question
question_percentage_neutral=qa_data.loc[qa_data.question_polarity==0,['question']].values
len(question_percentage_neutral)
print("Neutral Percentage In Question:{} %".format(100*len(question_percentage_neutral)/len(qa_data.question_polarity)))
#o/p:Neutral Percentage In Question:59.2198254328381 %
#3.6.b:neutral percentage for answer
answer_percentage_neutral=qa_data.loc[qa_data.answer_polarity==0,['answer']].values
len(answer_percentage_neutral)
print("Neutral Percentage In Answer:{} %".format(100*len(answer_percentage_neutral)/len(qa_data.answer_polarity)))
#o/p:Neutral Percentage In Answer:38.12984665709931 %
