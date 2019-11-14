import os
os.chdir("C://Users//Prakat-L-041//Desktop//ir//group2")

import pandas as pd
import whoosh
from whoosh import scoring
from whoosh.fields import Schema, TEXT
#from whoosh import index
import os, os.path
#from whoosh import index
from whoosh import qparser
#from whoosh.scoring import *
#Below module contains implementations of various scoring algorithms.Default is BM25F

#Load the data
qa_df = pd.read_csv("D:/qa_Electronics.csv")
#update the null values answer field with default value
qa_df["answer"].fillna("Please Provide more information", inplace = True)

#os.chdir("H:/Project NLP-2")
#Schema is created to index on question and answer fields
schema = Schema(question = TEXT (stored = True,  field_boost = 2.0),
                answer = TEXT (stored = True,  field_boost = 2.0),
                text = TEXT)
#Functions to create index for the search fields
def add_stories(i, dataframe, writer):   
    writer.update_document(question = str(dataframe.loc[i, "question"]),
                           answer = str(dataframe.loc[i, "answer"]))

# create and populate index
from whoosh import index
def populate_index(dirname, dataframe, schema):
    # Checks for existing index path and creates one if not present
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    print("Creating the Index")
    ix = index.create_in(dirname, schema)
    with ix.writer() as writer:
        # Imports stories from pandas df
        print("Populating the Index")
        for i in dataframe.index:
            add_stories(i, dataframe, writer)    
            
#Populate index of the csv file           
populate_index("QAData_Index", qa_df, schema)          

##creates index searcher
#query search based on index
from whoosh import index
def index_search(dirname, search_fields, search_query):
    ix = index.open_dir(dirname)
    schema = ix.schema
    
    # Create query parser that looks through designated fields in index
    og = qparser.OrGroup.factory(0.9)
    mp = qparser.MultifieldParser(search_fields, schema, group = og)
    # This is the user query
    q = mp.parse(search_query)
    # Actual searcher, prints top 10 hits
        
    with ix.searcher(weighting=scoring.BM25F(B=0.75,K1=1.5)) as s:
        results = s.search(q, limit = None)
        print("Total Documents: ",ix.doc_count_all())
        print("Retrieved Documents: ",results.estimated_length())
        print(results._get_scorer())
        for i,result in enumerate(results[0:5]):
            print("Search Results: ",result.rank,"Score: ",result.score)
            print("Question: ",result['question'])
            print("Answer: ",result['answer'])
            print("------------------------")
                            
        
index_search("QAData_Index", ['question', 'answer'], u"mac")
