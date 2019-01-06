'''
    Sources used: 
        1) https://www.youtube.com/watch?v=FrmrHyOSyhE
        2) 
'''
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

'''
    # Document : A single piece of text information. 
        - Tweet, email, book, lyrics
        - Equivlent to 1 row or observations
    # Corpus : a collection of documents.
        - A whole data set of rows/observations
    # Token : Word, phrase, symbols derived from a document

----------------------------------------------------------------

    # CountVectorizer takes whats called the bag of words approach
        - Each message is seperated into tokens and the number of 
            times each token occurs in a message is counted

    # TfidfVectorizer also creates a document term matrix from our messages
        - Instead of filling the DTM with token counts it calculates
            term frequency-inverse document frequency value for each word

        - TF-IDF : Is the product of two weights :
                1) Term frequency  2)Inverse document frequency

        - Term frequency : a weight representing how often a word occurs
            in a document. 
                - Several occurrences of a word in one doc --> TF-IDF will increase
        - Inverse document frequency: a weight representing how common a 
            word is across documents.
                - A common word in many documents --> TF-IDF will decrease        

        - The goal is to scale down the effect of tokens/words that occur
            very frequently in a given corpus 
        
'''
documents = ['Hello, how are you!',
                'Win money, win from home.',
                'Call me now.',
                'Hello, Call hello you tomorrow?']

count_vector = CountVectorizer()
count_vector.fit(documents)
print("*"*70,'\n',count_vector.get_feature_names())

#Numpy array
# print('-'*70)
# print(type(count_vector.transform(documents)))
# print(count_vector.transform(documents))
# print('-'*70)

doc_array = count_vector.transform(documents).toarray()
print(doc_array)

frequency_matrix = pd.DataFrame(doc_array,columns=count_vector.get_feature_names())
print('-'*70)
print(frequency_matrix)
print('-'*70)


###########################################
corpus_train = ['Hello, how are you!',
                'Win money, win from home.',
                'Call me now.',
                'Hello, Call hello you tomorrow?']

corpus_test = ['Hello, how are you doing!',
                'Win cash, win from work.']   

count_vector2 = CountVectorizer()

# Either
# Fit/ Learn a vocabulary dictionary of all tokens in the raw documents.
# count_vector2.fit(corpus_train)
'''
    -Transform documents to document-term matrix.
    - Extract token counts out of raw text documents using the vocabulary
     fitted with fit or the one provided to the constructor.
'''
# training_data = count_vector2.transform(corpus_train)

training_data = count_vector2.fit_transform(corpus_train)

testing_data = count_vector2.transform(corpus_test)

frequency_matrix_train = pd.DataFrame(training_data.toarray(),columns=count_vector2.get_feature_names())
frequency_matrix_test = pd.DataFrame(testing_data.toarray(),columns=count_vector2.get_feature_names())
print('-'*70)
print("----------------- Train Data ------------------")
print(frequency_matrix_train)
print('-'*70)
print('-'*70)
print("----------------- Test Data ------------------")
print(frequency_matrix_test)
print('-'*70)