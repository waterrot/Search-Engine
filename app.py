from flask import Flask, render_template, request

import numpy as np
import csv
import math

#pre-proccessing libaries
import nltk
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('omw-1.4')

app = Flask(__name__)

@app.route('/')
def search_page():
    return render_template('search.html')

@app.route('/results', methods=['POST'])
def search_results():
    query_string = request.form['query']

    # the query split into a list
    query = query_string.split(" ")

    # get the perprocessing values
    outputs = perprocessing()
    matrix = outputs[0]
    page_rank_matrix = outputs[1]
    tf_idf = outputs[2]

    # calculate the matrix results
    resultsBooleanAndPageRankModel = CalcBooleanPageRankModel(query, page_rank_matrix, matrix)
    resultTFIDFModel = CalcTfIdf(query, tf_idf)

    return render_template('results.html', query=query_string, resultsBooleanAndPageRankModel=resultsBooleanAndPageRankModel, resultTFIDFModel=resultTFIDFModel)


def perprocessing():
  #termincidences
  stop_words = set(stopwords.words("dutch"))

  def clean_words(words):
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    words = [re.sub(r'[^\w\s]','',word) for word in words]
    words = [word.lower() for word in words]
    words = [word for word in words if word not in stop_words]
    stemmed_words = [stemmer.stem(word) for word in words]
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return lemmatized_words

  wordincidences = {}
  filenames = ['document1.txt', 'document2.txt', 'document3.txt', 'document4.txt', 'document5.txt', 'document6.txt', 'document7.txt' , 'document8.txt', 'document9.txt', 'document10.txt']
  for filename in filenames:
    with open(filename, "r") as file:
      words = file.read().split()
      cleaned_words = clean_words(words)
      wordincidences[filename] = {word: 1 for word in cleaned_words}

  #term-document matrix
  filenames = set()
  terms = set()

  for filename, term_dict in wordincidences.items():
    filenames.add(filename)
    for term in term_dict:
        terms.add(term)

  matrix = [[''] + list(filenames)]
  for term in terms:
    row = [term]
    for filename in filenames:
        if term in wordincidences[filename]:
            row.append(wordincidences[filename][term])
        else:
            row.append(0)
    matrix.append(row)

  #PageRank value
  damping = 0.9
  file_path = "documents_graph.txt"

  def compute_rank(file_path, damping): 
    file = open(file_path)
    reader = csv.reader(file)
    rank_dict = {} 
    matrix = [] 
    split_data = []
    for line in reader: 
        matrix.append(line)
    for line in matrix:
        for words in line:
            split_data.append(words.split())
    
    for i in range(50):
        for sub in split_data:
            rank_dict[sub[0]] = 1 - damping
            for inner_sub in split_data: 
                if inner_sub[0] != sub[0]: 
                    if sub[0] in inner_sub:
                        if inner_sub[0] in rank_dict:
                            rank_dict[sub[0]] += damping * (rank_dict[inner_sub[0]]/(len(inner_sub)-1)) 
                        else:
                            rank_dict[sub[0]] += damping * (1/(len(inner_sub)-1))  
    return rank_dict

  page_rank_matrix = compute_rank(file_path, damping)

  #td-idf model

  #termfreq
  wordcounts = {}
  filenames = ['document1.txt', 'document2.txt', 'document3.txt', 'document4.txt', 'document5.txt', 'document6.txt', 'document7.txt' , 'document8.txt', 'document9.txt', 'document10.txt']
  for filename in filenames:
    with open(filename, "r") as file:
        words = file.read().split()
        cleaned_words = clean_words(words)
        word_frequency = {}
        for word in cleaned_words:
            if word in word_frequency:
                word_frequency[word] += 1
            else:
                word_frequency[word] = 1
        wordcounts[filename] = word_frequency

  #term freq matrix
  unique_words = set()
  for doc in wordcounts.values():
    unique_words.update(doc.keys())

  matrix = [[''] + list(wordcounts.keys())]
  for word in unique_words:
    row = [word]
    for filename in wordcounts:
        if word in wordcounts[filename]:
            row.append(wordcounts[filename][word])
        else:
            row.append(0)
    matrix.append(row)

  #freq matrix to weight matrix
  N = len(filenames)

  tf_idf = [['']+filenames]
  for row in matrix[1:]:
    term = row[0]
    n = sum([1 for value in row[1:] if value != 0])
    idf = math.log2(N/n)
    tf_idf_values = [value * idf for value in row[1:]]
    tf_idf.append([term]+tf_idf_values)

  return matrix, page_rank_matrix, tf_idf


#function to calculate the Boolean en PageRank matrix value
def CalcBooleanPageRankModel(query, page_rank_matrix, matrix):
    matching_queries = []

    i = 0
    while i < len(matrix):
        single_list = matrix[i]

        for q in query:
            if q == single_list[0]:
                inner_list = []
                j = 1
                while j < len(single_list):
                    if single_list[j] == 1:
                        item = matrix[0]
                        inner_list.append(item[j])
                        j = j + 1
                    else:
                        j = j + 1
                    matching_queries.extend([inner_list])
            i = i + 1

    # find the double elements inside the nested list, these items then a hit for
    # all the queries
    common_elements = list(set.intersection(*map(set,matching_queries)))

    # collect all the ranks with the element names in this list
    list_ranks = []

    # if the item of common_elements is equal to the first item of a nested list
    # on the pagerank.csv document, display the value that belongs to that item
    for item in page_rank_matrix:
        for element in common_elements:
            if element == item[0]:
                single_rank = [element, item[1]]
                list_ranks.extend([single_rank])

    #rank the matrix based on the score of each item
    unranked_page_score_matrix = np.array(list_ranks)
    ranked_page_score_matrix = np.array(sorted(unranked_page_score_matrix, key=lambda score:score[1], reverse=True))

    return ranked_page_score_matrix


#function to calculate the TF-IDF matrix
def CalcTfIdf(query, tf_idf):
    # make a nested list with the item name and all the associated vector lengths
    j = 0
    nested_list = []
    for i in range(1,len(tf_idf[j])):
        inner_list = []
        for j in range(len(tf_idf)):
            single_list = tf_idf[j]
            if type(single_list[i]) == str:
                # add the name of the item to the nested list
                inner_list.append(single_list[i])
            else:
                # add the square frequencies to the nested list 
                inner_list.append(single_list[i]**2)
            nested_list.append(inner_list)

    # add all the vector lengths of the nested lists together per item
    vector_lengths_list = []
    for new_list in nested_list:
        sum = 0
        j = 0
        inner_list = []
        for j in range(len(new_list)):
            single_list = new_list[j]
            # if the value inside the list of the nested list is a number, 
            # then add them to the sum
            if type(single_list) == int or type(single_list) == float:
                sum += single_list
                # if the value inside the list of the nested list is a string,
                # then add this string to the list, because this will be the name 
                # of the vector length
            elif type(single_list) == str:
                inner_list.append(single_list)
        inner_list.append(math.sqrt(sum))
        vector_lengths_list.append(inner_list)

    # make the product between the document en query vector. The query vector is 
    # one if it is in the query request and zero if it is not. the new_nested_list
    # will give the name of the item with all the corresponding values
    j = 0
    new_nested_list = []
    for i in range(1,len(tf_idf[j])):
        inner_list = []
        for j in range(len(tf_idf)):
            single_list = tf_idf[j]
            if single_list[0] in query:
                inner_list.append(single_list[i])
            elif type(single_list[i]) == int or type(single_list[i]) == float:
                inner_list.append(0)
            else:
                inner_list.append(single_list[i])
            new_nested_list.append(inner_list)

    # add all the products of the nested lists together per item
    product_document_query_vector = []
    for products in new_nested_list:
        sum = 0
        j = 0
        inner_list = []
        for j in range(len(products)):
            single_product = products[j]
            # if the value inside the list of the nested list is a number, 
            # then add them to the sum
            if type(single_product) == int or type(single_product) == float:
                sum += single_product
            # if the value inside the list of the nested list is a string,
            # then add this string to the list, because this will be the name 
            # of the product
            elif type(single_product) == str:
                inner_list.append(single_product)
        inner_list.append(sum)
        product_document_query_vector.append(inner_list)

    # calculate the cosine similarity
    i = 0
    cosine_similarity = []
    while i < len(product_document_query_vector):
        for length in vector_lengths_list:
            inner_list = []
            if length[0] == product_document_query_vector[i][0]:
                result_number = product_document_query_vector[i][1] / (length[1] * math.sqrt(len(query)))
                inner_list.append(length[0])
                inner_list.append(result_number)
                i = i + 1
            cosine_similarity.append(inner_list)

    #rank the matrix based on the score of the cosine similarity
    unranked_cosine_similarity_matrix = np.array(cosine_similarity)
    ranked_cosine_similarity_matrix = np.array(sorted(unranked_cosine_similarity_matrix, key=lambda score:score[1], reverse=True))

    return ranked_cosine_similarity_matrix

if __name__ == '__main__':
    app.run(debug=True)