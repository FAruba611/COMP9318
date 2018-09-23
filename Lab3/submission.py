## import modules here 
import pandas as pd
from collections import Counter

def tokenize(sms):
    return sms.split(' ')

def get_freq_of_tokens(sms):
    tokens = {}
    for token in tokenize(sms):
        if token not in tokens:
            tokens[token] = 1
        else:
            tokens[token] += 1
    return tokens


################# Question 1 #################

def multinomial_nb(training_data, sms):# do not change the heading of the function
    data_set = {}
    # Algorithm references: ppt: 7class-a.pdf page 63:
    # Step1: Generating the docs_j which is subset of documents for which the target class is c_j(ham, spam, ...)
    for i in training_data:
        if i[1] not in data_set:
            data_set[i[1]] = i[0]
        else:
            data_set[i[1]] = Counter(i[0]) + Counter(data_set[i[1]])

    #print(data_set)
    size_of_dataset_ham = sum(data_set['ham'].values())
    size_of_dataset_spam = sum(data_set['spam'].values())
    # Step2: Generating the smooth vocabulary(this shouldn't include tokens in sms)
    vocabulary = set()
    vocabulary = vocabulary.union(list(data_set['spam'].keys()), list(data_set['ham'].keys()))
    size_of_smooth_vocab = len(vocabulary)
    # Step3: Calculate likelihood P(c_j) of class
    class_set = dict()
    for data_item in training_data:
        key = data_item[1]
        if key not in class_set.keys():
            class_set.update({key:1})
        else:
            class_set[key] += 1
    pr_ham = class_set['ham']/sum(class_set.values())
    pr_spam = class_set['spam']/sum(class_set.values())

    # Step4: Calculate likelihood P(x_k | c_j) of occur times of each word s_j
    pr_word_class = 1
    for word in sms:
        if word not in data_set['ham'] and word not in data_set['spam']:
            ham_occur = 0
            spam_occur = 0
            continue

        elif word not in data_set['ham'] and word in data_set['spam']:
            ham_occur = 0
            spam_occur = 0 + data_set['spam'][word]

        elif word in data_set['ham'] and word not in data_set['spam']:
            ham_occur = 0 + data_set['ham'][word]
            spam_occur = 0

        else:
            ham_occur = 0 + data_set['ham'][word]
            spam_occur = 0 + data_set['spam'][word]

        add_smooth = 1

        pr_word_ham = pow((ham_occur + add_smooth) / (size_of_smooth_vocab + size_of_dataset_ham), add_smooth)
        pr_word_spam = pow((spam_occur + add_smooth) / (size_of_smooth_vocab + size_of_dataset_spam), add_smooth)
        pr_word_class = pr_word_class * (pr_word_spam / pr_word_ham)

    pr_class_word = pr_word_class * (pr_spam / pr_ham)
    return pr_class_word


raw_data = pd.read_csv('./asset/data.txt', sep='\t')
training_data = []
for index in range(len(raw_data)):
    training_data.append((get_freq_of_tokens(raw_data.iloc[index].text), raw_data.iloc[index].category))
print(raw_data)
#print(training_data)

#sms = 'I am not spam'
sms = 'to is to'
print(multinomial_nb(training_data, tokenize(sms)))
