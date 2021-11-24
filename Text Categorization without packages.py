import numpy as np
import re
def import_data(file_name):
    df = np.loadtxt(file_name, delimiter=',', skiprows=1, dtype=str)
    # np.random.shuffle(df)
    return(df)

def data_preprocessing(abstract):
    #     without_stopwords = w for w in abstract if not w in stop_words
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
                  'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
                  'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
                  'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 
                  'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 
                  'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
                  'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                  'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
                  'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                  'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've',
                  'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn',
                  'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
    
    words = abstract.split()
    abstract_text = []
    for x in words:
        if x not in stop_words:
            abstract_text.append(x)

    abstract_non_number = re.sub("[^a-zA-Z]"," ",abstract)
    words = abstract.lower()
    return(words)

def GetVocabulary(data): 
    vocab_dict = {}
    wid = 0
    for document in data:
        words = document.split() 
        for word in words:
            word = word.lower() 
            if word not in vocab_dict:
                vocab_dict[word] = wid
                wid += 1
    return vocab_dict

def Document2Vector(vocab_dict, data):
    word_vector = np.zeros(len(vocab_dict.keys()))
    words = data.split()
    out_of_voc = 0
    for word in words:
        word = word.lower()
        if word in vocab_dict:
            word_vector[vocab_dict[word]] += 1
        else:
            out_of_voc += 1
    return word_vector, out_of_voc

def FreqList(txt_numeric):
    freqList = np.zeros(len(txt_numeric[0]))
    for i in txt_numeric:
        no_zero = np.flatnonzero(i)
        for j in no_zero:
            freqList[j] += 1
    freqList = np.array(freqList)
    return freqList

def calculte_wordtimes(train_matrix):
    new = np.array(train_matrix)
    train_matrix_one = np.int64(new>0)
    num_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    
    word_times,a_word_times, b_word_times, e_word_times, v_word_times = np.zeros(num_words),np.zeros(num_words),np.zeros(num_words),np.zeros(num_words),np.zeros(num_words)
    
    for i in range(num_docs):
        word_times += train_matrix_one[i]
        if labels_train[i] == 'A':
            a_word_times += train_matrix_one[i]
        if labels_train[i] == 'B':
            b_word_times += train_matrix_one[i]
        if labels_train[i] == 'E':
            e_word_times += train_matrix_one[i]
        if labels_train[i] == 'V':
            v_word_times += train_matrix_one[i]
    return word_times

def NaiveBayes_train(train_matrix,labels_train):
    num_docs = len(train_matrix)
    num_words = len(train_matrix[0]) 
    
    a_word_counter = np.ones(num_words)
    b_word_counter = np.ones(num_words)
    e_word_counter = np.ones(num_words)
    v_word_counter = np.ones(num_words)

    a_total_count = 0;
    b_total_count = 0;
    e_total_count = 0;
    v_total_count = 0;
    
    a_count = 0
    b_count = 0
    e_count = 0
    v_count = 0
    for i in range(num_docs):
        if labels_train[i] == 'A':
            a_word_counter += train_matrix[i]
            a_total_count += sum(train_matrix[i])
            a_count += 1
        if labels_train[i] == 'B':
            b_word_counter += train_matrix[i]
            b_total_count += sum(train_matrix[i])
            b_count += 1
        if labels_train[i] == 'E':
            e_word_counter += train_matrix[i]
            e_total_count += sum(train_matrix[i])
            e_count += 1
        if labels_train[i] == 'V':
            v_word_counter += train_matrix[i]
            v_total_count += sum(train_matrix[i])
            v_count += 1
    
    p_a_vector = np.log(a_word_counter/(a_total_count + num_words)) 
    p_b_vector = np.log(b_word_counter/(b_total_count + num_words))  
    p_e_vector = np.log(e_word_counter/(e_total_count + num_words))  
    p_v_vector = np.log(v_word_counter/(v_total_count + num_words))  
    
    return p_a_vector, np.log(a_count/num_docs), p_b_vector, np.log(b_count/num_docs),p_e_vector, np.log(e_count/num_docs), p_v_vector, np.log(v_count/num_docs), a_total_count, b_total_count,e_total_count, v_total_count

    
def extension_NaiveBayes_train(train_matrix,labels_train):
    num_docs = len(train_matrix)
    num_words = len(train_matrix[0]) 
    
    a_word_counter = np.ones(num_words)
    b_word_counter = np.ones(num_words)
    e_word_counter = np.ones(num_words)
    v_word_counter = np.ones(num_words)
    one = np.ones(num_words)
    a_total_count = 0;
    b_total_count = 0;
    e_total_count = 0;
    v_total_count = 0;
    
    a_count = 0
    b_count = 0
    e_count = 0
    v_count = 0

    fluency = np.log(num_docs/FreqList(train_matrix))
    
    for i in range(num_docs):
        if labels_train[i] == 'A':
            a_word_counter += train_matrix[i] * fluency
            a_total_count += sum(train_matrix[i] * fluency)
            a_count += 1
        if labels_train[i] == 'B':
            b_word_counter += train_matrix[i] * fluency
            b_total_count += sum(train_matrix[i] * fluency)
            b_count += 1
        if labels_train[i] == 'E':
            e_word_counter += train_matrix[i] * fluency
            e_total_count += sum(train_matrix[i] * fluency)
            e_count += 1
        if labels_train[i] == 'V':
            v_word_counter += train_matrix[i] * fluency
            v_total_count += sum(train_matrix[i] * fluency)
            v_count += 1
    
    p_a_vector = np.log((a_word_counter + 1) /(a_total_count + num_words)) 
    p_b_vector = np.log((b_word_counter + 1) /(b_total_count + num_words))  
    p_e_vector = np.log((e_word_counter + 1) /(e_total_count + num_words))  
    p_v_vector = np.log((v_word_counter + 1) /(v_total_count + num_words))  
    
    return p_a_vector, np.log(a_count/num_docs), p_b_vector, np.log(b_count/num_docs),p_e_vector, np.log(e_count/num_docs), p_v_vector, np.log(v_count/num_docs), a_total_count, b_total_count,e_total_count, v_total_count



def Predict(test_word_vector, p_a_vector, p_a, p_b_vector, p_b,p_e_vector, p_e, p_v_vector, p_v, a_smoothing, b_smoothing,e_smoothing, v_smoothing):
    
    a = sum(test_word_vector * p_a_vector) + p_a + a_smoothing
    b = sum(test_word_vector * p_b_vector) + p_b + b_smoothing
    e = sum(test_word_vector * p_e_vector) + p_e + e_smoothing
    v = sum(test_word_vector * p_v_vector) + p_v + v_smoothing
    if a > b and a>e and a>v:
        return 'A'
    if b >a and b >e and b>v:
        return 'B'
    if e >a and e >b and e>v:
        return 'E'
    if v >a and v >b and v>e:
        return 'V'


def get_accuracy(y, y_hat):
    return sum(yi == yi_hat for yi, yi_hat in zip(y, y_hat)) / len(y)

'''
Standard: Mutinomial naive bayes
Cross validation use 10-fold cross validation to train and get accuracy 

'''
# read file
file_name = 'trg.csv'
df = import_data(file_name)
# for cross-validation(10-fold-crossvalidation)
df0,df1,df2,df3,df4,df5,df6,df7,df8,df9 = np.split(df, 10, axis=0)
subset = [df0,df1,df2,df3,df4,df5,df6,df7,df8,df9]
accuracy_list = []
for x in range(10):
    name = 'df'+ str(x)
    test = locals()[name]
    array = np.arange(10)
    new_array = np.delete(array,x)
    train = []
    for y in new_array:
        name1 = 'df'+ str(y)
        train_name = locals()[name1]
        train.extend(train_name)
        training = np.array(train)
    data_train1, data_test, labels_train, labels_test = training[:,2], test[:,2], training[:,1],test[:,1]
    data_train = []
    for x in data_train1:
        data_train.append(data_preprocessing(x))
    vocab_dict = GetVocabulary(data_train)
    train_matrix = []
    for document in data_train:
        word_vector, _ = Document2Vector(vocab_dict, document)
        train_matrix.append(word_vector)
    p_a_vector, p_a, p_b_vector, p_b,p_e_vector, p_e, p_v_vector, p_v, a_total_count, b_total_count, e_total_count, v_total_count = NaiveBayes_train(train_matrix, labels_train)
    num_words = len(vocab_dict.keys())
    predictions = []
    i = 0
    for document in data_test:
        # if i % 200 == 0:
        #     print ('Test on the doc id:' + str(i))
        i += 1    
        test_word_vector, out_of_voc = Document2Vector(vocab_dict, document)
        # Add smoothing for out_of_vocbulary words
        if out_of_voc != 0:
            a_smoothing = np.log(out_of_voc/(a_total_count + num_words))
            b_smoothing = np.log(out_of_voc/(b_total_count + num_words))
            e_smoothing = np.log(out_of_voc/(e_total_count + num_words))
            v_smoothing = np.log(out_of_voc/(v_total_count + num_words))
        else:
            a_smoothing = 0
            b_smoothing = 0
            e_smoothing = 0
            v_smoothing = 0
        ans = Predict(test_word_vector, p_a_vector, p_a, p_b_vector, p_b,p_e_vector, p_e, p_v_vector, p_v, a_smoothing, b_smoothing,e_smoothing, v_smoothing)
        predictions.append(ans)
    accuracy = get_accuracy(labels_test,predictions)
    accuracy_list.append(accuracy)
    print(accuracy)
average_accuracy = np.mean(accuracy_list)
print('standard multinomial average_accuracy')
print(average_accuracy)

'''
extension: Transforming by document frequency
Cross validation use 10-fold cross validation to train and get accuracy 

'''
# read file
file_name = 'trg.csv'
df = import_data(file_name)
# for cross-validation(10-fold-crossvalidation)
df0,df1,df2,df3,df4,df5,df6,df7,df8,df9 = np.split(df, 10, axis=0)
subset = [df0,df1,df2,df3,df4,df5,df6,df7,df8,df9]
accuracy_list = []
for x in range(10):
    name = 'df'+ str(x)
    test = locals()[name]
    array = np.arange(10)
    new_array = np.delete(array,x)
    train = []
    for y in new_array:
        name1 = 'df'+ str(y)
        train_name = locals()[name1]
        train.extend(train_name)
        training = np.array(train)
    data_train1, data_test, labels_train, labels_test = training[:,2], test[:,2], training[:,1],test[:,1]
    data_train = []
    for x in data_train1:
        data_train.append(data_preprocessing(x))
    vocab_dict = GetVocabulary(data_train)
    train_matrix = []
    for document in data_train:
        word_vector, _ = Document2Vector(vocab_dict, document)
        train_matrix.append(word_vector)
    p_a_vector, p_a, p_b_vector, p_b,p_e_vector, p_e, p_v_vector, p_v, a_total_count, b_total_count, e_total_count, v_total_count = extension_NaiveBayes_train(train_matrix, labels_train)
    num_words = len(vocab_dict.keys())
    predictions = []
    i = 0
    for document in data_test:
        # if i % 200 == 0:
        #     print ('Test on the doc id:' + str(i))
        i += 1    
        test_word_vector, out_of_voc = Document2Vector(vocab_dict, document)
        # Add smoothing for out_of_vocbulary words
        if out_of_voc != 0:
            a_smoothing = np.log(out_of_voc/(a_total_count + num_words))
            b_smoothing = np.log(out_of_voc/(b_total_count + num_words))
            e_smoothing = np.log(out_of_voc/(e_total_count + num_words))
            v_smoothing = np.log(out_of_voc/(v_total_count + num_words))
        else:
            a_smoothing = 0
            b_smoothing = 0
            e_smoothing = 0
            v_smoothing = 0
        ans = Predict(test_word_vector, p_a_vector, p_a, p_b_vector, p_b,p_e_vector, p_e, p_v_vector, p_v, a_smoothing, b_smoothing,e_smoothing, v_smoothing)
        predictions.append(ans)
    accuracy = get_accuracy(labels_test,predictions)
    accuracy_list.append(accuracy)
    print(accuracy)
average_accuracy = np.mean(accuracy_list)
print('extension average_accuracy')
print(average_accuracy)

'''
train the whole dataset and give the reslut
'''

# read file
file_name = 'trg.csv'
df = import_data(file_name)
#use all trainset to train
X, y = df[:,2], df[:,1]
X_train = []
for documents in X:
    X_train.append(data_preprocessing(documents))
vocab_dict = GetVocabulary(X_train)
train_matrix = []
for document in X_train:
    word_vector, _ = Document2Vector(vocab_dict, document)
    train_matrix.append(word_vector)
p_a_vector, p_a, p_b_vector, p_b,p_e_vector, p_e, p_v_vector, p_v, a_total_count, b_total_count, e_total_count, v_total_count = NaiveBayes_train(train_matrix,y)
num_words = len(vocab_dict.keys())

test_file_name = 'tst.csv'
test = import_data(test_file_name)
predictions = []
for document in test[:,1]:     
    test_word_vector, out_of_voc = Document2Vector(vocab_dict, document)
    if out_of_voc != 0:
        a_smoothing = np.log(out_of_voc/(a_total_count + num_words))
        b_smoothing = np.log(out_of_voc/(b_total_count + num_words))
        e_smoothing = np.log(out_of_voc/(e_total_count + num_words))
        v_smoothing = np.log(out_of_voc/(v_total_count + num_words))
    else:
        a_smoothing = 0
        b_smoothing = 0
        e_smoothing = 0
        v_smoothing = 0
    ans = Predict(test_word_vector, p_a_vector, p_a, p_b_vector, p_b,p_e_vector, p_e, p_v_vector, p_v, a_smoothing, b_smoothing,e_smoothing, v_smoothing)
    predictions.append(ans)
result = []
for i in range(1,len(predictions)+1):
    result.append([i,predictions[i-1]])
# print(result)
np.savetxt("jguo811-test.csv", result,fmt='%s', delimiter=',')