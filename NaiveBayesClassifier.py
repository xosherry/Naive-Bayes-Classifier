from __future__ import division
from __future__ import print_function
import string
import time
import sys
import math
from stemmer import word_stem

positive_words = {}
negative_words = {}

stop_words = ["1", "0", "movie", "film", "show", "have"]

# Read in training data
def parse(filename, train_or_test=None):
    reviews = []
    # create a list of lists, where each inner list is a review
    with open(filename) as f:
        lines = f.read().splitlines()
        print(lines[0][:-1])
        for review in lines:
            review = review[:-1]
            review = review.lower().translate(None, string.punctuation).split()

            # transform "review", a review, into a list of unigrams and bigrams
            review = uni_bi_tri_grams(review)
            #print(review)
            reviews.append(review)

    if train_or_test == 'train':
        for i in range(len(reviews)):
            if lines[i][-1] == '1':
                for word in reviews[i]:
                    positive_words[word] = positive_words.get(word, 0) + 1
            else:
                for word in reviews[i]:
                    negative_words[word] = negative_words.get(word, 0) + 1

        remove_highly_correlated_features(len(reviews))
        remove_rare_words()


    f.close()
    return reviews

def clean(list_of_words):
    clean_list = []
    for word in list_of_words:
        if word not in stop_words:
            word.strip("\xc2\x85")
            word.strip("br")
            clean_list.append(word)

    return word_stem(clean_list)

def uni_bi_tri_grams(list_of_words):
    n_grams = clean(list_of_words)
    enhance_words = ["veri", "realli", "super", "so", "extrem", "truli", "highli", "pretti"]
    if len(n_grams) > 1:
        n_grams.extend(b for b in zip(n_grams[:-1], n_grams[1:]))
        n_grams.extend(b for b in zip(n_grams[:-1], n_grams[1:], n_grams[2:]))

        for b in zip(n_grams[:-1], n_grams[1:]):
            if b[0] in enhance_words:
                n_grams.append(b[1])
                n_grams.append((b[0], b[1]))

        return n_grams
    else:
        return n_grams

def remove_highly_correlated_features(total_reviews):
    for word in positive_words.keys():
        diff = abs(positive_words.get(word) - negative_words.get(word, 0))
        max_count = max(positive_words.get(word), negative_words.get(word, 0), positive_words.get(word))  # (arg1, arg2, default)
        if positive_words.get(word) > total_reviews * 0.3:
            if diff < max_count * 0.3:
                del positive_words[word]
                del negative_words[word]
        else:
            if diff < max_count * 0.1:
                del positive_words[word]
                del negative_words[word]

def remove_rare_words():
    for word in positive_words.keys():
        if positive_words.get(word) < 1:
            del positive_words[word]
    for word in negative_words.keys():
        if negative_words.get(word) < 1:
            del negative_words[word]

def get_accuracy(file_with_true_labels, prediction_labels):
    labels = []
    with open(file_with_true_labels) as f:
        lines = f.read().splitlines()
        for line in lines:
            labels.append(int(line[-1]))

    correct = 0
    for i in range(len(labels)):
        if labels[i] == prediction_labels[i]:
            correct += 1
    return correct / len(labels)

# list_of_words_by_review is a list of lists, in which each inner list is a review broken into clean words
# ex: I love it -> [['love']]
def make_prediction(list_of_words_by_review):
    positive_count = sum(positive_words.values())
    negative_count = sum(negative_words.values())
    vocab_size = positive_count + negative_count

    # P(class)
    prob_pos = positive_count / (positive_count + negative_count)
    prob_neg = negative_count / (positive_count + negative_count)

    predictions = []
    for review in list_of_words_by_review:
        prob_word_given_pos = 0
        prob_word_given_neg = 0

        # P(doc | class) w/ smoothing
        for word in review:
            prob_word_given_pos += math.log((positive_words.get(word, 0) + 2.6) / (positive_count + 2.6 * vocab_size))
            prob_word_given_neg += math.log((negative_words.get(word, 0) + 2.6) / (negative_count + 2.6 * vocab_size))

        # Cmap = P(doc | class)P(class)
        pos_prediction = math.log(prob_pos) + prob_word_given_pos
        neg_prediction = math.log(prob_neg) + prob_word_given_neg

        if pos_prediction > neg_prediction:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions


def main():
    #train:
    startTrain = time.time()
    reviews_train = parse(sys.argv[1], 'train')
    endTrain = time.time()

    startClassifyTrain = time.time()
    predictions_train = make_prediction(reviews_train)
    endClassifyTrain = time.time()

    accuracy_train = get_accuracy(sys.argv[1], predictions_train)

    #test:
    reviews_test = parse(sys.argv[2])
    startClassifyTest = time.time()
    predictions_test = make_prediction(reviews_test)
    endClassifyTest = time.time()
    accuracy_test = get_accuracy(sys.argv[2], predictions_test)

    training_time = int(endTrain - startTrain)
    labeling_time = int((endClassifyTrain - startClassifyTrain) + (endClassifyTest - startClassifyTest))
    #print(*predictions_test, sep='\n')
    print(training_time, 'seconds', '(training)')
    print(labeling_time, 'seconds', '(labeling)')
    print(accuracy_train, '(training)')
    print(accuracy_test, '(testing)')

    # # print top features:
    # a = sorted(((v, k) for k, v in positive_words.iteritems()), reverse=True)
    #
    # for i in range(10):
    #     print(a[i])
    #
    # b = sorted(((v, k) for k, v in negative_words.iteritems()), reverse=True)
    #
    # for i in range(10):
    #     print(b[i])



main()


