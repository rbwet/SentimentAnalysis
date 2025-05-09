'''
File: sentiment_analysis.py
Author: Aidan Brownell, Robbie Wtzel
Date: 

This provides sentiment analysis functions for processing tweets in particular,
but relies on tweet_processing to handle the cleanup of the tweets. Analysis is
done using Naive Bayes.

'''
import random
import tweet_processor as tp
from openai import OpenAI


import numpy as np

client = OpenAI()

def get_llm_response(client : OpenAI, prompt : str) -> str:
    completion = client.chat.completions.create(
        model = 'gpt-3.5-turbo',
        messages = [
            {'role' : 'user', 'content' : prompt}
        ],
        # temperature is the randomness of your result 
        temperature=0
    )
    return completion

def partition_training_and_test_sets(pos_tweets : list[str],
                                     neg_tweets : list[str], 
                                     split : float = .8) -> tuple[list[str], 
                                                                  np.ndarray[float], 
                                                                  list[str], 
                                                                  np.ndarray[float], 
                                                                  int, int, int, int]:
    '''
    Partition our sets of tweets into positive and negative tweets based
    on a split factor. 

    Parameters: 
        pos_tweets -- list of strings that are positive tweets
        neg_tweets -- list of strings that are negative tweets
        split -- factor to split the training and partition sets into.
            Defaults to .8, or 80% training, 20% testing.

    Returns:
        A list of training tweets
        A list of the same size of training labels, which will be 1 or 0 for positive or negative tweets
        A list of testing tweets
        A list of testing labels, which 
    '''
    if split < 0 or split > 1:
        raise Exception('split must be between 0 and 1')
    
    # multiply the length of the list of tweets by our split factor and convert to an int
    pos_train_size = int(split * len(pos_tweets))
    neg_train_size = int(split * len(neg_tweets))
    
    # split our sets
    pos_x = pos_tweets[:pos_train_size]
    neg_x = neg_tweets[:neg_train_size]
    
    # test sets
    test_pos = pos_tweets[pos_train_size:]
    test_neg = neg_tweets[neg_train_size:]

    # combine the sets for training and testing
    train_x = pos_x + neg_x
    test_x = test_pos + test_neg

    # our labels are 1 for positive, 0 for negative, so we'll create
    # arrays of 1s and 0s for the training and test sets
    train_y = np.append(np.ones(len(pos_x)), np.zeros(len(neg_x)))
    test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))

    pos_test_size = len(pos_tweets) - pos_train_size
    neg_test_size = len(neg_tweets) - neg_train_size
    return (train_x, train_y, test_x, test_y, pos_train_size, 
            neg_train_size, pos_test_size, neg_test_size)


# takes a list of tweets        
def build_word_freq_dict(tweets : list[list[str]], labels : np.ndarray[int]) -> dict[(str, int),  int]:
    '''
    Creates a frequency dictionary based on the tweets. The frequency dictionary
    has keys which are (word, label) pairs, for example, ('happi', 1), while the
    value associated with it is the number of times that word was seen in a given
    class. For example, if 'happi' is seen 10 times in positive tweets, then we'd 
    see freqs[('happi', 1)] = 10. If it were seen 3 times in negative tweets, we'd
    see freqs[('happi', 0)] = 3.

    Parameters: 
    tweets -- A list of strings, each a tweet
    labels -- A list of integers either 0 or 1 for negative or positive classes

    Note that the number of tweets and labels must match. 

    Return: 
    A dictionary containing (word, class) keys mapping to the number of 
    times that word in that class appears in the data set
    '''
    dict = {}
    vocab = set()

    
    # create the dictionary and vocabulary here
    for tweet, label in zip(tweets, labels):
        for word in tweet:
            key = (word, label)
            dict[key] = dict.get(key, 0) + 1
            if word not in vocab: 
                vocab.add(word)
    # return the frequency dictionary
    return dict, vocab

def test_word_freq_dict():
    '''
    Simple function that tests some tweets and if your build_word_freq_dict is built correctly
    '''
    tweets = [['i', 'am', 'happi'], ['i', 'am', 'trick'], ['i', 'am', 'sad'], 
              ['i', 'am', 'tire'], ['i', 'am', 'tire']]
    labels = [1, 0, 0, 0, 0]
    print("testing build_word_freq_dict, should get {('i', 1): 1, ('am', 1): 1, ('happi', 1): 1, ('i', 0): 4, ('am', 0): 4, ('trick', 0): 1, ('sad', 0): 1, ('tire', 0): 2}")
    print(f'test of word frequency: {build_word_freq_dict(tweets, labels)}')


def analyze_misclassified_tweets(client, mislabeled_tweets):
    print("Analyzing misclassified tweets...")
    results = []  #stores each result similar to outTwo in the inferring function from lab 5
    for tweet_text, (expected_label, nb_prediction) in mislabeled_tweets.items():
        prompt = f"Is the sentiment of this tweet positive or negative? Limit your response to one word, positive or negative'{tweet_text.strip()}'"
        #the response using the adjusted get_llm_response function
        response = (get_llm_response(client, prompt)).choices[0].message.content
        #formatted dict
        formatted_response = {
            "Tweet": tweet_text,
            "Expected Label": expected_label,
            "Naive Bayes Prediction": nb_prediction,
            "LLM Prediction": response
        }
        results.append(formatted_response)  # Append the response as a dictionary
        print(f"Tweet: {tweet_text}\nExpected Label: {expected_label}\nNaive Bayes Prediction: {nb_prediction}\nLLM Prediction: {response}\n")
        if (response == "positive" and expected_label < 0) or (response == "negative" and expected_label > 0):
            prompt = f"Why did you decide that:'{tweet_text.strip()} is {nb_prediction}'"
            response = (get_llm_response(client, prompt)).choices[0].message.content
            print(f"Still wrong, the LLM says: \n{response}")
            print("\n\n\n")
    return results  #we can optionally return results for further processing or testing



def count_pos_neg(freqs : dict[(str, int),  int]) -> tuple[int, int]:
    '''
    Count the number of positive and negative words in the
    frequency dictionary.

    Parameters:
    freqs -- a dictionary of ((str, int), int) pairs, where the key is a
    word and label of 0 or 1 for negative or positive sentiment, and the value
    associated with the key is the number of times it was seen.

    Returns:
    Returns two values, the number of times any positive word was seen (i.e., the
    total number of positive events), and the number of times a negative word was
    seen. 
    '''
   
    num_pos = num_neg = 0
    # calculate the number of times each word appears in 
    # particular class of positive or negative 
    for key in freqs:
        if key[1] == 1:
            num_pos += freqs[key]
        else:
            num_neg += freqs[key]

    return num_pos, num_neg



def build_loglikelihood_dict(freqs : dict[(str, int),  int], N_pos : int, N_neg : int, vocab : list[str]) -> dict[str, float]:

    '''
    Create a dictionary based on the frequency of each word in each class appearing
    of the probability of that word occuring, using Laplacian smoothing by adding
    1 to each occurrence and the size of the vocabulary. 

    Thus, we'd calculate (freq(w_i, class) + 1) / (N_class + V_size)

    Parameters:
        freqs -- dictionary from (word, class) to occurrence count mapping
        N_pos -- number of positive events for all words
        N_neg -- number of negative events for all words
        vocab -- list vocabulary of words

    Returns:
        A dictionary of words to the ratio of positive and negative usage of the word
    '''
    loglikelihood_dict = {}
    V_size = len(vocab)

    for word in vocab:
        # Get the count of the word occurring in positive and negative classes
        freq_pos = freqs.get((word, 1), 0)
        freq_neg = freqs.get((word, 0), 0)

        # Calculate the log-likelihood ratio using Laplacian smoothing
        loglikelihood = ((freq_pos + 1) / (N_pos + V_size))/((freq_neg + 1) / (N_neg + V_size))

        # Store the log-likelihood ratio in the dictionary
        loglikelihood_dict[word] = np.log(loglikelihood)

    return loglikelihood_dict


def naive_bayes_predict(loglikelihood : dict[str, float], log_pos_neg_ratio : float, tweet : list[str]) -> float:
    '''
    Calculates the prediction based on our dictionary of log-likelihoods of each
    word in a tweet added to the log of the ratio of positive and negative tweets

    Parameters:
        loglikelihood -- a dictionary of words to the ratio of postive/negative probabilities of the words
        log_pos_neg_ratio -- the log of the ratio of total positive to total negative events
        tweet -- a list of tokens (likely from process_tweet)
    '''
    val = 0
    for word in tweet:
        val += loglikelihood.get(word, 0)
    return val+log_pos_neg_ratio


def main():
    mislabeledTweets = {}

#setup samples
    pos_tweets, neg_tweets, stopwords = tp.process_tweets('positive_tweets.json', 'negative_tweets.json', 'english_stopwords.txt')
    train_x, train_y, test_x, test_y, _, _, _, _ = partition_training_and_test_sets(pos_tweets, neg_tweets, .8)
    freq_train, vocab = build_word_freq_dict(train_x, train_y)
    num_pos, num_neg = count_pos_neg(freq_train)
    log_pos_neg_ratio = np.log(num_pos / num_neg)
    log_likelihood = build_loglikelihood_dict(freq_train, num_pos, num_neg, vocab)
    

    for i in range(len(test_x)):
        #idx = random.randint(0, len(test_x) - 1)
        tweet = test_x[i]
        label = test_y[i]
        prediction = naive_bayes_predict(log_likelihood, log_pos_neg_ratio, tweet)
        print(f'Tweet: {tweet} | Label: {label} | Prediction: {prediction}')
        
        #adjust threshold if needed
        prediction_threshold = 0  
        mislabeled = (label == 0.0 and prediction > prediction_threshold) or (label == 1 and prediction < -prediction_threshold)
        
        if mislabeled:
            tweet_text = ' '.join(tweet)
            mislabeledTweets[tweet_text] = (label, prediction)
            print(f'Mislabeled Tweet Detected: {tweet_text}')

    #analyze and display mislabeled tweets
    print(f"Mislabeled Tweets Count: {len(mislabeledTweets)}")
    for tweet, data in mislabeledTweets.items():
        print(f"Tweet: {tweet}, Label: {data[0]}, Prediction: {data[1]}")

    analyze_misclassified_tweets(client, mislabeledTweets)

if __name__ == '__main__':
    main()
