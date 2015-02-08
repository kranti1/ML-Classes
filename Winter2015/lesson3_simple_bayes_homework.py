################################################################################
#
#  MLEARN 210 : homework #3
#
################################################################################


################################################################################
################################################################################

########
#
# GLOBAL DATA
#



#
# We just use a hard-coded model.
# This 'bag-of-words' model is supposedly trained from labeled articles on 3 topics.
#

########################################################################
# Each topic model has a prior probability : this is just its ratio vs all other article topics.
#
# Each topic model has conditional probabilities for a set of words.
# Each of these conditional probabilities corresponds to P(Bi | A), where A is the topic and Bi is the occurrence of the word.
#
# These probabilities are NOT AT ALL REALISTIC:
# We're assuming a tiny sliver of English vocabulary here, so weights will appear large.
# Another reason I've made the weights large is so the [Python built-in] floating point math doesn't become chaotic.
#
# You may consider each conditional probability as a weight on a conceptual "die" used to generate an article on that topic,
# where the weight of the die gives the likelihood of the die generating the word on a given roll.
#
g_topics_model = {
      'SPORTS' : { 'prior':0.05,  'weights': { 'injury':0.012, 'football':0.02, 'basketball':0.02, 'player':0.03, 'team':0.03, 'game':0.02  }  },
      'MEDICINE' : { 'prior':0.02, 'weights': { 'virus':0.04, 'injury':0.05, 'hospital':0.02, 'vaccine':0.03 }    },
      'TECH' : { 'prior':0.03, 'weights': { 'virus':0.02, 'online':0.04, 'web':0.03, 'website':0.03, 'server':0.04, 'computer':0.03  }   },
    }

#
# These are assumed to be the 'prior' probabilities of individual words, based on relative appearance in all articles
# We're assuming a tiny sliver of English vocabulary here, so weights will appear large.
#
g_word_priors = { 'the':0.1, 'will':0.05, 'a':0.05, 'to':0.05, 'an':0.05, 'is':0.05, 'this':0.05, 'and':0.04, 'went':0.02 }
g_word_default_prior = 0.01     # apply this weight to any word for which we don't have a prior or conditional probability




#
# These are new "articles" that your code will label with a topic.
#
# In real-world NLP, punctuation would first need to be stripped;
# words would be normalized to lower case, then stemmed and "lemmatized"; etc.
#
g_article_A_text = "the local football team will play a home game this saturday"
g_article_B_text = "online learning is being delivered on the web"
g_article_C_text = "a vaccine to combat the new virus was developed and administered in trials at the local hospital"
g_article_D_text = "a computer virus brought down the university server"
g_article_E_text = "the basketball player suffered an injury and went to the hospital"   # mixes sports and medical terms

g_all_articles_map = {
      'article_A' : g_article_A_text,
      'article_B' : g_article_B_text,
      'article_C' : g_article_C_text,
      'article_D' : g_article_D_text,
      'article_E' : g_article_E_text,
    }    

#g_all_articles_map = {
#      'article_A' : g_article_A_text,
#    }    

################################################################################
#
# ComputeArticleScoreForTopic
#
# Given a list of words in an article, and a topic model, compute the probability that the article is of the given topic,
# using the "bag-of-words" assumption and Naive Bayes.
#
# For any word that does not have a conditional probability wrt to the topic, use the default word 'prior', if there is one.
# For any word that does not have a prior or conditional probability, assume a default weight of [g_word_default_prior].
#
# NOTE:  For this simplified problem, you can just use Python's built-in floating-point math for probability calculation.
#            For an even slightly more complex problem, the probabilities would get tiny, and the math (divisions) would become chaotic.
#            You would have to use a numeric library like NUMPY.
#
def ComputeArticleScoreForTopic(article_words_list, topic_prior, topic_word_weights):
  global g_word_priors
  global g_word_default_prior
  
  #
  #  COMPLETE THIS CODE
  #
  # To compute score for an article score = P(A | B1,B2,...,Bn) = PRODUCT[ P(Bi | A) ] * P(A) / PRODUCT[ P(Bi) ]
  # Numerator: 
  #
  #]
  score = 1
  Pab =1
  Pb = 1
  for word in article_words_list:

    if (word in topic_word_weights):
        Pnum = topic_word_weights[word]
    elif word in g_word_priors:   
        Pnum = g_word_priors[word]
    else:
        Pnum = g_word_default_prior

    if word in g_word_priors:   
        Pdin = g_word_priors[word]
    else:
        Pdin = g_word_default_prior
    
    Pab = Pab * Pnum
    Pb = Pb * Pdin
    score = score * Pnum/Pdin
  
  score = score * topic_prior    
  score2 = Pab/Pb * topic_prior
  
  #print "{}  = {}".format("Pab",Pab) 
  #print "{}  = {}".format("Pb",Pb) 
  #print "{}  = {}".format("score2",score2) 
     
  return score2 

################################################################################
################################################################################

def GetWordProbability(article_word, topic, prior=0):
    global g_word_priors
    global g_word_default_prior
    article_words_list = g_article_A_text.split(' ')
    topic_prior = g_topics_model[topic]['prior']
    topic_word_weights = g_topics_model[topic]['weights']

    
    if (article_word in topic_word_weights and prior ==0):
        return topic_word_weights[article_word]
    elif article_word in g_word_priors:   
        return g_word_priors[article_word]
    else:
        return g_word_default_prior

    
    

################################################################################
#
# RunModels
#
def RunModels():

  article_names_list = g_all_articles_map.keys()
  article_names_list.sort()
  
  for article_name in article_names_list:
    article_text = g_all_articles_map[article_name]
    article_words_list = article_text.split(' ')

    max_topic_score = 0.0
    max_topic_name = ""

    for topic_name in g_topics_model:
      topic_prior = g_topics_model[topic_name]['prior']
      topic_word_weights = g_topics_model[topic_name]['weights']

      score = ComputeArticleScoreForTopic(article_words_list, topic_prior, topic_word_weights)

      print(" article '%s' score for topic '%s' is : %f " % (article_name, topic_name, score))
      
      if score > max_topic_score:
        max_topic_score = score
        max_topic_name = topic_name

    print(" TOP SCORE for article '%s' = %f : topic = %s \n" % (article_name, max_topic_score, max_topic_name))

        

################################################################################
#
# This is the main() entrypoint of every top-level Python program. 
#
if __name__ == '__main__':
    RunModels()




################################################################################
################################################################################


