#!/usr/bin/env python

"""
Implemention of the WordNet score propagation algorithm of 

@inproceedings{Blair-Goldensohn-etal08,
  Address = {Beijing, China},
  Author = {Blair-Goldensohn, Sasha and Hannan, Kerry and McDonald,
  Ryan and Neylon, Tyler and Reis, George A. and Reynar, Jeff},
  Booktitle = {{WWW} Workshop on {NLP} in the Information Explosion Era (NLPIX)},
  Title = {Building a Sentiment Summarizer for Local Service Reviews},
  Year = {2008}}

For additional details, see the 'Sentiment lexicons' handout from
Linguist 287 / CS 424P: Extracting Social Meaning and Sentiment,
Stanford, Fall 2010.

The class ParallelWnSentimentPropagator breaks the job up into parts
and runs them using ParallelPython.  This is necessary for all
categories except adverb ('r'), because the matrices get too big to
fit into memory.

The class WnSentimentPropagator handles the actual propagation, over
the part of the lexicon it is given.

To further save space and speed things along, both classes process
specific WordNet parts of speech.

Users should check that WN_ROOT points to their local copy of WordNet.

The method uses the NLTK interface to WordNet:

  http://www.nltk.org/

The parallelization uses ParallelPython:

  http://www.nltk.org/

If this program is called with no arguments, then it runs a small
experiment with adverbial seed-sets and prints the non-null scores it
finds to the command line in tab-separated format.

---Chris Potts.
"""

######################################################################

import sys
if sys.version_info < (2, 5) or sys.version_info >= (2, 7):
    print "This program requires a version of Python between 2.5 and 2.6.*. \
    You're running version %s.%s.%s" % sys.version_info[0:3]
    sys.exit(2)
import os
import copy
from operator import itemgetter
from collections import defaultdict
from itertools import repeat
try:
    import numpy
except:
    sys.stderr.write("Couldn't find Numpy. To get it: http://numpy.scipy.org/.\n")
try:
    import pp
except ImportError:
    sys.stderr.write("Couldn't find the ParallelPython library. To get it: http://www.parallelpython.com/.\n")
try:
    from nltk.corpus import wordnet as wn
except ImportError:
    sys.stderr.write("Couldn't find an NLTK installation. To get it: http://www.nltk.org/.\n")
    sys.exit(2)

######################################################################

class ParallelWnScorePropagator:
    """
    Class for parallelizing word-net sentiment propagation using
    WnSentimentPropagator.  This is often necessary because of memory
    management and speed.
    """
    def __init__(self, positive, negative, neutral, pos, weight=0.2, rescale=True, job_count=100):
        """
        Builds a number of SentimentLexicon jobs in parallel.  The
        method runs() handles the parallelization.  The initialization
        arguments correspond to those for SentimentLexicon:
        
        Arguments
        positive (list) -- the positive seed set
        negative (list) -- the negative seed set
        neutral (list)  -- the neutral seed set
        pos (str) -- WordNet pos value: a, v, r, n
        
        Keyword argument
        weight (float) -- the biasing weight used by the algorithm (default: 0.2)
        rescale -- should sentiment scores be rescaled logarithmically? (default: True)
        job_count -- the number of jobs to run in parallel (default: 200)
        """
        self.positive = positive
        self.negative = negative
        self.neutral = neutral
        self.pos = pos
        self.weight = weight
        self.job_count = job_count
        self.rescale = rescale

    def run(self, runs=5):
        """
        A poor linguist's map/reduce function. Uses ParallelPython to
        divide the lemmas into job_count number of independent jobs,
        which are run in parallel using map_func and then merged.
        Each job actually involves the entire vocabulary, but just to
        compute the values for its assigned subset.

        Keyword argument
        runs -- passed down through WnSentimentPropagator to its runs
                function to determine the number of iterations (default: 5)
        """        
        def map_func(runs, params):
            """
            Builds a sentiment lexicon over this part of the corpus
            and returns the sentiment dictionary obtained via run() on
            that lexicon.
            """
            sent_lex = wnscores.WnScorePropagator(params["positive"], params["negative"], params["neutral"],
                                                  params["pos"],
                                                  weight=params["weight"], rescale=params["rescale"],
                                                  start=params["start"], finish=params["finish"])
            return sent_lex.run(runs=runs)

        # Divide up the lexicon into smaller jobs.
        job_splits = self.job_splits()
        # Set up the parallel zerver.        
        ppservers = ()
        job_server = pp.Server(ppservers=ppservers)
        # Build the jobs.
        jobs = []
        for start, finish in job_splits:            
            params = { # parameters for map_func --- for building SentimentLexicon objects.
                "positive":self.positive,
                "negative":self.negative,
                "neutral":self.neutral,
                "pos":self.pos,
                "weight":self.weight,
                "rescale":self.rescale,
                "start":start,
                "finish":finish}
            # args to job_server.submit: function, (function's args), (external functions), (external libraries)            
            jobs.append(job_server.submit(map_func, (runs, params), (), ("wnscores",)))
        # Reduce function: merge the results of the jobs.
        all_sentiment = {}
        for i, job in enumerate(jobs):
            sentiment = job()
            for key,val in sentiment.items():
                if key in all_sentiment:
                    raise Exception("%s already seen in an earlier job, suggesting that the jobs are partly overlapping." % key)
                all_sentiment[key] = val
            print "Job %s of %s completed." % (i+1, len(jobs))
        return all_sentiment

    def job_splits(self):
        """
        Divides the lemmas up into self.job_length jobs. The output is
        a list of tuples (start, finish), which pick out
        (non-overlapping, collectively exhaustive) regions of the list
        SentimentLexicon.lemmas.
        """
        lc = self.lemma_count()
        job_length = int(round(float(lc)/float(self.job_count)))
        start = 0
        finish = job_length
        splits = []
        for i in range(self.job_count-1):
            splits.append((start,finish))
            start = finish
            finish += job_length
        splits.append((start, lc))
        return splits

    def lemma_count(self):
        """
        Count the number of lemmas in WordNet restricted to
        self.pos, for the purposes of job division.
        """        
        synsets = list(WordNetCorpusReader(WN_ROOT).all_synsets(pos=self.pos))
        lc = {}
        for synset in synsets:
            for lemma in synset.lemmas:
                lc[lemma] = True
        lc = len(lc.keys())
        return lc

######################################################################

class WnScorePropagator:
    """
    Wordnet score propagator.  The method runs() actually propagates
    the scores.
    """    
    def __init__(self, positive, negative, neutral, pos, weight=0.2, rescale=True, start=0, finish=None):
        """
        Arguments
        positive (list of strings) -- the positive seed set
        negative (list of strings) -- the negative seed set
        neutral (list of strings)  -- the neutral seed set
        pos (string) -- WordNet pos value: a, v, r, n, s
        
        Keyword arguments
        weight (float) -- the biasing weight used by the algorithm. 
        rescale (boolean) -- should sentiment scores be rescaled logarithmically? (default: True)
        start (int) -- the initial lemma to use (default: 0)
        finish (int or None) -- the final lemma to use (default: None, which becomes the final index of self.lemmas)
        """
        self.positive = positive
        self.negative = negative
        self.neutral = neutral
        self.pos = pos
        self.weight = weight
        self.rescale = rescale                                
        # The s0 matrix is retained for comparison at the end.  The
        # matrix s is modified during matrix multiplication.
        self.s = {}
        self.s0 = {}
        self.initialize_s()
        # Sort the set of lemmas by name to ensure a stable sort
        # throughout. If the sorting is at all unstable, then the jobs
        # might overlap, resulting in gaps in the cumulative coverage
        # of parallel jobs.
        self.lemmas = sorted(self.s.keys(), cmp=(lambda x, y : cmp(x.name, y.name)))
        self.lemma_count = len(self.lemmas)
        # Start and end points, for parallelization. The default is to
        # span the entire space of lemmas.
        self.start = start
        self.finish = finish
        if self.finish == None or self.finish > self.lemma_count:
            self.finish = self.lemma_count
        # Builds the initial sentiment-score matrix, which is modeled
        # as a two-dimensional dictionary with default value 0.0.        
        self.a = defaultdict(lambda : defaultdict(float))
        self.initialize_a()

    def run(self, runs=5):
        """
        Iterative matrix multiplication to propagate the
        scores. Returns a dictionary mapping the full vocabulary of
        lemma names (strings) to their (rescaled) scores.

        Keyword argument:
        runs -- the default number of matrix multiplications (default: 5)
        """
        # Matrix mutiplications.
        for i in range(runs):
            self.multiply_matrices()
        # Build the final dictionary.
        sentiment = {}        
        for lemma,score in self.s.items():
            # This check on inclusion in a is crucial: during parallel
            # runs, we need to take care not to assign values to the
            # whole set of lemmas, but rather only to the current
            # subset.
            if self.a[lemma]:
                sentiment[lemma.name] = self.rescale_score(score)
        return sentiment

    def initialize_s(self):
        """
        Builds the vectors s, as a dictionary mapping words to
        reals. The domain of the dictionary is the full vocabulary
        (restricted to self.pos).
        """
        synsets = list(wn.all_synsets(pos=self.pos))
        for synset in synsets:
            for lemma in synset.lemmas:
                # Items in the positive seed set are assigned to +1 initially.
                if lemma.name in self.positive:
                    self.s0[lemma] = 1.0
                    self.s[lemma]  = 1.0                    
                # Items in the negative seed set are assigned to -1 initially.
                elif lemma.name in self.negative:
                    self.s0[lemma] = -1.0
                    self.s[lemma]  = -1.0                    
                # Items outside of the seed sets are assigned 0 initially.
                else:
                    self.s0[lemma] = 0.0
                    self.s[lemma]  = 0.0

    def initialize_a(self):
        """
        Builds the matrix a, the sentiment scores for the words
        outside of the initial seed sets.

        This is memory intensive; it works for the adverb ('r')
        without parallelization, but the larger word classes need to
        be parallelized.

        Bug correction by Travis Brown, 2010-03-03.
        """
        for index in range(self.start, self.finish):
            lemma1 = self.lemmas[index]
            # Begin iteration.
            self.a[lemma1][lemma1] = 1 + self.weight
            if lemma1.name not in self.neutral:
                # For 'a', the related 's' elements cause an error here.
                this_pos = lemma1.synset.pos
                if this_pos == "s":
                    this_pos = "a"
                syns = wn.synsets(lemma1.name, this_pos)                    
                # Propagate scores.
                for syn in syns:
                    for lemma2 in syn.lemmas:
                        if lemma1 != lemma2:
                            self.a[lemma1][lemma2] = self.weight
                            ants = [ant for syn in syns
                                    for lem in syn.lemmas
                                    for ant in lem.antonyms() if lem == lemma1]
                            for lemma2 in ants:
                                self.a[lemma1][lemma2] = -self.weight
                                
    def multiply_matrices(self):
        """
        Matrix multiplicaton with sign correction. The matrix here is
        a two-dimensional dictionary. The function modifies self.s
        directly; self.s0 is retained for sign correction. The checks
        on non-0 values significantly speed up the process.
        """
        for lemma1 in self.lemmas:
            if self.a[lemma1]:
                lemma1_vals = self.a[lemma1]
                colsum = sum(self.s[lemma2] * lemma1_vals[lemma2]
                             for lemma2 in self.lemmas if lemma1_vals[lemma2] != 0.0 and self.s[lemma2] != 0.0)
                self.s[lemma1] = self.sign_correct(lemma1, colsum)

    def rescale_score(self, score):
        """
        Logarithmic rescaling of scores. If self.rescale=True, then
        rescaling takes place, else it simply returns the score.
        """
        if self.rescale:
            if abs(score) <= 1:
                return 0.0
            else:
                return numpy.log(abs(score)) * numpy.sign(score)
        else:
            return score
 
    def sign_correct(self, lemma, colsum):
        """
        Sign correction via reference to the s0 matrix. If the final
        matrix assigns a score with different sign that the s0 matrix,
        then we simply flip the sign of the final matrix.
        
        Arguments:
        lemma -- string
        colsum -- the score delivered by the matrix multiplications
        """
        if numpy.sign(self.s0[lemma]) != numpy.sign(colsum):
            return -colsum
        else:
            return colsum

######################################################################
# QUICK TEST

def tiny_adv_experiment():
    """
    Small experiment involving tiny adverbial seeds-sets. Uses the
    parallelization functionality. To try a non-parallel version, just
    change ParallelWnSentimentPropagator to WnSentimentPropagator and
    remove the keyword job_count=50. The output is the set of words
    with non-0 scores, sorted from smallest to largest, in
    tab-separated format.
    """    
    # Part of speech is adverb.
    pos = 'r'
    # Seed sets.
    positive = ["easily","wonderfully","generously","happily","joyfully"]
    negative = ["terribly","cruelly","angrily","sadly","wrongly"]
    neutral = ["administratively","financially","geographically","legislatively","managerial"]
    # Parallel propagator.
    parallel_propagator = ParallelWnScorePropagator(positive, negative, neutral, pos, weight=0.2, rescale=False, job_count=50)
    # Run.
    sentiment_dict = parallel_propagator.run(runs=5)
    # Display just the words with non-null scores, sorted.
    non_null = filter((lambda x : x[1] != 0.0), sentiment_dict.items())
    for key, val in sorted(non_null, key=itemgetter(1)):
        print "%s\t%s" % (key, val)
    print "Words with non-null scores:", len(non_null)

if __name__ ==  "__main__":
    tiny_adv_experiment()

######################################################################   
    
