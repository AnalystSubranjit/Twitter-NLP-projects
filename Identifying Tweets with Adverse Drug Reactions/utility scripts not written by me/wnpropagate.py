#!/usr/bin/env python

"""
Implementation of the WordNet sense propagation algorithm described in
the 'Sentiment lexicons' handout from Linguist 287 / CS 424P:
Extracting Social Meaning and Sentiment, Stanford, Fall 2010.

This is a modified version of the procedure described in section 2.1 of

@inproceedings{EsuliSebastiani06,
  Address = {Genova},
  Author = {Esuli, Andrea and Sebastiani, Fabrizio},
  Booktitle = {Proceedings of the  5th Conference on Language Resources and Evaluation},
  Pages = {417-422},
  Title = {Senti{W}ord{N}et: A Publicly Available Lexical Resource for Opinion Mining},
  Year = {2006}}

The modifications:

  * It generalizes to any number of input synsets.

  * The set of WordNet relations to use is a parameter. Users can
    specify subsets of METHODS below.

  * The user can remove overlap between the propagated synsets.

Implementation note:

The functions use the NLTK interface to WordNet:

  http://www.nltk.org/

Its Synset objects are not hashable, so they can't be added to sets or
used as dictionary keys.  To get around this with relative efficiency,
same_polarity() and other_polarity() manipulate dictionaries mapping
synset.name (string) to synset (NLTK Synset objects).  Set union and
set subtract are then handled by two custom methods that use
dictionaries.

The __main__ method runs a demo using the seed sets from the
experiment described in the associated handout, with 4 iterations.
Thus,

  python wnpropagate.py

will run the experiment (assuming you have NLTK installed). The
positive and negative seed sets are from

@article{TurneyLittman03,
  Author = {Turney, Peter D and Littman, Michael L},
  Journal = {ACM Transactions on Information Systems},
  Number = {4},
  Pages = {315--346},
  Title = {Measuring Praise and Criticism: Inference of Semantic Orientation from Association},
  Volume = {21},
  Year = {2003}}

and were also used in the creation of SentiWordNet:

  http://sentiwordnet.isti.cnr.it/

The random seed set is a more or less random selection from Harvard
Inquirer's non-Positive/Negativ set.

This software is distributed under Eclipse Public License - v 1.0:

  http://www.eclipse.org/legal/epl-v10.html

---Chris Potts
"""

######################################################################

import os
import sys
try:
    import nltk.corpus
except ImportError:
    sys.stderr.write("Couldn't find an NLTK installation. To get it: http://www.nltk.org/.\n")
    sys.exit(2)

WN_ROOT = "/Volumes/CHRIS/Documents/data/corpora/nltk_data/corpora/wordnet/"
#WN_ROOT = "/Volumes/CHRIS/Documents/data/corpora/WordNet/WordNet-2.0/dict/"
if not os.path.exists(WN_ROOT):
    raise IOError("Couldn't find a WordNet dictionary at %s. \
    Please change the value of the variable WN_ROOT to point to the \
    dict subdirectory of your copy of WordNet." % WN_ROOT)
    
######################################################################

METHODS          = set(["also_sees", "similar_tos", "derivationally_related_forms", "pertainyms", "antonyms"])
SYNSET_METHODS   = set(["also_sees", "similar_tos"])
LEMMA_METHODS    = set(["derivationally_related_forms", "pertainyms"])
OPPOSING_METHODS = set(["antonyms"])

def wordnet_sense_propagate(synsets_list, iterations, remove_overlap=True, methods=METHODS):
    """
    The propagation algorithm iself. See create_seed_synsets() for
    moving from strings to synsets.

    Arguments:

    synsets_list -- A list of synsets (the function create_seed_synsets()
    will move you from lists of strings to synsets).

    iterations (int) -- The number of iterations

    Keyword arguments:

    remove_overlap (boolean) -- Should we remove overlap between the
    propagated sets at each iteration (default: True)?

    methods (set) -- Set of methods to use (default: METHODS, as
    specified above).

    Output

    t -- list of lists of synsets. The number of rows is len(synsets_list),
    and the number of columns is iterations+1. Each row is a sequence of
    propagations. The 0th member of t[i] is the ith seed set in synsets_list,
    and the final (i.e., (iterations+1)th) member (index = iterations) is the
    final output of propagation. Thus, to get at all these final propagations,
    iterate over the members of t, grapping their final elements. Those final
    elements are sets of synsets.
    """
    # Intialize the output matrix.
    t = [[synsets] for synsets in synsets_list]
    for i in xrange(iterations):
        new_vals = {}
        for j in xrange(len(synsets_list)):
            # Same-polarity relations from the current synset.
            new_same = same_polarity(t[j][i], methods=methods)
            # Get the union of all the opposing synsets.
            others = {}
            for k in xrange(len(synsets_list)):
                if j != k:
                    for other_synset in t[k][i]:
                        others[other_synset.name] = other_synset
            # Other-polarity relations from others.
            new_diff = other_polarity(others.values(), methods)
            # Append the new synset. We append the values, which are
            # seed sets. We don't need the keys, which are synset
            # names used only for book-keeping.
            new_vals[j] = dict_union(new_same, new_diff)
        if remove_overlap:
            # This will be the pairwise intersection of all the sets
            # propagated at this level.
            overlap = {}
            for j in xrange(len(new_vals)-1):
                for k in xrange(j+1, len(new_vals)):
                    overlap = dict_union(overlap, dict_intersect(new_vals[j], new_vals[k]))                    
        for j in xrange(len(synsets_list)):
            t[j].append(dict_subtract(new_vals[j], overlap).values())
    return t

def same_polarity(synsets, methods):
    """
    Arguments:
    synsets -- a list of synsets

    methods -- the methods to use, presumed to be a subset of
    (methods & SYNSET_METHODS)
    
    As mentioned in the opening docstring, the method uses
    dictionaries, with synset.name mapped to synset, to get around
    limitations on NLTK Synset objects.
    """
    new_synsets = {}
    for synset in synsets:
        new_synsets[synset.name] = synset
        # Synset-level relations.
        for synset_method in (methods & SYNSET_METHODS):
            for related_synset in eval("synset." + synset_method + "()"):
                new_synsets[related_synset.name] = related_synset
        # Lemma-level relations.
        for lemma in synset.lemmas:
            for lemma_method in (methods & LEMMA_METHODS):
                for related_lemma in eval("lemma." + lemma_method + "()"):
                    new_synsets[related_lemma.synset.name] = related_lemma.synset
    return new_synsets

def other_polarity(synsets, methods):
    """
    Arguments:
    synsets -- a list of synsets

    methods -- the methods to use, presumed to be a subset of
    (methods & OPPOSING_METHODS)
    
    As mentioned in the opening docstring, the method uses
    dictionaries, with synset.name mapped to synset, to get around
    limitations on NLTK Synset objects.
    """
    new_synsets = {}
    for synset in synsets:
        for lemma in synset.lemmas:
            for lemma_method in (methods & OPPOSING_METHODS):
                for alt_lemma in eval("lemma." + lemma_method + "()"):
                    new_synsets[alt_lemma.synset.name] = alt_lemma.synset
    return new_synsets

######################################################################
# HELPER METHODS

def create_seed_synsets(word_list, pos=None):
    """
    Use the WordNet functionality to turn a string list into a set of synsets.
        
    Argument:    
    word_list -- a list of strings

    Keyword argument
    pos -- a WordNet part of speech (default: None)
    
    Output:
    s -- a list of synsets
    """
    s = []
    corpus_reader = nltk.corpus.WordNetCorpusReader(WN_ROOT)
    for word in word_list:
        if pos:
            s += corpus_reader.synsets(word, pos)
        else:
            s += corpus_reader.synsets(word)
    return s

def dict_subtract(d1, d2):
    """Subtract dictionary d1 from dictionary d2, to yield a new dictionary d."""    
    d = {}
    for key, val in d1.iteritems():
        if key not in d2:
            d[key] = val    
    return d

def dict_intersect(d1, d2):
    """
    The intersection of dictionary d1 with dictionary d2, to yield a
    new dictionary d. Since this merges keys, it presumes that d1 and
    d2 do not map the same key to different values.
    """
    d = {}
    for key, val in d1.iteritems():
        if key in d2:
            d[key] = val    
    return d

def dict_union(d1, d2):
    """
    The union of dictionary d1 with dictionary d2, to yield a new
    dictionary d. Since this merges keys, it presumes that d1 and d2
    do not map the same key to different values.
    """   
    return dict(d1, **d2)
        
######################################################################
# QUICK TEST

def quicktest():
    # Seed sets.
    turney_littman_positive = ["excellent", "good", "nice", "positive", "fortunate", "correct", "superior"]
    turney_littman_negative = ["nasty","bad", "poor", "negative", "unfortunate", "wrong", "inferior"]
    objective = ["administrative", "financial", "geographic", "constitute", "analogy", "ponder",
                 "material", "public", "department", "measurement", "visual"]

    # Synsets.
    synsets_list = [
        create_seed_synsets(turney_littman_positive, pos='a'),
        create_seed_synsets(turney_littman_negative, pos='a'),
        create_seed_synsets(objective) ]

    # Print the initial seed set, wrapped in <set n=i iteration=0>
    # tags.
    for i, synsets in enumerate(synsets_list):
        print "<synsets n=%s iteration=0>" % (i+1)
        for synset in sorted(synsets, cmp=(lambda x, y : cmp(x.name, y.name))):
            print "\t" + synset.name
        print "</synsets>\n"

    # Run.
    iterations = 4
    t = wordnet_sense_propagate(synsets_list, iterations, remove_overlap=True, methods=METHODS)

    # Print the final list from each iteration, wrapped in <set n=i
    # iteration=j> tags, where i is the index of the corresponding
    # initial set and j = iterations.
    for i, seq in enumerate(t):
        print "<synsets n=%s iteration=%s>" % (i+1, iterations)
        for synset in sorted(seq[-1], cmp=(lambda x, y : cmp(x.name, y.name))):
            print "\t" + synset.name
        print "</synsets>\n"

    # Sizes of the final iterations.
    for i, seq in enumerate(t):
        print "Items in the final iteration for class %s: %s" % (i+1, len(seq[-1]))
    

if __name__ == "__main__":
    quicktest()
    
######################################################################
