# Lexical Normalisation of Twitter Data

- Definition of [Text Normalization](https://en.wikipedia.org/wiki/Text_normalization), [Approximate String Matching
](https://en.wikipedia.org/wiki/Approximate_string_matching), [Edit Distance](https://en.wikipedia.org/wiki/Edit_distance) and [Levenshtein Distance](https://en.wikipedia.org/wiki/Levenshtein_distance) from Wikipedia
- [**How to Write a Spelling Corrector**](http://norvig.com/spell-correct.html) by Peter Norvig 
- A step-by-step [guide](http://www.nltk.org/howto/twitter.html) to twitter NLP using NLTK
- The project data set comes from [this paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.207.6218&rep=rep1&type=pdf) 
  > Bo Han and Timothy Baldwin (2011) Lexical normalisation of short text messages: Maknsens a #twitter. In _Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics_, Portland, USA. pp. 368–378.
- The lexical normalisation dictionary `emnlp_dict.txt` makes use of data from [here](https://people.eng.unimelb.edu.au/tbaldwin/etc/emnlp2012-lexnorm.tgz) and `Test_Set_3802_Pairs.txt` from [here](http://www.hlt.utdallas.edu/~yangl/data/Text_Norm_Data_Release_Fei_Liu/Test_Set_3802_Pairs.txt)
  >Bo Han, Paul Cook, and Timothy Baldwin. 2012. Automatically constructing a normalisation dictionary for microblogs. In Proceedings of EMNLP.  
  
  >Fei Liu, Fuliang Weng, Bingqing Wang, Yang Liu. Insertion, Deletion,
    or Substitution? Normalizing Text Messages without Pre-categorization
    nor Supervision. In Proceedings of the 49th Annual Meeting of the 
    Association for Computational Linguistics (ACL 2011), short paper, 
    pages 71-76.  
    
  >Fei Liu, Fuliang Weng, Xiao Jiang. A Broad-Coverage Normalization
    System for Social Media Language. In Proceedings of the 50th Annual
    Meeting of the Association for Computational Linguistics (ACL 2012), 
    pages 1035-1044.
    
`{labelled, unlabelled}-tokens.txt`:  

A list of tokens, one per line. The tokens are drawn from real tweets, excepts that tokens not containing at least one (English) alphabetical character - like "." or "!!" - have been excluded.  

`labelled-tokens.txt` has the form:  

  `Token` `Code`  `Canonical_Form`  
  
Where `Token` is drawn from the tweet text (suitably down-cased), `Canonical_Form` is the normalised version of the token, and `Code` can take one of three values: 
    
  + **IV**: "in vocabulary", such that the form from the tweet was found in the `dict.txt`, and is consequently not a candidate for normalisation.  
  + **OOV**: "out of vocabulary", such that the form of the token from the tweet was not found in the `dict.txt`, and thus the token was a candidate for normalisation. In some cases, the canonical form is homologous (equivalent) to the un-normalised form. In other cases, they are different --- these are the "spelling mistakes" that need to be "corrected".
  + **NO**: "not a normalisation candidate", such that the token was not considered in the normalisation process.
    
`unlabelled-tokens.txt` has the form:    

`Token` `???`  

where `Code` and `Canonical_Form` are omitted and it is used as test data only
