### Identifying Tweets with Adverse Drug Reactions Using Text Classification
---
- Definition of [Adverse drug reaction](https://en.wikipedia.org/wiki/Adverse_drug_reaction)
- A nice intro to [Text Classification](http://www.nltk.org/book/ch06.html)
- source of the [data set](http://diego.asu.edu/downloads/twitter_annotated_corpus/), please follow the [download instructions](http://diego.asu.edu/downloads/twitter_binary_data_readme.txt) and use the [python script](http://diego.asu.edu/downloads/download_binary_twitter_data.py) to download raw tweets if needed  


There are three sub-directories --- train, dev, and test --- each containing a .txt file and a .arff file:

 - The .txt file contains the raw tweets, with the following format:
     tweet-id TAB class TAB text-of-tweet
   where "class" is one of Y or N, indicating whether the tweet depicts an ADR or not (respectively).

 - The .arff file is a representation of the above, suitable for use with Weka. We have used the method of
   Mutual Information to identify the 92 best terms from the training tweets, and have recorded their 
   frequency in each tweet.
   The ARFF file format is essentially just comma-separated values (CSV) with a header, as follows:
     An @RELATION line, which names the dataset
     One @ATTRIBUTE line for each attribute, including (in this case) the ID and the Class. Most of the
       given attributes are NUMERIC; nominal attributes (like "class") can be indicated by enumerating 
       the possible attribute values in braces {}. (The Class is conventionally the final-listed attribute.)
     One @DATA line, indicating that the following lines are the instances in CSV.

	 	
