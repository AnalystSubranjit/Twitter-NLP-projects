This file describes the data files for COMP90049 Project 2, 2017S2.

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

	 	 
	 
Observe that the "Correctly Classified Instances", based on the dataset without the ID attribute, are 881 (81.9%). 
This is actually (very slightly) worse than before! 
You might like to ask which instances are being classified incorrectly now, which were correct before, and think about why that might have happened.
Or, there might be some instances that were incorrect before, but are correct now (and why?).

- Did Naive Bayes classify the data well? Why or why not?
- Was it more effective on the "Y" class or the "N" class? Why? What does that mean?
- What are the useful attributes? What are the useless attributes? Is there some combination of attributes that works particularly well? Why or why not?
- What other attributes might be helpful here?