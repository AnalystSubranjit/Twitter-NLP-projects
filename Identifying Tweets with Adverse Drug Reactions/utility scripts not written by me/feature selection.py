import pandas as pd


#%% 
f1 = open("3.txt", 'r')
f2 = open("1.txt", 'r')
f3 = open("4.txt", 'r')

def read_file(f):
    file = []
    for line in f:
        file.append(line)
    f.close()
    return file

a = read_file(f1)
b = read_file(f2)
c = read_file(f3)

a = a[21:345]
b = b[21:345]
c = c[21:345]

a = [x.split()[2] for x in a ]
b = [x.split()[2] for x in b ]
c = [x.split()[2] for x in c ]

    
features = list(set(a) & set(b) & set(c))


#%% Parse tweets
f1 = open("train.txt", 'r',  encoding='utf-8')
f2 = open("dev.txt", 'r',  encoding='utf-8')
f3 = open("test.txt", 'r',  encoding='utf-8')

train = parse_tweets(f1)
dev = parse_tweets(f2)
test = parse_tweets(f3)
all_tweets = train + dev + test

f1.close()
f2.close()
f3.close()


#%% Tokenizing
train = [tknzr.tokenize(x) for x in train]
dev = [tknzr.tokenize(x) for x in dev]
test = [tknzr.tokenize(x) for x in test] 

#%% Stemming
train = [[stemmer.stem(word) for word in words] for words in train]
dev = [[stemmer.stem(word) for word in words] for words in dev]
test = [[stemmer.stem(word) for word in words] for words in test]

#%% Parsing csv
train_csv = pd.read_csv('train_new.csv')
dev_csv = pd.read_csv('dev_new.csv')
test_csv = pd.read_csv('test_new.csv')

#%%
train_csv = train_csv[features + ['class']]
test_csv = test_csv[features + ['class']]
dev_csv = dev_csv[features + ['class']]

#%% write to csv
train_csv.to_csv('train_new.csv', sep=',', encoding='utf-8', index=False)
dev_csv.to_csv('dev_new.csv', sep=',', encoding='utf-8', index=False)
test_csv.to_csv('test_new.csv', sep=',', encoding='utf-8', index=False)

#%% Generate n-gram of tags
def bigrams(tags):
    bigram = []
    for tag in tags:
        bigram.append([tuple(x) for x in TextBlob(tag).ngrams(n = 2)])     
    return bigram

def trigrams(tags):
    trigram = []
    for tag in tags:
        trigram.append([tuple(x) for x in TextBlob(tag).ngrams(n = 3)])     
    return trigram


#%% read all tags
def readtags(f):
    all_tags=[]
    for line in f:
        all_tags.append(line.split('\t')[1])
    f.close()
    return all_tags

f1 = open('all_tags.txt','r', encoding = 'utf-8')
f2 = open('train_tags.txt','r', encoding = 'utf-8')
f3 = open('dev_tags.txt','r', encoding = 'utf-8')
f4 = open('test_tags.txt','r', encoding = 'utf-8')

all_tags = readtags(f1)
train_tags = readtags(f2)
dev_tags = readtags(f3)
test_tags = readtags(f4)



#%% continue
bigram_all = bigrams(all_tags)
bigram_all = [item for sublist in bigram_all for item in sublist]
bigram_all = list(set(bigram_all))


#%% continue
trigram_all = trigrams(all_tags)
trigram_all = [item for sublist in trigram_all for item in sublist]
trigram_all = list(set(trigram_all))


#%% continue
train_tags = bigrams(train_tags)
dev_tags = bigrams(dev_tags)
test_tags = bigrams(test_tags)


#%% continue
train_tags = trigrams(train_tags)
dev_tags = trigrams(dev_tags)
test_tags = trigrams(test_tags)


#%% count n-gram
def count_gram(tweets, allgram):
    counts=[]
    for tweet in tweets:
        count=[]
        for gram in allgram:
            count.append(tweet.count(gram))
        counts.append(count)
    return pd.DataFrame(counts, columns = [x[0] + x[1] + x[2] for x in trigram_all])
        
train_tags = count_gram(train_tags, trigram_all)
dev_tags = count_gram(dev_tags, trigram_all)
test_tags = count_gram(test_tags, trigram_all)


#%% drug name
drug = []
def parse_drug(drug):
    f = open('drug.txt','r')
    for line in f:
        drug.append(line.strip())
    return 
parse_drug(drug)
drug = list(set(drug) - set(list(train_csv.columns)))


#%% narrow down
def count_drug():
    a=[]
    for y in drug:
        for x in train+dev+test:
            if y.isalnum() == False:
                break
            if re.match(y, x) != None:
                a.append(y)
                break
    return a
drug = count_drug()



#%% continue
def drug_count(tweets):
    b=[]
    for x in tweets:
        a=[]
        for y in drug:
            if re.match(y, x) != None:
                a.append(1)
            else:
                a.append(0)
        b.append(a)
    return pd.DataFrame(b, columns = drug)

train_drug = drug_count(train)
dev_drug = drug_count(dev)
test_drug = drug_count(test)


#%%
train_csv = pd.concat([train_csv, train_tags], axis=1)
dev_csv = pd.concat([dev_csv, dev_tags], axis=1)
test_csv = pd.concat([test_csv, test_tags], axis=1)


#%% ADR terms
def readf():
    f = open("adr.txt", 'r')
    se=[]
    for x in f:
        se.append(x.strip())
    f.close()
    return se
se = readf()
se = list(set(se) - set(list(train_csv.columns)))


#%% remove symbols 
def Textcleaner(tweets):
    cleaned = [re.sub(r'\W+', ' ', x).strip().lower() for x in tweets]
    return cleaned

train = Textcleaner(train)
dev = Textcleaner(dev)
test = Textcleaner(test)


#%% narrow down again
def se1():
    a=[]
    for y in se:
        for x in train+dev+test:
            if re.match(y, x) != None:
                a.append(y)
                break
    return a
se = se1()


#%%
def count_se(tweets):
    b=[]
    for x in tweets:
        a=[]
        for y in se:
            if re.match(y, x) != None:
                a.append(1)
            else:
                a.append(0)
        b.append(a)
    return b
train_se = count_se(train)
dev_se = count_se(dev)
test_se = count_se(test)

train_se = pd.DataFrame(train_se, columns = se)
dev_se = pd.DataFrame(dev_se, columns = se)
test_se = pd.DataFrame(test_se, columns = se)



#%%

# =============================================================================
# a = list(train_csv.columns)
# 
# c = np.delete(a, [1521, 1522, 1523, 1524, 1525]).tolist()
# 
# a = c + ['polarity','subjectivity','compound_score','sentiWordNet','class']
# 
# 
# train_csv = train_csv[a]
# test_csv = test_csv[a]
# dev_csv = dev_csv[a]
# =============================================================================
