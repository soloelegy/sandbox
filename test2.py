"""
    Test clustering for texts
"""

# Import packages
import csv
from nltk import word_tokenize
# from scipy import sparse
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
from string import punctuation
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

#### 1. Load data

# Load raw data
with open('test_all.csv', encoding="utf8", errors="replace") as csvfile:
    file_obj = csv.reader(csvfile)
    file_list = list(file_obj)

print(len(file_list))
# 5017

# Create dataframe
col_names = file_list[0]
file_df = pd.DataFrame(file_list[1:], columns=col_names)


#### 2. ETL process for clustering

# Convert to words
ftr_df = file_df[['Key','Description']]
ftr_df['words'] = ftr_df['Description'].apply(lambda txt: word_tokenize(txt), 1)

# Uncomment to check the max and min number of words
#max(ftr_df['words'].apply(lambda w: len(w), 1))  #5720
#min(ftr_df['words'].apply(lambda w: len(w), 1))  #1


# Vectorize
vectorizer = TfidfVectorizer(max_df=0.5
                             , max_features=1000
                             , min_df=2
                             , stop_words='english'
                             ,use_idf=True)
# Perform PCA and scaling
svd = TruncatedSVD(10)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

stage1 = vectorizer.fit_transform(ftr_df['Description'])
stage2 = lsa.fit_transform(stage1)

#### 3. Clustering

# K-means cluster 
n_clstr = 15 # number of clusters

km = KMeans(n_clusters = n_clstr
            , init = 'k-means++'
            , max_iter = 100, n_init=1
            , verbose = 0)

km.fit(stage2)
#clstr_rslt = km.fit_transform(stage2)

print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(stage2, km.labels_, sample_size=1000))

#### 4. Export results

# Attach labels and eport to csv
file_df["cluster_num"] = pd.Series(km.labels_)
file_df.to_csv('C:\\Users\\waner.liang\\nlp_poc\\cluster_result.csv', sep = ',', encoding='utf-8')

# Count number in each cluster
summary_df = file_df.groupby("cluster_num").count()
summary_df.to_csv('C:\\Users\\waner.liang\\nlp_poc\\summary.csv', sep = ',', encoding='utf-8')



original_space_centroids = svd.inverse_transform(km.cluster_centers_)
order_centroids = original_space_centroids.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()

clstr_info = []
for i in range(n_clstr):
    top10_terms = []
    for ind in order_centroids[i, :10]:
        top10_terms.append(terms[ind])
    clstr_info.append((i, top10_terms))

pd.DataFrame(clstr_info, columns=['cluster', 'top10_terms']) \
             .to_csv('C:\\Users\\waner.liang\\nlp_poc\\cluster_info.csv'\
                     , sep = ',', encoding='utf-8')

for i in range(n_clstr):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end='')
    print()
    
    
    
