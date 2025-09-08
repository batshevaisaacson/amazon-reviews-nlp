#!/usr/bin/env python
# coding: utf-8

# In[4]:


from google.colab import drive
drive.mount('/content/drive')


# In[5]:


import pandas as pd

file_path = '/content/drive/MyDrive/Capstone/Amazon_Reviews.csv'
reviews = pd.read_csv(file_path)
print(reviews.shape)


# In[6]:


#check for duplicates in the text column of Amazon reviews
duplicates = reviews[reviews.duplicated(subset=['Text'])]
print(duplicates.shape)


# In[7]:


#remove duplicates and save to a new file
reviews = reviews.drop_duplicates(subset=['Text'])
print(reviews.shape)


# In[8]:


import re

def clean_review(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"<.*?>", "", text)    # remove HTML tags
    text = re.sub(r"[^\x00-\x7F]+", "", text)  # remove emojis/non-ASCII
    # Insert space after punctuation if missing
    text = re.sub(r"([.!?])(?=\S)", r"\1 ", text)
    #remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    #remove numbers
    text = re.sub(r"\d+", "", text)
    text = text.strip()
    return text


# In[9]:


#apply function to reviews
reviews['Cleaned_Text'] = reviews['Text'].apply(clean_review)

#view the first cleaned review
print(reviews['Cleaned_Text'][0])

#count the number of cleaned reviews
print(reviews.shape)


# In[10]:


reviews = reviews.reset_index(drop=True)


# In[ ]:


#load a sentence-BERT model
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

#generate embeddings (each review gets assigned a vector)
embeddings = model.encode(reviews['Cleaned_Text'], show_progress_bar=True)


# In[ ]:


print(f"Number of embeddings: {len(embeddings)}")


# In[ ]:


#Save embeddings to a file
import numpy as np
from google.colab import files
np.save('/content/drive/MyDrive/Capstone/embeddings2.npy', embeddings)


# In[14]:


#load the embeddings file
import numpy as np
from google.colab import files
embeddings = np.load('/content/drive/MyDrive/Capstone/embeddings2.npy')


# In[15]:


print(f"Number of embeddings: {len(embeddings)}")


# In[16]:


#standardize embeddings using standard scaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_embeddings = scaler.fit_transform(embeddings)


# In[17]:


#apply PCA to reduce dimentions before clustering
from sklearn.decomposition import PCA

pca = PCA(n_components=50, random_state=42)
reduced_embeddings = pca.fit_transform(scaled_embeddings)

print(f"Original shape: {scaled_embeddings.shape}")
print(f"Reduced shape: {reduced_embeddings.shape}")


# In[ ]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Take a random sample of 20,000 from the new feature space
sample_size = 20000
if normalized_features.shape[0] > sample_size:
    np.random.seed(42)
    sample_indices = np.random.choice(reduced_embeddings.shape[0], sample_size, replace=False)
    sample_embeddings = reduced_embeddings[sample_indices]
else:
    sample_embeddings = reduced_embeddings

# Set range of k to test
k_values = range(2, 13)

silhouette_scores = []
inertia_scores = []

# Run k-means on sample for each k
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(sample_embeddings)

    s_score = silhouette_score(sample_embeddings, kmeans.labels_)
    silhouette_scores.append(s_score)
    inertia_scores.append(kmeans.inertia_)


# In[ ]:


import matplotlib.pyplot as plt

#Plot silhouette scores results
plt.plot(k_values, silhouette_scores, marker='o')
plt.title('Silhouette Scores')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()


# In[ ]:


#Plot inertia (elbow plot)
plt.plot(k_values, inertia_scores, marker='o')
plt.title('Elbow Plot (Inertia)')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()


# In[18]:


#fit final K-means model with k=7
from sklearn.cluster import KMeans
final_kmeans = KMeans(n_clusters=7, random_state=42, n_init='auto')
#Train the model on the reduced embedding vectors (from PCA)
final_kmeans.fit(reduced_embeddings)
#get the cluster labels assigned to each review
cluster_labels = final_kmeans.labels_
#Add the cluster labels as a new column to the reviews data frame
#This associates each review with the cluster it belongs to
reviews['cluster'] = cluster_labels
#print the first few reviews along with their cluster labels
reviews[['Cleaned_Text', 'cluster']].head()


# In[19]:


#count how many reviews were assigned to each cluster
reviews['cluster'].value_counts()


# In[20]:


#find closest reviews to each cluster centroid
def closest_reviews_to_centroid(reduced_embeddings, cluster_labels, reviews, final_kmeans, top_n=20):
  closest_reviews = []
  #for each cluster
  for cluster in range(final_kmeans.n_clusters):
    #find the centroid of the cluster
    centroid = final_kmeans.cluster_centers_[cluster]
    #find the indicies of reviews in this cluster
    cluster_indices = np.where(cluster_labels == cluster)[0]
    #get embeddings of those reviews
    cluster_embeddings = reduced_embeddings[cluster_indices]
    #calculate distances from each embedding to the centroid
    distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
    #get the indicies of the closest reviews
    closest_indices = np.argsort(distances)[:top_n]
    #Map positions of closest reviews back to their original indices in the full dataset
    closest_review_indices = cluster_indices[closest_indices]
    #add the closest reviews to the list
    closest_reviews.append(reviews.iloc[closest_review_indices]['Cleaned_Text'].tolist())
  return closest_reviews


# In[21]:


print(reduced_embeddings.shape)  # e.g., (500000, 384) or (500000, 50)
print(final_kmeans.cluster_centers_.shape)  # Should match embedding dim


# In[23]:


import random

closest_reviews = closest_reviews_to_centroid(reduced_embeddings, cluster_labels, reviews, final_kmeans, top_n=20)

for cluster_id, review_texts in enumerate(closest_reviews):
    print(f"\n--- Cluster {cluster_id} ---\n")
    # Randomly sample 5 unique reviews from the closest 20
    sample_reviews = random.sample(review_texts, k=min(10, len(review_texts)))
    for i, review in enumerate(sample_reviews, 1):
        print(f"{i}. {review[:300]}...\n")


# In[ ]:


#extract top keywords per cluster
from sklearn.feature_extraction.text import TfidfVectorizer

#loop through each cluster
for i in range(reviews['cluster'].nunique()):
    #get the cleaned text of all reviews in the cluster
    cluster_texts = reviews[reviews['cluster'] == i]['Cleaned_Text'].dropna().astype(str)

    #initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=20, #return the top 20 important words
                                 stop_words='english' #remove common stopwords
                                 )
    #fit the vectorizer on the text and transform into TF-IDF features
    X = vectorizer.fit_transform(cluster_texts)
    #extract the top keywords
    keywords = vectorizer.get_feature_names_out()
    #print the keywords for this cluster
    print(f"Cluster {i} top keywords: {keywords}")


# In[ ]:


#calculate the average star rating by cluster
avg_ratings = reviews.groupby('cluster')['Score'].mean().reset_index()
avg_ratings.columns = ['cluster', 'Average Star Rating']
print(avg_ratings)


# In[ ]:


import matplotlib.pyplot as plt

plt.bar(avg_ratings['cluster'], avg_ratings['Average Star Rating'])
plt.xlabel('Cluster')
plt.ylabel('Average Star Rating')
plt.title('Average Star Rating per Cluster')
plt.ylim(1, 5)
plt.show()


# In[21]:


#distribution of star ratings
reviews['Score'].value_counts(normalize=True) * 100


# In[24]:


import matplotlib.pyplot as plt
import seaborn as sns

# Create a crosstab: counts of star ratings per cluster
ct = pd.crosstab(reviews['cluster'], reviews['Score'], normalize='index')

#Plot the stacked bar chart
ct.plot(kind='bar', stacked=True, colormap='viridis', figsize=(10,6))
plt.title('Star Rating Distribution by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Proportion of Star Ratings')
plt.legend(title='Star Rating', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()


# In[26]:


cluster_rating_pct = (
    reviews.groupby('cluster')['Score']
    .value_counts(normalize=True)
    .rename('percentage')
    .reset_index()
)

#sort by cluster and Score
cluster_rating_pct = cluster_rating_pct.sort_values(['cluster', 'Score'])

print(cluster_rating_pct)

