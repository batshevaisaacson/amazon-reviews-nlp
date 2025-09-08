# Amazon Reviews NLP

### Overview  
Applied **NLP** and **unsupervised clustering** to ~390,000 Amazon food product reviews to uncover customer themes and insights.  
This project was developed as part of a **Data Science Capstone course**.  

### Objectives  
- Identify recurring themes in reviews using clustering  
- Gain deeper insights beyond star ratings  
- Provide actionable recommendations for improving products and service  

### Methodology  
- Preprocessing: cleaned and standardized review text  
- Embeddings: Sentence-BERT  
- Dimensionality reduction: PCA  
- Clustering: K-means (k = 7, selected via silhouette & elbow method)  
- Theme extraction: TF-IDF keywords + sample reviews  

### Results  
- 7 key clusters representing customer themes (taste, quality, packaging, shipping, etc.)  
- Clusters revealed insights not visible in star ratings alone
- Findings directly informed **practical recommendations for sellers**  

### Files / Links  
- [Python Script](amazon_reviews.py)  
- [Presentation PDF](Amazon_Reviews_Capstone.pdf)  
