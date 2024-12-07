import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import torch
import torch.nn as nn
import plotly.express as px
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from gensim.models import Word2Vec

# Conditional NLTK Downloads
nltk_packages = ['stopwords', 'punkt', 'wordnet', 'omw-1.4']

for package in nltk_packages:
    try:
        if package in ['wordnet', 'omw-1.4']:
            nltk.data.find(f'corpora/{package}')
        else:
            nltk.data.find(f'tokenizers/{package}')
    except LookupError:
        nltk.download(package)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

##############################################
# Data Loading and Preprocessing
##############################################
categories = ['sci.space', 'sci.crypt', 'sci.electronics', 'sci.med']
newsgroups_data = fetch_20newsgroups(subset='all', categories=categories)

papers = pd.DataFrame({'text': newsgroups_data.data, 'target': newsgroups_data.target})
papers['subject'] = [newsgroups_data.target_names[i] for i in newsgroups_data.target]

papers = papers.sample(n=1000, random_state=42).reset_index(drop=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [re.sub(r'[^a-zA-Z]', '', word.lower()) for word in tokens]
    tokens = [t for t in tokens if t not in stop_words and t != '']
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

papers['tokens'] = papers['text'].apply(preprocess_text)

def extract_title(text):
    match = re.search(r'^Subject:\s*(.*)', text, re.MULTILINE | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        for line in text.split('\n'):
            line = line.strip()
            if line:
                return (line[:50] + '...') if len(line) > 50 else line
        return "No Title"

def extract_num_citations(text):
    lower_text = text.lower()
    return lower_text.count("cite") + lower_text.count("citation")

papers['title'] = papers['text'].apply(extract_title)
papers['num_citations'] = papers['text'].apply(extract_num_citations)

##############################################
# Embedding with Word2Vec
##############################################
w2v_model = Word2Vec(papers['tokens'], vector_size=50, window=5, min_count=2, workers=4, seed=42)

def get_doc_embedding(tokens):
    embeddings = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(w2v_model.vector_size)

papers['w2v_embedding'] = papers['tokens'].apply(get_doc_embedding)
doc_embeddings = np.vstack(papers['w2v_embedding'].values)

##############################################
# Dimensionality Reduction (PCA + TSNE)
##############################################
scaler = StandardScaler()
embeddings_norm = scaler.fit_transform(doc_embeddings)

pca = PCA(n_components=30, random_state=42)
embeddings_pca = pca.fit_transform(embeddings_norm)

tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings_pca)

papers['dim1'] = embeddings_2d[:,0]
papers['dim2'] = embeddings_2d[:,1]

subject_colors = {
    'sci.space': 'red',
    'sci.med': 'purple',
    'sci.electronics': 'brown',
    'sci.crypt': 'coral'
}

##############################################
# Load the Trained MLP Model
##############################################
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Ensure the model file path is correct
mlp = MLP(input_dim=50, hidden_dim=64, output_dim=2)
mlp.load_state_dict(torch.load("mlp_xy_predictor.pth", map_location=torch.device('cpu')))
mlp.eval()

##############################################
# Streamlit Interface
##############################################
st.title("Semantic Document Visualization")

st.write("This app shows documents from four scientific categories in a reduced semantic space. Enter your own document below, along with its title, and see where it would appear among the others.")

user_title = st.text_input("Article Title:", value="My Custom Document")
user_text = st.text_area("Article Abstract:", value="Type your document content here...")

if st.button("Predict and Plot"):
    user_tokens = preprocess_text(user_text)
    user_embedding = get_doc_embedding(user_tokens)
    user_embedding_tensor = torch.tensor(user_embedding, dtype=torch.float).unsqueeze(0)

    with torch.no_grad():
        predicted_coords = mlp(user_embedding_tensor).numpy()[0]

    user_doc = pd.DataFrame({
        'dim1': [predicted_coords[0]],
        'dim2': [predicted_coords[1]],
        'subject': ['User Document'],
        'title': [user_title]
    })

    plot_data = pd.concat([papers[['dim1','dim2','subject','title']], user_doc], ignore_index=True)

    fig = px.scatter(
        plot_data,
        x='dim1',
        y='dim2',
        color='subject',
        hover_data=['title'],
        title='Documents Positioned in a Semantic Space',
        color_discrete_map=subject_colors  # Use predefined colors
    )
    fig.update_xaxes(title_text='Semantic Dimension 1')
    fig.update_yaxes(title_text='Semantic Dimension 2')

    st.plotly_chart(fig, use_container_width=True)
