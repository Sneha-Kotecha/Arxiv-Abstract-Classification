# Load Relevant Packages

import csv
import random
import numpy as np
import pandas as pd
from collections import defaultdict
import re
import time
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from transformers import BertGenerationEncoder, BertTokenizer, BertGenerationDecoder, EncoderDecoderModel, BertForMaskedLM, BertConfig, AutoModelForMaskedLM
from tqdm import tqdm
import string
from bertviz import model_view, head_view
import datetime

from torch import nn, optim
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import re
import ast
import time
import json
import pickle
import random
import inspect
import datetime as dt
import multiprocessing
from typing import Dict, Any, Generator, Tuple, Optional
from sklearn.metrics import f1_score
from tabulate import tabulate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification
import torch

import nltk
nltk.download('stopwords')
nltk.download('punkdt')

import warnings
warnings.filterwarnings("ignore")

class Datum:
    def __init__(self, id, title, categories, abstract, topic, sub_topic):
        self.id = id
        self.title = title
        self.categories = categories
        self.abstract = abstract
        self.topic = topic
        self.sub_topic = sub_topic

    def getid(self):
        return self.id
    
    def gettitle(self):
        return self.title
    
    def getcategories(self):
        return self.categories
    
    def getabstract(self):
        return self.abstract
    
    def gettopic(self):
        return self.topic
    
    def getsubtopic(self):
        return self.sub_topic
    
    def setid(self, id):
        self.id = id

    def settitle(self, title):
        self.title = title
    
    def setcategories(self, categories):
        self.categories = categories    

    def setabstract(self, abstract):
        self.abstract = abstract
    
    def settopic(self, topic):
        self.topic = topic  
    
    def setsubtopic(self, sub_topic):    
        self.sub_topic = sub_topic
        
    def get_unique_topics(self):
        return list(set(self.gettopic().split(', ')))
    
    def single_category(self):
        return len(self.gettopic().split(', ')) == 1

DATASET_PATH = "/data/NLP_Summative/arxiv-metadata-oai-snapshot.json"
NUM_PAPERS = 1000000

import json

def load_arxiv_data(dataset_path="/Users/snehakotecha/Desktop/Oxford/Introduction to NLP/Summative /arxiv-metadata-oai-snapshot.json", num_papers=None):
    """Loads arXiv metadata from a JSON file.

    Args:
        dataset_path: Path to the JSON file.
        num_papers: If specified, loads only the first `num_papers`.

    Returns:
        A list of metadata entries (dictionaries).
    """

    data = []
    with open(dataset_path, 'r') as f:
        for line in f:
            try:
                data_obj = json.loads(line)
                data.append(data_obj)
                if num_papers and len(data) == num_papers:
                    break
            except json.JSONDecodeError:
                print(f"Error decoding JSON on line: {line}")

    return data

# Example usage:
arxiv_data = load_arxiv_data("arxiv-metadata-oai-snapshot.json", num_papers=NUM_PAPERS)  # Load 100 papers

category_map = {
    'astro-ph': ('Physics', 'Astrophysics'),
    'astro-ph.CO': ('Physics', 'Astrophysics - Cosmology and Nongalactic Astrophysics'),
    'astro-ph.EP': ('Physics', 'Astrophysics - Earth and Planetary Astrophysics'),
    'astro-ph.GA': ('Physics', 'Astrophysics - Astrophysics of Galaxies'),
    'astro-ph.HE': ('Physics', 'Astrophysics - High Energy Astrophysical Phenomena'),
    'astro-ph.IM': ('Physics', 'Astrophysics - Instrumentation and Methods for Astrophysics'),
    'astro-ph.SR': ('Physics', 'Astrophysics - Solar and Stellar Astrophysics'),
    'acc-phys': ('Physics', 'Acceleator Physics'),
    'cond-mat.dis-nn': ('Physics', 'Condensed Matter Physics - Disordered Systems and Neural Networks'),
    'cond-mat.mes-hall': ('Physics', 'Condensed Matter Physics - Mesoscale and Nanoscale Physics'),
    'cond-mat.mtrl-sci': ('Physics', 'Condensed Matter Physics - Materials Science'),
    'mtrl-th': ('Physics', 'Condensed Matter Physics - Materials Science'),
    'cond-mat.other': ('Physics', 'Condensed Matter Physics - Other Condensed Matter'),
    'cond-mat.quant-gas': ('Physics', 'Condensed Matter Physics - Quantum Gases'),
    'cond-mat.soft': ('Physics', 'Condensed Matter Physics - Soft Condensed Matter'),
    'cond-mat.stat-mech': ('Physics', 'Condensed Matter Physics - Statistical Mechanics'),
    'cond-mat.str-el': ('Physics', 'Condensed Matter Physics - Strongly Correlated Electrons'),
    'cond-mat.supr-con': ('Physics', 'Condensed Matter Physics - Superconductivity'),
    'supr-con': ('Physics', 'Condensed Matter Physics - Superconductivity'),
    'cs.AI': ('Computer Science', 'Artificial Intelligence'),
    'cs.AR': ('Computer Science', 'Hardware Architecture'),
    'cs.CC': ('Computer Science', 'Computational Complexity'),
    'cs.CE': ('Computer Science', 'Computational Engineering, Finance, and Science'),
    'cs.CG': ('Computer Science', 'Computational Geometry'),
    'cs.CL': ('Computer Science', 'Computation and Language'),
    'cmp-lg': ('Computer Science', 'Computation and Language'),
    'cs.CR': ('Computer Science', 'Cryptography and Security'),
    'cs.CV': ('Computer Science', 'Computer Vision and Pattern Recognition'),
    'cs.CY': ('Computer Science', 'Computers and Society'),
    'cs.DB': ('Computer Science', 'Databases'),
    'cs.DC': ('Computer Science', 'Distributed, Parallel, and Cluster Computing'),
    'cs.DL': ('Computer Science', 'Digital Libraries'),
    'cs.DM': ('Computer Science', 'Discrete Mathematics'),
    'cs.DS': ('Computer Science', 'Data Structures and Algorithms'),
    'cs.ET': ('Computer Science', 'Emerging Technologies'),
    'cs.FL': ('Computer Science', 'Formal Languages and Automata Theory'),
    'cs.GL': ('Computer Science', 'General Literature'),
    'cs.GR': ('Computer Science', 'Graphics'),
    'cs.GT': ('Computer Science', 'Computer Science and Game Theory'),
    'cs.HC': ('Computer Science', 'Human-Computer Interaction'),
    'cs.IR': ('Computer Science', 'Information Retrieval'),
    'cs.IT': ('Computer Science', 'Information Theory'),
    'cs.LG': ('Computer Science', 'Machine Learning'),
    'cs.LO': ('Computer Science', 'Logic in Computer Science'),
    'cs.MA': ('Computer Science', 'Multiagent Systems'),
    'cs.MM': ('Computer Science', 'Multimedia'),
    'cs.MS': ('Computer Science', 'Mathematical Software'),
    'cs.NA': ('Computer Science', 'Numerical Analysis'),
    'cs.NE': ('Computer Science', 'Neural and Evolutionary Computing'),
    'cs.NI': ('Computer Science', 'Networking and Internet Architecture'),
    'cs.OH': ('Computer Science', 'Other Computer Science'),
    'cs.OS': ('Computer Science', 'Operating Systems'),
    'cs.PF': ('Computer Science', 'Performance'),
    'cs.PL': ('Computer Science', 'Programming Languages'),
    'cs.RO': ('Computer Science', 'Robotics'),
    'cs.SC': ('Computer Science', 'Symbolic Computation'),
    'cs.SD': ('Computer Science', 'Sound'),
    'cs.SE': ('Computer Science', 'Software Engineering'),
    'cs.SI': ('Computer Science', 'Social and Information Networks'),
    'cs.SY': ('Computer Science', 'Systems and Control'),
    'econ.EM': ('Economics', 'Econometrics'),
    'econ.TH': ('Economics', 'Theoretical'),
    'econ.GN': ('Economics', 'General Economics'),
    'eess.AS': ('Electrical Engineering and Systems Science', 'Audio and Speech Processing'),
    'eess.IV': ('Electrical Engineering and Systems Science', 'Image and Video Processing'),
    'eess.SP': ('Electrical Engineering and Systems Science', 'Signal Processing'),
    'eess.SY': ('Electrical Engineering and Systems Science', 'Systems and Control'),
    'gr-qc': ('Physics', 'General Relativity and Quantum Cosmology'),
    'hep-ex': ('Physics', 'High Energy Physics - Experiment'),
    'hep-lat': ('Physics', 'High Energy Physics - Lattice'),
    'hep-ph': ('Physics', 'High Energy Physics - Phenomenology'),
    'hep-th': ('Physics', 'High Energy Physics - Theory'),
    'math.AC': ('Mathematics', 'Commutative Algebra'),
    'math.AG': ('Mathematics', 'Algebraic Geometry'),
    'alg-geom': ('Mathematics', 'Algebraic Geometry'), 
    'math.AP': ('Mathematics', 'Analysis of PDEs'),
    'math.AT': ('Mathematics', 'Algebraic Topology'),
    'math.CA': ('Mathematics', 'Classical Analysis and ODEs'),
    'math.CO': ('Mathematics', 'Combinatorics'),
    'math.CT': ('Mathematics', 'Category Theory'),
    'math.CV': ('Mathematics', 'Complex Variables'),
    'math.DG': ('Mathematics', 'Differential Geometry'),
    'dg-ga': ('Mathematics', 'Differential Geometry'),
    'math.DS': ('Mathematics', 'Dynamical Systems'),
    'math.FA': ('Mathematics', 'Functional Analysis'),
    'funct-an': ('Mathematics', 'Functional Analysis'),
    'math.GM': ('Mathematics', 'General Mathematics'),
    'math.GN': ('Mathematics', 'General Topology'),
    'math.GR': ('Mathematics', 'Group Theory'),
    'math.GT': ('Mathematics', 'Geometric Topology'),
    'math.HO': ('Mathematics', 'History and Overview'),
    'math.IT': ('Mathematics', 'Information Theory'),
    'math.KT': ('Mathematics', 'K-Theory and Homology'),
    'math.LO': ('Mathematics', 'Logic'),
    'math.MG': ('Mathematics', 'Metric Geometry'),
    'math.MP': ('Mathematics', 'Mathematical Physics'),
    'math.NA': ('Mathematics', 'Numerical Analysis'),
    'math.NT': ('Mathematics', 'Number Theory'),
    'math.OA': ('Mathematics', 'Operator Algebras'),
    'math.OC': ('Mathematics', 'Optimization and Control'),
    'math.PR': ('Mathematics', 'Probability'),
    'math.QA': ('Mathematics', 'Quantum Algebra'),
    'q-alg': ('Mathematics', 'Quantum Algebra'),
    'math.RA': ('Mathematics', 'Rings and Algebras'),
    'math.RT': ('Mathematics', 'Representation Theory'),
    'math.SG': ('Mathematics', 'Symplectic Geometry'),
    'math.SP': ('Mathematics', 'Spectral Theory'),
    'math.ST': ('Mathematics', 'Statistics Theory'),
    'math-ph': ('Physics', 'Mathematical Physics'),
    'nlin.AO': ('Physics', 'Nonlinear Sciences - Adaptation and Self-Organizing Systems'),
    'nlin.CD': ('Physics', 'Nonlinear Sciences - Chaotic Dynamics'),
    'chao-dyn': ('Physics', 'Nonlinear Sciences - Chaotic Dynamics'),
    'nlin.CG': ('Physics', 'Nonlinear Sciences - Cellular Automata and Lattice Gases'),
    'comp-gas': ('Physics', 'Nonlinear Sciences - Cellular Automata and Lattice Gases'),
    'nlin.PS': ('Physics', 'Nonlinear Sciences - Pattern Formation and Solitons'),
    'patt-sol': ('Physics', 'Nonlinear Sciences - Pattern Formation and Solitons'),
    'nlin.SI': ('Physics', 'Nonlinear Sciences - Exactly Solvable and Integrable Systems'),
    'solv-int': ('Physics', 'Nonlinear Sciences - Exactly Solvable and Integrable Systems'),
    'adap-org': ('Physics', 'Nonlinear Sciences - Adaptation and Self-Organizing Systems'),
    'nucl-ex': ('Physics', 'Nuclear Experiment'),
    'nucl-th': ('Physics', 'Nuclear Theory'),
    'physics.acc-ph': ('Physics', 'Accelerator Physics'),
    'physics.ao-ph': ('Physics', 'Atmospheric and Oceanic Physics'),
    'ao-sci': ('Physics', 'Atmospheric and Oceanic Physics'),   
    'physics.app-ph': ('Physics', 'Applied Physics'),
    'physics.atm-clus': ('Physics', 'Atomic and Molecular Clusters'),
    'physics.atom-ph': ('Physics', 'Atomic Physics'),
    'atom-ph': ('Physics', 'Atomic Physics'),
    'physics.bio-ph': ('Physics', 'Biological Physics'),
    'physics.chem-ph': ('Physics', 'Chemical Physics'),
    'chem-ph': ('Physics', 'Chemical Physics'),
    'physics.class-ph': ('Physics', 'Classical Physics'),
    'physics.comp-ph': ('Physics', 'Computational Physics'),
    'physics.data-an': ('Physics', 'Data Analysis, Statistics and Probability'),
    'bayes-an' : ('Physics', 'Data Analysis, Statistics and Probability'),
    'physics.ed-ph': ('Physics', 'Physics Education'),
    'physics.flu-dyn': ('Physics', 'Fluid Dynamics'),
    'physics.gen-ph': ('Physics', 'General Physics'),
    'physics.geo-ph': ('Physics', 'Geophysics'),
    'physics.hist-ph': ('Physics', 'History and Philosophy of Physics'),
    'physics.ins-det': ('Physics', 'Instrumentation and Detectors'),
    'physics.med-ph': ('Physics', 'Medical Physics'),
    'physics.optics': ('Physics', 'Optics'),
    'physics.plasm-ph': ('Physics', 'Plasma Physics'),
    'plasm-ph': ('Physics', 'Plasma Physics'),
    'physics.pop-ph': ('Physics', 'Popular Physics'),
    'physics.soc-ph': ('Physics', 'Physics and Society'),
    'physics.space-ph': ('Physics', 'Space Physics'),
    'quant-ph': ('Physics', 'Quantum Physics'),
    'q-bio': ('Quantitative Biology'),
    'q-bio.BM': ('Quantitative Biology', 'Biomolecules'),
    'q-bio.CB': ('Quantitative Biology', 'Cell Behavior'),
    'q-bio.GN': ('Quantitative Biology', 'Genomics'),
    'q-bio.MN': ('Quantitative Biology', 'Molecular Networks'),
    'q-bio.NC': ('Quantitative Biology', 'Neurons and Cognition'),
    'q-bio.OT': ('Quantitative Biology', 'Other Quantitative Biology'),
    'q-bio.PE': ('Quantitative Biology', 'Populations and Evolution'),
    'q-bio.QM': ('Quantitative Biology', 'Quantitative Methods'),
    'q-bio.SC': ('Quantitative Biology', 'Subcellular Processes'),
    'q-bio.TO': ('Quantitative Biology', 'Tissues and Organs'),
    'q-fin.CP': ('Quantitative Finance', 'Computational Finance'),
    'q-fin.EC': ('Quantitative Finance', 'Economics'),
    'q-fin.GN': ('Quantitative Finance', 'General Finance'),
    'q-fin.MF': ('Quantitative Finance', 'Mathematical Finance'),
    'q-fin.PM': ('Quantitative Finance', 'Portfolio Management'),
    'q-fin.PR': ('Quantitative Finance', 'Pricing of Securities'),
    'q-fin.RM': ('Quantitative Finance', 'Risk Management'),
    'q-fin.ST': ('Quantitative Finance', 'Statistical Finance'),
    'q-fin.TR': ('Quantitative Finance', 'Trading and Market Microstructure'),
    'stat.AP': ('Statistics', 'Applications'),
    'stat.CO': ('Statistics', 'Computation'),
    'stat.ME': ('Statistics', 'Methodology'),
    'stat.ML': ('Statistics', 'Machine Learning'),
    'stat.OT': ('Statistics', 'Other Statistics'),
    'stat.TH': ('Statistics', 'Statistics Theory'),
    'cond-mat':('Physics', 'Condensed Matter')
}

# Assuming arxiv_data is a list of dictionaries containing paper information
for paper in arxiv_data:
    # Get the categories from the paper
    categories = paper.get('categories', '').split()
    
    # Map the categories to their main category and subcategory
    mapped_categories = [(category_map[cat][0], category_map[cat][1]) for cat in categories]
    
    # Extract the main category and subcategory from the mapped categories
    topics = [cat[0] for cat in mapped_categories]
    sub_topics = [cat[1] for cat in mapped_categories]
    
    # Remove duplicate main categories
    unique_topics = set(topics)
    unique_sub_topics = set(sub_topics)
    
    # Add the topic and sub_topic fields to the paper dictionary
    paper['main_categories'] = ', '.join(unique_topics)
    paper['sub_categories'] = ', '.join(unique_sub_topics)
# Assuming arxiv_data is a list of dictionaries
for d in arxiv_data:
    if 'main_categories' in d and isinstance(d['main_categories'], str):
        d['main_categories'] = d['main_categories'].split(",")[0]

arxiv_data_objects = []
for entry in arxiv_data:
    datum = Datum(entry['id'], entry['title'], entry['categories'], entry['abstract'], entry['main_categories'], entry['sub_categories'])
    arxiv_data_objects.append(datum)


arxiv_data_objects = [obj for obj in arxiv_data_objects if "Q" not in obj.get_unique_topics()]
arxiv_data_objects = [obj for obj in arxiv_data_objects if "Economics" not in obj.get_unique_topics()]
arxiv_data_objects = [obj for obj in arxiv_data_objects if "Electrical Engineering and Systems Science" not in obj.get_unique_topics()]

topics = ["Physics", "Mathematics", "Computer Science", "Statistics", "Quantitative Biology", "Quantitative Finance"]
arxiv_data_objects = [obj for obj in arxiv_data_objects if any(any(topic in unique_topic for topic in topics) for unique_topic in obj.get_unique_topics())]
unique_topics = set(topic for obj in arxiv_data_objects for topic in obj.get_unique_topics())
print(unique_topics)

main_categories_counts = pd.Series([datum.get_unique_topics() for datum in arxiv_data_objects]).value_counts()
print(main_categories_counts)




arxiv_data_objects = [obj for obj in arxiv_data_objects if any(any(topic in unique_topic for topic in topics) for unique_topic in obj.get_unique_topics())]
unique_topics = set(topic for obj in arxiv_data_objects for topic in obj.get_unique_topics())
print(unique_topics)
import random

# Get the number of samples in the least represented category
min_samples = min(len([obj for obj in arxiv_data_objects if category in obj.get_unique_topics()]) 
                  for category in ['Mathematics', 'Computer Science', 'Physics', 'Quantitative Finance', 'Statistics', 'Quantitative Biology'])

# Initialize an empty list to store the undersampled data objects
undersampled_data_objects = []

# For each category, undersample to the minimum sample size
for category in ['Mathematics', 'Computer Science', 'Physics', 'Quantitative Finance', 'Statistics', 'Quantitative Biology']:
    # Filter the samples for the current category
    category_samples = [obj for obj in arxiv_data_objects if category in obj.get_unique_topics()]
    
    # Randomly undersample the category samples
    random.shuffle(category_samples)
    category_samples = category_samples[:min_samples]
    
    # Add the undersampled category samples to the list of all undersampled data objects
    undersampled_data_objects.extend(category_samples)

# Replace the original data objects with the undersampled data objects
arxiv_data_objects = undersampled_data_objects

main_categories_counts = pd.Series([datum.get_unique_topics() for datum in arxiv_data_objects]).value_counts()
print(main_categories_counts)

from sklearn.model_selection import train_test_split
random.seed(0)
# Shuffle the data
shuffled_data = arxiv_data_objects.copy()
random.shuffle(shuffled_data)

# Split the data into train, dev, and test sets
train_data, test_data = train_test_split(shuffled_data, test_size=0.2, random_state=42)
train_data, dev_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Print the sizes of the train, dev, and test sets
print(f"Train set size: {len(train_data)}")
print(f"Dev set size: {len(dev_data)}")
print(f"Test set size: {len(test_data)}")
from collections import Counter

# Assuming `data` is your entire dataset
unique_topics = [datum.get_unique_topics() for datum in arxiv_data_objects]

# Flatten the list of lists into a single list
unique_topics = [topic for sublist in unique_topics for topic in sublist]

# Count the occurrences of each topic
topic_counts = Counter(unique_topics)

print(topic_counts)
unique_topics = {entry.gettopic() for entry in arxiv_data_objects}
unique_topics = list(set(unique_topics))
unique_topics
topics = ["Physics", "Mathematics", "Computer Science", "Quantitative Finance", "Statistics", "Quantitative Biology"]
import random
from sklearn.metrics import f1_score


def clean_text(text):
    # Replace mentions and URLs with special token
    text = re.sub(r"@[A-Za-z0-9_-]+", 'USR', text)
    text = re.sub(r"https?\S+", 'URL', text)

    # Remove newline and tab characters
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')

    # Strip whitespace
    text = text.strip()

    # Make lowercase
    text = text.lower()

    return text

for datum in arxiv_data_objects:
    datum.abstract = clean_text(datum.abstract)

import nltk
from nltk.tokenize import word_tokenize

# Tokenize the abstracts
for datum in arxiv_data_objects:
    datum.tokens = word_tokenize(datum.abstract)

for datum in arxiv_data_objects:
    datum.abstract = ' '.join(datum.tokens)
shuffled_data = arxiv_data_objects.copy()
random.shuffle(shuffled_data)

# Split the data into train, dev, and test sets
train_data, test_data = train_test_split(shuffled_data, test_size=0.2, random_state=42)
train_data, dev_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Print the sizes of the train, dev, and test sets
print(f"Train set size: {len(train_data)}")
print(f"Dev set size: {len(dev_data)}")
print(f"Test set size: {len(test_data)}")

all_topics = ["Physics", "Mathematics", "Computer Science", "Quantitative Finance", "Statistics", "Quantitative Biology"]
pred = []
true = []

for datum in test_data:
    datum_topics = datum.get_unique_topics()
    # Randomly select a topic from all possible topics
    pred_topic = random.choice(all_topics)
    for topic in datum_topics:
        if topic in all_topics:
            true.append(topic)
    pred.append(pred_topic)

f1 = f1_score(true, pred, average='macro')
print("The F1 score for the random baseline is: ", f1)

pred[:5]
true[:5]


# Logistic Regression
class Config:
    max_length = 500
    learning_rate = 0.01
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class LogisticRegression(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LogisticRegression, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size).to(Config.device)
        self.linear = nn.Linear(hidden_size * Config.max_length, 6).to(Config.device)

    def forward(self, abstract):
        abstract = self.embedding(abstract)
        input = abstract
        input = torch.flatten(input, start_dim=1)
        linear_output = self.linear(input)
        return linear_output
SOS_token = 0
EOS_token = 1

class LanguageStores:
    def __init__(self):
        self.word2index = {'<UNK>': 0}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        # Write your code here
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        # Write your code here
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.n_words += 1

    def print_stats(self):
        print('Vocabulary size of %d' % (self.n_words))


def prepareLanguageDictionaries():
    lang = LanguageStores()

   
    for datum in train_data:
        lang.add_sentence(datum.getabstract())
    return lang


lang = prepareLanguageDictionaries()
lang.print_stats()
num_classes = len(topics)
num_classes

def indexesFromSentence(lang, sentence):
    # Write your code here
    unknown = lang.word2index['<UNK>']
    return [lang.word2index.get(word, unknown) for word in sentence.split(' ')[:Config.max_length]]

def get_dataloader(batch_size, data):
    lang = prepareLanguageDictionaries()

    n = len(data)
    input_ids = np.zeros((n, Config.max_length), dtype=np.int32)
    target_ids = np.zeros((n, num_classes), dtype=np.int32)

    for idx, datum in enumerate(data):
        inp_indices = indexesFromSentence(lang, datum.getabstract())
        input_ids[idx, :len(inp_indices)] = inp_indices
        target_ids[idx, [topics.index(topic) for topic in datum.get_unique_topics()]] = 1

    train_tensor_data = TensorDataset(torch.LongTensor(input_ids).to(Config.device),
                               torch.Tensor(target_ids).to(Config.device))

    train_sampler = RandomSampler(train_tensor_data)
    train_dataloader = DataLoader(train_tensor_data, sampler=train_sampler, batch_size=batch_size)
    return lang, train_dataloader



def train_epoch(dataloader, classifier, optimizer, criterion):
    total_loss = 0
    for data in tqdm(dataloader):
        input_tensor, target_tensor = data 
        optimizer.zero_grad()
        outputs = classifier(input_tensor)

        loss = criterion(outputs, target_tensor)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)
def train_classifier(train_dataloader, classifier, criterion, optimizer, n_epochs):

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, classifier, optimizer, criterion)
        print('Average loss for epoch %d: %.4f' % (epoch, loss))
def test_classifier(test_data, classifier, criterion):
    total_loss = 0
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for datum in tqdm(test_data):
            input_tensor, target_tensor = datum
            classifier_outputs = classifier(input_tensor)
            loss = criterion(classifier_outputs, target_tensor)
            total_loss += loss.item()
            classifier_outputs = classifier_outputs.cpu().numpy()
            outputs = np.argmax(classifier_outputs, axis=1)
            targets = np.argmax(target_tensor.cpu().numpy(), axis=1)
            all_outputs.extend(outputs)
            all_targets.extend(targets)
    return f1_score(all_targets, all_outputs, average='macro')
len(train_data)
embedding_size = 256
batch_size = 32

lang, train_dataloader = get_dataloader(batch_size, train_data)


criterion = nn.CrossEntropyLoss()

classifier = LogisticRegression(lang.n_words, embedding_size).to(Config.device)
optimizer = optim.Adam(classifier.parameters(), lr=Config.learning_rate)


train_classifier(train_dataloader, classifier, criterion, optimizer, 6)

_, test_data_loader = get_dataloader(batch_size, test_data)
f1 = test_classifier(test_data_loader, classifier, criterion)
print('The macro F1 score of the logistic regression classifier is: ', f1)

# Gated Logistic Model
class Config:
    max_length = 50
    learning_rate = 0.01
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 32

    @classmethod
    def update_max_length(cls, new_length):
        cls.max_length = new_length
        
class LogisticRegression(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LogisticRegression, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size).to(Config.device)
        self.linear = nn.Linear(hidden_size * Config.max_length, 6).to(Config.device)

    def forward(self, abstract):
        abstract = self.embedding(abstract)
        input = abstract
        input = torch.flatten(input, start_dim=1)
        linear_output = self.linear(input)
        return linear_output
SOS_token = 0
EOS_token = 1

class LanguageStores:
    def __init__(self):
        self.word2index = {'<UNK>': 0}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        # Write your code here
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        # Write your code here
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.n_words += 1

    def print_stats(self):
        print('Vocabulary size of %d' % (self.n_words))


def prepareLanguageDictionaries():
    lang = LanguageStores()

   
    for datum in train_data:
        lang.add_sentence(datum.getabstract())
    return lang


lang = prepareLanguageDictionaries()
lang.print_stats()
num_classes = len(topics)
num_classes

def indexesFromSentence(lang, sentence):
    # Write your code here
    unknown = lang.word2index['<UNK>']
    return [lang.word2index.get(word, unknown) for word in sentence.split(' ')[:Config.max_length]]

def get_dataloader(batch_size, data, max_length):
    lang = prepareLanguageDictionaries()

    n = len(data)
    input_ids = np.zeros((n, Config.max_length), dtype=np.int32)
    target_ids = np.zeros((n, num_classes), dtype=np.int32)

    for idx, datum in enumerate(data):
        inp_indices = indexesFromSentence(lang, datum.getabstract())
        input_ids[idx, :len(inp_indices)] = inp_indices
        target_ids[idx, [topics.index(topic) for topic in datum.get_unique_topics()]] = 1

    train_tensor_data = TensorDataset(torch.LongTensor(input_ids).to(Config.device),
                               torch.Tensor(target_ids).to(Config.device))

    train_sampler = RandomSampler(train_tensor_data)
    train_dataloader = DataLoader(train_tensor_data, sampler=train_sampler, batch_size=batch_size)
    return lang, train_dataloader



def train_epoch(dataloader, classifier, optimizer, criterion):
    total_loss = 0
    for data in tqdm(dataloader):
        input_tensor, target_tensor = data 
        optimizer.zero_grad()
        outputs = classifier(input_tensor)

        loss = criterion(outputs, target_tensor)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def train_classifier(train_dataloader, classifier, criterion, optimizer, n_epochs):

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, classifier, optimizer, criterion)
        print('Average loss for epoch %d: %.4f' % (epoch, loss))

def test_classifier(test_data, classifier, criterion):
    total_loss = 0
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for datum in tqdm(test_data):
            input_tensor, target_tensor = datum
            classifier_outputs = classifier(input_tensor)
            loss = criterion(classifier_outputs, target_tensor)
            total_loss += loss.item()
            classifier_outputs = classifier_outputs.cpu().numpy()
            outputs = np.argmax(classifier_outputs, axis=1)
            targets = np.argmax(target_tensor.cpu().numpy(), axis=1)
            all_outputs.extend(outputs)
            all_targets.extend(targets)
    return f1_score(all_targets, all_outputs, average='macro')
len(train_data)
embedding_size = 128
batch_size = 32
def train_and_test_classifier(max_length):
    lang, train_dataloader = get_dataloader(batch_size, train_data, max_length)
    classifier = LogisticRegression(lang.n_words, embedding_size).to(Config.device)
    optimizer = optim.Adam(classifier.parameters(), lr=Config.learning_rate)
    train_classifier(train_dataloader, classifier, criterion, optimizer, 6)
    _, test_data_loader = get_dataloader(batch_size, test_data, max_length)  # Pass max_length argument here
    f1 = test_classifier(test_data_loader, classifier, criterion)
    print(f'The macro F1 score of the logistic regression classifier at abstract length {Config.max_length} is: ', f1)


# Test the model at different abstract lengths
for max_length in range(50, 201, 10):
    Config.update_max_length(max_length)
    print("Abstract length:", max_length)
    train_and_test_classifier(max_length)
