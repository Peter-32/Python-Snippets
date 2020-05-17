import pandas as pd
import zenpy
from zenpy import Zenpy
from zenpy.lib.api_objects import Comment
import os
import re
import nltk
import torch
import networkx as nx
import argparse
import wikipedia
import numpy as np
from models import InferSent
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Get processed_tickets
with open('/Users/petermyers/work/success_zendesk/tickets_processed.txt', 'r') as file:
    processed_tickets = [int(x) for x in file.read().split("\n") if len(x) > 0]

# Infersent
infersent = InferSent({'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                       'pool_type': 'max', 'dpout_model': 0.0, 'version': 1})
infersent.load_state_dict(
    torch.load('/Users/petermyers/Desktop/Other/data/InferSent/encoder/infersent1.pkl'))
infersent.set_w2v_path('/Users/petermyers/Desktop/Other/data/GloVe/glove.840B.300d.txt')

# Helpers
footer_text = ["thanks", "best regards", "thank", "thank you", "sincerely", "thanks peter",
               "thank you peter", "--", "---", "----", "-----", "------"]
flatten = lambda l: [item for sublist in l for item in sublist]
invalid_sentences = ['Filename not specified.', "Error!", 'Best,', 'Thank you,', '--',
                     'Peter Myers',
                     'Technical Service Engineer',
                     'peter@impact.com | ',
                     'https://impact.com',
                     '(https://www.linkedin.com/company/impact-partech/)   Error!',
                     '(https://www.facebook.com/ImpactParTech/)   Error!',
                     '(https://twitter.com/impactpartech)  Error!',
                     '(https://www.youtube.com/c/impactpartech)',
                     '(https://go.impact.com/Event-Webinar-PC-20200219-Partnerships-Drive-2X-Faster-Growth-3X-ROI_RegistrationPage.html)', ]

# Create a Zenpy instance
credentials = {'subdomain': 'impactradius', 'email': 'peter@impact.com',
               'password': 'PeterMyers0!'}
zenpy_client = Zenpy(**credentials)


def process_ticket(ticket):
    my_ticket = ticket.to_dict()
    id = ticket.to_dict()['id']
    if id not in processed_tickets:

        description = my_ticket['description']

        # Sentences and embeddings
        sents = [x for x in
                 flatten([x.split("\n\n") for x in nltk.sent_tokenize(my_ticket['description'])[:-1]])
                 if x.strip() != '' and x not in invalid_sentences]
        last_sentence = min([i for (x, i) in zip(sents, range(len(sents))) if
                             x.strip().lower().replace(",", "") in footer_text] + [len(sents)])
        sents = sents[0:last_sentence]
        infersent.build_vocab(sents, tokenize=True)
        embeddings = infersent.encode(sents, tokenize=True, verbose=False)

        ticket.comment = Comment(body=description, public=False)
        #### Write that this ticket is processed
