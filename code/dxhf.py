import os
import re
import ast
import httpx
import json
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from openai import OpenAI
from nltk.tokenize import sent_tokenize

class DxHF:
    def __init__(self):
        os.environ['AALTO_OPENAI_API_KEY'] = '#YOUR_OPENAI_API_KEY#'
        assert (
            "AALTO_OPENAI_API_KEY" in os.environ and os.environ.get("AALTO_OPENAI_API_KEY") != ""
        ), "you must set the `AALTO_OPENAI_API_KEY` environment variable."
        self.client = OpenAI(
            base_url="https://aalto-openai-apigw.azure-api.net",
            api_key=False, # API key not used, and rather set below
            default_headers = {
                "Ocp-Apim-Subscription-Key": os.environ.get("AALTO_OPENAI_API_KEY"),
            },
            http_client=httpx.Client(
                event_hooks={ "request": [self.update_base_url] }
            ),
        )
        self.sentenceTransformer = SentenceTransformer("stsb-roberta-base-v2")
        # self.crossEncoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.crossEncoder = CrossEncoder('cross-encoder/stsb-roberta-base')

    def update_base_url(self, request: httpx.Request) -> None:
        if request.url.path == "/chat/completions":
            request.url = request.url.copy_with(path="/v1/openai/deployments/gpt-4o-2024-08-06/chat/completions")
            # request.url = request.url.copy_with(path="/v1/chat") # chat/gpt4-8k /chat
            # request.url = request.url.copy_with(path="/v1/openai/gpt4-turbo/chat/completions")
            # request.url = request.url.copy_with(path="/v1/openai/gpt4o/chat/completions")

    def decompose(self, sentence):
        self.message = [
            {
                "role": "system", 
                "content": "You work as a text message checker. You duty is to extract all the claims from a given sentence. A sentence may contain multiple claims. Each claim is a sentence composed of the words from the original sentence. It should not change the meaning in the sentence. It should only copy text, not add new words. Each claim should try to be of the form <subject> <predicate> <object>, and should have the first occurrence of any pronouns replaced by their antecedents."
            },
            {
                "role": "user", 
                "content": "Extract all the claims from a given sentence by copying the words from the original text. A sentence may contain multiple claims. Each claim is a sentence composed of the words from the original sentence. It should not change the meaning in the sentence. It should only copy text, not add new words. Each claim should try to be of the form <subject> <predicate> <object>, and should have the first occurrence of any pronouns replaced by their antecedents.\
                \n\
                Sentence: You can then add water and mix everything until you have a firm dough.\
                \n\
                Claim: You can then add water\
                \n\
                Claim: You can mix everything until you have a firm dough\
                \n\
                Sentence: That’s why the driver needs to be paying attention, and must still be able to see clearly, and still must have his or her own ideas about what to do.\
                \n\
                Claim: driver needs to be paying attention\
                \n\
                Claim: driver must still be able to see clearly\
                \n\
                Claim: driver must have his or her own ideas about what to do\
                \n\
                Sentence: %s"%(sentence)
            }
        ]
        # print("====================================")
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini", # mini
            messages=self.message
        )
        message_content = completion.choices[0].message.content
        claims = message_content.split("\n")
        return claims
    
    def keyword(self, context, claim_1, claim_2):
        self.message = [
            {
                "role": "system", 
                "content": "You work is to summarize two similar sentences with one word as keywork. It should be a word can summarize the shared meaning of the two sentences."
            },
            {
                "role": "user", 
                "content": "Given the conversation as the context: %s\
                \n\
                Here are two claims. \
                ``%s'' \
                and\
                ``%s'' \
                \n\
                Please summarize the two claims. No other words"%(context, claim_1, claim_2)
            }
        ]
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini", # mini
            messages=self.message
        )
        message_content = completion.choices[0].message.content
        return message_content

    def relevance(self, query_sentence, sentences_group):
        prediction_list = []
        for sentence in sentences_group:
            prediction_list.append([query_sentence, sentence])
        logits = self.crossEncoder.predict(prediction_list)
        for i in range(len(sentences_group)):
            print("{ \"claim\": \"%s\", \"relevance\": %s },"%(sentences_group[i], logits[i]))

        return logits
    
    def linking(self, sentences_group1, sentences_group2, threshold = 0.5):
        embeddings_1 = self.sentenceTransformer.encode(sentences_group1)
        embeddings_2 = self.sentenceTransformer.encode(sentences_group2)
        similarity_matrix = cosine_similarity(embeddings_1, embeddings_2)
        links = np.argwhere(similarity_matrix > threshold)
        connections = []
        for idx_1, idx_2 in links:
            print(f"{idx_1} <-> {idx_2} : {similarity_matrix[idx_1, idx_2]}")
            connections.append({ "accept_claim_index": int(idx_1), "reject_claim_index": int(idx_2), "similarity": float(similarity_matrix[idx_1, idx_2]) }),

        return connections