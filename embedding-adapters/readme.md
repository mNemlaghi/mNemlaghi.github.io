# Decoding Annual Reports: Embedding Adapters for Precise 10-K Question Answering

## The challenge


### A human history of wasting time.

Today I'll talk about boredom. For instance, who reads corporate annual reports without a sense of wasted time? Who does read annual reports with _genuine pleasure_? Don't get me wrong, I'm not talking about litterary pleasure, but more generally, intellectual pleasure or a sentiment of work accomplishment. In this planet, who did experience a strong sense of usefulness reading an annual report? Reading annual report is the corporate equivalent of slowly watching paint dry – if the paint was made of accounting jargon and regulatory compliance. The thing is, they're of paramount importance for a plethora of actors. They provide insights for multiple, crucial decisions.


### Is AI saving our time ?

AI, especially with the LLM behemoths, seems relevant to end this endless boredom, by a deep understanding of the underlying language. But as we all know now, GenAI comes with hallucination. So, RAG (retrieval augmented generation) comes to the rescue (an intro [here with Amazon Bedrock Knowledge Bases](https://mnemlaghi.github.io/cloud-embeddings/part-four-store)) and ground LLMs with factuality, acting as a "cheatsheet", feeding updated data to the LLM, so that LLM doesn't state that, for instance, Georges V is the *current* king of United Kingdom.

RAG has therefore become a proverbial framework in today's generative AI era. But how can we ensure that:
1. the "cheatsheet" contains relevant, accurate data (maximize *relevancy*) ?
2. the "cheatsheet" minimizes distracting content  (minimize *distractors*) ?

We are going to focus on the "R" part of RAG -a.k.a retriever. Retrievers are in charge of forming the cheatsheet. From a cost perspective, the challenge of reducing distractors from retrievers will improve your RAG system, because cheatsheet will contain noise reduced data. It is a cost opportunity, as a better retriever system will likely feed a lesser amount of tokens to the LLM.


## A solution: _adapt_ embeddings !

This is where embeddings come into play. If you're not familiar, please stop reading this series and go [here](https://mnemlaghi.github.io/cloud-embeddings/) first. Now, off-the-shelf embeddings, either open-source or proprietary, have been commoditized. This time, we are going to adjust off-the-shelf embeddings with a technique called "adapters".  Adapters are learned processes that don't touch the embedding *per se*, but rather transform an embedding of dimension _d_ into another embedding of dimension _d_. They can be expressed as matrices. For instance, if we have a proprietary embedding _X_, then we just have to learn a matrix of dimension (d,d) that transform _X_ into another embedding Y, such as _Y=AX_.

And yes, indeed, it can be as simple as a simple matrix. And, yes, indeed, we don't touch to the initial embedding.

At the end of the journey, we'll be able to maximize relevant documents and minimize distractions to a  RAG system specialized for 10-K! Picture the system we're going to build as a smart cat handling dozens of documents for us.


![Your cat robot](catrobot.png)


Here is how we are going to proceed.


## The code

Just with a important but slight detour, I've also written a companion repository, called [rag-adapters](https://github.com/mNemlaghi/rag-adapters) where all the parts will be tied together inside a notebook.


## The steps

Let's see the menu for today, lots to uncover between practice and concepts. 

1. _Hors d'oeuvre_ 🥗 is [Synthetic data generation](part-one-data-generation): we're going to go through a 10-K earnings document, in order to create accurate question/answers pairs. Rather a practical guide on how to generate data from an LLM and how to enforce JSON output
2. Part 1 🥘 is our *plat de résistance*: we're going to [explain & design the adapters](part-two-adapters-training). Conceptual but contains some practical tips.
3. _Fromage_  🧀: in part 3(coming soon), we'll evaluate and interpret the performances during part 2. 
4. And finally, the  _dessert_ 🍰:  we're going to perform a RAG system benchmark over an annual report to highlight, in practice, the added value of adapters (coming soon as well)


## Conclusions

We're going to learn a lot in this journey! Who knew that understanding annual reports can be fun?


__Notes__:The information provided in this series is for educational purposes only and should not be considered as financial or investment advice. Please read our full [Investment Disclaimer](disclaimer) for more details.