# Catechism ✝️
A repository that contains the scripts for vectorizing the Catechism in a Supabase DB and a LangGraph agent that queries that database to find relevant answers.

# Overview 👁️
This repository began as an effort to be able to query and find answers about the Catholic church faster than googling and with a definitive source -- the Catechsim itself. 

Answers provided are directly pulled from the Catechism and the LLM agent has been specifically instructed to retain semantic meaning of the words of the Catechism, while explaining the contents in language that is more easily understood by the general public. The goal is to lower the threshold for the average person to come to understand Catholicism.

While there is absolutely room for error, the use of an LLM instructed to explain the beliefs of the Catholic Church while retaining the semantic meaning of the text is hardly different (and potentially better) than a human attempting to do the same thing. In fact, it is possible that this agent performs better, because it can reference the Catechism directly to answer every single query.