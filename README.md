# Word embeddings reveal how fundamental sentiments structure natural language
All of the publicly available information and tools pertaining to "Word embeddings reveal how fundamental sentiments structure natural language", which has been conditionally accepted at American Behavioral Scientist. Previously called "Can we distill fundamental sentiments from natural language use? Evaluating word embeddings as a complement to survey-based ratings of affective meaning" and "Towards a more computational affect control theory:  Automating and democratizing the measurement of the affective meaning of concepts". 

More information on the tools and information in each folder can be found in the README file in each folder.

**DEPRECIATION ISSUE** This code used an older version of scikit-learn and uses features that were depreciated at or after version 0.21. Therefore, an older version of Scikit-learn will be needed to run this code at the moment. We are working on creating updated models and scripts that will work with the newest version of scikit-learn.

## Analysis Scripts
In this folder you can find the scripts we use to analyze the data for the paper, which should produce exact replication of the results offered in the paper.

## ScikitLearn Models
In this folder you'll find saved versions of the models we train to produce the final results. These models are saved in a .pkl format, which can be loaded using the Scikit Learn library in Python. Coming soon: an example script using these models to predict the EPA ratings of words as if they were included in the 2015 US Online Dictionary (cited in the paper) that is easily adaptable to new terms.

## WebApp
In this folder you'll find all of the R Shiny scripts we hosted to collect our original dataset. Additionally, you'll find the settings, concepts, and an excerpt of the results Google Sheets that were tied to our web application when we collected the data.

