# Observation-Undersampling for Fraud Prediction
## A Python implementation of the Observation Under-sampling model proposed by Perols et al. (2017)

### Background
The phenomenon of fraud is rare but expensive (to the industry and its customers). However it is a fascinating aspect of human behavior, and it's a worthy subject of study. Why? The problems that englobe its understanding are far from trivial.

Fraud, to be possible, needs three elements:

A supply of motivated fraudsters.
Availability of suitable targets.
The absence of vigilantes.
Motivated offenders and visible targets will always be available. And we (the scientists, engineers, thinkers, etc.), could be seen as one of the layers of 'vigilantes,' adept enough to prevent the attacks of these sophisticated thieves.

One of the main challenges is that fraudsters adapt to our defenses, and we (the vigilantes) have to engineer new barriers, playing always an arms race with the fraudsters. This phenomenon is known as the Red Queen Effect by evolutionary biologists and could be seen by some people as a headache that burns resources. I myself, think of it as a drive to innovation.

One of the solutions to overcome this problem is to detect fraud instances in real-time, or even forecast them. This can be done by constructing predictive models with math as our building blocks, computer operations as our cement, and science and literature as our blueprints. Because of the Red Queen Effect, these models will need a constant re-think.

Hence, here is my attempt to demonstrate how this can be done.

José P. Barrantes,
Data Scientist

17/Dec/2019


### About this project and methodology
The strength of this methodology is in the data preprocessing phase, it is designed to address the class unbalance problem. This pre-processing architecture consists in generating several training subsets, each one will contain a copy of all the instances of the minority class (the frauds in this case) available in the training set, and several random instances of the majority class.

Each of these subsets is used to train a classifier, hoping that it will learn better to recognise the minority class. Later, groups of classifiers are combined in a global (ensembled) classifier who can discern better between classes. An analogy can be used to explain this: each individual model is a voter, and it chooses to which class belongs the instance we are testing, in the end, we count the votes to take our decision.

Here we will use a 'soft-voting' system, as part of our process of decision making. This means that we run our models with an example transaction, we get the probability (of the example belonging to the fraud class) of each model, and we average them. If the value excedes a fixed threshold, we classify the instance as a positive case. This threshold is determined by an algorithm that minimizes our performance metric, the Expected Cost of Misclassification.

So, we should have an ensemble algorithm who is able to recognize fraud cases and differenciate them from true transactions. You can find more details about the model design and methodology in the file README.pdf.
