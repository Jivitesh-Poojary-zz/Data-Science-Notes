# Data-Science-Notes
It contains notes I have formed after reading blogs and seeing conference talks

Situations where we should avoid Machine Learning:
•	It is important to remember that ML is not a solution for every type of problem. There are certain cases where robust solutions can be developed without using ML techniques. For example, you don’t need ML if you have a simple problem to solve. Here can determine a target value by using simple rules, computations, or predetermined steps that can be programmed without needing any data-driven learning.
•	You don't have enough data: Machine Learning is designed to work with huge amounts of data. Really huge. 100k records is a good start. If the training data set is too small, then the system's decisions will be biased. That does not imply that if you have enough data you should be using Machine Learning. 
•	Data are too noisy: "Noise" in ML is the irrelevant information in a dataset. If there is too much of it, the computer might memorize noise. This becomes critical that steps we cannot take enough steps to maintain the integrity of the data.
•	You don't have much time (and money): ML is time- and resource-intensive. First, data scientists need to prepare a dataset (if they don't do it, see point no. 3). Then, the computer needs some time to learn. Then the IT team performs test and adjusts the algorithm. Then, the computer needs some time to learn, again. IT does some testing, and adjusts the algorithm. The computer goes back to learning... The cycle repeats over and over again. The more time is needed, the more you need to pay IT specialists.
Situations where we can use Machine Learning:
•	Situations where there is no human expert.
Source: Bond graph of a new molecule
Target: Predicted binding strength to AIDS protease molecule
•	Situations where humans can perform the task but can’t describe how they do it.
Source: Picture of a hand-written character
Target: ASCII code of the character
•	Situations where the desired function is changing frequently. 
Source: Description of stock prices and trades for last 10 days. 
Target: Recommended stock transactions
•	Situations where each user needs a customized mapping function.
Source: Incoming email message
Target: Importance score for presenting to the user (or deleting without presenting)
•	Situations where you cannot accurately code the rules (there will always be some edge case you are missing).
•	Situations where you cannot scale the system (the rules become too convoluted or size and variety of data is huge).
 
Data Science Project Work Themes

Work Theme	Risks/Pitfalls
Construct a data map to understand what data sources are related and potential needed to measure the client journey	  Miss a critical data source, delaying analysis and subsequent steps
Define goals with Lab and external stakeholders such as sponsors	  Lab does not commit to goal and you cannot proceed with subsequent work themes
Define measurable KPIs that are aligned with measuring Lab goals
•	Revenue growth goals
•	Cost reduction ("scale") goals
•	Client experience improvement goals	  KPIs are not related to business goals
Construct measurable hypotheses about how the intervention (e.g. the MVP) will impact users	  Expected lift from the intervention not agreed to, so cannot calculate experiment run time
Where appropriate, design a controlled web experiment to test the above hypotheses.	  Poor experiment design results in intervention (e.g. MVP) that can't be measured
In priority order, request data ETL from Analytics Enablement Lab to support measuring the KPIs and testing the hypotheses about the intervention.	  Lab does not have data in needs in time for collecting experiment data
Build a repeatable measurement system that results in a dashboard	  If not repeatable, regular experiment updates are too costly in terms of time and probability of error higher.
Launch the experiment, collect data and monitor data quality.	  Overlook errors in data quality generated while collecting experiment data, so new experiment data must be collected
Analyze the experiment results and provide a recommendation based on statistical analysis on whether the intervention should be scaled or discontinued	  Inappropriate use of statistical tools/techniques (e.g. not applying Bonferroni correction) result in an inappropriate recommendation to the Lab

 
Data Analytics Project Approach:
Figuring out what you need to know is often difficult and the process to get there can be hard. But it's not impossible, there are keys to good measurement that I've seen as I've worked with businesses and leaders across the company to develop measurement that matters:
•	Understand what success looks like: Be willing to pause the initiative until you can define success and be sure that you can reasonably expect to tie your actions to the definition you choose. We often see projects that plan to change huge sweeping things like division level cash flow or Net Promoter Score. While this would be great and may happen, in most cases given the scale and complexity of the outcomes, it will be near impossible to tease out your contribution. Think in terms of interim outcomes that you know or at least think you can connect to the broader goal.
•	Use the best data you have while building what you need: In the perfect world you have direct line of site from action to business outcome, but most of us don't live there. If you don't understand the connection between certain activities and business outcomes it's ok to go with well-reasoned "gotta believes". Things like, increasing repeat web visits from targeted groups or broad engagement seem to be pretty safe bets and get you moving in the right direction. But don't stay there, define what you really want and work to get the data or develop the analytics you need to get there. Being data and insight-driven is like any other skill, you need to practice it to perfect it, don't expect to be perfect day-one or ever.  
•	Focus on actionable metrics that drive decisions: Look to measure things that inform your strategy or daily activities. If the information isn't used to make a decision or you see a huge change in the data but don't know what to do, it's probably the wrong measure.
•	Avoid vanity metrics: The opposite of actionable metrics are vanity metrics. They make you feel good when they go up, but don't drive decisions or actions. Metrics that come "out-of-the-box" with most reporting packages usually fall into this vanity metric category. Things like total page views and site visits are commonly used to determine the health of a website, but what actions do they drive?
•	Invest in measurement like your job depends on it: We've all been there, budget is tight and you need to trade-off client facing capabilities with measurement. Historically, the client facing capability wins, but that is short sighted. Without data how do you know if it's working, get funding for the next capability, or explain to your boss what a great job the team did.
•	Measure outcomes not activities: I sent 8,000 emails last week. That statement is a great measure if my goal was to send emails, but chances are I intended my emails to engender some action. Measuring the activity ignores why I am doing the activity in the first place.

So while knowledge is indeed good, focusing on data and insights that answer key business questions or inform important decisions is truly great. And if you feel like it's hard, that's because it often is. But just like we didn't give up when the "Japanese bombed Pearl Harbor", I encourage you to take on the journey to meaningful data.
 
Machine Learning Family:
 

Supervised Learning	Unsupervised Learning	Semi-supervised machine learning
Classification
-	Used for systems where output value being predicted is a category / class

Examples:
-	Support Vector Machines
-	Decision Tree
-	Logistic Regression
-	Naïve Bayes
-	Nearest Neighbor
-	Neural Network
-	Random Forrest
-	Ensemble Methods
-	Adaboost
-	XGBoost

 	Clustering
-	Learning Algorithm tries to group given set of data points into different groups based on similarities in predetermined features.

Examples:
-	K-means
-	DBSCAN
-	Hierarchical Agglomerative
-	HCS clustering
-	Gaussian Mixture Models

 	Active Learning
-	Learning algorithm is able to interactively query the user (or some other information source) to obtain the desired outputs at new data points.
-	There are situations in which unlabeled data is abundant but manually labeling is expensive.

Examples:
-	All versions of supervised learning workflows.
-	

 
Regression
-	Used for systems where the value being predicted falls somewhere on a continuous real number range

Examples:
-	Linear Regression
-	Generalized Linear Model
-	Generalized Additive Model
-	Bayesian Regression
-	Nonparametric Regression
-	Time series regression

 	Dimensionality Reduction:
-	Learning algorithm tries to reduce the number of variables (columns) required for analysis. 
-	This can be achieved by transforming data to smaller dimension or maintaining relevant dimensions without transformation.

Examples:
-	Latent Discriminant Analysis
-	Principle Component Analysis
-	Singular Value Decomposition
-	Non-negative Matrix Factorization
-	Canonical Correlation Analysis
-	Auto-Encoders
-	Manifold Learning

 	Transductive learning 
-	Transductive inference is reasoning from observed, specific (training) cases to specific (test) cases.

Examples:
-	Transductive Support Vector Machine

	Rule-based machine learning
-	Association rule learning is a rule-based machine learning method for discovering interesting relations between variables in large databases. It is intended to identify strong rules discovered in databases using some measures of interestingness

Examples:
-	Apriori
-	Eclat 
-	FP-Growth	Inductive learning
-	Inductive reasoning is a method of reasoning in which the premises are viewed as supplying some evidence for the truth of the conclusion.

-	Most of the classification based algorithms can be converted to regression based algorithms based on the data and the use-case.
-	Besides these broad categories we also have a fourth category of machine learning called reinforcement learning. Reinforcement learning (RL) is an area of machine learning concerned with how software agents ought to take actions in an environment so as to maximize some notion of cumulative reward. It will use concepts like dynamic programming, policy and value based optimization.

 
Components of a typical Machine Learning Algorithm:
Representation	Evaluation	Optimization
Instances
-	K-nearest neighbor
-	Support Vector machines
Hyperplanes
-	Naïve Bayes
-	Logistic regression
Decision trees
Set of rules
-	Propositional rules
-	Logic programs
Neural Networks
Graphical models
-	Bayesian networks
-	Conditional random fields	Accuracy
Error rate
Precision and Recall
Square error likelihood
Posterior probability
Information gain
K-L divergence
Cost / Utility Margin	Combinational optimization
-	Greedy search
-	Beam search
-	Branch-and-bound
Continuous optimization
-	Unconstrained
    Gradient descent
    Sub-gradient descent
    Coordinate descent
    Mirror descent
    Conjugate gradient
    Quasi-Newton methods
-	Constrained
    Linear programming
    Quadratic programming

-	This is a non-exhaustive list of combinations items that can be considered. 
 
Steps in Machine Learning:
•	Data Exploration and Preparation
-	Exploration - Every data scientist devotes serious time and effort to exploring data to sanity-check its most basic properties, and to expose unexpected features. Such detective work adds crucial insights to every data-driven endeavor.
-	Preparation - Many datasets contain anomalies and artifacts. Any data-driven project requires mindfully identifying and addressing such issues. Responses range from reformatting and recoding the values themselves, to more ambitious pre-processing, such as grouping, smoothing, and subsetting.

 

•	Data Representation and Transformation
-	Data scientists very often find that a central step in their work is to implement an appropriate transformation restructuring the originally given data into a new and more revealing form. Data Scientists develop skills in two specific areas:
-	Modern Databases. The scope of today's data representation includes everything from homely text files and spreadsheets to SQL and noSQL databases, distributed databases, and live data streams. Data scientists need to know the structures, transformations, and algorithms involved in using all these different representations.
-	Mathematical Representations. These are interesting and useful mathematical structures for representing data of special types, including acoustic, image, sensor, and network data. For example, to get features with acoustic data, one often transforms to the cepstrum or the Fourier transform; for image and sensor data the wavelet transform or some other multi scale transform (e.g. pyramids in deep learning). Data scientists develop facility with such tools and mature judgement about deploying them.

•	Computing with Data
-	Every data scientist should know and use several languages for data analysis and data processing. These can include popular languages like R and Python, but also specific languages for transforming and manipulating text, and for managing complex computational pipelines. It is not surprising to be involved in ambitious projects using a half dozen languages in concert.
-	Beyond basic knowledge of languages, data scientists need to keep current on new idioms for efficiently using those languages and need to understand the deeper issues associated with computational efficiency.
-	Cluster and cloud computing and the ability to run massive numbers of jobs on such clusters has become an overwhelmingly powerful ingredient of the modern computational landscape. To exploit this opportunity, data scientists develop workflows which organize work to be split up across many jobs to be run sequentially or else across many machines.

•	Data Modeling
-	Generative modeling, in which one proposes a stochastic model that could have generated the data, and derives methods to infer properties of the underlying generative mechanism. This roughly speaking coincides with traditional Academic statistics and its offshoots
-	Predictive modeling, in which one constructs methods which predict well over some given data universe (i.e. some very specific concrete dataset). This roughly coincides with modern Machine Learning, and its industrial offshoots.

•	Data Leakage
-	You can cause data leakage if you include data from outside the training data set that allows a model or machine-learning algorithm to make unrealistically good predictions. 
-	Leakage is a common reason why data scientists get nervous when they get predictive results that seem too good to be true. These dependencies can be hard to detect. 
-	To avoid leakage often requires iterating between building an analysis data set, creating a model, and evaluating the accuracy of the results.

•	Training and testing the model
-	How a model "learns" from the data is relatively model dependent- some go through one entry at a time, check their answer, and learn from each row individually. Others consider all of the training data in aggregate to find predictive features. Evaluating models is much more standard. 
-	Before training the model, data is divided up into 3 segments- training, validation, and test. The training data is what models are built with, initially. The validation data set is then used to evaluate and tweak the model- for instance, if you wanted to look at the effect of the number of trees in a random decision forest (if those words are gibberish, just imagine we are adjusting the model training process, with the same training data), you would retrain it using the training data, and then compare it's performance on the validation data. Finally, once you are happy with the model, the test data is used to give a final estimate on how well the model will perform on new data (which is what we want). 

 
-	It is important to keep the three categories of data completely separate. If you train using the same data that you measure the model on, the model will have "cheated" because it would have seen the examples before. Imagine a student studying for a test, but some of his flashcards are directly from the test. When the test time comes, he can just write down memorized answers instead of actually knowing the material enough to generalize to new questions.

•	Data Visualization and Presentation
-	Data visualization at one extreme overlaps with the very simple plots of EDA - histograms, scatterplots, time series plots - but in modern practice it can be taken to much more elaborate extremes. Data scientists often spend a great deal of time decorating simple plots with additional color or symbols to bring in an important new factor, and they often crystallize their understanding of a dataset by developing a new plot which codifies it. Data scientists also create dashboards for monitoring data processing pipelines that access streaming or widely distributed data. Finally they develop visualizations to present conclusions from a modeling exercise.

•	Data Science Workflow:
-	Ultimately, data science doesn’t fit neatly into a pure software development workflow and will, over time, need to create its own best practices and not just borrow them from other fields.
-	“How will things from traditional software development need to be modified (for data science)?” asks Drew Conway. “I think one of them is time. Time is so core to how all success is measured inside an agile development process. That’s a little bit harder to do with data science because there’s a lot more unknowns in a world where you’re running an experiment.”
-	Software development processes and workflows have evolved over decades to integrate roles like design, support methodologies like Agile development, and accommodate the fact that software is now created by teams, not by solo coders. Data science will undergo a similar evolution before it reaches maturity. In the meantime, data scientists will continue to build in order to learn. 
What are the benefits are Anomaly detection to business?
•	Typically a Data Scientist works on cleaning data, feature engineering, implementing Machine Learning models, developing evaluation and deployment strategies. A disproportionate amount of time is spent on cleaning data. It is said that in some cases Data Scientist spend around 80% of their time in this process. 
•	If we were to help the team in this process it would help the Data Scientist in being more useful in model development.
•	It helps us in maintaining the virtuous cycle of:
-> Great models that are delivered 
-> Business uses them and comes up additional requirements 
-> Data scientist improve the existing models. 
By removing the bottleneck we increase the pace of this cycle thus helping business.
•	In a recent Kaggle survey it was seen that the main pain point for Data Scientist for a machine learning project was dirty data - https://www.kaggle.com/surveys/2017
 
•	We can look at Anomaly Detection as a process to maintain data quality in the context of validating correct transformations from legacy system as well as finding errors which have crept in the data due to some error in application logic. With these dual benefits we can use add more value to downstream teams (Data Science and Business).
•	To be considered “intelligent”, the data archive of the future should operate effectively with minimal human guidance, anticipate important events, and adapt its behavior in response to changes in data content, user needs, or available resources. To do this, the intelligent archive will need to learn from its own experience and recognize hidden patterns in incoming data streams and data access requests.
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.461.5846&rep=rep1&type=pdf
•	From the perspective of data quality assessment, this notion of intelligence would be manifested primarily in the ability of an archive to recognize data quality problems solely from its experience with past data, rather than having to be told explicitly the rules for recognizing such problems. 
•	For example, the archive could flag suspect data by recognizing departures from past norms. Even better, it could categorize data based on the type or severity of data quality problem. The archive could learn to recognize problems either from explicit examples or simply its own observation of different types of data.
•	Another manifestation of intelligence would be the ability of an archive to respond automatically to data quality problems. For example, significant increases in the amount of data flagged as bad or missing might indicate that the data are exceeding the bounds expected by science algorithms.
 
Data Quality:
•	Dirty data is a serious problem for businesses leading to incorrect decision making, inefficient daily operations, and ultimately wasting both time and money. Dirty data often arises when domain constraints and business rules, meant to preserve data consistency and accuracy, are enforced incompletely or not at all in application code. 
•	Data Quality needs to be responsible for the current state of Enterprise Analytics. Disruptive technologies have the power to capture, collate, and synthesize disparate data formats (physical, transactional, geospatial, sensor-driven, or social), but these rich collections of data will fail to deliver useful insights unless they are appropriately prepared and cleansed as input to advanced AI or ML tools.
•	Data Quality directly impacts the outcome of Machine Learning algorithms, and data testing has proved that good data can actually refine the ML algorithms during the development phase. There is a close connection between Data Quality and ML tools and the long-range monetization prospects of “high-quality data” used in the industry.
•	Statistical techniques such as missing data imputation, outlier detection, data transformations, dimensionality reduction, robust statistics, cross-validation & bootstrapping play a critical role in data quality management.
•	Many firms have faced a time lag in making data-driven decisions – by the time the data is located, tidied, sorted and applied, it is virtually out of date and no longer relevant. Firms can run into significant issues – both regulatory and business-related – if their data quality is not up to scratch.
•	Data quality issues can take many forms. For example, particular properties in a specific object have invalid or missing values, a value coming in an unexpected or corrupted format; duplicate instances; inconsistent references or unit of measures; incomplete cases; broken URLs; corrupted binary data, missing packages of data, gaps in the feeds, miss mapped properties and more.
•	Where big data is combined with machine learning, additional quality issues emerge. Changes made to normalize the data can lead to bias in interpretation by a machine learning algorithm. A relatively low frequency of errors in huge data stores arguably makes the need for data-quality scrutiny less important, but the reality is that quality issues are simply shifted to other areas. Automated corrections and general assumptions can introduce hidden bias across an entire data set.
•	The ultimate test of data quality is whether it produces the required result. This demands rigorous testing, as well as consideration of potential sources for introduced error. Although tools for data cleansing, normalization, and wrangling are growing increasingly popular, the diversity of possible factors means that these processes will not be completely automated anytime soon. As automation spreads, you must ensure that an automated solution is not introducing new problems into the data stream as a result of transformation rules.
•	The benefits of improving data quality go far beyond reduced costs. It is hard to imagine any sort of future in data when so much is so bad. Thus, improving data quality is a gift that keeps giving — it enables you to take out costs permanently and to more easily pursue other data strategies.
 
Types of Anomalies:
•	Systematic Errors:
-	Systematic errors are those that appear regularly in the data under a given set of conditions. An example systematic error is incorrectly classified pixels at land/water boundaries, perhaps due to detector response characteristics or errors in the science algorithm. 21 Such errors can be very hard to detect (in a general sense) using machine learning approaches because identifying a problem may require relatively deep scientific knowledge about what can be induced from the data. At the same time, such errors are often easily detected by the users because if the governing conditions occur frequently, the problem will manifest itself frequently. 
-	Thus, detecting systematic errors does not initially look like a good application for machine learning. There are three mitigating factors. First, some systematic errors can be subtle in the context of normal uses of the data but easily identified from a fresh perspective. For example, data that appears normal when viewed as a spatial image at a given point in time can easily be seen to have severe clipping or discontinuities when viewed as a time series at a given location. We speculate that machine learning algorithms, particularly regression tress and unsupervised classifiers, can provide this fresh perspective. 
-	Second, the conditions under which a systematic error occurs can be so complex that the error appears to users to be random: identifying such complex patterns is the forte of machine learning. Finally, a substantial amount of time can elapse before users examine the data and discover an error, at which point opportunities to re-acquire the data may have passed or the erroneous data may have already been incorporated into numerous other data products or decisions. Thus, in cases where there is a significant cost to latent errors in the data, there may be an argument for automated quality assessment even when the error would certainly be found later by users.

•	Random Errors:
-	Random errors are those that appear irregularly in the data under a given set of conditions. An example is a sudden variation in a data value caused by data corruption. 22 Such errors are relatively easy to detect using statistical approaches because one only need detect sudden changes from normal values. At the same time, these errors can easily hide from users in the large volumes of data in the archive if they occur infrequently. In the financial domain, detecting random errors is significantly complicated by the natural fluctuations present in the data, which can have many of the same attributes as anomalies. 
-	We speculate that machine learning techniques can be applied in several ways to assist with this problem. For example, they could be applied to adaptively determine what values are “normal” at a given time or location, and thus provide a baseline for exposing anomalies.

•	Other Anomalies:
-	Data anomalies can occur even without errors. When collected over a long period of time, distributions can be observed to drift as the underlying behaviors change. For example, a typical financial institution may see the proportion of its population with low credit bureau scores change along with changing economic conditions. This population change could have more widespread impact among attributes defining the behavior of population like income, credit available to them etc. Fundamental data shifts like these can adversely affect the performance of key models and tools deployed by the organization.

•	Genuine Outliers:
-	Anomalous data attributes could be due to the presence of genuine outliers in the data. Such cases, despite being a reality of the data, have to be treated properly before being processed through a typical modeling framework. Although many machine learning techniques are robust to the presence of outliers, their presence can still affect key steps like feature engineering and feature selection.
 
Output of Anomaly Detection:
An important aspect for any anomaly detection technique is the manner in which the anomalies are reported. 
Output	Description
Scores	Scoring techniques assign an anomaly score to each instance in the test data depending on the degree to which that instance is considered an anomaly. Thus the output of such techniques is a ranked list of anomalies. An analyst may choose to either analyze the top few anomalies or use a cutoff threshold to select the anomalies.

Scoring-based anomaly detection techniques allow the analyst to use a domain specific threshold to select the most relevant anomalies.
Labels	Techniques in this category assign a label (normal or anomalous) to each test instance.

Techniques that provide binary labels to the test instances do not directly allow the analysts to put domain specific threshold, though this can be controlled indirectly through parameter choices within each technique.
 
Violation of Logical Constraints vs. Deviation from Norms
Violations of logical constraints covers a broad set of errors, some of which are good candidates for data quality assessment based on machine learning, and some of which are not. Simple constraints (e.g., a set of percentages should not total more than 100%) are probably best checked by simple rules based on the domain of a variable; there is no clear value for machine learning here. 
At the other end of the spectrum are logical constraints that require relatively deep knowledge about business processes that cannot be derived from the data stream itself; rules developed by human experts’ maybe best here. 
In the middle, however, are a large number of moderately complex constraints that could be learned from the data because they represent deviations from the norm. While such rules will often be intuitive to a person with even modest knowledge of the domain, machine learning techniques (particularly unsupervised classifiers) could relieve the data quality analyst of the burden of identifying a comprehensive set of rules, and could automatically adapt to observed changes in the data as well. This is a fundamentally different way of thinking as we are targeting to reduce the time taken to develop a data quality system. It also considers a notion of a Quality Engineer whose job to maintain data quality for downstream utilization.
A related approach would be to store normative examples (rather than derived rules) and use statistical or lazy learning techniques to identify deviations from the norm or similarities to known bad datasets.  

 
Challenges in Anomaly Detection:
At an abstract level, an anomaly is defined as a pattern that does not conform to expected normal behavior. A straightforward anomaly detection approach, therefore, is to define a region representing normal behavior and declare any observation in the data that does not belong to this normal region as an anomaly. But several factors make this apparently simple approach very challenging: 
-	Defining a normal region that encompasses every possible normal behavior is very difficult. In addition, the boundary between normal and anomalous behavior is often not precise. Thus an anomalous observation that lies close to the boundary can actually be normal, and vice versa. 
-	When anomalies are the result of malicious actions, the malicious adversaries often adapt themselves to make the anomalous observations appear normal, thereby making the task of defining normal behavior more difficult. 
-	In many domains normal behavior keeps evolving and a current notion of normal behavior might not be sufficiently representative in the future. 
-	The exact notion of an anomaly is different for different application domains. For example, in the medical domain a small deviation from normal (e.g., fluctuations in body temperature) might be an anomaly, while similar deviation in the stock market domain (e.g., fluctuations in the value of a stock) might be considered as normal. Thus applying a technique developed in one domain to another, is not straightforward. 
-	Availability of labeled data for training/validation of models used by anomaly detection techniques is usually a major issue. 
-	Often the data contains noise that tends to be similar to the actual anomalies and hence is difficult to distinguish and remove. 
-	Anomaly detection techniques typically assume that anomalies in data are rare when compared to normal instances. Though this assumption is generally true, anomalies are not always rare. For example, when dealing with worm detection in computer networks, the anomalous (worm) traffic is actually more frequent than the normal traffic. Unsupervised techniques are not suited for such bulk anomaly detection. Techniques operating in supervised or semi-supervised modes can be applied to detect bulk anomalies
Due to these challenges, the anomaly detection problem, in its most general form, is not easy to solve. In fact, most of the existing anomaly detection techniques solve a specific formulation of the problem. The formulation is induced by various factors such as the nature of the data, availability of labeled data, type of anomalies to be detected, and so on. Often, these factors are determined by the application domain in which the anomalies need to be detected. Researchers have adopted concepts from diverse disciplines such as statistics, machine learning, data mining, information theory, spectral theory, and have applied them to specific problem formulations. 
Data Management Challenges in production Machine Learning:
Training Phase	Production Phase

 
Ref: - https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46178.pdf
Training Input Data: 
Generating data for our model 	Serving Input Data:
This involves making predictions on the new data
Prepare:
Cleaning the data, performing feature engineering, transforming the data	Prepare:
Perform the same process that was used in Training phase
Validate:
Checking if the prepare process was done correctly. 	
Fix: 
Making changes to the data / updating rules used in the prepare step	
Training Data:
Transformed and validated data is used as an input for the machine learning model	Serving Data:
Serving data is the transformed new data that is coming to the environment
Train:
Training Machine Learning model	Serve:
Perform predictions on the serving data
Model:
Store the different models obtained after hyperparameter tuning and testing different approaches.	Model:
Use the trained model to make predictions
Evaluate:
Evaluate the model for the necessary use case. Optimizing the KPI / Accuracy / Error rate / Other indicators. 	
 
Steps to be taken for Data Validation and maintaining Data Quality:
•	Check a feature’s min, max, and most common value
-	Ex: Latitude values must be within the range [-90, 90] or [-π/2, π/2]
-	Ex: A FICO score of 8000
-	Ex: Develop a trim mean which is more robust to outlier by ignoring few extreme values in mean calculation.

•	Check a feature’s distribution if a continuous variable 
-	Ex: The values fall within few standard deviations of the mean. (Z-score method). A downfall of this method is that it requires the assumption of a normal data distribution. Usually, this assumption holds true as the sample size gets larger, though a formal test such as the Andersen–Darling method can be used to test the assumption
-	Ex: Use modified Z-Score if there are fewer values or using median makes more sense
-	Ex: Use Interquartile range to find upper and lower bounds for good values (Box-plot)
 
-	Ex: Perform Dixon-type tests, they are based on the ratio of the ranges. These tests are flexible enough to allow for specific observations to be tested. They also perform well with small sample sizes. Because they are based on ordered statistics, there is no need to assume normality of the data. Depending on the number of suspected outliers, different ratios are used to identify potential outliers.
-	Ex: Cohen’s d and h statistic measure the effect size (in units of the standard deviation) of changes in mean and proportion. These statistics are preferable to standard hypothesis testing methods like the Student’s t-test because, in the case of big data, extremely small changes in the distribution will reach statistical significance. For the purpose of data quality control, we are interested only in changes that are large enough to have practical significance, which can be achieved with the effect size statistics. 
http://proceedings.mlr.press/v71/love18a/love18a.pdf

•	Apply a model which is robust to outliers
-	Ex: Regression analysis is highly sensitive to outliers and influential observations. Outliers in regression can overstate the coefficient of determination (R2), give erroneous values for the slope and intercept, and in many cases lead to false conclusions about the model. Outliers in regression usually are detected with graphical methods such as residual plots including deleted residuals. We can apply Loess model as it is robust to outliers and check for values which have a large error rate.
 
-	Ex: A common statistical test is Cook's distance measure, which provides an overall measure of the impact of an observation on the estimated regression coefficient. Just because the residual plot or Cook's distance measure test identifies an observation as an outlier does not mean one should automatically eliminate the point. One should fit the regression equation with and without the suspect point and compare the coefficients of the model and review the mean-squared error and R2 from the two models.

•	The histograms of continuous or categorical values are as expected
-	Ex: Histograms give us a visual perspective of the distribution. We can check to see if we need to perform any transformation.
 
 

•	Use Anscombe’s Quartet
-	Ex: Anscombe's quartet is a great example where relying on statistical alone may not be the right approach always.
 
•	Time series Anomaly detection:
-	Ex: We can check for change in the magnitude or range of values over time like Upward and downward level changes. 
-	Ex: We can also observe Positive or negative trends changes over time. 
  

-	Ex: Perform STL decomposition, this technique gives you an ability to split your time series signal into three parts: seasonal, trend and residue. Pros of this approach are in its simplicity and how robust it is. It can handle a lot of different situations and all anomalies can still be intuitively interpreted. It’s good mostly for detecting additive outliers. To detect level changes you can analyze some rolling average signal instead of the original one. The cons of this approach are in its rigidity regarding tweaking options. All you can tweak is your confidence interval using the significance level.
 
-	Ex. Another approach is use unsupervised learning to teach CART to predict the next data point in your series and have some confidence interval or prediction error as in the case of the STL decomposition approach. By using Generalized ESD test, or Grubbs’ test to check if your data point lies inside or outside the confidence interval. The strength of this approach is that it’s not bound in any sense to the structure of your signal, and you can introduce many feature parameters to perform the learning and get sophisticated models. The weakness is a growing number of features can start to impact your computational performance fairly quickly. In this case, you should select features consciously.

•	Develop a contingency table for categorical features
-	Ex: it is a cross table for two categorical features. The table can be for absolute values or percentage values, it can be used to compare the observed and expected values.  
  

•	Whether a feature is present in enough examples
-	Ex: Country code must be in at least 70% of the examples

•	Whether a feature has the right number of values (i.e., cardinality)
-	Ex: There cannot be more than one age of a person
-	Ex: If the field is a checklist then validate if there are any maximum or minimum number of values.

•	Whether a feature has the right value for Categorical value:
-	Ex: If the field is a dropdown it will have a single value
-	Ex: A Boolean field should have TRUE or FALSE if it is NOT NULL.

•	Check to see if there are missing values or NULL or 0
-	Ex: Some transformations may change the original value to something which is undesirable

•	In case of text features we can see if there are any special character or incorrect words.
-	Ex: We can see if there are any special characters by maintaining a dictionary
-	Ex: A dictionary of permissible words can also be maintained with the help of domain knowledge about the data 
Machine Learning Algorithms for Anomaly Detection:
Supervised	Unsupervised	Semi-supervised
The anomalous instances are far fewer compared to the normal instances in the training data. Issues that arise due to imbalanced class distributions have been addressed in the data mining and machine learning literature.

Second, obtaining accurate and representative labels, especially for the anomaly class is usually challenging.	The techniques in this category make the implicit assumption that normal instances are far more frequent than anomalies in the test data. If this assumption is not true then such techniques suffer from high false alarm rate.	A limited set of anomaly detection techniques exists that assumes availability of only
the anomaly instances for training

Such techniques are not commonly used, primarily because it is difficult to obtain a training data set that covers every possible anomalous behavior that can occur in the data.

Unsupervised Anomaly Detection will help us in separating the data into two distributions (classes) but it will not tell us anything about what is correct and what is wrong. This is something only a human can understand. It can find patterns which humans may not be aware of. 
However having said that if we have enough manually annotated records we can use active learning / supervised learning systems to develop machine learning classifiers.

Factors to be for selecting Anomaly Detection Algorithm:
Factor	Description
Com	
	
	
 
Some Popular Anomaly Detection Algorithms:
•	One-Class Support Vector Machine: 
-	Support vector machines (SVMs) are supervised learning models that analyze data and recognize patterns, and that can be used for both classification and regression tasks.
-	Typically, the SVM algorithm is given a set of training examples labeled as belonging to one of two classes. An SVM model is based on dividing the training sample points into separate categories by as wide a gap as possible, while penalizing training samples that fall on the wrong side of the gap. The SVM model then makes predictions by assigning points to one side of the gap or the other.
-	Sometimes oversampling is used to replicate the existing samples so that you can create a two-class model, but it is impossible to predict all the new patterns of fraud or system faults from limited examples. Moreover, collection of even limited examples can be expensive.
-	Therefore, in one-class SVM, the support vector model is trained on data that has only one class, which is the “normal” class. It infers the properties of normal cases and from these properties can predict which examples are unlike the normal examples. This is useful for anomaly detection because the scarcity of training examples is what defines anomalies: that is, typically there are very few examples of the network intrusion, fraud, or other anomalous behavior.

•	PCA-based: 
-	This algorithm uses PCA to approximate the subspace containing the normal class. The subspace is spanned by orthonormal eigenvectors associated with the top eigenvalues of the data covariance matrix. 
-	For each new input, the anomaly detector first computes its projection on the eigenvectors, then it computes the normalized reconstruction error. This norm error is the anomaly score. The higher the error, the more anomalous the instance is.

•	Isolation Forest :
-	Good for high-dimensional data
-	‘isolates’ observations by randomly selecting a feature and randomly selecting a split value between the max and min values of the selected feature. Random partitioning produces shorter tree paths for anomalies.

•	Local Outlier Factor (LOF) :
-	Measures the local deviation of density of a given row with respect to its neighbors. Based on k-Nearest Neighbor.
-	It is local in that the anomaly score depends on how isolated the object is with respect to its surrounding neighborhood.

•	Double Median Absolute Deviance (MAD):
-	Symmetric and Asymmetric distributions
-	The median absolute deviation from the median of all points less than or equal to the median and the median absolute deviation from the median of all points greater than or equal to the median.
-	Not practical for boolean or near-constant data
-	Scales up to large datasets

•	Anomaly Detection Blender:
-	Base models are the Isolation Forest and Double MAD
-	A tunable mean , minimum or maximum blender of the 2 base models

•	Anomaly Detection with Supervised Learning (XGB):
-	Base models are the Isolation Forest and Double MAD
-	The average score of the base models are taken and a percentage are labeled as Anomaly and the rest as Normal. The percentage labeled as Anomaly is defined by the expected outlier fraction parameter. 
Different Categories of Anomaly Detection:
Types	Advantages	Disadvantages
Classification Based

 	Classification-based techniques, especially the multi-class techniques, can make use of powerful algorithms that can distinguish between instances belonging to different classes. 

The testing phase of classification-based techniques is fast, since each test instance needs to be compared against the precomputed model.	Multi-class classification-based techniques rely on the availability of accurate labels for various normal classes, which is often not possible.

Classification-based techniques assign a label to each test instance, which can also become a disadvantage when a meaningful anomaly score is desired for the test instances. Some classification techniques that obtain a probabilistic prediction score from the output of a classifier, can be used to address this issue
Nearest Neighbor-Based

 	A key advantage of nearest neighbor -based techniques is that they are unsupervised in nature and do not make any assumptions regarding the generative distribution for the data. Instead, they are purely data driven. 

Semi-supervised techniques perform better than unsupervised techniques in terms of missed anomalies, since the likelihood that an anomaly will form a close neighborhood in the training data set is very low. 

Adapting nearest neighbor-based techniques to a different data type is straightforward, and primarily requires defining an appropriate distance measure for the given data.	For unsupervised techniques, if the data has normal instances that do not have enough close neighbors, or if the data has anomalies that have enough close neighbors, the technique fails to label them correctly; resulting in missed anomalies.

For semi-supervised techniques, if the normal instances in the test data do not have enough similar normal instances in the training data, the false positive rate for such techniques is high.

The computational complexity of the testing phase is also a significant challenge since it involves computing the distance of each test instance—with all instances belonging to either the test data itself, or to the training data—to compute the nearest neighbors.

Performance of a nearest neighbor-based technique greatly relies on a distance measure, defined between a pair of data instances, which can effectively distinguish between normal and anomalous instances. Defining distance measures between instances can be challenging when the data is complex, for example, graphs, sequences, and so on.
Clustering-Based

 
Clustering-based techniques can operate in an unsupervised mode. 

Such techniques can often be adapted to other complex data types by simply plugging in a clustering algorithm that can handle the particular data type. 

The testing phase for clustering-based techniques is fast since the number of clusters against which every test instance needs to be compared is a small constant.	Performance of clustering-based techniques is highly dependent on the effectiveness of clustering algorithms in capturing the cluster structure of normal instances. 

Many techniques detect anomalies as a byproduct of clustering, and hence are not optimized for anomaly detection. 

Several clustering algorithms force every instance to be assigned to some cluster. This might result in anomalies getting assigned to a large cluster, thereby being considered as normal instances by techniques that operate under the assumption that anomalies do not belong to any cluster. 

Several clustering-based techniques are effective only when the anomalies do not form significant clusters among themselves. 

The computational complexity for clustering the data is often a bottleneck, especially if O(N2d) clustering algorithms are used.
Statistical

 	If the assumptions regarding the underlying data distribution hold true, statistical techniques provide a statistically justifiable solution for anomaly detection.

The anomaly score provided by a statistical technique is associated with a confidence interval, which can be used as additional information while making a decision regarding any test instance.

If the distribution estimation step is robust to anomalies in data, statistical techniques can operate in an unsupervised setting without any need for labeled training data.	The key disadvantage of statistical techniques is that they rely on the assumption that the data is generated from a particular distribution. This assumption often does not hold true, especially for high dimensional real data sets.

Even when the statistical assumption can be reasonably justified, there are several hypothesis test statistics that can be applied to detect anomalies; choosing the best statistic is often not a straightforward. In particular, constructing hypothesis tests for complex distributions that are required to fit high dimensional data sets is nontrivial.

Histogram-based techniques are relatively simple to implement, but a key shortcoming of such techniques for multivariate data is that they are not able to capture the interactions between different attributes. An anomaly might have attribute values that are individually very frequent, but whose combination is very rare, however an attribute-wise histogram-based technique would not be able to detect such anomalies.
Information Theoretic	They can operate in an unsupervised setting.

They do not make any assumptions about the underlying statistical distribution for the data.	The performance of such techniques is highly dependent on the choice of the information theoretic measure. Often, such measures can detect the presence of anomalies only when there is a significantly large number of anomalies present in the data.

Information theoretic techniques applied to sequences, and spatial data sets rely on the size of the substructure, which is often nontrivial to obtain.

It is difficult to associate an anomaly score with a test instance using an information theoretic technique.
Spectral	Spectral techniques automatically perform dimensionality reduction and hence are suitable for handling high dimensional data sets. Moreover, they can also be used as a preprocessing step followed by application of any existing anomaly detection technique in the transformed space.

Spectral techniques can be used in an unsupervised setting.	Spectral techniques are useful only if the normal and anomalous instances are separable in the lower dimensional embedding of the data. 

Spectral techniques typically have high computational complexity
Contextual	The key advantage of contextual anomaly detection techniques is that they allow a natural definition of an anomaly in many real-life applications where data instances tend to be similar within a context. Such techniques are able to detect anomalies that might not be detected by point anomaly detection techniques that take a global view of the data.	They are applicable only when a context can be defined
 
Feature Engineering:
-	Feature engineering is the process of transforming raw data into features that better represent the underlying problem to the predictive models, resulting in improved model accuracy on unseen data.
-	Not every single feature will be relevant to the problem, and some of the features are highly correlated, nonetheless, having too many features is a better problem than having too few.
-	Finding all relevant features is NP-hard – Possible to construct a distribution that demands an exhaustive search through all the subsets of features.

Type	Data type	Description
Normalization	Continuous	
Binarization	Continuous	Threshold numerical values to get boolean values 
Bucketization	Continuous	Convert continuous features to discrete features
Winsorizing		
One-hot Encoding	Categorical	
Feature crosses		
Transfer learning		
Extrapolation		
Scaling of values		
Move the center		

