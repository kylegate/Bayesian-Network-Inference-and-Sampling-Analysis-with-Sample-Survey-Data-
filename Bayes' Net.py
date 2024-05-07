import pandas as pd
from BayesNet import *

"""

Name:(Kyle Webb)
Date:(3/18/24)
Assignment:(Assignment #8)
Due Date:(3/17/24)
About this project:(Statistic information based on Star Wars data set)
Assumptions:(N/A)
All work below was performed by (Kyle Webb)

"""

''' Part 1 : '''

T, F = True, False

file = r"C:\\Users\\kylew\\OneDrive\\Desktop\\StarWars.xlsx"
dataFrame = pd.read_excel(file)

# Preprocess column names
dataFrame.columns = dataFrame.columns.str.replace(' ', '_').str.replace('?', '_').str.replace('-', '_')

# lines 15-48 find percentages for each attribute using pandas from StarWars.xlsx
# Gender: Male
dfMale = dataFrame.query('Gender==\'Male\'')
PercentMale = dfMale.shape[0] / dataFrame.shape[0]
# Gender: != Male
dfNotMale = dataFrame.query('Gender!=\'Male\'')
PercentNotMale = dfNotMale.shape[0] / dataFrame.shape[0]
# Ages: 18-29
dfAge = dataFrame.query('Age==\'18-29\'')
PercentAge = dfAge.shape[0] / dataFrame.shape[0]
# Ages: != 18-29
dfNotAge = dataFrame.query('Age!=\'18-29\'')
PercentNotAge = dfNotAge.shape[0] / dataFrame.shape[0]

# Seen Star Wars : Yes  and Gender : Male , Age : 18-29 , and Education : Bachelor Degree
dfSeenStarWars = dataFrame.query('Have_you_seen_any_of_the_6_films_in_the_Star_Wars_franchise_==\'Yes\'')
PercentSeenStarWars = dfSeenStarWars.shape[0] / dataFrame.shape[0]
# Seen Star Wars : != Yes   and Gender : Male , Age : 18-29 , and Education : Bachelor Degree
dfNotSeenStarWars = dataFrame.query('Have_you_seen_any_of_the_6_films_in_the_Star_Wars_franchise_!=\'Yes\'')
PercentNotSeenStarWars = dfNotSeenStarWars.shape[0] / dataFrame.shape[0]
# Star Wars Fan : Yes and Seen Star Wars Yes
dfStarWarsFan = dataFrame.query('Do_you_consider_yourself_to_be_a_fan_of_the_Star_Wars_film_franchise_==\'Yes\'')
PercentStarWarsFan = dfStarWarsFan.shape[0] / dataFrame.shape[0]
# Star Wars Fan : != Yes    and Seen Star Wars Yes
dfNotStarWarsFan = dataFrame.query('Do_you_consider_yourself_to_be_a_fan_of_the_Star_Wars_film_franchise_!=\'Yes\'')
PercentNotStarWarsFan = dfNotStarWarsFan.shape[0] / dataFrame.shape[0]

# Implement the BayesNet you created on paper in step 1 using the BayesNet Class from python
# script provided for you in the python example for this module in a python script (20 points)
StarWarsNetwork = BayesNet([
    ('Male', '', PercentMale),
    ('Age_Range', '', PercentAge),
    ('Seen_Star_Wars', 'Male Age_Range',
     {(T, T): (PercentMale * PercentAge),
      (T, F): (PercentMale * PercentNotAge),
      (F, T): (PercentNotMale * PercentAge),
      (F, F): (PercentNotMale * PercentNotAge)
      }),
    ('Star_Wars_Fan', 'Seen_Star_Wars', {T: PercentStarWarsFan, F: PercentNotStarWarsFan})
])

# using the python script provided for you in the python example for this module compute P(of the same variable)
# using enumeration in a python script (10 points)
print("\nenumeration")
print("P(Age_Range|SeenStarWars=T)", enumeration_ask('Age_Range', dict(Seen_Star_Wars=T), StarWarsNetwork).show_approx())


''' Part 2 : '''

# Using the python script provided for you in the python example for this module compute P(of the same variable)
# using enumeration in a python script (5 points)
print("\nenumeration")
print("P(SeenStarWars|Male=T)", enumeration_ask('Seen_Star_Wars', dict(Male=T), StarWarsNetwork).show_approx())

# Using the python script provided for you in the python example for this module compute P(of the same variable)
# using elimination in a python script (5 points)
print("\nvariable elimination")
print("P(Seen_Star_Wars|Male=T)", elimination_ask('Seen_Star_Wars', dict(Male=T), StarWarsNetwork).show_approx())

# Using the python script provided for you in the python example for this module compute P(of the same variable)
# using rejection sampling in a python script (5 points)
print("\nrejection_sampling")
random.seed(47)
print("P(Seen_Star_Wars|Male=F)", rejection_sampling('Seen_Star_Wars', dict(Male=F), StarWarsNetwork, 10000).show_approx())


# Using the python script provided for you in the python example for this module compute P(of the same variable)
# using likelihood weighting sampling in a python script (5 points)
print("\nlikelihood_weighting")
random.seed(1017)
print("P(Seen_Star_Wars|Male=F)", likelihood_weighting('Seen_Star_Wars', dict(Male=F), StarWarsNetwork, 10000).show_approx())

# Using the python script provided for you in the python example for this module compute P(of the some variable)
# using prior sampling in a python script (5 points)
print("\nprior_sampling")
random.seed(42)
all_obs = [prior_sample(StarWarsNetwork) for x in range(1000)]
male_true = [observation for observation in all_obs if observation['Male'] == True]
Male_and_Seen_Star_Wars = [observation for observation in male_true if observation['Seen_Star_Wars'] == True]
print("P(Seen_Star_Wars,Male=T)", len(Male_and_Seen_Star_Wars) / len(male_true))
