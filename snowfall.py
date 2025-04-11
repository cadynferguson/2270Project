import numpy as np
import random
import json
"""

0" (No Snow)
0.1"–0.9"
1"–2.9"
3"–4.9"
5"–6.9"
7"–8.9"
9"–10.9"
11"–13.9"
14"–17.9"
18"+
10 Categories of snowfall

1. Parse the JSON 
2. Create an array of all of the new_snow values
3. Using this array we will loop through an empty np array and increment the values based on the previous state
3a. So if previous state was no snow and the current one is 4inches than we will increment row 0 column 4 by 1
"""
def put_into_range(val):
    thresholds = [0, 0.9, 2.9, 4.9, 6.9, 8.9, 10.9, 13.9, 17.9]
    for i, t in enumerate(thresholds):
        if val <= t:
            return i
    return 9

def get_value_out_of_range(val):
    values = [0, 0.5, 2.0, 4.0, 6.0, 8.0, 10.0, 12.5, 16.0, 20.0]
    return values[val]

# start with all 0's up to the longest season that alta has had
snow_amt = [0] * 211
with open('snowfall-history.json') as f:
    data = json.load(f)
    # for each year we will loop through the days and add the new_snow to the snow_amt array
    for year in data['snowfallHistory']:
        day_ct = 0
        print(len(year))
        for day in year:
            snow_amt[day_ct] += (day['new_snow'])
            day_ct += 1
# we now take the average of the snow amount for each day and then put that into our ranges
new_snow = [put_into_range((x * .96 // 4)) for x in snow_amt]

# we split it into months so the whole system isnt decided by one month but more on a per month basis
new_snow = np.array_split(new_snow, 6)

# self explanatory
def generate_matrix(input):
    return_mat = np.zeros((10, 10))
    previous = input[0]
    for i in range(1, len(input)):
        current = input[i]
        return_mat[previous][current] += 1
        previous = current
    return return_mat

# for each month we will generate a matrix
for i in range(len(new_snow)):
    new_snow[i] = generate_matrix(new_snow[i])

# normalize the matrix to turn it into a stochastic matrix
def normalize(matrix):
    for i in range(len(matrix)):
        row_sum = np.sum(matrix[i])
        if row_sum != 0:
            matrix[i] /= row_sum
    return matrix

# normalize each matrix
for i in range(len(new_snow)):
    new_snow[i] = normalize(new_snow[i])

# initialize our starting state which is no snow as it's early october
current_state = np.zeros((10,1))
# 5% chance to start off the season with some snow we might be able to see this once after like 50 plots and could be minimal or could be a record snow season
current_state[0 if random.random() > 0.02 else 2] = 1
states = []

days_per_month = [31, 30, 31, 31, 28, 31]

current_category = 0
snow_day_categories = []

# for each month
for month_index in range(6):
    matrix = new_snow[month_index]
    for _ in range(days_per_month[month_index]):
        # get the current category
        transition_probs = matrix[current_category]

        # make sure it isnt all 0's
        if np.sum(transition_probs) == 0:
            transition_probs = np.ones(10) / 10

        # get the next category based on the probabilities of the current row
        next_category = np.random.choice(10, p=transition_probs)
        # randomly have a chance to increase the storm strength. this is configurable and highly noticable
        if random.random() < 0.05 and next_category != 9:
            next_category += 1
        # also have  achance to decrease the storm strength. this is configurable and highly noticable
        if random.random() > 0.9 and next_category != 0:
            next_category -= 1
        snow_day_categories.append(next_category)
        current_category = next_category

# get an average of the value of the snow day categories for our estimate
daily_snowfall = [get_value_out_of_range(cat) for cat in snow_day_categories]

# sum up the daily snowfall cummulatively to build a graph ie [1, 2, 3, 4] = [1, 3, 6, 10]
cumulative_snowfall = np.cumsum(daily_snowfall)

# plot it
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(cumulative_snowfall, label='Cumulative Snowfall', color='blue')
plt.fill_between(np.arange(len(cumulative_snowfall)), cumulative_snowfall, color='blue', alpha=0.3)

plt.title("Snow Accumulation Over the Season (Markov Chain Simulation). Total Snowfall: {:.2f} inches".format(cumulative_snowfall[-1]))
plt.xlabel("Day of Season")
plt.ylabel("Total Snowfall (inches)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()