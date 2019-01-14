import pandas as pd
import numpy as np

# Examples from: https://www.dataquest.io/blog/naive-bayes-tutorial/

days = [["ran", "was tired"], ["ran", "was not tired"], ["didn't run", "was tired"], ["ran", "was tired"], ["didn't run", "was not tired"], ["ran", "was not tired"], ["ran", "was tired"]]

n_days = len(days)
n_days_tired = len([d for d in days if d[1] == "was tired"])
n_days_ran = len([d for d in days if d[0] == "ran"])
# Calculate P["tired" | "ran"] = (p["ran"|"tired"] p["tired"]) / p("ran")

prob_tired = n_days_tired / n_days

prob_ran = n_days_ran / n_days

prob_ran_given_tired = len([d for d in days if d[0] == "ran" and  d[1] == "was tired"]) / n_days_tired

prob_tired_given_ran = (prob_ran_given_tired * prob_tired) / prob_ran

print("P['tired'|'ran'] = {}".format(prob_tired_given_ran))

#####################################################################

days = [["ran", "was tired", "woke up early"], ["ran", "was not tired", "didn't wake up early"], ["didn't run", "was tired", "woke up early"], ["ran", "was tired", "didn't wake up early"], ["didn't run", "was tired", "woke up early"], ["ran", "was not tired", "didn't wake up early"], ["ran", "was tired", "woke up early"]]

# predict wether on the new day this person was tired:
new_day = ["ran", "didn't wake up early"]

def calc_y_probability(y_label, days):
    return( len( [d for d in days if d[1] == y_label]) / len(days))

def calc_ran_given_y(ran_label, y_label, days):
    return(len([d for d in days if d[1] == y_label and d[0] == ran_label]) / len(days))

def calc_woke_early_given_y(woke_label, y_label, days):
    return(len([d for d in days if d[1] == y_label and d[2] == woke_label]) / len(days))

denominator = len([d for d in days if d[0] == new_day[0] and d[2] == new_day[1]]) / len(days)

prob_tired = (calc_y_probability("was tired", days) * calc_ran_given_y(new_day[0], "was tired", days) * calc_woke_early_given_y(new_day[1], "was tired", days)) / denominator

prob_not_tired = (calc_y_probability("was not tired", days) * calc_ran_given_y(new_day[0], "was not tired", days) * calc_woke_early_given_y(new_day[1], "was not tired", days)) / denominator


classification = "was tired"
if prob_not_tired > prob_tired:
    classification = "was not tired"
print("Final classification for new day: {0}. Tired probability: {1}. Not tired probability: {2}.".format(classification, prob_tired, prob_not_tired))
