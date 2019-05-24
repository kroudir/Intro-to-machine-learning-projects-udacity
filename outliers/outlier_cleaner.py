#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    errors = (net_worths - predictions) * (net_worths - predictions)
    for age, net_worth, error in zip(ages, net_worths, errors):
        cleaned_data.append((age,net_worth,error))
    
    cleaned_data.sort(key = lambda tup: tup[2])
    
    return cleaned_data[:81]

