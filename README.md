# Project Overview

The purpose of this project is to create a model to predict the outcome of UFC (Ultimate Fighting Championship) events. The UFC was created in 1993 with the expressed purpose of identifying the most effective martial art(s). This project is an extension of my curiosity into the original purpose of the UFC; what makes a more effective martial artist?

# Data Overview

## Sources:
 - [UFC.com](https://www.ufc.com/events)
 - [UFCStats.com](https://www.ufcstats.com)
 - [BestFightOdds.com](https://www.bestfightodds.com)


### Data Description

The majority of the data I used for this project I scraped from UFCStats.com. This website contains more statistics than any other website, but does not have some key metrics such as fighter sizes, bios, and odds. 

Thus, the odds were scraped from bestfightodds.com, while the fighter sizes and bios were scraped from ufc.com. 

Further, individual events and fights, and much from the final streamlit application, are scraped from ufc.com.

The data itself is fight-by-fight based data, originally from over 8,000 fights (which, after dropping for lack of data, decreased to around 5,000 by the final testing dataframe).

There are around 400 features in this dataset, after all are added.


### Define Target Variable

The target variable in this project is if a fighter won an individual fight or not. 

### Define Scoring Metric

Because our data is evenly split between wins and losses, and there is no relative advantage between false negatives and false positives, accuracy is my chosen scoring metric.

## Project Structure

As there was no up-to-date database available, a good portion of this notebook is scraping and saving data using various methods with beautiful soup and selenium. 

The following features were created, either by the scrape itself or calculations done after:

- Fighters A & B 
- Fighter Odds
- Event Date, Name, Urls, Fighter URLs
- Descriptive statistics (mean, median, minimum, maximum, standard deviation) for metrics for each fighter, such as:
    - Knockdowns (attempts, successes, average success rate)
    - Significant Strikes (attempts, successes, average success rate)
    - Total Strikes (attempts, successes, average success rate)
    - Takedowns (attempts, successes, average success rate)
    - Submissions (attempts, successes, average success rate)
    - Control time (attempts, successes, average success rate)
    - Head Strikes (attempts, successes, average success rate)
    - Body Strikes (attempts, successes, average success rate)
    - Leg Strikes (attempts, successes, average success rate)
    - Distance Strikes (attempts, successes, average success rate)
    - Clinch Strikes (attempts, successes, average success rate)
    - Ground Stikes (attempts, successes, average success rate)
- Height, Weight, Reach, Leg Reach, and fighter differences in these metrics.

All in all, there were aproximately 350 features in the final test set.



## Testing

The initial model (decision tree) achieved an accuracy of 60%. 

After iterating on a variety of models, including decision tree, logistic regression, bagged trees, extrra trees, KNN, and random forest, I found the best performing model to be an extra trees model which tested at 70% accuracy. 

## Project Conclusion

The final 
