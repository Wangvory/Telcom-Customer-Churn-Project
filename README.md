# Telcom Customer Churn Project
## Part 1 Data Description
I use the data tells the churn rate of a certain Telcom company, containing 21 variables. The dependent variable is churn or not. And the detailed description of the data is listed in PDF. 

## Part 2 Data Visualization
Then did the data visualization to understand the data.Choose the variables: Tenure, Senior Citizen, Contract type and monthly charges.

## Part 3 Model Selection
In this project, aim to find the best model for the real-world, Telecom customer churn rate dataset. Using the same metric, mean test score, to decide the best model, I want to know if the conclusion I achieved in Problem 3 Question 4, which used a synthesized dataset, also works for a real-world dataset. To maximize the accuracy of our conclusion, I conducted cross validation and collected the best accuracy for each model. 
From the result above, I can find that the best model is Logistic regression, folloId by SVC which has a very close test accuracy. This is very different from what I got in Problem Set 3, where SVC performed the worst. It’s excited to find that logistic regression gives the best model, because I can achieve good interpretation without sacrificing for accuracy.
### KNN classifier
performed pretty Ill when I made a loop and tested different k from 3 to 13 and found that k=11 gives the best test accuracy of 0.7837, which is better than the Random Forest model.
### Decision tree
was the best model in Problem Set 3, but here it performs just almost as Ill as the Linear Discriminant. I’ve imagined that Linear Discriminant would do better when restricted to real values, and our result shows that it does.
### Random Forest model
In this run, the Random Forest model didn’t do better compared to the maximum test accuracy of Decision Tree, when I have tested different depth of trees from 1 to 20. But the result of Random Forest could change due to the randomness.
### Quadratic discriminant
is even worse than the Gaussian Naïve Bayes, therefore, it should not be considered a good model.