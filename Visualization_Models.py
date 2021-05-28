import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd

#Storing the data into lists
#Random Forest
accuracy_RF = [97.8, 97.6, 98.3, 97.1, 97.5, 97.6, 97.89999999999999, 97.2, 97.1, 98.1]
time_RF = [2.7911531925201416, 2.2163238525390625, 2.4514999389648438, 2.3436572551727295, 2.2993648052215576, 2.2126851081848145, 2.283093214035034, 2.232358932495117, 2.142333507537842, 2.276837110519409]

#Logistic Regression
accuracy_LR = [90.85, 91.05, 90.75, 90.7, 89.9, 90.64999999999999, 90.64999999999999, 90.75, 90.14999999999999, 90.3];
time_LR = [1.377514123916626, 1.3241703510284424, 1.3009059429168701, 1.2602698802947998, 1.312192678451538, 1.2653284072875977, 1.2184617519378662, 1.4264214038848877, 1.4082610607147217, 1.265329360961914]

#KNN
accuracy_KNN = [91.10000000000001, 91.45, 91.25, 91.10000000000001, 91.35, 91.55, 91.05, 90.60000000000001, 90.3, 90.64999999999999]
time_KNN = [37.61703276634216, 38.47673058509827, 37.616586685180664, 36.09958291053772, 34.27668237686157, 34.20232105255127, 33.554439067840576, 34.14704084396362, 34.068636894226074, 34.29947352409363]

avg_acc_RF = sum(accuracy_RF)/10
avg_acc_LR = sum(accuracy_LR)/10
avg_acc_KNN = sum(accuracy_KNN)/10

avg_time_RF = sum(time_RF)/10
avg_time_LR = sum(time_LR)/10
avg_time_KNN = sum(time_KNN)/10

DATA = [[avg_acc_RF, avg_time_RF], [avg_acc_LR, avg_time_LR], [avg_acc_KNN, avg_time_KNN]]
df = pd.DataFrame(data = DATA, columns = ['Average Accuracy', 'Average Time'], index = ['Random Forest', 'Logistic Regression', 'KNN'])
df

plt.figure(figsize = (8,8))
sns.barplot(data = df, y = df['Average Accuracy'], x = df.index)

plt.figure(figsize = (8,8))
sns.barplot(data = df, x = df['Average Time'], y = df.index)

plt.figure(figsize = (8,8))
sns.lineplot(data= {'RF': accuracy_RF,'LR': accuracy_LR, 'KNN': accuracy_KNN})

plt.figure(figsize = (8,8))
sns.lineplot(data= {'RF': time_RF, 'LR': time_LR, 'KNN': time_KNN})
