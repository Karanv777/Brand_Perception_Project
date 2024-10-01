import pandas as pd
from matplotlib import pyplot as plt  
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sentiment = SentimentIntensityAnalyzer()

def analyze(text):
    try:
        sent_1 = sentiment.polarity_scores(text)

        if sent_1["compound"] >= 0.05:
            print("\nThe Perception of Brand is Positive 😍\n")

        elif sent_1["compound"] <= -0.05:
            print("\nThe Perception of Brand is Negative 😡\n")

        else:
            print("\nThe Perception of Brand is Neutral 😐\n")

    except Exception as e:
        print(e)


Text = pd.read_csv("Dataset_1.csv")
analyze(Text)
print(Text)


plt.rcParams["figure.figsize"] = [13.00, 5.50]
plt.rcParams["figure.autolayout"] = True
columns = ["userName", "score"]

Text_2 = pd.read_csv("Dataset_1.csv", usecols=columns)
plt.plot(Text.userName, Text.score)
plt.show()

