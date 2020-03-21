import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# define the current working directory
here = os.path.dirname(os.path.abspath(__file__))

# create the path to our data file
path_to_train = os.path.join(here, "..", "data", "train.csv")

# read in our data to a pandas dataframe
data = pd.read_csv(path_to_train)

# inspect the top of the dataframe to see what it looks like
print(data.head())


def print_digit(row_number: int, dataframe: pd.DataFrame):
    """
    Function which will display what a digit looks like
    :param row_number: the row number of the digit in the dataset we want to display
    :param dataframe: the pandas dataframe containing all of our data
    """
    row = dataframe.iloc[row_number, 1:].values
    row = row.reshape(28, 28).astype('uint8')
    plt.imshow(row)
    plt.show()


# we seperate out the first column containing what each digit is
df_x = data.iloc[:, 1:]
df_y = data.iloc[:, 0]

# we split up our data into a train and test set with 20% forming the test set
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)

# we create our RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)

# we train our classifier on the training data
rf.fit(x_train, y_train)

# we use our classifier to predict our test values
predictions = rf.predict(x_test)

# we extract those into a list for comparison with our predictions
expected_values = y_test.values

# we create a zero value for our counter of correct predictions
count = 0

# we iterate through our predictions
for i in range(len(predictions)):
    # if our prediction is correct, we add 1 to our count of correct predictions
    if predictions[i] == expected_values[i]:
        count += 1

# we calculate our overall prediction accuracy
percentage_accuracy = count/len(predictions)*100

# we print this out for the user
print(f"""Our prediction accuracy is {round(percentage_accuracy,2)}%""")
