import pandas as pd

header = ['Outlook', 'Temperature', 'Humidity', 'Wind', 'Play']

df = pd.read_csv('./income_train.csv', header=None, names=header)

data_size = df.shape[0]


def check_value(value, column_name, type):
    count = 0
    for i in range(data_size):
        if df[column_name].values[i] == value and df['Play'].values[i] == type:
            count += 1
    return count


def naive_bayes(test):
    Play_count = df.Play.value_counts()

    proba_yes = Play_count['Yes'] / data_size
    proba_no = Play_count['No'] / data_size

    # yes
    proba_outlook_yes = check_value(test[0], 'Outlook', 'Yes') / Play_count['Yes']
    proba_temp_yes = check_value(test[1], 'Temperature', 'Yes') / Play_count['Yes']
    proba_humidity_yes = check_value(test[2], 'Humidity', 'Yes') / Play_count['Yes']
    proba_wind_yes = check_value(test[3], 'Wind', 'Yes') / Play_count['Yes']

    _yes = proba_outlook_yes * proba_temp_yes * proba_humidity_yes * proba_wind_yes * proba_yes

    # no
    proba_outlook_no = check_value(test[0], 'Outlook', 'No') / Play_count['No']
    proba_temp_no = check_value(test[1], 'Temperature', 'No') / Play_count['No']
    proba_humidity_no = check_value(test[2], 'Humidity', 'No') / Play_count['No']
    proba_wind_no = check_value(test[3], 'Wind', 'No') / Play_count['No']

    _no = proba_outlook_no * proba_temp_no * proba_humidity_no * proba_wind_no * proba_no

    return round(_yes / (_no + _yes), 2), round(_no / (_no + _yes), 2)
    pass


data_test = ['Sunny', 'Hot', 'High', 'Weak']
res_y, res_n = naive_bayes(data_test)
print("Yes: ", res_y)
print("No: ", res_n)
