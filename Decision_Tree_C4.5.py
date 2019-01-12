import numpy as np
import pandas as pd
# code is based on this resource - http://gabrielelanaro.github.io/blog/2016/03/03/decision-trees.html

def partition(a, y):
    if is_real(a):
        split, _ = best_split(a, y)
        return {"<= {}".format(split): np.where(a >= split)[0],
                "> {}".format(split): np.where(a < split)[0]}, split
    else:
        return {c: (a == c).nonzero()[0] for c in np.unique(a)}, None


def entropy(s):
    val, counts = np.unique(s, return_counts=True)
    freqs = counts.astype('float') / len(s)
    return v(freqs)


def best_split(x, y):
    x = np.sort(x)
    splits = []
    n = len(x)
    for i in range(n - 1):
        splits.append((x[i] + x[i + 1]) / 2.0)
    splits = np.array(splits)
    impurities = []
    for split in splits:
        imp = (len(x[x >= split]) / n) * entropy(y[x >= split]) + \
              (len(x[x < split]) / n) * entropy(y[x < split])
        impurities.append(imp)
    impurities = np.array(impurities)
    max_idx = np.argmax(impurities)
    return splits[max_idx], impurities[max_idx]


def information_gain(y, x):

    res = entropy(y)

    # We partition x, according to attribute values x_i
    if is_real(x):
        split, impurity = best_split(x, y)
        res -= impurity
        return res, split
    else:
        val, counts = np.unique(x, return_counts=True)
        freqs = counts.astype('float') / len(x)

        # We calculate a weighted average of the entropy
        for p, v in zip(freqs, val):
            res -= p * entropy(y[x == v])

        return res, None

def v(freqs):
    res = 0
    for p in freqs:
        if p != 0.0:
            res -= p * np.log2(p)
    return res

def intrinsic_value(x, split):
    n = len(x)
    if split is not None:
        freqs = np.array([len(x[x >= split]), len(x[x < split])], dtype=float) / n
    else:
        _, counts = np.unique(x, return_counts=True)
        freqs = counts.astype('float') / n
    return v(freqs)


def information_gain_ratio(y, x):
    res, split = information_gain(y, x)
    v = intrinsic_value(x, split)
    return res / v if res != 0 else 0


def is_pure(s):
    return len(set(s)) == 1


def recursive_split(x, y, fields):
    # If there could be no split, just return the original set
    if is_pure(y) or len(y) == 0:
        return y

    # We get attribute that gives the highest information gain
    gain = np.array([information_gain_ratio(y, x_attr) for x_attr in x.T])
    selected_attr = np.argmax(gain)

    # If there's no gain at all, nothing has to be done, just return the original set
    if np.all(gain < 3e-2):
        return y

    # We split using the selected attribute
    sets, split = partition(x[:, selected_attr], y)

    res = {}
    for k, v in sets.items():
        y_subset = y.take(v, axis=0)
        x_subset = x.take(v, axis=0)

        form = "{} = {}" if split is None else "{} {}"
        res[form.format(fields[selected_attr], k)] = recursive_split(
            x_subset, y_subset, fields)

    return res


def is_real(x):
    return type(x[0]) == float

def predict_one(tree, fields, x):
    while type(tree) is not np.ndarray:
        changed = False
        for i in range(len(fields)):
            key = "{} = {}".format(fields[i], x[i])
            if key in tree:
                changed = True
                tree = tree[key]
                break
            key_format = "{} <=".format(fields[i])
            flag = False
            for key in tree:
                if key.startswith(key_format):
                    changed = True
                    split = float(key.split("<= ")[1])
                    key_format = ("{} <= {}" if x[i] >= split else "{} > {}").format(fields[i], split)
                    tree = tree[key_format]
                    flag = True
                    break
            if flag: break
        if not changed:
            tree = get_answer_rec(tree)
            break

    (values, counts) = np.unique(tree, return_counts=True)
    ind = np.argmax(counts)
    return values[ind]

def get_answer_rec(data):
    if type(data) is np.ndarray: return data
    ans = None
    for _, value in data.items():
        ret = get_answer_rec(value)
        ans = ret if ans is None else np.concatenate((ans, ret))

    return ans

def predict(tree, fields, x):
    return np.array([predict_one(tree, fields, x_i) for x_i in x])

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

train_set = pd.read_csv('titanic_modified.csv')
train_set = train_set.dropna()

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

X = train_set.iloc[:, :6].values
y = train_set.iloc[:, 6].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)

fields = list(train_set.columns.values)[:-1]
tree = recursive_split(X_train, y_train, fields)

y_predicted = predict(tree, fields, X_test)
print(classification_report(y_pred=y_predicted, y_true=y_test))
print("Testing accuracy:", accuracy_score(y_true=y_test, y_pred=y_predicted))

y_predicted = predict(tree, fields, X_train)
print("Training accusracy: ", accuracy_score(y_true=y_train, y_pred=y_predicted))

#Preprunning used by limiting the growth of the tree if the gain is less than 0.003