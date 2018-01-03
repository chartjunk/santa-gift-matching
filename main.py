import pandas as pd
import numpy as np


def solve():
    wish = pd.read_csv('data/child_wishlist_v2.csv', header=None).as_matrix()[:,1:]
    #good = pd.read_csv('data/gifts_goodkids_v2.csv', header=None).as_matrix()[:,1:]
    pred = np.ones((len(wish)), dtype=np.int32)*-1
    gifts_given = np.zeros((1000), dtype=np.int32)
    crappy_kids = list()

    print("Predict triplets")
    for kid in range(0, 5000, 3):
        gift = wish[kid, 1]
        pred[kid] = gift
        pred[kid+1] = gift
        pred[kid+2] = gift
        gifts_given[gift] += 3

    print("Predict twins")
    for kid in range(5001, 45000, 2):
        gift = wish[kid, 1]
        pred[kid] = gift
        pred[kid+1] = gift
        gifts_given[gift] += 2

    print("Predict singles")
    # * 1000 gifts limit must be considered
    for kid in range(45001, len(wish)):
        is_gift_given = False
        for gift in wish[kid]:
            if gifts_given[gift] < 1000:
                pred[kid] = gift
                gifts_given[gift] += 1
                is_gift_given = True
                break
        if is_gift_given is False:
            crappy_kids.append(kid)

    print("Distribute rest of the gifts randomly")
    gifts_not_wanted = np.argwhere(gifts_given < 1000).flatten()
    gift_ix = 0
    for kid in crappy_kids:
        gift = gifts_not_wanted[gift_ix]
        pred[kid] = gift
        gifts_given[gift] += 1
        if gifts_given[gift] == 1000:
            gift_ix += 1

    assert len(pred) == len(wish)
    assert pd.value_counts(pred).max() == 1000


if __name__ == '__main__':
    solve()