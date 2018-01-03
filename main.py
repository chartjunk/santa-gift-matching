import pandas as pd
import numpy as np


def get_avg_norm_happiness(child_gifts, wish, good):

    n_child = len(wish)
    n_gift = len(good)

    max_child_happiness = 2*100
    max_gift_happiness = 2*1000
    total_norm_child_happiness = 0
    total_norm_gift_happiness = 0

    for child in range(0,len(wish)):

        gift = child_gifts[child]

        child_happiness = -1
        if gift in wish[child]:
            child_happiness = 100-np.argwhere(wish[child] == gift)
        total_norm_child_happiness += child_happiness / max_child_happiness

        gift_happiness = -1
        if child in good[gift]:
            gift_happiness = 1000-np.argwhere(good[gift] == child)
        total_norm_gift_happiness += gift_happiness / max_gift_happiness

    avg_norm_child_happiness = (1./n_child)*total_norm_child_happiness
    avg_norm_gift_happiness  = (1./n_gift)*total_norm_gift_happiness
    avg_norm_happiness = avg_norm_child_happiness**3 + avg_norm_gift_happiness**3
    return avg_norm_happiness


def solve():
    wish = pd.read_csv('data/child_wishlist_v2.csv', header=None).as_matrix()[:,1:]
    good = pd.read_csv('data/gift_goodkids_v2.csv', header=None).as_matrix()[:,1:]
    pred = np.ones((len(wish)), dtype=np.int32)*-1
    gifts_given = np.zeros((1000), dtype=np.int32)
    crappy_children = list()

    print("Predict triplets")
    for child in range(0, 5000, 3):
        gift = wish[child, 1]
        pred[child] = gift
        pred[child+1] = gift
        pred[child+2] = gift
        gifts_given[gift] += 3

    print("Predict twins")
    for child in range(5001, 45000, 2):
        gift = wish[child, 1]
        pred[child] = gift
        pred[child+1] = gift
        gifts_given[gift] += 2

    print("Predict singles")
    # * 1000 gifts limit must be considered
    for child in range(45001, len(wish)):
        is_gift_given = False
        for gift in wish[child]:
            if gifts_given[gift] < 1000:
                pred[child] = gift
                gifts_given[gift] += 1
                is_gift_given = True
                break
        if is_gift_given is False:
            crappy_children.append(child)

    print("Distribute rest of the gifts randomly")
    gifts_not_wanted = np.argwhere(gifts_given < 1000).flatten()
    gift_ix = 0
    for child in crappy_children:
        gift = gifts_not_wanted[gift_ix]
        pred[child] = gift
        gifts_given[gift] += 1
        if gifts_given[gift] == 1000:
            gift_ix += 1

    assert len(pred) == len(wish)
    assert pd.value_counts(pred).max() == 1000

    print("Calculate avg norm happiness")
    print("Avg norm happiness = " + str(get_avg_norm_happiness(pred, wish, good)))

if __name__ == '__main__':
    solve()