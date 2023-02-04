from sklearn.metrics import auc

xs = [0, 0, 0.5, 0.5, 1]
ys = [0, 0.5, 0.5, 1, 1]

print(auc(xs, ys))
