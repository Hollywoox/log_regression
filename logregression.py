def distance(w1, w2, w1_, w2_):
    return np.sqrt(np.square(w1 - w2) + np.square(w1_ - w2_))

def sigmoid(w1, w2, x1, x2):
    return 1 / (1 + np.exp(-w1*x1 - w2*x2))

def log_regression(w1, w2, X, y, k, C, iter, broad):
    for i in range(iter):
        w1_ = w1 + k * np.mean(y * X[:, 0] * (1 - 1 / (1 + np.exp(-y * (w1 * X[:, 0] + w2 * X[:, 1]))))) - k * C * w1
        w2_ = w2 + k * np.mean(y * X[:, 1] * (1 - 1 / (1 + np.exp(-y * (w1 * X[:, 0] + w2 * X[:, 1]))))) - k * C * w2
        if distance(w1, w2, w1_, w2_) <= broad:
            break
        w1 = w1_
        w2 = w2_
    answers = sigmoid(w1_, w2_, X[:, 0], X[:, 1])
    print(w1_, w2_)
    return answers
