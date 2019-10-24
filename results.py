from graphs.random_features import random_features as graph
import model.tensorflow_model as model
import numpy as np

method, labels = 2, 10
n_lab, n_train, n_unlab, n_feat = 160, 640, 200, 10
num_epochs, lr = 400, 0.01

def frac(G):
    labs = sum(np.concatenate((G.Ylabel,G.Ytarget,G.Ytest)).astype("int32"))
    return (max(labs)/sum(labs))

with open("results/report.csv", "w") as arq:
    arq.write("p,train_loss,train_acc,test_acc,frac\n")
    for p in [0.001, 0.01, 0.1]:
        for i in range(200):
            print(p, i)
            G = graph(n_lab, n_train, n_unlab, n_feat, labels, method, p=p)
            val = frac(G)
            arq.write("{},{},{},{},{}\n".format(p, *(model.train(G, num_epochs, lr, method, False, True)), val))
