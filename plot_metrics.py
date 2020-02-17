import matplotlib.pyplot as plt
import plotly.express as px
import json
import os

metrics = dict()
for dir in os.listdir("evaluations/"):
    with open("evaluations/{}/history.json".format(dir), "r") as rfile:
        hist = rfile.read().replace("'", '"')
    dir_reformat = dir.replace("\uf00d", "")
    hist = json.loads(hist[1:-1])
    metrics[dir_reformat] = hist["val_acc"]
    list_best = [max(o) for o in list(metrics.values())]

plt.figure()
fig = px.scatter(y=list_best, x=list(metrics.keys()))
fig.write_image("images/fig1.png")
print(list_best)

# plt.plot(hist["val_acc"])
# plt.show()
# print(hist)