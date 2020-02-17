import matplotlib.pyplot as plt
import plotly.express as px
import json
import os

dict_val_acc = dict()
dict_partut_acc = dict()
dict_gsd_acc = dict()
for dir in os.listdir("evaluations/"):
    with open("evaluations/{}/history.json".format(dir), "r") as rfile:
        hist = rfile.read().replace("'", '"')
    dir_reformat = dir.replace("\uf00d", "")
    hist = json.loads(hist[1:-1])
    dict_val_acc[dir_reformat] = hist["val_acc"]
    list_best = [max(o) for o in list(dict_val_acc.values())]

    for corpus in os.listdir("evaluations/{}".format(dir)):
        if corpus.endswith("ud-test"):
            with open("evaluations/{}/{}".format(dir, corpus), "r") as rfile:
                lines = rfile.readlines()
                # if corpus == "partut":
                if corpus == "evaluation_score_fr_partut-ud-test":
                    dict_partut_acc[dir_reformat] = lines[1]
                elif corpus == "evaluation_score_fr_gsd-ud-test":
                    # elif corpus == "gsd":
                    dict_gsd_acc[dir_reformat] = lines[1]
plt.figure()
fig = px.scatter(y=list_best, x=list(dict_val_acc.keys()))
fig.write_image("images/fig1.png")
print(list_best)

fig = px.scatter(y=list(dict_gsd_acc.values()), x=list(dict_gsd_acc.keys()))
fig.write_image("images/gsd_accuracy.png")

fig = px.scatter(y=list(dict_partut_acc.values()), x=list(dict_partut_acc.keys()))
fig.write_image("images/partut_accuracy.png")
# plt.plot(hist["val_acc"])
# plt.show()
# print(hist)