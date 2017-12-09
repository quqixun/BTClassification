import pandas as pd


#
# pip install xlrd
#
xsl = pd.read_excel("Data Collection final US.xlsx")
csv = pd.read_csv("total.csv")

studyids = xsl["patient_studyID"].values.tolist()
idh1mts = xsl["IDH1 mutation (0=no, 1=yes, 9=missing)"].values.tolist()
qcodels = xsl["1p19qcodel (0=no, 1=yes, 9=missing)"].values.tolist()

subjectids = csv["subject"].values.tolist()

subj_idh1mts, subj_qcodels, subj_labels = [], [], []

for subjid in subjectids:
    idx = studyids.index(subjid)

    idh1 = int(str(idh1mts[idx])[0])
    qcod = int(str(qcodels[idx])[0])
    subj_idh1mts.append(idh1)
    subj_qcodels.append(qcod)

    if idh1 == 0 and qcod == 0:
        label = 0
    elif idh1 == 1 and qcod == 0:
        label = 1
    elif idh1 == 0 and qcod == 1:
        label = 2
    elif idh1 == 1 and qcod == 1:
        label = 3
    else:
        label = -1
    subj_labels.append(label)

csv["IDH1 mutation"] = pd.Series(subj_idh1mts, index=csv.index)
csv["1p19qcodel"] = pd.Series(subj_qcodels, index=csv.index)
csv["label"] = pd.Series(subj_labels, index=csv.index)
csv.to_csv("total.csv", index=False)
