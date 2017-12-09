import os
import glob
import pandas as pd


# data_dir = "/home/user1/Desktop/prep/new_data"
# subjects = os.listdir(data_dir)
# csv_paths = []
# for subject in subjects:
#     csv_paths += glob.glob(os.path.join(data_dir, subject, "new") + "/*.csv")

# df = pd.concat(map(pd.read_csv, csv_paths))
# df.to_csv("total.csv", index=False)

# for i in range(len(csv_paths)):
#     df = pd.read_csv(csv_paths[i])
#     print i, df["subject"].values

total = pd.read_csv("total.csv")
des = total["info"].values.tolist()
subjid = total["subject"].values.tolist()


def query(l, t, nt=[]):
    indices = []
    for i in range(len(l)):
        tnum = 0
        for it in t:
            if it in l[i].lower():
                tnum += 1
        
        ntnum = 0
        for n in nt:
            if n not in l[i].lower():
                ntnum += 1

        if tnum == len(t) and ntnum == len(nt):
            indices.append(i)

    scan_type = "_".join(t)
    print scan_type, ":", len(indices)

    return indices


# print "T1:", len(query(des, "t1"))
# print "T2:", len(query(des, "t2"))
# print "Flair:", len(query(des, "flair"))
# print "AX:", len(query(des, "ax"))
# print "SAG:", len(query(des, "sag"))
# print "COR:", len(query(des, "cor"))

# all_types = {}
# for scan_type in ["t1", "t2", "flair"]:
#     for scan_view in ["ax", "cor", "sag"]:
#         type_list = query(des, scan_type)
#         view_list = query(des, scan_view)
#         and_list = set(type_list) & set(view_list)
#         print " ".join([scan_type, scan_view, ":"]), len(list(and_list))
#         all_types[scan_type + scan_view] = list(and_list)

# print all_types

# t1cor = query(des, ["t1", "cor"], ["flair"])
# t2cor = query(des, ["t2", "cor"], ["flair"])

# t1 = query(des, ["t1"], ["flair"])
# t2 = query(des, ["t2"], ["flair"])
# flair = query(des, ["flair"], ["t1", "t2"])
# t1flair = query(des, ["t1", "flair"])
# t2flair = query(des, ["t2", "flair"])


t1ax = query(des, ["t1", "ax"], ["flair", "spgr"])
subj_t1ax = [int(subjid[i]) for i in t1ax]
print len(set(subj_t1ax))


t1cor = query(des, ["t1", "cor"], ["flair"])
t1sag = query(des, ["t1", "sag"], ["flair"])

t2ax = query(des, ["t2", "ax"], ["flair", "spgr"])
t2cor = query(des, ["t2", "cor"], ["flair"])
t2sag = query(des, ["t2", "sag"], ["flair"])

flairax = query(des, ["flair", "ax"], ["t1", "t2", "spgr"])
flaircor = query(des, ["flair", "cor"], ["t1", "t2"])
flairsag = query(des, ["flair", "sag"], ["t1", "t2"])



# t1cor = []
# for i in range(len(des)):
#     lw = des[i].lower()
#     if ("t1" in lw) and ("cor" in lw) and ("flair" not in lw):
#     	t1cor.append(i)
# print t1cor