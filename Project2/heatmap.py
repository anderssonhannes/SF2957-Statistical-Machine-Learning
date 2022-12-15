import numpy as np
import pandas as pd

arr = pd.read_csv("Project2//sumQdata/sumQ_1.csv",
                    names=["sum_a", "sum_d","usable_ace","action_reward"])

arr["sum_a"] = arr["sum_a"].str[1:]
arr["usable_ace"] = arr["usable_ace"].str[:-1]

arr["sum_a"] = arr["sum_a"].astype("int")
arr["usable_ace"] = arr["usable_ace"].astype("bool")

# Change later for better intervals [10,21]
arr = arr[arr.sum_a.between(10,21)]
arr = arr[arr.sum_d.between(2,11)]
arr["New_action_reward"] = ""
Temp=[]
for row in arr.iterrows():
    tmp = row[1].action_reward.strip("][").split(" ")
    while '' in tmp:
        tmp.remove("")
    
    tmp = list(map(float,tmp))
    Temp.append(tmp)
    #arr.loc[row[0],"New_action_reward"] = tmp
    #tmp = arr.loc[row[0],"action_reward"]
arr["C"]=Temp
    


print(arr)