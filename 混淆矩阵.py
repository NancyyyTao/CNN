import pandas as pd
from sklearn.metrics import confusion_matrix

data_path='E:/17-20/2.1/抽穗前/RF/AND 间接计算.xlsx'
sheetName='Sheet2'
df=pd.read_excel(data_path,sheet_name=sheetName)
y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
 y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])
