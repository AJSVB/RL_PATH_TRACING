import os

for j in [2]:
 for i in ["vanilla","uni","notp","notpuni","dasr","grad","ntsr"]:
  os.system("python main.py " +str(j) +" "+ i)

#os.system("python denoisingscript.py")
