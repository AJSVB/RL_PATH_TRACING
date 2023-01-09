import os

for j in [4]:
 #for i in ["vanilla","uni","notp","notpuni","dasr","grad","ntas"]:
 for i in ["vanilla"]:

  print()
  print(i)
  print()
  os.system("python main.py " +str(j) +" "+ i)

#os.system("python denoisingscript.py")
