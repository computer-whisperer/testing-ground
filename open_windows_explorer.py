import subprocess

print("before open")
p = subprocess.Popen(r'explorer /select,"C:/Users/Christian/Desktop"')
p.wait()
print("after open")