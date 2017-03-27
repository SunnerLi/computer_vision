import os

for name in os.listdir('./'):
    if name[-4:]=='.png':
        os.rename(name, 'cc'+name)

count = 0
for name in os.listdir('./'):
    if name[-4:]=='.png':
        os.rename(name, str(count)+'.png')
        count += 1