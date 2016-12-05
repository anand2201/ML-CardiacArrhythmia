import matplotlib.pyplot as plt

x = [1,2,3,4,5]
y = [10,20,30,40,50]

for idx, val in enumerate(x):
    print(idx)

plt.plot(x,y)
plt.xlabel('pca component value')
plt.ylabel('accuracy')
plt.show()

x = [1,2,3,4,5]
y = [10,20,30,40,50]

plt.plot(x,y)
plt.xlabel('pca component value')
plt.ylabel('accuracy')
plt.show()