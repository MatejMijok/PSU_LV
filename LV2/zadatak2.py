import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(open("mtcars.csv", "rb"), usecols=(1,2,3,4,5,6),
delimiter=",", skiprows=1)

plt.scatter(data[:,0], data[:,3], linewidth = data[:,5], marker=".", c = "g")

print("Minimalni mpg iznosi: " + str(min(data[:,0])))
print("Maksimalni mpg iznosi: " + str(max(data[:,0])))
print("Srednja vrijednost iznosi: " + str(sum(data[:,0])/data[:,0].size))

car6cyl = []

for i,car in enumerate(data[:,1]):
    if car == 6:
        car6cyl.append(data[i,0])

print("Minimalni mpg 6 cyl auta iznosi: " + str(min(car6cyl)))
print("Maksimalni mpg 6 cyl auta iznosi: " + str(max(car6cyl)))
print("Srednja vrijednost mpg 6 cyl auta iznosi: " + str(sum(car6cyl)/len(car6cyl)))

plt.show()
