import string
file = open("song.txt", "r")

rjecnik = {}


for x in file:
    x = x.split(" ")
    
    for rijec in x:
        rijec = rijec.translate(str.maketrans('','', string.punctuation))
        rijec = rijec.strip()

        if rijec.lower() in rjecnik:
            rjecnik[rijec.lower()] += 1
        else:
            rjecnik[rijec.lower()] = 1

print(rjecnik)
br = 0
for x in rjecnik:
    if rjecnik.get(x) == 1:
        br = br + 1


print("Broj rijeci koje se ponavljaju samo jednom: " + str(br))
