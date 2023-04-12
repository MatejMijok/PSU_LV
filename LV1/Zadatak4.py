imeDatoteke = input("Ime datoteke: ")

file = open(imeDatoteke, "r")

suma = 0
brojac = 0

for x in file:
    linija = file.readline()
    if linija.startswith("X-DSPAM-Confidence: "):
       element = linija.find(":")
       suma += float(linija[element+1:])
       brojac += 1

file.close()

suma /= brojac

print("Average X-DSPAM-CONFIDENCE: " + str(suma))
