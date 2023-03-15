listaBrojeva = []
brojac = 0
while True:
    unos = input("Unesite vrijednost za upis u listu ili rijec Done: ")
    if unos == "Done":
        break
    else:
        try:
            listaBrojeva.append(float(unos))
            brojac += 1
        except ValueError:
            print("Vrijednost nije broj!")
            continue

print("Uneseno je " + str(brojac) + " brojeva")
listaBrojeva.sort()
print("Sortirana lista: " + str(listaBrojeva))
print("Minimalni element: " + str(listaBrojeva[0]))
print("Maksimalni element: " + str(listaBrojeva[-1]))
print("Srednja vrijednost: " + str(sum(listaBrojeva)/brojac))