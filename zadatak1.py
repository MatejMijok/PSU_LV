radniSati = int(input("Unesite broj radnih sati: "))
satnica = float(input("Unesite satnicu u eurima: "))

def total_euro():
  ukupno = radniSati * satnica
  return ukupno

print("Ukupno iznosi: " + str(total_euro())) 
