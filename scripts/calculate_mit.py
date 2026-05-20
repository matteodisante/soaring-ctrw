import numpy as np
from scipy.optimize import root_scalar
from pymittagleffler import mittag_leffler

def trova_argomento_mittag_leffler(alpha, valore_target):
    # Definiamo la funzione di cui vogliamo trovare lo zero
    def equazione(z):
        # Calcoliamo la Mittag-Leffler e sottraiamo il valore target.
        # Usiamo np.real per estrarre la parte reale e aiutare la convergenza.
        risultato_ml = np.real(mittag_leffler(z, alpha, 1.0))
        return risultato_ml - valore_target
        
    # Usiamo root_scalar per trovare il valore di z che azzera l'equazione.
    # Forniamo due valori iniziali (x0 e x1) per avviare l'algoritmo di ricerca.
    # Sapendo che per alpha=0.5 il valore in z=0 è 1, per arrivare a 0.5
    # l'argomento z dovrà essere negativo.
    soluzione = root_scalar(equazione, x0=0.0, x1=-1.0)
    
    if soluzione.converged:
        return soluzione.root
    else:
        raise ValueError("L'algoritmo non è riuscito a trovare una soluzione.")

# 1. Inserisci il valore dell'esponente
esponente = 0.6 

# 2. Inserisci il valore della funzione che vuoi ottenere
valore_desiderato = 1/2

# Calcolo
argomento_z = trova_argomento_mittag_leffler(esponente, valore_desiderato)

print("Risultati:")
print("Esponente alpha:", esponente)
print("Valore target desiderato:", valore_desiderato)
print("Argomento z calcolato:", argomento_z)

# Verifica opzionale per confermare che il risultato sia corretto
verifica = np.real(mittag_leffler(argomento_z, esponente, 1.0))
print("Verifica del calcolo:", verifica)