#!/usr/bin/python2.7
#-*- coding: utf-8 -*-
print('import peak_search : extrema_of, create_plage_from_extrema, peak_finder')

from general import *
from plage   import *


# TODO :
# ignorer des max sous une valeur (absolue) donnée en paramètre (optionel)
# ignorer des max sous une valeur (relative) donnée en paramètre (optionel)

# hidden peak :
# • trier par importance de la variations, et ne conserver que celles qui manquent.
# • supperposition de deux max en normal & hidden.
# • rechercher et utiliser la plage (plus large) issue de la fourier et non du gradient
# ( plage du gradiant toujours égal aux petites variations, alors que cette plage ne sert que pour le argmin )

def extrema_of( data ):
    """
    À partir d'un tableau de données, renvois une liste d'extremum sous la forme :
    [ indice , boolean ]
    La liste est dans l'ordre des indices croissant
    Le booléen indique s'il s'agit d'un maximum (True) ou d'un minimum (False)

    Note : Les maximums et minimum sont toujours alternés.
    """
    # initialisation
    taille = len(data)
    windows = []
    croissant = data[1]>=data[0] # croissant -> cherche maxium. sinon cherche minimum

    windows.append([0, not croissant]) # si croissant au début, premier point = début de fenêtre

    avant = data[1]
    # boucle
    for i in range(2, taille):
        val = data[i]
        if croissant:
            if(val<=avant):
                windows.append([i-1, True])
                croissant = False
        else :
            if(val>=avant) :
                windows.append([i-1, False])
                croissant = True
        avant = val

    windows.append([taille-1, croissant]) # si décroissant à la fin, dernier point = fin de fenêtre

    return windows


def create_plage_from_extrema( extrema, ofMax=True ):
    """
    À partir d'une liste d'extremum alternées ( cf  "extrema_of(data)" ),
    renvois une liste de plage sous forme de tableau.

    entrée : [ [xMin, False], [xMax, True], [xMin, False], [xMax, True]… ]
    sortie : [ [xMin, xMax, xMin ], [xMin, xMax, xMin ], … ]

    paramètre ofMax :
    par défaut à True : renvois une liste de plage avec les maximums entre deux minimums
    si False : liste de plage avec les minimums entre deux maximums
    """

    if extrema[0][1]==ofMax : extrema = extrema[1:] # si premier est maximum -> on déplace la borne au premier minimum
    if extrema[-1][1]==ofMax: extrema = extrema[:-1]# si dernier est maximum -> on déplace la borne au dernier minimum

    retour = []

    size = len(extrema)
    # note : le tableau commence et fini toujours par un minimum (ou max si ofMin ) ( au moins 3 éléments )
    if(size >= 3):
        for i in range(1, size, 2) :
            retour.append( [extrema[i-1][0], extrema[i][0], extrema[i+1][0]] )

    return retour



def peak_finder(data, x=None, max_nb_of_peaks=3, finesse=5, edge=0.1, withGradient=True, hidden_slope=[1, 33]):
    """
    spectrum : la courbe à analyser
    max_nb_of_peak : le nombre max de pic à détecter.
    finesse : the finesse value reflects the width of the expected peaks proportionally to the length of the spectrum, in percent.
    For example a finesse of 5 will only detect peaks larger than 5% of the spectrum length.
    edge : pour effacer les effets de bords
    hidden_slope : (min, max) slope to find hidden peaks in % on the gradient
    """

    taille = len(data)
    frequencies = int(100/finesse)
    la_fourier = fourier(data, frequencies) # filtre passe bas aux fréquences faibles.

    pb = {}
    extrema = extrema_of( la_fourier )              # recherche des extrema

    result = []
    for p in create_plage_from_extrema( extrema ) : # transformation en plages
        result.append( Plage( p[0], p[1], p[2], ref=la_fourier, state='max' ) ) # création des Peaks

    gradient = None
    # ajout des pics par la méthode du gradient de la fourier
    if withGradient :

        la_fourier2 = fourier(data, max(3, int(frequencies*0.75)) ) # lissage fourier (détection des pentes, pas des pics)
        # gradient, cc, high_freq = fourier(gradient, max(2, int(frequencies/2))) # lissage gradient (détection des pentes, pas des pics)

        gradient = np.gradient(la_fourier2) if x is None else np.gradient(la_fourier2, x)
        # gradient = fourier(gradient, frequencies, withoutCC=False) # lissage gradient (détection des pentes, pas des pics)
        gradient = fourier(gradient, max([ 2, int(frequencies) ] )) # lissage gradient (détection des pentes, pas des pics)

        extrema = extrema_of( gradient )

        abs_gradient = np.abs(gradient) # /!\ np.abs ne marche qu'avec le rfft ( pas de complexes )
        slope = np.max(abs_gradient) * hidden_slope
        hidden_plages = []
        if len(result) :
            iParent, parent, maxParent = 0, result[0], len(result) -1
        else :
            iParent, parent, maxParent = 0, None, -1
        for i in range(1, len(extrema)-1) :
            x = extrema[i][0]
            isMax = extrema[i][1]
            while( parent and not parent.hasInside(x) ) :
                if iParent < maxParent :
                    iParent += 1
                    parent = result[iParent]
                else : # iParent >= maxParent. no other plages
                    parent = None
            if( slope[0] < abs_gradient[x] < slope[1] ) :
                if ( (gradient[x] > 0) ^ isMax ) :
                    hidden_plages.append( Plage(
                        extrema[i-1][0],
                        x,
                        extrema[i+1][0],

                        ref=gradient,
                        state='hidden',
                        parent=parent.dichotomy(x) if parent else None
                    ))
        #hidden_plages = filter( lambda plage: filtre_plage_lineaire(plage, gradient, variation), hidden_plages )
        #hidden_plages = filter( f_win, hidden_plages )
        result += hidden_plages

    # edge filter left & right
    borneA = int( edge * taille )
    borneB = taille-borneA

    edged = []
    for p in result:
        if not ( borneA < p.milieu < borneB ) :
            p.addState('edge')
            edged.append(p)
    if len(edged) :
        pb['edge'] = edged

    # test overflow
    taille = len(result) - len(edged)
    if taille >= max_nb_of_peaks : pb['overflow'] = taille

    return result, la_fourier, gradient, pb
