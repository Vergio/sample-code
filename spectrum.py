#!/usr/bin/python2.7
#-*- coding: utf-8 -*-
print('import spectrum : listSpectrum, Spectrum, f_sort_big_peaks,_arrayOfNone')

import time
import inspect      # to debug lambda : inspect.getsource(myfunc) return the code used

from general        import *
from peak           import Peak
from peak_search    import peak_finder
from model          import Model
from modelisedPeak  import ModelisedPeak

import spectrumBuilder as sb

listSpectrum = set()

# TODO
# fourier, gradient, polynome -> tableau à chaque niveau de modélisation.
# ajouter un statut montrant le niveau de modélisation obtenu
#
# optimisation : modeliser a la volee
#
# Définir les ordres successifs de fit par le nombre d'éléments dans le PARAM['get models']
#
# Décomposer en decory d'un Spectrum pour les fits
# re-définir le background
# -> doit re-calculer les fit si re-définition du background
#
# Sous-spectrum -> extrais sur des plages différentes (ex : x-ray)
# Spectrum correlés -> cooling / warming des hystérésis
# -> ils doivent avoir le même background


f_sort_big_peaks = attrgetter('area') #lambda x: x.area
_arrayOfNone = [None]*4 # for empty _fit array

def correctData (dataX, spectrum, param) :
    """ Effectue une vérification et une correction de données entre abscices et ordonnées

        - Assure que la longueure des tableaux est la même. (assert)
        - Élimine les valeurs possédant un nan (en abscices ou ordonnée)
        - Si 'search_pic', ne conserve qu'un nombre pair d'éléments. (élimine le dernier si ça n'est pas le cas)

        Interdit ensuite la modification des tableaux numpy retournées.
        ( assure la non modification des données brutes )
    """
    L = len(spectrum)
    assert L and L==len(dataX), 'dataX and spectrum must be the same length'

    notNan = ~np.isnan(spectrum) & ~np.isnan(dataX)
    if not np.all(notNan) :
        dataX    = dataX[notNan]
        spectrum = spectrum[notNan]
        L = len(spectrum)

    if L%2 and 'search peak' in param and param['search peak'] :
        # nécessaire pour la fonction fourier, qui sinon retourne un élément de moins
        dataX    = dataX[:-1]
        spectrum = spectrum[:-1]

    dataX.flags.writeable    = False
    spectrum.flags.writeable = False

    return dataX, spectrum

def calcArea(x,y) :
    """ Calcule l'aire entre deux courbes """
    return np.abs( np.trapz( np.abs(y), x))

class Spectrum(object):
    """ Objet rassemblant un tableau de données (abscices/ordonnées),
        Et effectuant une analyse de ses pics. (recherche de pic et fit)
    """

    __slots__ = (
        'id',
        'creation',
        'coordinates',
        'units',
        'fileInfo',
        'PARAM',
        'spectrum',
        'dataX',
        '_f_nrj',
        'bg',
        'bg_param',
        '_area',
        '_peaks',
        '_fitting_level', # { level, validPeaks:[] }
        'big_peaks',
        'virtualPeaks',
        '_spectrum_norm',
        '_spectrum_bgSub',
        '_spectrum_bgSub_norm',
        '_fourier',
        '_gradient',
        '_fit',      # [ [f_fit, fit, diff, div], [f_fit, fit, diff, div], …   ]
        '_firstFit', # [f_fit, fit, diff, div]
        '_secondFit',# [f_fit, fit, diff, div]
        '_polynome',
        'plages',
        'warnings'
    )

    # ======================================================================== #
    #                               Initialisation
    # ======================================================================== #

    def __init__(self, dataX, spectrum, coordinates, units, fileInfo, param, builder=None):
        """ Initialise un spectrum
            - dataX     : tableau des abscices
            - spectrum  : tableau des ordonnées
            - coordonées du spectrum dans un ensemble de sepctrum ( tuple si plusieurs dimenssions )
            - units     : unités abscisses / ordonnées ( tuple x,y )
            - fileInfo  : méta-données du fichier d'extraction. (souvent commun à plusieurs spectrum)
            - param     : Objet de paramettrage
            - builder   : (defaut None), fabrique externe utilisée pour initialiser le spectrum.

        """
        dataX, spectrum = correctData(dataX, spectrum, param)

        listSpectrum.add(self)
        self.id             = get_new_id()
        self.creation       = time.time()
        self.warnings       = {}
        self.coordinates    = coordinates
        self.units          = units
        self.fileInfo       = fileInfo
        self.PARAM          = param
        self.spectrum       = spectrum
        self.dataX          = dataX
        self._firstFit      = None
        self._secondFit     = None
        self._fitting_level = { 'level':0, 'validPeaks':[] }
        self._peaks         = []
        self.big_peaks      = []
        self.virtualPeaks   = []
        self.plages         = set()

        if isinstance(builder, sb.SpectrumBuilder):
            builder.initialise( self )
        else :
            # modelisation
            self.build_f_nrj()

            f_bg = self.PARAM['f_extractBg'](dataX, spectrum)
            if isinstance(f_bg,tuple) :
                f_bg, info = f_bg
            else :
                info = { 'code':inspect.getsource(self.PARAM['f_extractBg']) }
            self.initalise_bg( f_bg, info )

        if not self.PARAM['testing'] :
            self.clearMemory()

    def initalise_bg (self, f_bg, param=None) :
        """ Set la fonction de background et les paramètres,
            et (re)-lance l'initialisation des pics et du fit
        """

        #re-initalise
        self.clearMemory()

        # sub background
        self.bg       = f_bg
        self.bg_param = param if param!=None else { 'code' : inspect.getsource(f_bg) }

        # defines self.plages, self.fourier, self.gradient, self.polynome before create Peaks
        self.initialisePeaks()
        self.initialiseFit()

    def clearMemory(self):
        """
        clear cache
            area : with and without bg
            polynome
            fourier
            gradient
            spectrum : norm, without bg and both
        """
        if hasattr(self, '_area')               : del self._area
        if hasattr(self, '_polynome')           : del self._polynome
        if hasattr(self, '_fourier')            : del self._fourier
        if hasattr(self, '_gradient')           : del self._gradient
        if hasattr(self, '_spectrum_norm')      : del self._spectrum_norm
        if hasattr(self, '_spectrum_bgSub')     : del self._spectrum_bgSub
        if hasattr(self, '_spectrum_bgSub_norm'): del self._spectrum_bgSub_norm

    def initialisePeaks(self) :
        """ Efface les pics, et si la recherche est autorisé en paramètre,
            (re-)effectue la recherche des plages puis lance la création des objets pics à partir de ces hidden_plages
            (cf create_main_peaks)
        """

        searchPeak = self.PARAM['search peak']

        # discard and re-initialise Peaks for re-initialisation
        for p in self._peaks : p.discard()
        self._peaks     = []
        self.plages    = set()
        self.big_peaks = []
        self.fitting_level = 0 # reset fitting_level and validPeaks

        if searchPeak :
            plages = self.peak_finder()    # defines self.plages, self.fourier  and self.gradient
            self.create_main_peaks(plages) # defines self.polynome, before create Peaks

    def initialiseFit(self):
        """ Effectue le fit des pics à partir des modèles en paramètre.

            Effectue un fit de premier ordre par modèle en paramètre.
            puis orchestre le lancement d'un second ordre si le premier n'est pas suffisant.

            Pour l'instant, un seul fit de premier ordre, et pas de second ordre.
        """
        # TODO meilleur fitting
        #   parcourir le tableau
        #       1er ordre :
        #           effectuer un fit par GetModels
        #           choisir le modèle le plus adapté
        #       2eme ordre :
        #           effectuer le fit sur la différence / division ?
        #           effectuer le fit sur chaque pic modélisé du premier ordre ?

        getModelsList = self.PARAM['get models'] if 'get models' in self.PARAM else None
        if isinstance(getModelsList, (list, tuple, set) ) :
            for i, f_getModel in enumerate(getModelsList) :
                self.first_order_model(f_getModel, i) # defines _firstFit
                break # only 1 if exist for the moment

        return # no second order for the moment
        if(not self.has_validFirstFitting()):
            self.second_order_model()


    def build_f_nrj(self):
        """ Construit la fonction de conversion des abscices-valeur en abscices-indices
            A supprimer pour utiliser plutôt une définition par Builder
        """
        # La mesure en longueure d'onde est linéraire. ( a.x + b )
        # Donc la mesure déduite en énergie est en 1/ (a.x + b)
        dataX = self.dataX
        if not 'ordered type' in self.PARAM or self.PARAM['ordered type'] == 'linear' :
            b = float(dataX[0])
            a = (float(dataX[-1]) - b)/float( len(dataX) )
            div = b/float(a)

            self._f_nrj = [
                lambda x: a*x + b,
                lambda y: int( y/a - div )
            ]
        elif self.param['ordered type'] == 'invert' :
            # La mesure en longueure d'onde est linéraire. ( a.x + b )
            # Donc la mesure déduite en énergie est en 1/ (a.x + b)
            b = 1/float(dataX[0])
            a = (1/float(dataX[-1]) - b)/float( len(dataX) )
            div = b/float(a)

            self._f_nrj = [
                lambda x: 1/(a*x + b),         # y = 1/( a.x + b )
                lambda y: int( 1/(a*y) - div ) # x = ( 1-b.y ) / a.y
            ]

    # ======================================================================== #
    #                   Méthodes accesseurs ou raccourcis                      #
    # ======================================================================== #

    def get_info(self):
        """ ( Adapté "TP-OP phase transition for MAPbI3 films of varying grain size" )
            Renvois une chaine de caractères précisant les caractéristiques de l'échantillon
        """
        return '%s : %s %sK %s (%s) - %d'%(
            self.experiment,
            self.typeEchantillon,
            int(self.temperature) if self.temperature and hasattr(self, 'temperature') else np.nan,
            'cooling' if self.cooling else 'warming',
            str(self.coordinates),
            self.id,
        )

    def get_peaks           (self) : return list(self._peaks)
    def get_energy_array    (self) : return self.dataX
    def get_nbPeaks         (self) : return len(self.peaks)
    def get_nbPeaksWithFwhm (self) : return len( self.peaksWithFwhm )
    def get_nbValidPeaks    (self) : return len( self.validPeaks )
    def get_nbBigPeaks      (self) : return len( self.big_peaks )
    def get_typeEchantillon (self) : return self.fileInfo['type']
    def get_experiment      (self) : return self.fileInfo['experiment']
    def get_filepath        (self) : return self.fileInfo['filepath']
    def get_cooling         (self) : return self.fileInfo['cooling']
    def get_warming         (self) : return not self.fileInfo['cooling']
    def get_mapWidth        (self) :
        return self.fileInfo['map_w'] if 'map_w' in self.fileInfo else None
    def get_temperature     (self) :
        return self.fileInfo['temperature'] if 'temperature' in self.fileInfo else None
    def get_physicalQuantity(self) :
        return self.fileInfo['physical quantity'] if 'physical quantity' in self.fileInfo else None

    def get_x (self) :
        return self.coordinates[0] if hasattr(self,'coordinates') and len(self.coordinates)   else 0
    def get_y (self) :
        return self.coordinates[1] if hasattr(self,'coordinates') and len(self.coordinates)>1 else 0
    def get_z (self) :
        return self.coordinates[2] if hasattr(self,'coordinates') and len(self.coordinates)>2 else 0

    def is_typeEchantillon  (self, type_echantillon) :
        return type_echantillon==self.fileInfo['type']

    def get_area (self, withBg=False) :
        if not hasattr(self, '_area') :
            self._area = (
                calcArea(self.dataX, self.get_spectrum(withBg=False)),
                calcArea(self.dataX, self.get_spectrum(withBg=True)),
            )
        return self._area[ 1 if withBg else 0 ]

    def get_spectrum(self, withBg=False, norm=False):
        """ Calcul et renvois le tableau des ordonnées. ( tableau numpy non modifible )
            - withBg (False) : Sans le background
            - norm   (False) : Normalisé.

            Méthode utilisant des caches. ( _spectrum_norm, _spectrum_bgSub, _spectrum_bgSub_norm )
            Les tableaux en sortie ne sont calculés qu'une seule fois.
        """
        if withBg :
            if not norm :
                return self.spectrum
            if not hasattr(self, '_spectrum_norm') :
                self._spectrum_norm = np.array( funclib.norm(self.spectrum) )
                self._spectrum_norm.flags.writeable = False
            return self._spectrum_norm
        if not hasattr(self, '_spectrum_bgSub') :
            self._spectrum_bgSub = self.spectrum - self.bg(self.dataX)
            self._spectrum_bgSub.flags.writeable = False
        if not norm :
            return self._spectrum_bgSub
        if not hasattr(self, '_spectrum_bgSub_norm') :
            self._spectrum_bgSub_norm = np.array( funclib.norm( self._spectrum_bgSub ))
            self._spectrum_bgSub_norm.flags.writeable = False
        return self._spectrum_bgSub_norm

    def get_background(self):
        """ Renvois un tableau des ordonnées du background """
        return self.get_spectrum(withBg=True)-self.get_spectrum(withBg=False)

    def get_indiceFromNRJ(self, nrj):
        """ Convertit une abscice en valeur en une abscice en indice.
            ( Note : les abscices sont supposés monotones )

            Utilise pour ce faire une fonction définie lors de l'initialisation ( _f_nrj )
        """
        if np.isnan(nrj) : return np.nan
        return self._f_nrj[1](nrj)

    def get_peaksWithFwhm(self) : return filter(attrgetter('fwhm'), self._peaks)

    def addVirtualPeak(self, Peak) :
        """ Ajoute un pic virtuel à la liste des pics """
        self.virtualPeaks.append(Peak)
        self._peaks.append(Peak)

    def get_validPeaks(self, withVirtual=True) :
        """ Renvois une liste des pics valides
            - withVirtual (=True) : complète avec les pics virtuels
        """
        retour = [self._peaks[i] for i in self._fitting_level['validPeaks'][:]]
        if withVirtual : retour += self.virtualPeaks
        return retour



    def get_fitting_level(self): return self._fitting_level['level']
    def set_fitting_level(self, nb):
        """ Défini le nombres de pics à renvoyer dans le validPeaks,
            en les complétant le cas échéchéant par les hiddens pics

            (re)Définie le cache de la liste de validPeaks au moment du set.

            Note :
            Les pics non hiddens sont toujours renvoyés, quel que soit leur nombres
            Mettre à 0 le fitting_level permet de ne jamais renvoyer les hidden pics.
            Les hiddens pics ne sont pas "inventés" s'ils n'ont pas été détectés,
            ce mécanisme n'assure pas la présence du nombre de pic fixé dans les validPeaks

            Ce mécanisme permet d'incrémenter progressivement le nombre de hidden pic à utiliser lors d'un fit invalide.
        """
        assert isinstance(nb, int) and nb >= 0
        # peaks    = filter(lambda p: not p.hasState('invalid'), self._peaks)
        # maxPeaks = filter(lambda p: p.hasState('max'), peaks)
        # add      = filter( lambda p:p.hasState('hidden'), peaks )
        peaks    = []
        maxPeaks = []
        add      = []
        for p in self._peaks:
            if not p.hasState('invalid') :
                peaks.append(p)
                if p.hasState('max'):
                    maxPeaks.append(p)
                elif p.hasState('hidden'):
                    add.append(p)
        if nb :
            if len(add) < nb : nb = len(add)
            add = sorted( add, key=attrgetter('area'), reverse=True ) # lambda p:p.area
            maxPeaks += add[0:nb]

        maxPeaks.sort( key=attrgetter('xMax') ) # lambda p:p.xMax
        self._fitting_level['level']      = nb
        self._fitting_level['validPeaks'] = [self._peaks.index(p) for p in maxPeaks]

    def get_bothPeak(self, f_compare=False):
        both = self.big_peaks[:]
        if len(both)<2 or not f_compare :
            return both
        if f_compare(both[1], both[0]) : both.reverse()
        return both
    def get_firstPeak(self, f_compare=False):
        both = self.get_bothPeak(f_compare)
        return both[0] if len(both) else None
    def get_secondPeak(self, f_compare=False):
        both = self.get_bothPeak(f_compare)
        return both[1] if len(both)>1 else None

    def get_bothPeakHeight  (self): return self.get_bothPeak  (lambda p1,p2 : cmp(p1.yMax, p2.yMax) )
    def get_firstPeakHeight (self): return self.get_firstPeak (lambda p1,p2 : cmp(p1.yMax, p2.yMax) )
    def get_secondPeakHeight(self): return self.get_secondPeak(lambda p1,p2 : cmp(p1.yMax, p2.yMax) )

    def get_bothPeakNRJ     (self): return self.get_bothPeak  (lambda p1,p2 : cmp(p1._xMax, p2._xMax ) )
    def get_firstPeakNRJ    (self): return self.get_firstPeak (lambda p1,p2 : cmp(p1._xMax, p2._xMax) )
    def get_secondPeakNRJ   (self): return self.get_secondPeak(lambda p1,p2 : cmp(p1._xMax, p2._xMax) )

    def get_bothPeakWidth   (self): return self.get_bothPeak  (lambda p1,p2 : cmp(p1.fwhm, p2.fwhm ) )
    def get_firstPeakWidth  (self): return self.get_firstPeak (lambda p1,p2 : cmp(p1.fwhm, p2.fwhm) )
    def get_secondPeakWidth (self): return self.get_secondPeak(lambda p1,p2 : cmp(p1.fwhm, p2.fwhm) )

    def get_bothPeakSigma   (self): return self.get_bothPeak  (lambda p1,p2 : cmp(p1.sigma, p2.sigma ) )
    def get_firstPeakSigma  (self): return self.get_firstPeak (lambda p1,p2 : cmp(p1.sigma, p2.sigma) )
    def get_secondPeakSigma (self): return self.get_secondPeak(lambda p1,p2 : cmp(p1.sigma, p2.sigma) )

    def get_goodPeaksFilters(self):
        """ ( Adapté "TP-OP phase transition for MAPbI3 films of varying grain size" )
            Renvois les filtres de phases définis en paramètres
            ( filtre définisant si un pic fait ou non partie d'une phase )
        """
        # TODO : rendre générique cette recherche des filtres dans les params

        key = ( self.cooling,  self.temperature )
        filters = self.PARAM['goodPeaks'][self.experiment][self.typeEchantillon]
        if key in filters :
            filters = filters[key]
        elif 'default' in filters :
            filters = filters['default']
        else:
            raise Exception('filter not defined')
        return filters

    def get_goodPeaks (self, indice=None):
        """ Renvois l'ensemble des pics valides inclus dans les phases définies dans les paramètres
            - indice (=None) : sélection d'une phase particulière
                L'indice dépend de la définition des filtres de phases dans les paramètres.
                ( numérique pour une liste, només pour les dictionnaires )
        """
        validPeaks = self.validPeaks
        filters = self.get_goodPeaksFilters()
        result = set()

        if indice == None :
            if isinstance(filters, dict) : filters = filters.values()
            for f in filters :
                result.update( filter(f, validPeaks) )
        elif isinstance(filters, dict):
            if indice in filters :
                result.update(filter(filters[indice], validPeaks))
        elif indice < len(filters):
            result.update(filter(filters[indice], validPeaks))

        return result


    #### Fonctions de modelisation ####

    def has_firstFit(self):         return bool(np.any(self._firstFit))
    def get_firstFit(self):         return (self._firstFit or _arrayOfNone)[1]
    def get_firstFitFunction(self): return (self._firstFit or _arrayOfNone)[0]
    def get_firstFitDiff(self):     return (self._firstFit or _arrayOfNone)[2]
    def get_firstFitDiv(self):      return (self._firstFit or _arrayOfNone)[3]
    def has_validFirstFitting(self):return bool(np.any(self._firstFit)) and self._firstFit[2] < self.PARAM['validFitting']

    def has_secondFit(self):        return bool(np.any(self._secondFit))
    def get_secondFit(self):        return (self._secondFit or _arrayOfNone)[1]
    def get_secondFitFunction(self):return (self._secondFit or _arrayOfNone)[0]
    def get_secondFitDiff(self):    return (self._secondFit or _arrayOfNone)[2]
    def get_secondFitDiv(self):     return (self._secondFit or _arrayOfNone)[3]
    def has_validSecondFitting(self):return bool(np.any(self._secondFit)) and self._secondFit[2] < self.PARAM['validFitting']

    def has_fit(self):        return bool(np.any(self._firstFit) or np.any(self._secondFit))
    def get_fit(self):        return (self._secondFit or self._firstFit or _arrayOfNone)[1]
    def get_fitFunction(self):return (self._secondFit or self._firstFit or _arrayOfNone)[0]
    def get_fitDiff(self):    return (self._secondFit or self._firstFit or _arrayOfNone)[2]
    def get_fitDiv(self):     return (self._secondFit or self._firstFit or _arrayOfNone)[3]

    def has_validFitting(self):
        return self.has_fit() and (self._secondFit or self._firstFit)[2] < self.PARAM['validFitting']


    # ======================================================================== #
    #                         Méthodes de modélisation                         #
    # ======================================================================== #


    def get_polynome(self,withBg=False):
        """ Calcul et renvois les ordonnées d'une régression polynomiale
            - withBg (False) : sans le background

            Utilise un cache (_polynome)
            Ne calcule qu'une seule fois le polynome
        """
        if not hasattr(self,'_polynome') or not np.any(self._polynome) :
            spectrum = self.get_spectrum(withBg=False)
            poly_order = self.PARAM['poly_order']
            smoothing_window = int( len(spectrum)*self.PARAM['search peak finesse']/500 )
            if smoothing_window < 2*poly_order :
                smoothing_window = 2*poly_order+1
            elif not smoothing_window%2 :
                smoothing_window += 1 # must be odd
            self.set_ploynome( savitzky_golay(spectrum, smoothing_window, poly_order) )
        return self._polynome

    def set_ploynome(self, polynome) :
        """ Set le cache des ordonnées de la régression polynomiale (et verrouille les modifications du tableau) """
        self._polynome = np.array(polynome)
        self._polynome.flags.writeable = False

    def get_fourier(self):
        """ Calcul et renvois les ordonnées d'un lissage par fourier

            Utilise un cache (_fourier)
            Ne calcule qu'une seule fois le filtrage
        """
        if not hasattr(self, '_fourier') or not np.any(self._fourier) :
            self.set_fourier( fourier(self.spectrum, self.PARAM['max_nb_of_peaks']) )
        return self._fourier

    def set_fourier(self, fourier) :
        """ set le cache des ordonnées du filtrage par fourier (et verrouille les modifications du tableau) """
        self._fourier = np.array(fourier)
        self._fourier.flags.writeable = False

    def get_gradient(self):
        """ Calcule et renvois le gradiant d'un lissage par fourier des ordonnées

            Utilise un cache (_gradient), ainsi que la méthodes get_fourier elle-même cachée
        """
        if not hasattr(self, '_gradient') or not np.any(self._gradient) :
            gradient = np.gradient(self.get_fourier(), self.dataX )
            # TODO l'échantillonnage n'est pas linéaire car convertit en 1/xself.
            # Il faudrait multiplier ce gradient par une fonction qui compenserait la variation de l'échantillonage.
            self.set_gradient(  gradient )
        return self._gradient

    def set_gradient(self, gradient):
        """ set le cache des ordonnées du gradient (et verrouille les modifications du tableau)"""
        self._gradient = np.array(gradient)
        self._gradient.flags.writeable = False


    def peak_finder(self, cache=True):
        """ Effectue une recherche de pic et renvois une liste de plage potentielle
            Utilise la fonction "peak_finder"

            - cache (=True) : Pour optimisation, initialise le cache de la fourier et du gradient
            par ceux calculé lors de la recherche de pic.
            PRUDENCE :
            Les méthodes définies dans get_fourier ou get_gradient, initialement identiques
            à celles utilisées dans peak_finder peuvent avoir évoluée différement.

            Note : Ne fait qu'effectuer la recherche, ne modifie pas les plages du spectrum.
        """

        plages, fourier, gradient, pb = peak_finder(
            self.get_spectrum(withBg=False),
            x               = self.dataX,
            finesse         = self.PARAM['search peak finesse'],
            max_nb_of_peaks = self.PARAM['max_nb_of_peaks'],
            edge            = self.PARAM['edge suppression'],
            hidden_slope    = self.PARAM['hidden_peak_slope']
        )
        if cache :
            self.set_fourier(fourier)
            self.set_gradient(gradient)

        if 'edge' in  pb :
            self.addWarning('peak finder', 'plage with max in edge', pb['edge'])
        if 'overflow' in pb :
            self.addWarning('peak finder', 'too many peaks found (%d/%d)'%(pb['overflow'], self.PARAM['max_nb_of_peaks']))
        return plages

    def create_main_peaks(self, plages):
        """ Crée les objets Peak à partir des plages fournies

            Effectue une validation des plages, et effectue parfois des fusions de plages
        """
        # TODO développer la description.

        plages = set(plages)
        # reset peaks id allready done
        self.plages    = plages
        self._peaks     = []
        self.big_peaks = []

        # ------- invalidation avant création ------- #

        hidden   = []
        notHidden= []
        width_limit = 0.5*self.PARAM['search peak finesse'] / float(100)

        if len(plages) :
            spectrum = self.get_spectrum(withBg=False, norm=True) # without Bg and normalized
            for p in plages:
                invalid    = p.hasState('invalid')
                edge       = p.hasState('edge')
                flat_limit = p.middle_angle( spectrum ) > self.PARAM['flat limit']
                too_thin   = p.get_width(norm=True) < width_limit
                is_hidden  = p.hasState('hidden')

                if edge or flat_limit or too_thin or invalid :
                    if not invalid: p.addState('invalid')
                    if flat_limit : p.addState('flat limit')
                    if too_thin   : p.addState('width too thin')
                elif is_hidden : hidden.append( p )
                else : notHidden.append(p)

        '''
        for cause, f in {
            'edge'         : methodcaller('hasState', 'edge') # lambda p:p.hasState('edge'),
            'flat limit'   : lambda p:p.middle_angle( self.get_spectrum(withBg=False, norm=True) ) > self.PARAM['flat limit'],
            'width to thin': lambda p:p.get_width(norm=True) < width_limit,
        }.items() :
            for p in filter(f, plages) :
                p.addState( cause, 'invalid')

        plages    = set( filter(lambda p: not p.hasState('invalid'), plages ) )
        notHidden = set( filter(lambda p: not p.hasState('hidden'),  plages ) )
        plages.difference_update(notHidden)
        '''
        # test max trop proches. ( 1/4 de search peak finesse, en indices )
        # TODO améliorer la recherche de proximité

        search = notHidden + hidden
        if len(search) > 1 :
            search.sort(key = attrgetter('milieu')) #lambda p:p.milieu)
            width_limit = 0.25 * len( self.spectrum ) * self.PARAM['search peak finesse'] / float(100)
            tooClose = []

            taille = len(search)
            i=0
            while( i < taille - 1 ):
                i += 1
                groupe = [search[i-1]]
                while( i < taille and search[i].milieu < search[i-1].milieu + width_limit ):
                    groupe.append( search[i] )
                    i += 1
                if len(groupe)>1 :
                    tooClose.append(groupe)

            # fusion des groupes
            # TODO fusioner les plages et les placer comme parent ?
            #      -> mais pb sur l'analyse des Peaks qui ont besoin d'une plage lisse
            if len(tooClose) :
                for groupe in tooClose :
                    width = groupe[-1].milieu - groupe[0].milieu

                    # sort : max prioritaire sur hidden, plage plus grande prioritaire
                    # method : decorate, sort, undecorate
                    decorated = [(int(plage.hasState('max')), plage.width, i, plage) for i, plage in enumerate(groupe)]
                    decorated.sort()
                    groupe = [plage for state, width, i, plage in decorated]

                    if width < width_limit :
                        for i in range(0,len(groupe)-1): # keep the last
                            groupe[i].addState('too close', 'invalid')
                    else :
                        while len(groupe) :
                            good = groupe.pop() # take and keep the last. it will not be invalidate
                            good = good.milieu
                            for plage in groupe : # invalid too close plage of this ( and first plages before… but its not usefull here)
                                if np.abs(plage.milieu - good) < width_limit :
                                    groupe.remove(plage)
                                    plage.addState('too close', 'invalid')

        # ------- création des Peaks ------- #

        for plage in notHidden :
            try:
                self._peaks.append(Peak( Spectrum=self, plage=plage ))
            except Exception as error :
                e_type, value, tb = sys.exc_info()
                self.addWarning('Exception in create Peak', {
                    'plage':plage,
                    'error':error,
                    'traceback':traceback.format_tb(tb)
                })

        for plage in hidden :
            try:
                self._peaks.append(Peak( Spectrum=self, plage=plage ))
            except Exception as error :
                e_type, value, tb = sys.exc_info()
                self.addWarning('Exception in create Peak', {
                    'plage':plage,
                    'error':error,
                    'traceback':traceback.format_tb(tb)
                })

        # ------- invalidation après création ------- #
        if not len(self._peaks):
            self.addWarning('no peak fund', self.plages)
        else :
            # limit_area = self.area*self.PARAM['area_selected']
            limit_height = np.max( map( attrgetter('yMax'), self._peaks ) ) * self.PARAM['peaks min-height']
            for cause, f in {
                # 'area-limited' : lambda p : p.area < limit_area
                'micro-peak': lambda p: p.yMax < limit_height
            }.items():
                for p in filter(f, self._peaks) :
                    p.addState( cause, 'invalid')

        # ------- finalisation ------- #

        self.fitting_level = 0 # reset fitting_level and validPeaks
        self.sortPeaks()

    def first_order_model(self, f_getModel, order=0):
        """ Effectue un fit à partir d'un modèle """

        self.fitting_level = 0 # reset fitting_level and validPeaks
        avant = 0

        # read PARAM
        maxNbOfPeaks = self.PARAM['max_nb_of_peaks']

        firstFit = [None, None, None, None]
        cherche = True
        i = 0
        while( cherche and i<100 ): # i : anti-infinite-loop
            i +=1
            self.cleanPeaks()
            model = f_getModel if isinstance(f_getModel, Model) else f_getModel(self)
            try:
                model.fit(self, transform=True)

                firstFit[0] = model.saveFit()
                firstFit[1] = firstFit[0]( self.dataX )
                spectrum = self.get_spectrum(withBg=False)
                firstFit[2] = 100*calcArea(self.dataX, spectrum - firstFit[1] ) / float(self.area)
                firstFit[3] = 100*np.mean( np.abs( (firstFit[1]/spectrum) -1 ) )

            except Exception as error:
                e_type, value, tb = sys.exc_info()
                self.addWarning('Exception in first order modelisation', {
                    'model used':model,
                    'error':error,
                    'traceback':traceback.format_tb(tb)
                })
                cherche = False
            else:
                self._firstFit = firstFit
                self.sortPeaks()

                if(self.has_validFirstFitting()):
                    cherche = False
                elif( self.nbValidPeaks >= maxNbOfPeaks ) :
                    cherche = False
                    self.addWarning('first fitting failed', 'max_nb_of_peaks reached' )
                else:
                    avant += 1
                    self.fitting_level += 1
                    if( avant != self.fitting_level ):
                        cherche = False
                        self.addWarning('first fitting failed', 'not enough hidden peaks to continue' )


    def second_order_model(self, get_models):
        """ Effectue un fit de deuxième ordre

            ( second ordre : à partir d'une différence d'avec le premier ordre )
        """
        if( len(self.PARAM['get models'])<2) : return
        self._secondFit = None

        # self.second_fit = model.saveFit()

        # 2ème ordre en différence ( spectrum - 1er ordre )
        # -> détection des pics cachés.

        # 2ème ordre en division ( spectrum / ( 1er ordre + 2ème ordre en différence) )
        # -> détection d'une enveloppe ( exponentielle ? )

        # self.sortPeaks()


    # ======================================================================== #
    #                              Autres Méthodes                             #
    # ======================================================================== #


    def addWarning(self, cause, valeur, ref=None):
        if ref==None : ref = self
        if not cause in self.warnings:
            self.warnings[cause] = [{'val':valeur,'ref':ref}]
        else:
            self.warnings[cause].append({'val':valeur,'ref':ref})


    def reload(self):
        self.__init__(self.dataX, self.spectrum, self.x, self.y, self.fileInfo)

    def discard(self):
        """ S'efface de la liste des spectrums, et efface ses pics de la listes des peaks
            Permet de liébérer de la mémoire.
        """
        self.cleanPeaks()
        for p in self._peaks :
            p.discard()
        listSpectrum.discard(self)

    def sortPeaks(self):
        self.big_peaks = []
        if len(self._peaks) :
            self._peaks.sort( key=f_sort_big_peaks, reverse=True )
            self.fitting_level = self.fitting_level # reset validPeaks
            area = self.PARAM['area_selected'] * self._peaks[0].area
            for p in sorted( self.validPeaks, key=f_sort_big_peaks, reverse=True ) :
                if p.area >= area :
                    self.big_peaks.append(p)

    def replacePeak(self, replacePeak) :
        """ Permet de remplacer une instance de pic détenu par le spectrum par un de ses decory
            ( note, les pics doivent être "égaux" selon la méthode .equals )
        """
        assert isinstance( replacePeak, Peak ), 'replacePeak is not a %s (%s)'%(Peak, type(replacePeak))
        for i in range(0, len(self._peaks)) :
            if self._peaks[i].equals(replacePeak) :
                self._peaks[i] = replacePeak


    def cleanPeaks(self):
        """ Re référence les originaux des pics et non leur decory (fitté), et les re-trie

            Note : Les décory fitté des pics seront libérés de la mémoire s'ils ne sont pas utilisé ailleurs
        """
        for i in range(0, len(self._peaks)):
            if self._peaks[i].isDecory() :
                self._peaks[i] = self._peaks[i].get_originalPeak(True)
        self.sortPeaks()

    def __getattr__(self, name):
        if name not in self.fileInfo :
            raise AttributeError('%s not in spectrum %s'%(name, self.id))
        else:
            return self.fileInfo[name]


    # ======================================================================== #
    #                               Accesseurs                                 #
    # ======================================================================== #


    info                = property( get_info )
    typeDataObject      = 'Spectrum'
    x                   = property( get_x )
    y                   = property( get_y )
    z                   = property( get_z )
    energy_array        = property( get_energy_array )
    peaks               = property( get_peaks )
    nbPeaks             = property( get_nbPeaks )
    nbPeaksWithFwhm     = property( get_nbPeaksWithFwhm )
    nbValidPeaks        = property( get_nbValidPeaks )
    nbBigPeaks          = property( get_nbBigPeaks )
    typeEchantillon     = property( get_typeEchantillon )
    temperature         = property( get_temperature )
    experiment          = property( get_experiment )
    physicalQuantity    = property( get_physicalQuantity )
    filepath            = property( get_filepath )
    cooling             = property( get_cooling )
    mapWidth            = property( get_mapWidth )
    warming             = property( get_warming )

    polynome            = property( get_polynome )
    fourier             = property( get_fourier )
    gradient            = property( get_gradient )
    area                = property( get_area )

    peaksWithFwhm       = property ( get_peaksWithFwhm )
    validPeaks          = property ( get_validPeaks )
    fitting_level       = property ( get_fitting_level, set_fitting_level )

    bothPeakHeight      = property( get_bothPeakHeight )
    bothPeakNRJ         = property( get_bothPeakNRJ )
    bothPeakWidth       = property( get_bothPeakWidth )
    bothPeak            = property( get_bothPeakNRJ )

    firstPeakHeight     = property( get_firstPeakHeight )
    secondPeakHeight    = property( get_secondPeakHeight )
    firstPeakNRJ        = property( get_firstPeakNRJ )
    secondPeakNRJ       = property( get_secondPeakNRJ )
    firstPeakWidth      = property( get_firstPeakWidth )
    secondPeakWidth     = property( get_secondPeakWidth )
    firstPeak           = property( get_firstPeak )
    secondPeak          = property( get_secondPeak )

    is_firstFitted      = property( has_firstFit )
    firstFit            = property( get_firstFit )
    f_firstFit          = property( get_firstFitFunction )
    firstFitDiff        = property( get_firstFitDiff )
    firstFitDiv         = property( get_firstFitDiv )

    is_secondFitted     = property( has_secondFit )
    secondFit           = property( get_secondFit )
    f_secondFit         = property( get_secondFitFunction )
    secondFitDiff       = property( get_secondFitDiff )
    secondFitDiv        = property( get_secondFitDiv )

    is_fitted           = property ( has_fit )
    fit                 = property ( get_fit )
    f_fit               = property ( get_fitFunction )
    fitDiff             = property ( get_fitDiff )
    fitDiv              = property ( get_fitDiv )
    is_validFitted      = property ( has_validFitting )

    goodPeaks           = property ( get_goodPeaks )




    # ============================================ #

    # ================ plot ====================== #

    def plotSpectrum(self, withBg=False, color=None, label=''):
        return plt.plot(self.dataX, self.get_spectrum(withBg), color=color, label=label )

    def plotBg(self, color=None, label=''):
        return plt.plot(self.dataX, self.bg(self.dataX), color=color, label=label )

    def plotPolynome(self,withBg=False, color=None, label=''):
        return plt.plot(self.dataX, self.get_polynome(withBg), color=color, label=label )

    def plotFourier(self, color=None, label=''):
        fourier = self.fourier
        fourier = fourier - min(fourier)
        fourier = funclib.norm(fourier)*np.max(self.get_spectrum(False))
        return plt.plot(self.dataX, fourier, color=color, label=label )

    def plotGradient(self, color=None, label='', factor=False):
        gradient = self.get_gradient()
        if factor : gradient *= max(self.spectrum)
        return plt.plot(self.dataX, gradient, color=color, label=label)

    def plotFit(self, color=None, label='', withPeaks=True, extrapol=0):
        plot = []
        if self.is_fitted :
            dataX, fit = self.dataX, self.fit
            if extrapol > 1 :
                dataX = np.linspace(dataX[0], dataX[-1], extrapol*len(dataX))
                fit   = self.f_fit(dataX)
            plot += plt.plot( dataX, fit, label=label, color=color )
            if withPeaks and len(self.validPeaks)>1 :
                for p in self.validPeaks:
                    plot += p.plotFit(extrapol=extrapol)
        return plot



    def plotMax(self, withFwhm=True, marker='X', color='r', withVertical=True, withDots=True):
        tab = self.peaksWithFwhm if withFwhm else self._peaks
        plot = []
        for peak in tab:
            plot += peak.plotMax(marker, color, withVertical, withDots)
        return plot

    def plotFwhm(self, marker='X', color='g', withVertical=True, withDots=True):
        plot = []
        for peak in self.peaksWithFwhm :
            plot += peak.plotFwhm(marker, color, withVertical, withDots)
        return plot

    def plot(self, withBg=False, spectrumOnly = False, withFwhm=True, withMax=True, withVertical=True, withDots=True, withPeakFit=True, withPoly=False, withFourier=False, withFit=True, label=''):
        plot = self.plotSpectrum(withBg=withBg, label=label)
        if withFit :     plot += self.plotFit(withPeaks=False, extrapol=float(withFit) )
        if withPoly :    plot += self.plotPolynome()
        if withFourier : plot += self.plotFourier()
        if not spectrumOnly:
            if withPeakFit :
                for p in self.validPeaks:
                    plot += p.plotFit(extrapol=float(withPeakFit))
            if withMax  : plot += self.plotMax( withFwhm=withFwhm, withVertical=withVertical, withDots=withDots)
            if withFwhm : plot += self.plotFwhm( withVertical=withVertical, withDots=withDots )
        return plot


    def imshowMap(self, f_map, f_filter=False, matrice=True, vmin=None, vmax=None, colorbar=True) :
    	import visualisation.mapping as mapping

        matrice = mapping.get_fileData(
            self.experiment,
            data_type=self.typeEchantillon,
            temperature=self.temperature,
            cooling=self.cooling,

            f_filter=f_filter,
            f_map=f_map,
            matrice=matrice
        )
        im = plt.imshow(matrice, vmax=vmax, vmin=vmin)
        if colorbar : plt.colorbar(im, fraction=0.046, pad=0.04)
        return im


    def printWarnings(self):
        pp.pprint(self.warnings)

    def hasWarning( self, causes=False, filterCauses=[], details=False ):
        if isinstance(causes, str) : causes = [causes]
        if details : # return array of matched causes
            warnings = self.warnings.keys()
            if filterCauses :
                for c in filterCauses :
                    if c in warnings :
                        warnings.remove(c)
            if causes :
                for c in warnings :
                    if not c in causes :
                        warnings.remove(c)
            return warnings
        else : # return bool

            if not causes :
                warnings = self.warnings.keys()
                if filterCauses :
                    for c in filterCauses :
                        if c in warnings :
                            warnings.remove(c)
                return len(warnings) > 0

            # if causes
            for c in causes :
                if filterCauses and c in filterCauses :
                    raise Exception( 'search warning : %s in causes and filter cause'%(c) )
                if c in self.warnings :
                    return True
            return False

    def __repr__(self):return "Spectrum() %s"%(self.info)
    def __str__(self): return self.info
