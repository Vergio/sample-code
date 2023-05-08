#!/usr/bin/python2.7
#-*- coding: utf-8 -*-
print('import peak : Peak, listPeak')

from general import *
from plage import Plage
import math

listPeak = set()

sigma_factor = 1/float(2*math.sqrt(2*math.log(2)))

class Peak(object):
    __slots__ = (
        'id',
        'plage',
        '_xMax',
        '_xLeft',
        '_xRight',
        '_fitted_values',
        'quality',
        'halfMax',
        'Spectrum',
        'warnings',
        '_dataX',
        '_extract',
        '_polynome'
    )

    # TODO
    # si max -> identique.
    #
    # si hidden peak :
    # - détection plateaux : la plus patite valeur de la dérivée [ debut / fin ] dépasse un minimum.
    # - pics confondus : pas trop proche d'un max.

    def __init__(self, Spectrum, plage):
        listPeak.add(self)

        self.id = get_new_id()
        self.warnings = {}
        self.Spectrum = Spectrum
        self._fitted_values = [np.nan,np.nan,np.nan]
        self.initialiseFromPlage(plage)
        self.testTolerance()


    def initialiseFromPlage(self, plage) :
        assert isinstance( plage, Plage )
        if hasattr(self, '_xMax')   : del self._xMax
        if hasattr(self, '_xLeft')  : del self._xLeft
        if hasattr(self, '_xRight') : del self._xRight
        if hasattr(self, 'quality') : del self.quality
        if hasattr(self, 'halfMax') : del self.halfMax
        if hasattr(self, '_dataX')  : del self._dataX
        if hasattr(self, '_extract'): del self._extract
        if hasattr(self, '_polynome'):del self._polynome

        self.plage = plage
        debut, arg_maximum, fin = plage.debut, plage.milieu, plage.fin
        self.quality = self.Spectrum.PARAM['f_testQuality'](self.extract)
        self._xMax = arg_maximum

        # régression polynomiale

        # recalcul du maximum si signal / bruit de qualité
        #if self.quality and not plage.hasState('hidden') :
        #    polynome = self.polynome
        #    _xMax = np.where(polynome == polynome.max())[0][0] + debut
        #    if debut+1 > _xMax < fin -1 :
        #        self._xMax = _xMax
        #    else : # sinon
        #        self.addState('hidden') # si polynome monotone (alors que la fourier ne l'est pas) -> hidden peak
        #        self.removeState('max')

        # if peak at start or end of the plage, it's not a peak
        if  self._xMax <= debut or self._xMax >= fin-1 :
            self.warningNoPeak()
            self._xRight = np.nan
            self._xLeft  = np.nan
            self.halfMax = np.nan
            self.addState('invalid', 'peak on border')

        else :
            polynome = self.polynome
            # mi hauteur sur polynome
            _xMax = self._xMax-debut
            self.halfMax = 0.5*polynome[_xMax]

            # en indice
            leftSlice = np.abs( polynome[0:_xMax]- self.halfMax )
            val = np.min( leftSlice ) # np.min car monotone sur le slice (même si la polynome est bruitée )
            self._xLeft = debut + np.where( leftSlice == val )[0][0]

            rightSlice = np.abs( polynome[_xMax:]- self.halfMax )
            val = np.min( rightSlice )
            self._xRight  = np.where( rightSlice == val )[0][0] + self._xMax
            # after here can use auto-calcul type : self.yLeft, self.xLeft

            if( plage.hasState('hidden') ) :
                # les pics cachés sont situées sur des parties entièrement monotonnes
                # ce qui empêche la détection de la mi-hauteur au moins d'un côté.
                # Mais sa plage, calculée sur le gradient, se termine sur les points d'inflexion de la courbe.
                # (max locaux du gradient)
                # En première approximation, ces points peuvent être considérées comme des largeurs à mi-hauteur.
                # Le fitting retrouvera les valeurs correctes.
                #
                # La courbe étant entièrement monotone, la recherche du min( abs( data-halfMax ) ) projette automatiquement
                # un point sur _xMax ( le point à gauche si la courbe était décroissante, et inversement )
                # dans ce cas, on prend la largeur jusqu'au point d'inflexion (le point extrême de la plage)
                # On prend ensuite le plus petit écart des deux points ainsi trouvés, et on le reporte de l'autre côté.

                if self._xLeft == self._xMax : self._xLeft = plage.debut
                if self._xRight== self._xMax : self._xRight= plage.fin

                width = np.min((
                    self._xRight - self._xMax, # point trouvé à gauche
                    plage.fin - self._xMax,    # plage extrême gauche

                    self._xMax - self._xLeft,  # point trouvé à droite
                    self._xMax - plage.debut   # plage extrême à droite
                ))

                self._xLeft  = self._xMax - width
                self._xRight = self._xMax + width

        if self.PARAM['ordered type'] == 'invert' :
            self._xLeft, self._xRight = self._xRight, self._xLeft

    def get_nrj(self,x):
        if np.isnan(x): return np.nan
        if 0 < x < len(self.Spectrum.dataX) :
            return self.Spectrum.dataX[x]
        else :
            return self.Spectrum._f_nrj[0](x)


    def get_bg(self, x):
        if np.isnan(x): return np.nan
        return self.Spectrum.bg(x)

    def get_indiceFromNRJ(self, nrj, fromPlage=False):
        x = self.Spectrum.get_indiceFromNRJ(nrj)
        if np.isnan(x) : return np.nan
        if fromPlage : x = self.plage.indiceFrom(x)
        return x

    def get_max(self,withBg=False):
        if np.isnan(self._xMax): return np.nan
        return [ self.xMax, self.get_yMax(withBg) ]

    def get_xMax(self)  :
        if np.isnan(self._xMax): return np.nan
        return self.get_nrj( self._xMax )
    def get_xLeft(self) :
        if np.isnan(self._xLeft): return np.nan
        return self.get_nrj( self._xLeft )
    def get_xRight(self):
        if np.isnan(self._xRight): return np.nan
        return self.get_nrj( self._xRight )

    def get_xMax_fitted(self):
        x = self._fitted_values[1]
        return self.xMax if np.isnan(x) else x
    def get_xLeft_fitted(self):
        x = self._fitted_values[0]
        return self.xLeft if np.isnan(x) else x
    def get_xRight_fitted(self):
        x = self._fitted_values[2]
        return self.xRight if np.isnan(x) else x
    def set_xMax_fitted(self, value):
        self._fitted_values[1] = value
        self._xMax = self.get_indiceFromNRJ( value )
    def set_xLeft_fitted(self, value):
        self._fitted_values[0] = value
        self._xLeft = self.get_indiceFromNRJ( value )
    def set_xRight_fitted(self, value):
        self._fitted_values[2] = value
        self._xRight = self.get_indiceFromNRJ( value )
    def del_xMax_fitted(self):
        self._fitted_values[1] = np.nan
    def del_xLeft_fitted(self):
        self._fitted_values[0] = np.nan
    def del_xRight_fitted(self):
        self._fitted_values[2] = np.nan

    def get_yMax(self,withBg=False):
        if np.isnan(self._xMax): return np.nan
        if not 0 < self._xMax < len(self.Spectrum.spectrum): return np.nan
        yMax = self.Spectrum.spectrum[ self._xMax ]
        if(not withBg): yMax -= self.get_bg( self.xMax )
        return yMax

    def get_yLeft(self,withBg=False):
        if np.isnan(self._xLeft): return np.nan
        if not 0 < self._xLeft < len(self.Spectrum.spectrum): return np.nan
        yLeft = self.Spectrum.spectrum[ self._xLeft ]
        if(not withBg): yLeft -= self.get_bg( self.xLeft  )
        return yLeft

    def get_yRight(self,withBg=False):
        if np.isnan(self._xRight): return np.nan
        if not 0 < self._xRight < len(self.Spectrum.spectrum): return np.nan
        yRight = self.Spectrum.spectrum[ self._xRight ]
        if(not withBg): yRight -= self.get_bg( self.xRight )
        return yRight

    def get_fwhm(self):
        if np.isnan(self._xRight) or np.isnan(self._xLeft) : return np.nan
        return np.abs(self.xRight - self.xLeft)

    def get_fwhm_tuple(self):
        if np.isnan(self._xMax) : return (np.nan, np.nan)
        xMax = self.xMax
        return (
            np.nan if np.isnan(self._xLeft)  else np.abs(self.xLeft  - xMax),
            np.nan if np.isnan(self._xRight) else np.abs(self.xRight - xMax),
        )
    def get_fwhm_fitted(self):
        xRight = self.xRight_fitted
        xLeft  = self.xLeft_fitted
        if np.isnan(xRight) or np.isnan(xLeft) : return np.nan
        return np.abs(xRight - xLeft)

    def get_fwhm_tuple_fitted(self):
        xMax = self.xMax_fitted
        if np.isnan(xMax) : return (np.nan, np.nan)
        xLeft  = self.xLeft_fitted
        xRight = self.xRight_fitted
        return (
            np.nan if np.isnan(xLeft)  else np.abs(xLeft  - xMax),
            np.nan if np.isnan(xRight) else np.abs(xRight - xMax),
        )

    def get_fwhm_ratio(self):
        fwhm = self.fwhm
        if np.isnan(fwhm) or not fwhm : return np.nan
        fwhm_tuple = self.fwhm_tuple
        return (fwhm_tuple[0]/fwhm, fwhm_tuple[1]/fwhm)


    def get_aspectRatio(self):
        if np.isnan(self._xMax): return np.nan
        if np.isnan(self.fwhm): return np.nan
        return self.yMax / float(self.fwhm)

    def get_sigma(self) :
        fwhm = self.fwhm
        if not fwhm or np.isnan(fwhm): return np.nan
        return fwhm*sigma_factor

    def get_area(self):
        fwhm = self.fwhm
        if np.isnan(self._xMax) or np.isnan(fwhm): return np.nan
        return self.yMax * fwhm

    def get_areaFraction(self):
        area = self.area
        if np.isnan(area): return area
        return area / float( self.Spectrum.area )

    def testTolerance(self):
        yLeft = self.yLeft
        yRight= self.yRight


    def whatGoodPeakAreYou(self, what):
        filters = self.Spectrum.get_goodPeaksFilters()

        if isinstance(filters, list) :
            if not isinstance(what,int) or not ( 0 <= what < len(filters)) :
                return False
        elif isinstance(filters, dict) :
            if not what in filters :
                return False

        return filters[what](self)

    def whatKindOfGoodPeakAreYou(self, tab):
        return tuple([self.whatGoodPeakAreYou(i) for i in tab])

    def get_kindOfPeak(self) :
        tab = self.Spectrum.get_goodPeaksFilters()
        return set( [ key for key, f in enumerate(tab) if f(self) ] )

    def get_dataX(self):
        if not hasattr(self, '_dataX') or not np.any(self._dataX) :
            self._dataX = self.plage.extract( self.Spectrum.dataX )
            self._dataX.flags.writeable = False
        return self._dataX

    def get_energy_array(self): return self.get_dataX()

    def get_extract(self, withBg=False):
        if withBg : return self.plage.extract( self.Spectrum.get_spectrum(withBg) )
        if not hasattr(self, '_extract') or not np.any(self._extract):
            self._extract = self.plage.extract( self.Spectrum.get_spectrum(withBg=False) )
            self._extract.flags.writeable = False
        return self._extract

    def get_polynome(self):
        if not hasattr(self, '_polynome') or not np.any(self._polynome) :
            self._polynome = self.plage.extract( self.Spectrum.polynome )
        return self._polynome

    def get_fourier(self):
        if not hasattr(self, '_fourier') or not np.any(self._fourier) :
            self._fourier = self.plage.extract( self.Spectrum.fourier )
        return self._fourier



    def get_debut(self)  : return self.plage.debut
    def get_fin(self)    : return self.plage.fin
    def get_state(self)  : return self.plage.state
    def hasState(self, state): return self.plage.hasState(state)
    def addState   (self, *states): self.plage.addState(*states)
    def removeState(self, *states): self.plage.removeState(*states)

    def get_bound(self)  :
        plage = self.plage
        if plage.hasState('hidden') and plage.hasParent() :
            plage = plage.parent
        bound = plage.convert_in(self.Spectrum.energy_array)
        return sorted([ bound[2], bound[0] ]) # end can be before start with somme unit conversion


    def get_gaussianParam(self) :
        sigma2 = (self.fwhm*sigma_factor)**2
        return (
            self.yMax,#*math.sqrt(2*math.pi*sigma2),
            self.xMax,
            sigma2
        )

    def get_gaussianBound(self) :
        s = self.Spectrum
        bound = self.bound

        if (120, False, 'medium_grain', 'macro') == (s.temperature, s.cooling, s.typeEchantillon, s.experiment ) :
            A,mu,sigma2 = self.get_gaussianParam()
            yMax = self.yMax
            width = (bound[1]-bound[0])*sigma_factor
            return (
                [0.9*A, A],
                bound,
                [0, width**2]
            )
        else :
            return (
                [0, np.inf],#[0.7*A, A], # yMax / sigma2 ?
                bound,
                [0, np.inf]#[0, width**2]
            )


    def set_gaussianParam(self, param) :
        A, mu, sigma2 = param
        yMax = A #/math.sqrt(2*math.pi*sigma2)
        fwhm = math.sqrt(sigma2) / sigma_factor

        self.xMax_fitted   = mu
        self.xLeft_fitted  = mu - 0.5*fwhm
        self.xRight_fitted = mu + 0.5*fwhm
        # self.yMax = yMax
        self.halfMax = yMax/2



    def get_gaussianAsyParam(self) :
        fwhm = self.fwhm_tuple
        return (
            self.yMax,#*math.sqrt(2*math.pi*sigma2),
            self.xMax,
            (fwhm[0]*sigma_factor)**2,
            (fwhm[1]*sigma_factor)**2
        )

    def get_gaussianAsyBound(self) :
        s = self.Spectrum
        bound = self.bound
        #TODO remove this
        return (
            [0, np.inf],#[0.7*A, A], # yMax / sigma2 ?
            bound,
            [0, np.inf],#[0, width**2]
            [0, np.inf]
        )


    def set_gaussianAsyParam(self, param) :
        A, mu, sigma2Left, sigma2Right  = param
        yMax = A #/math.sqrt(2*math.pi*sigma2)
        fwhmLeft  = math.sqrt(sigma2Left) / sigma_factor
        fwhmRight = math.sqrt(sigma2Right) / sigma_factor

        self.xMax_fitted   = mu
        self.xLeft_fitted  = mu - 0.5*fwhmLeft
        self.xRight_fitted = mu + 0.5*fwhmRight
        # self.yMax = yMax
        self.halfMax = yMax/2




    def get_lorentzParam(self) :
        fwhm = self.fwhm
        return (
            self.yMax*math.pi*fwhm,
            self.xMax,
            fwhm
        )

    def get_lorentzBound(self) :
        A,mu,L = self.get_lorentzParam()
        bound = self.bound
        width = (bound[1]-bound[0])
        return (
            [0.7*A, A], # yMax / sigma2 ?
            bound,
            [0, width]
        )

    def set_lorentzParam(self, param) :
        A, mu, L = param
        yMax = A/(math.pi*L)
        fwhm = L

        self.xMax_fitted   = mu
        self.xLeft_fitted  = mu - 0.5*fwhm
        self.xRight_fitted = mu + 0.5*fwhm
        # self.yMax = yMax
        self.halfMax = yMax/2



    def get_voigtParam(self) :
        fwhm = self.fwhm
        return (
            self.yMax*math.pi*fwhm,
            self.xMax,
            fwhm
        )

    def get_voigtBound(self) :
        A,mu,L = self.get_lorentzParam()
        bound = self.bound
        width = (bound[1]-bound[0])
        return (
            [0.7*A, A], # yMax / sigma2 ?
            bound,
            [0, width]
        )

    def set_voigtParam(self, param) :
        A, mu, L = param
        yMax = A/(math.pi*L)
        fwhm = L

        self.xMax_fitted   = mu
        self.xLeft_fitted  = mu - 0.5*fwhm
        self.xRight_fitted = mu + 0.5*fwhm
        # self.yMax = yMax
        self.halfMax = yMax/2



    def discard(self): listPeak.discard(self)

    # from decory :

    def isDecory(self): return False
    def get_originalPeak(self, deep=False): return self
    def decoryDecompose(self): return [self]
    def equals(self, peak):
        assert isinstance(peak, Peak)
        return self == peak.get_originalPeak(True)


    # ================ Accesseurs ================ #

    def __getattr__(self, name):
        return getattr(self.Spectrum, name)


    typeDataObject= 'Peak'

    debut          = property( get_debut )
    fin            = property( get_fin )
    state          = property( get_state )

    xMax           = property( get_xMax )
    xLeft          = property( get_xLeft )
    xRight         = property( get_xRight )
    xMax_fitted    = property( get_xMax_fitted,   set_xMax_fitted,   del_xMax_fitted   )
    xLeft_fitted   = property( get_xLeft_fitted,  set_xLeft_fitted,  del_xLeft_fitted  )
    xRight_fitted  = property( get_xRight_fitted, set_xRight_fitted, del_xRight_fitted )
    fwhm           = property( get_fwhm )
    fwhm_tuple     = property( get_fwhm_tuple )
    fwhm_ratio     = property( get_fwhm_ratio )
    fwhm_fitted    = property( get_fwhm_fitted )
    fwhm_tuple_fitted = property( get_fwhm_tuple_fitted )

    yMax           = property( get_yMax )
    yLeft          = property( get_yLeft )
    yRight         = property( get_yRight )

    kindOfPeak     = property( get_kindOfPeak )
    extract        = property( get_extract )
    energy_array   = property( get_dataX )
    dataX          = property( get_dataX )
    polynome       = property( get_polynome )
    aspectRatio    = property( get_aspectRatio )
    sigma          = property( get_sigma )
    area           = property( get_area )
    areaFraction   = property( get_areaFraction )
    bound          = property( get_bound )
    gaussianParam  = property( get_gaussianParam, set_gaussianParam )
    gaussianBound  = property( get_gaussianBound )
    gaussianAsyParam = property( get_gaussianAsyParam, set_gaussianAsyParam )
    gaussianAsyBound = property( get_gaussianAsyBound )
    lorentzParam   = property( get_lorentzParam, set_lorentzParam )
    lorentzBound   = property( get_lorentzBound )


    # ================ warnings ================ #


    def addWarning(self, cause, valeur, ref=None):
        if ref == None : ref = self
        if not cause in self.warnings:
                self.warnings[cause] = [{ 'val':valeur, 'ref':ref }]
        else:   self.warnings[cause].append({ 'val':valeur, 'ref':ref })
        self.Spectrum.addWarning(cause, valeur, ref)

    def warningHalfMax(self, valeur, gauche=True):
        txt = 'max %f ecart a gauche de %s %% du halfMax' if gauche else 'max %f ecart a droite de %s %% du halfMax'
        self.addWarning('tolerence', txt%(self.xMax, valeur))

    def warningWithoutFwhm(self):
        self.addWarning('false peak', 'point without fwhm', self)


    def warningNoPeak(self):
        self.addWarning('false peak', 'no peak detected', self)

    # ================ Plot ====================== #

    def plotMax(self, marker='X', color='r', withVertical=True, withDots=True):
        xMax = self.xMax
        yMax = self.yMax

        label = '%s - amplitude-bg: %s | FWHM: %.3f meV' %(self.Spectrum.id, yMax, self.fwhm*1e3)
        plot = []
        if withVertical : plot.append( plt.axvline(xMax, color=color, label = label) )
        if withDots :
            plot.append( plt.scatter(xMax, yMax, marker=marker, color=color, label = label) )
            plot.append( plt.scatter(xMax, self.get_yMax(True), marker=marker, color=color) )
        return plot

    def plotFwhm(self, marker='X', color='g', withVertical=True, withDots=True ):
        xLeft = self.xLeft
        xRight= self.xRight
        plot = []
        if withVertical : plot.append( plt.axvline(xLeft, color=color) )
        if withDots :
            plot.append( plt.scatter(xLeft, self.yLeft, marker=marker, color=color ) )
            plot.append( plt.scatter(xLeft, self.get_yLeft(True),  marker=marker, color=color ) )

        if withVertical : plot.append( plt.axvline(xRight, color=color) )
        if withDots :
            plot.append( plt.scatter(xRight, self.yRight,  marker=marker, color=color) )
            plot.append( plt.scatter(xRight, self.get_yRight(True),  marker=marker, color=color) )
        return plot

    def plotFit(self, allSpectrum=True, label='', extrapol=0):
        return []

    def plotPlage(self, withBg=False,form='%0.1f'):
        return self.plage.plot(self.Spectrum.energy_array, self.Spectrum.get_spectrum(withBg),form=form)

    def plot(self, withBg=False, withFit=False, withMax=True, withFwhm=True, withVertical=True, withDots=True, withPlage=False, allSpectrum=True, label=''):
        plot = plt.plot(self.get_dataX(), self.get_extract(withBg), label=label)
        if withMax  : plot += self.plotMax(withVertical=withVertical, withDots=withDots)
        if withFwhm : plot += self.plotFwhm(withVertical=withVertical, withDots=withDots)
        if withFit  : plot += self.plotFit()
        if withPlage: plot += self.plotPlage(withBg)
        return plot

    def printWarnings(self): pp.pprint(self.warnings)


    def __repr__(self):
        return "Peak (%d) : "%(self.id)+str(self)
    def __str__(self):
        return "[%0.3f - %0.3f - (%0.3f) - %0.3f - %0.3f] (%s)"%(
            self.get_nrj(self.fin), # fin & debut inversés en énergie
            self.xLeft,
            self.xMax,
            self.xRight,
            self.get_nrj(self.debut),
            ', '.join(self.state)
        )
