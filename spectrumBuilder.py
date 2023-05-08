#!/usr/bin/python2.7
#-*- coding: utf-8 -*-
print('import spectrumBuilder ')

import time
import numpy as np
import copy

from general        import *
import spectrum


# TODO
# builder must be immutable

class SpectrumBuilder(object):

    __slots__ = (
        'id',
        'creation',
        'fileInfo',
        'units',
        'PARAM',
        'warnings',
        'dataX',
        '_f_nrj',
    )

    def __init__(self, fileInfo, param, units, dataX=None):
        assert len(units)==2, 'units need 2 values'
        t = t_tot = time.time()
        self.id             = get_new_id()
        self.creation       = time.time()
        self.warnings       = {}
        self.units          = tuple(units)
        self.fileInfo       = copy.deepcopy(fileInfo)
        self.PARAM          = copy.deepcopy(param)
        self.set_dataX(dataX)

    def initialise(self, spectrum) :
        if np.any(self.dataX) :
            spectrum._f_nrj = self._f_nrj
        else :
            spectrum.build_f_nrj()

        f_bg, info = self.build_f_bg( spectrum )
        spectrum.initalise_bg( f_bg, info )

    def build_f_bg(self, spectrum) :
        f_bg = self.PARAM['f_extractBg'](spectrum.dataX, spectrum.spectrum)
        if isinstance(f_bg,tuple) :
            f_bg, info = f_bg
        else :
            info = { 'code':inspect.getsource(self.PARAM['f_extractBg']) }
        return f_bg, info

    def build(self, data, dataX=None, coordinates=(0,0,0), fileInfoAdd=False) :
        if np.any(dataX) and len(dataX) :
            self.set_dataX(dataX)
        elif not hasattr(self, 'dataX') :
            raise Exception('no dataX for spectrum')

        fileInfo = self.fileInfo
        if isinstance(fileInfoAdd, dict) :
            fileInfo = copy.deepcopy(fileInfo)
            fileInfo.update(fileInfoAdd)

        data = np.array(data)
        data.flags.writeable = False

        return spectrum.Spectrum(
            self.dataX,
            data,
            coordinates = coordinates,
            units=self.units,
            fileInfo=fileInfo,
            param=self.PARAM,
            builder=self
        )


    def buildArray(self, data, dataX=None, fileInfoAdd=False, partialLoadTest=False, loadPrint=False):
        if np.any(dataX) and len(dataX) :
            self.set_dataX(dataX)
        elif not hasattr(self, 'dataX') :
            raise Exception('no dataX for spectrum')

        dataX = self.dataX

        fileInfo = self.fileInfo
        if isinstance(fileInfoAdd, dict) :
            fileInfo = copy.deepcopy(fileInfo)
            fileInfo.update(fileInfoAdd)

        PARAM = self.PARAM

        retourTab = []
        retourSet = set()
        i = 0
        for x in range(0, len(data)):
            if partialLoadTest and partialLoadTest(i) : break
            retourTab.append([])
            for y in range(0, len(data[x])) :
                if partialLoadTest and partialLoadTest(i) : break
                i+=1
                if loadPrint : loadPrint(i)
                dataY = np.array(data[x][y])
                dataY.flags.writeable = False
                try :
                    s = spectrum.Spectrum(
                        dataX,
                        dataY,
                        coordinates=(x,y),
                        units=self.units,
                        fileInfo=fileInfo,
                        param=PARAM,
                        builder=self
                    )
                    retourTab[x].append(s)
                    retourSet.add(s)
                except Exception as error:
                    e_type, value, tb = sys.exc_info()
                    self.addWarning('buildArray : %s'%(error), {
                        'file':fileInfo,
                        'coord':(x,y),
                        'raw':dataY,
                        'traceback':traceback.format_tb(tb)
                    })
                    retourTab[x].append(None)

        return retourTab, retourSet

    def set_dataX(self, dataX):
        if np.any(dataX) and len(dataX) :
            dataX = np.array(dataX)
            dataX.flags.writeable = False
            self.dataX = dataX
        elif hasattr(self,'dataX') :
            del self.dataX

        self.build_f_nrj()

    def clone(self):
        clone = copy.copy(self)
        clone.id             = get_new_id()
        clone.creation       = time.time()
        clone.warnings       = {}
        return clone

    def updateParam( self, update ):
        PARAM = copy.deepcopy(self.PARAM)
        PARAM.update(update)

        clone = self.clone()
        clone.PARAM = PARAM
        return clone

    def updateFileInfo(self, update):
        fileInfo = copy.deepcopy(self.fileInfo)
        fileInfo.update(update)

        clone = self.clone()
        clone.fileInfo = fileInfo
        return clone

    def updateUnits(self, units):
        assert len(units)==2, 'units need 2 values'
        units = tuple(units)
        if units == self.units : return self
        clone = self.clone()
        clone.units = units
        return clone

    def get_info(self):
        return 'Builder %s : %s %sK %s - %d'%(
            self.experiment,
            self.typeEchantillon,
            self.temperature,
            'cooling' if self.cooling else 'warming',
            self.id
        )

    def get_indiceFromNRJ(self, nrj):
        if np.isnan(nrj) : return np.nan
        assert hasattr(self,'dataX'), 'no dataX defined'
        return self._f_nrj[1](nrj)

    #### Fonctions de modelisation ####

    def build_f_nrj(self) :
        if not hasattr(self, 'dataX') :
            if hasattr(self, '_f_nrj') :
                del self._f_nrj
        else :
            dataX = self.dataX
            if not 'ordered type' in self.PARAM or self.PARAM['ordered type'] == 'linear' :
                b = float(dataX[0])
                a = (float(dataX[-1]) - b)/float( len(dataX) )
                div = b/float(a)

                self._f_nrj = [
                    lambda x: a*x + b,
                    lambda y: int( y/a - div )
                ]
            elif self.PARAM['ordered type'] == 'invert' :
                # La mesure en longueure d'onde est linéraire. ( a.x + b )
                # Donc la mesure déduite en énergie est en 1/ (a.x + b)
                b = 1/float(dataX[0])
                a = (1/float(dataX[-1]) - b)/float( len(dataX) )
                div = b/float(a)

                self._f_nrj = [
                    lambda x: 1/(a*x + b),         # y = 1/( a.x + b )
                    lambda y: int( 1/(a*y) - div ) # x = ( 1-b.y ) / a.y
                ]

    def addWarning(self, cause, valeur, ref=None):
        if not np.any(ref) : ref = self
        if not cause in self.warnings:
            self.warnings[cause] = [{'val':valeur,'ref':ref}]
        else:
            self.warnings[cause].append({'val':valeur,'ref':ref})


    def get_typeEchantillon (self) : return self.fileInfo['type']        if 'type'        in self.fileInfo else None
    def get_temperature     (self) : return self.fileInfo['temperature'] if 'temperature' in self.fileInfo else None
    def get_experiment      (self) : return self.fileInfo['experiment']  if 'experiment'  in self.fileInfo else None
    def get_filepath        (self) : return self.fileInfo['filepath']    if 'filepath'    in self.fileInfo else None
    def get_cooling         (self) : return self.fileInfo['cooling']     if 'cooling'     in self.fileInfo else None
    def get_warming         (self) : return not self.fileInfo['cooling'] if 'cooling'     in self.fileInfo else None
    def get_mapWidth        (self) : return self.fileInfo['map_w']       if 'map_w'       in self.fileInfo else None


    # ================ Accesseurs ================ #

    info                = property( get_info )
    typeDataObject      = 'SpectrumBuilder'

    typeEchantillon     = property( get_typeEchantillon )
    temperature         = property( get_temperature )
    experiment          = property( get_experiment )
    filepath            = property( get_filepath )
    cooling             = property( get_cooling )
    mapWidth            = property( get_mapWidth )
    warming             = property( get_warming )


    # ============================================ #

    # ================ plot ====================== #

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

    def __repr__(self):return "SpectrumBuilder() %s"%(self.info)
    def __str__(self): return self.info
