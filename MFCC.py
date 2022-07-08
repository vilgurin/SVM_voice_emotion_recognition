import librosa
import numpy as np
import scipy as sp

class MFCC:
    """
    MFCC implementation from scratch
    """
    def __init__(self, sample, sampleRate, filtersAmount, window_size = 512, hopsize=int(512/4), fftSize=512):
        self.sample = sample
        self.sampleRate = sampleRate
        self.filtersAmount = filtersAmount
        self.window_size = window_size
        self.hopsize = hopsize
        self.fftSize = fftSize

    def convertFrequencyToMelScale(self, frequency):
        return 1127 * np.log10(1 + (frequency / 700))

    def convertMelScaleToFrequency(self, mel):
        return 700 * (10**(mel / 1127) - 1)

    def roundFrequenciesToNearestFFTBin(self, fftSize, hertzFreqPoints, sampleRate):
        return np.floor((fftSize + 1) * hertzFreqPoints / sampleRate)

    def getPowerSpectrum(self, audio):
        
        getSTFT = librosa.stft(audio, hop_length=self.hopsize, win_length=self.window_size, n_fft=self.fftSize, window="hamming")
        magnitudeOfSTFT = np.abs(getSTFT)
        powerSpectrum = magnitudeOfSTFT ** 2 / self.fftSize
        return powerSpectrum
        
    def getValuesOfBanks(self, fbPrev, fbCurr, fbNext, filterBanks, fbIdx):
        
        fbPrev, fbCurr, fbNext = int(fbPrev), int(fbCurr), int(fbNext)
        for freq in range(fbPrev, fbCurr):
            filterBanks[fbIdx-1, freq] = (freq - fbPrev)/(fbCurr - fbPrev)
            
        for freq in range(fbCurr+1, fbNext):
            filterBanks[fbIdx-1, freq] = (fbNext - freq)/(fbNext - fbCurr)
        
        filterBanks[fbIdx-1, fbCurr] = 1
        return filterBanks
            
    def getFilterBanks(self, powerSpectrum, sampleRate, filtersAmount, fftSize=512):
        
        upperMel = self.convertFrequencyToMelScale(sampleRate/2)
        
        melPoints = np.linspace(0, upperMel, num=filtersAmount+2)
        hertzPoints = self.convertMelScaleToFrequency(melPoints)
        freqBinPoints = self.roundFrequenciesToNearestFFTBin(fftSize, hertzPoints, sampleRate)
        filterBanks = np.zeros((filtersAmount, int(fftSize/2)+1))
        
        for fbIdx in range(1, filtersAmount+1):
            
            fbPrev = freqBinPoints[fbIdx-1]
            fbCurr = freqBinPoints[fbIdx]
            fbNext = freqBinPoints[fbIdx+1]
            
            filterBanks = self.getValuesOfBanks(fbPrev, fbCurr, fbNext, filterBanks, fbIdx)
            
        return filterBanks
        
    def getDifferentialCoefficients(self, frames):
        
        diffCoefs = np.zeros(shape=frames.shape)
        for coefficient in range(1, frames.shape[1]-1):
            diffCoefs[:,coefficient] = 1/2 * (frames[:,coefficient+1] - frames[:,coefficient-1])
        
        return diffCoefs

    def getFilterBankEnergies(self, frames):
        
        return np.hstack(np.sqrt(np.sum(np.power(frames,2),axis=0)))


    def mfcc(self):
        
        powerSpectrum = self.getPowerSpectrum(self.sample)
        filterBanks = self.getFilterBanks(powerSpectrum, self.sampleRate, self.filtersAmount, fftSize=self.fftSize)  
        filterBanks[filterBanks==0] = np.finfo(float).eps # upper bound on the relative approximation error due to rounding in floating point arithmetic. 
        
        filterBanksRes = np.dot(filterBanks, powerSpectrum)
        filterBanksRes = filterBanksRes + np.finfo(float).eps
        filterBanksRes = np.log(filterBanksRes)
        
        cepstralCoefs = sp.fftpack.dct(filterBanksRes)

        lowerCoefs = cepstralCoefs[:13,:]    
        lowerDiffCoefs = self.getDifferentialCoefficients(cepstralCoefs)[:13,:]
        lowerAccelerCoefs = self.getDifferentialCoefficients(lowerDiffCoefs)
        
        energy = self.getFilterBankEnergies(lowerCoefs)
        energyDiff = self.getFilterBankEnergies(lowerDiffCoefs)
        energyAcceler = self.getFilterBankEnergies(lowerAccelerCoefs)
        
        return np.vstack((energy, energyDiff, energyAcceler, lowerCoefs, lowerDiffCoefs, lowerAccelerCoefs))