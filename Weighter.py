'''
Created on 26 Feb 2017

@author: jkiesele
'''

from __future__ import print_function

import matplotlib
#if no X11 use below
matplotlib.use('Agg')

class Weighter(object):
    '''
    contains the histograms/input to calculate jet-wise weights
    '''
    def __init__(self):

        self.Axixandlabel=[]
        self.axisX=[]
        self.axisY=[]
        self.hists =[]
        self.removeProbabilities=[]
        self.binweights=[]
        self.distributions=[]
        self.totalcounts=[]
        self.xedges=[]
        self.yedges=[]
        self.classes=[]
        self.refclassidx=0
        self.undefTruth=[]
        self.ignore_when_weighting=[]
    
    def __eq__(self, other):
        'A == B'
        def comparator(this, that):
            'compares lists of np arrays'
            return all((i == j).all() for i,j in zip(this, that))
        
        return self.Axixandlabel == other.Axixandlabel and \
           all(self.axisX == other.axisX) and \
           all(self.axisY == other.axisY) and \
           comparator(self.hists, other.hists) and \
           comparator(self.removeProbabilities, other.removeProbabilities) and \
           self.classes == other.classes and \
           self.refclassidx == other.refclassidx and \
           self.undefTruth == other.undefTruth and \
           comparator(self.binweights, other.binweights) and \
           comparator(self.distributions, other.distributions) and \
           self.totalcounts == other.totalcounts and \
           (self.xedges == other.xedges).all() and \
           (self.yedges == other.yedges).all()
    
    def __ne__(self, other):
        'A != B'
        return not (self == other)
        
    def setBinningAndClasses(self,bins,nameX,nameY,classes):
        self.axisX= bins[0]
        self.axisY= bins[1]
        self.nameX=nameX
        self.nameY=nameY
        self.classes=classes
        if len(self.classes)<1:
            self.classes=['']
        
    def addDistributions(self,Tuple,referenceclass='flatten'):
        import numpy
        selidxs=[]
        
        ytuple=Tuple[self.nameY]
        xtuple=Tuple[self.nameX]
        
        useonlyoneclass=len(self.classes)==1 and len(self.classes[0])==0
        
        if not useonlyoneclass:
            labeltuple=Tuple[self.classes]
            for c in self.classes:
                selidxs.append(labeltuple[c]>0)
        else:
            selidxs=[numpy.zeros(len(xtuple),dtype='int')<1]
            
        
        for i in range(len(self.classes)):
            if referenceclass not in ['lowest']:
                tmphist,xe,ye=numpy.histogram2d(xtuple[selidxs[i]],ytuple[selidxs[i]],[self.axisX,self.axisY],normed=True)
            else:
                tmphist,xe,ye=numpy.histogram2d(xtuple[selidxs[i]],ytuple[selidxs[i]],[self.axisX,self.axisY])
            self.xedges=xe
            self.yedges=ye
            if len(self.distributions)==len(self.classes):
                self.distributions[i]=self.distributions[i]+tmphist
                self.totalcounts[i] += numpy.sum(Tuple[self.classes[i]])
            else:
                self.distributions.append(tmphist)
                self.totalcounts.append(numpy.sum(Tuple[self.classes[i]]))

            
    def printHistos(self,outdir):
        import numpy
        def plotHist(hist,outname):
            import matplotlib.pyplot as plt
            H=hist.T
            fig = plt.figure()
            ax = fig.add_subplot(111)
            X, Y = numpy.meshgrid(self.xedges, self.yedges)
            ax.pcolormesh(X, Y, H)
            if self.axisX[0]>0:
                ax.set_xscale("log", nonposx='clip')
            else:
                ax.set_xlim([self.axisX[1],self.axisX[-1]])
                ax.set_xscale("log", nonposx='mask')
            #plt.colorbar()
            fig.savefig(outname)
            plt.close()
            
        for i in range(len(self.classes)):
            if len(self.distributions):
                plotHist(self.distributions[i],outdir+"/dist_"+self.classes[i]+".pdf")
                plotHist(self.removeProbabilities[i] ,outdir+"/remprob_"+self.classes[i]+".pdf")
                plotHist(self.binweights[i],outdir+"/weights_"+self.classes[i]+".pdf")
                reshaped=self.distributions[i]*self.binweights[i]
                plotHist(reshaped,outdir+"/reshaped_"+self.classes[i]+".pdf")
            
        
    def createRemoveProbabilitiesAndWeights(self,referenceclass='isB'):
        import numpy
        referenceidx=-1
        if referenceclass not in ['flatten','lowest']:
            try:
                referenceidx=self.classes.index(referenceclass)
            except:
                print('createRemoveProbabilities: reference index not found in class list {}'.format(referenceclass))
                raise Exception('createRemoveProbabilities: reference index not found in class list {}'.format(referenceclass))
            
        if len(self.classes) > 0 and len(self.classes[0]):
            self.Axixandlabel = [self.nameX, self.nameY]+ self.classes
        else:
            self.Axixandlabel = [self.nameX, self.nameY]
        
        self.refclassidx=referenceidx
        
        refhist=numpy.zeros((len(self.axisX)-1,len(self.axisY)-1), dtype='float32')
        refhist += 1
        
        if referenceidx >= 0:
            refhist=self.distributions[referenceidx]
            refhist=refhist/numpy.amax(refhist)
        
    
        def divideHistos(a,b):
            out=numpy.array(a)
            for i in range(a.shape[0]):
                for j in range(a.shape[1]):
                    if b[i][j]:
                        out[i][j]=a[i][j]/b[i][j]
                    else:
                        out[i][j]=-10
            return out
                
        probhists=[]
        weighthists=[]

        bin_counts = []
        for i in range(len(self.classes)):
            if self.classes[i] in self.ignore_when_weighting:  continue
            bin_counts.append(self.distributions[i])
        bin_min = numpy.array(numpy.minimum.reduce(bin_counts))
        
        for i in range(len(self.classes)):
            #print(self.classes[i])
            tmphist=self.distributions[i]
            #print(tmphist)
            #print(refhist)
            if referenceclass in ['lowest']:
                ratio = divideHistos(bin_min,tmphist)
            else:
                if numpy.amax(tmphist):
                    tmphist=tmphist/numpy.amax(tmphist)
                else:
                    print('Warning: class '+self.classes[i]+' empty.')
            ratio=divideHistos(refhist,tmphist)
            ratio=ratio/numpy.amax(ratio)#norm to 1
            #print(ratio)
            ratio[ratio<0]=1
            ratio[ratio==numpy.nan]=1
            weighthists.append(ratio)
            ratio=1-ratio#make it a remove probability
            probhists.append(ratio)
        
        self.removeProbabilities=probhists
        self.binweights=weighthists
        
        #make it an average 1
        #for i in range(len(self.binweights)):
        #    self.binweights[i]=self.binweights[i]/numpy.average(self.binweights[i])
    
    
        
        
    def createNotRemoveIndices(self,Tuple):
        import numpy
        if len(self.removeProbabilities) <1:
            print('removeProbabilities bins not initialised. Cannot create indices per jet')
            raise Exception('removeProbabilities bins not initialised. Cannot create indices per jet')
        
        tuplelength=len(Tuple)

        notremove=numpy.zeros(tuplelength)
        counter=0
        xaverage=[]
        norm=[]
        yaverage=[]
        
        useonlyoneclass=len(self.classes)==1 and len(self.classes[0])==0
        
        for c in self.classes:
            xaverage.append(0)
            norm.append(0)
            yaverage.append(0)
            
        

        for jet in iter(Tuple[self.Axixandlabel]):
            binX =  self.getBin(jet[self.nameX], self.axisX)
            binY =  self.getBin(jet[self.nameY], self.axisY)
            
            found = False
            for index, aclass in enumerate(self.classes):
                if  useonlyoneclass or 1 == jet[aclass]:
                    found = True
                    rand=numpy.random.ranf()
                    prob = self.removeProbabilities[index][binX][binY]
                    
                    if rand < prob and index != self.refclassidx:
                        #print('rm  ',index,self.refclassidx,jet[aclass],aclass)
                        notremove[counter]=0
                    else:
                        #print('keep',index,self.refclassidx,jet[aclass],aclass)
                        notremove[counter]=1
                        xaverage[index]+=jet[self.nameX]
                        yaverage[index]+=jet[self.nameY]
                        norm[index]+=1
            
                    counter=counter+1            
                    break
            if not found:
                notremove[counter] = 0
                counter += 1

        
            
        if not len(notremove) == counter:
            raise Exception("tuple length must match remove indices length. Probably a problem with the definition of truth classes in the ntuple and the TrainData class")
        
        
        return notremove

    
        
    def getJetWeights(self,Tuple):
        import numpy
        countMissedJets = 0  
        if len(self.binweights) <1:
            raise Exception('weight bins not initialised. Cannot create weights per jet')
        
        weight = numpy.zeros(len(Tuple))
        jetcount=0
        
        useonlyoneclass=len(self.classes)==1 and len(self.classes[0])==0
        
        for jet in iter(Tuple[self.Axixandlabel]):

            binX =  self.getBin(jet[self.nameX], self.axisX)
            binY =  self.getBin(jet[self.nameY], self.axisY)
            
            for index, aclass in enumerate(self.classes):
                if 1 == jet[aclass] or useonlyoneclass:
                    weight[jetcount]=(self.binweights[index][binX][binY])
                    
            jetcount=jetcount+1        

        print ('weight average: ',weight.mean())
        return weight
        
        
    def getBin(self,value, bins):
        """
        Get the bin of "values" in axis "bins".
        Not forgetting that we have more bin-boundaries than bins (+1) :)
        """
        for index, bin in enumerate (bins):
            # assumes bins in increasing order
            if value < bin:
                return index-1            
        #print (' overflow ! ', value , ' out of range ' , bins)
        return bins.size-2

        
        
