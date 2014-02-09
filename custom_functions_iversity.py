import numpy as np
import matplotlib.pyplot as plt

def graphicalComparisonPdf(X, modelPdf, scale = True, xMin = None, xMax = None):
    _X = X[np.logical_not(np.isnan(X))]
    if xMax is None:
        xMax = np.max(_X) # default parameter of xMax
    if xMin is None:
        xMin = np.min(_X) # default parameter of xMin
    nPlot = 1000
    xPlot = np.linspace(xMin, xMax, nPlot)
    yPlot = modelPdf(xPlot)
    
    nBins = np.min([np.sqrt(X.size), 40])  
    widthHistogram          = np.max(_X)- np.min(_X)
    averageHeightHistogram  = _X.size/nBins
    areaHistogram           = widthHistogram*averageHeightHistogram
    
    pdfScaleFactor = areaHistogram if not scale else 1 
    # if scale = False we rescale modelPDF(x) by the area of the histogram
    # if scale = True the histogram is scaled, such that its area is 1 (as is the case for modelPDF(x))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    _, _, p = plt.hist(_X, bins=nBins, normed = scale)
    l, = plt.plot(xPlot, yPlot * pdfScaleFactor, 'r', linewidth=3)
    
    ax.set_xlabel('x')
    ax.set_ylabel('pdf(x)')
    if scale:
        plt.legend([l, p[0]], ['pdf(x)', 'scaled histogram'])
    else:
        plt.legend([l, p[0]], ['scaled pdf(x)', 'histogram'])
    plt.show()