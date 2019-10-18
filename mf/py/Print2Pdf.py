#basefolder = "/homecentral/srao/Documents/code/mypybox"
import numpy as np
import code, sys, os
import pylab as plt
#sys.path.append(basefolder)
#import Keyboard as kb
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
#sys.path.append(basefolder + "/nda/spkStats")
#sys.path.append(basefolder + "/utils")
from DefaultArgs import DefaultArgs
#from reportfig import ReportFig
import matplotlib as mp


def Print2Pdf_Old(axHandle, figname, paperSize = [4.26, 3.26], figFormat = 'pdf', labelFontsize = 20.0, tickFontsize = 14.0):
#    [axHandle] = DefaultArgs(
    plt.rcParams['figure.figsize'] = paperSize[0], paperSize[1]
    plt.rcParams['axes.labelsize'] = labelFontsize
    yed = [tick.label.set_fontsize(tickFontsize) for tick in axHandle.yaxis.get_major_ticks()]
    xed = [tick.label.set_fontsize(tickFontsize) for tick in axHandle.xaxis.get_major_ticks()]
    plt.draw()
    plt.savefig(figname + '.' + figFormat, format=figFormat)

def Print2Pdf(figHandle, figname, paperSize = [5.26, 4.26], figFormat = 'pdf', labelFontsize = 12.0, tickFontsize = 12.0, titleSize = 12.0, IF_ADJUST_POSITION = False, axPosition = [0.125,  0.1, 0.775  ,  0.8], linewidth = 1):
#    plt.rcParams['figure.figsize'] = paperSize[0], paperSize[1]
 #   plt.rcParams['axes.labelsize'] = labelFontsize
    figHandle.set_figwidth(paperSize[0])
    figHandle.set_figheight(paperSize[1])
    axHandle = figHandle.get_axes()[0]
    axHandle.set_title(axHandle.get_title(), fontsize = titleSize); 
    axHandle.set_xlabel(axHandle.get_xlabel(), fontsize = labelFontsize);
    axHandle.set_ylabel(axHandle.get_ylabel(), fontsize = labelFontsize);
    yed = [tick.label.set_fontsize(tickFontsize) for tick in axHandle.yaxis.get_major_ticks()]
    xed = [tick.label.set_fontsize(tickFontsize) for tick in axHandle.xaxis.get_major_ticks()]
    #-- REMOVE right and top enclosing lines
    axHandle.spines['top'].set_visible(False) 
    axHandle.spines['right'].set_visible(False)
    axHandle.spines['bottom'].set_linewidth(0.5)
    axHandle.spines['left'].set_linewidth(0.5)
    axHandle.yaxis.set_ticks_position('left')
    axHandle.xaxis.set_ticks_position('bottom')
    axHandle.xaxis.set_tick_params(width = 0.5, direction = 'out')
    axHandle.yaxis.set_tick_params(width = 0.5, direction = 'out')
    axHandle.xaxis.set_tick_params(which = 'both', direction = 'out')
    axHandle.yaxis.set_tick_params(which = 'both', direction = 'out')    
    # axHandle.xaxis.set_minortick_params(width = 0.5, direction = 'out')
    # axHandle.yaxis.set_minortick_params(width = 0.5, direction = 'out')

    # mp.rcParams['axes.linewidth'] = 0.025
    
    if(IF_ADJUST_POSITION):
#        plt.gca().set_position(axPosition) #[0.15, 0.15, .8, .75]
         axHandle.set_position(axPosition) #[0.15, 0.15, .8, .75]
    figHandle.canvas.draw()
    if figFormat == 'png':
        figHandle.savefig(figname + '.' + figFormat, format=figFormat, dpi = 900)
    else:
        figHandle.savefig(figname + '.' + figFormat, format=figFormat)
    


def ProcessFigure(figHdl, filepath, figFormat = 'eps', paperSize = [4, 3], titleSize = 10, axPosition = [0.25, 0.25, .65, .65], tickFontsize = 10, labelFontsize = 12, nDecimalsX = 1, nDecimalsY = 1):
    # FixAxisLimits(figHdl)
    # FixAxisLimits(plt.gcf(), IF_XTICK_INT, nDecimalsX, nDecimalsY)
    Print2Pdf(plt.gcf(), filepath, paperSize, figFormat=figFormat, labelFontsize = labelFontsize, tickFontsize=tickFontsize, titleSize = titleSize, IF_ADJUST_POSITION = True, axPosition = axPosition)
    plt.show()

def FixAxisLimits(fig, IF_XTICK_INT = False, nDecimalsX = 1, nDecimalsY = 1):
    ax = fig.axes[0]
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.' + '%s'%(int(nDecimalsX)) + 'f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.' + '%s'%(int(nDecimalsY)) + 'f'))
    xmiddle = 0.5 * (xmin + xmax)
    xticks = [xmin, xmiddle, xmax]
    if IF_XTICK_INT:
	if xmiddle != int(xmiddle):
	    xticks = [xmin, xmax]
	ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.set_xticks(xticks)
    ax.set_yticks([ymin, 0.5 *(ymin + ymax), ymax])
    plt.draw()
