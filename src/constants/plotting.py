import matplotlib
import itertools

class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format

font = {'family': 'serif',
        'color': 'k',
        'weight': 'normal',
        'size': 14,
        }

marker = itertools.cycle((',', '.', 'o', '*'))
color = itertools.cycle(('b','r','g','k'))