import pylab as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable


def plot(plots, show=False, save=None, figsize=(16*0.5, 9*0.5), xlabel=None, ylabel=None, title=None, dpi=480, xlim=None, ylim=None, padding=0.75, xticks=None, yticks=None, style=None):
    pl.close()

    if style is not None:
        pl.style.use(style)

    fig = pl.figure(figsize=figsize)

    if not isinstance(plots, list):
        plots = [plots]

    xpadding = padding
    ypadding = padding
    if isinstance(padding, list):
        xpadding = padding[0]
        ypadding = padding[1]

    min_x = None
    max_x = None
    min_y = None
    max_y = None

    for p in plots:
        cminx = min(p[1])
        cmaxx = max(p[1])

        kwargs = {}
        ptype = p[0]
        if ptype=='plot':
            miny = p[2]
            maxy = p[2]
            minx = p[1]
            maxx = p[1]
        elif ptype=='bar':
            miny = p[2]
            maxy = p[2]
            minx = p[1]
            maxx = p[1]
            if len(p)>=4:
                kwargs = p[3]
            if 'bottom' in kwargs:
                miny = kwargs['bottom']
                maxy = [y1 + y2 for y1, y2 in zip(kwargs['bottom'], p[2])]
        elif ptype in ('fill_between', 'fillbetween', 'fill'):
            miny = p[2]
            maxy = p[3]
            minx = p[1]
            maxx = p[1]
        elif ptype in ('errorbar', 'error'):
            miny = p[2]
            maxy = p[2]
            minx = p[1]
            maxx = p[1]
            if len(p)>=4:
                kwargs = p[3]
            if 'yerr' in kwargs:
                if isinstance(kwargs['yerr'][0], list) or isinstance(kwargs['yerr'][0], tuple):
                    miny = [y-e for y, e in zip(p[2], kwargs['yerr'][0])]
                    maxy = [y+e for y, e in zip(p[2], kwargs['yerr'][1])]
                else:
                    miny = [y-e for y, e in zip(p[2], kwargs['yerr'])]
                    maxy = [y+e for y, e in zip(p[2], kwargs['yerr'])]
            if 'xerr' in kwargs:
                if isinstance(kwargs['xerr'][0], list) or isinstance(kwargs['xerr'][0], tuple):
                    minx = [x-e for x, e in zip(p[1], kwargs['xerr'][0])]
                    maxx = [x+e for x, e in zip(p[1], kwargs['xerr'][1])]
                else:
                    minx = [x-e for x, e in zip(p[1], kwargs['xerr'])]
                    maxx = [x+e for x, e in zip(p[1], kwargs['xerr'])]

        cminy = min(miny)
        cmaxy = max(maxy)
        cminx = min(minx)
        cmaxx = max(maxx)

        if min_x is None or cminx < min_x:
            min_x = cminx
        if max_x is None or cmaxx > max_x:
            max_x = cmaxx
        if min_y is None or cminy < min_y:
            min_y = cminy
        if max_y is None or cmaxy > max_y:
            max_y = cmaxy

    range_x = (max_x - min_x) / xpadding
    range_y = (max_y - min_y) / ypadding

    x0 = min_x - (range_x - (max_x - min_x)) / 2.0
    x1 = max_x + (range_x - (max_x - min_x)) / 2.0
    y0 = min_y - (range_y - (max_y - min_y)) / 2.0
    y1 = max_y + (range_y - (max_y - min_y)) / 2.0
    if ylim is not None:
        if ylim[0] is None:
            ylim[0] = y0
        if ylim[1] is None:
            ylim[1] = y1
        pl.ylim(ylim)
    else:
        pl.ylim([y0, y1])
    if xlim is not None:
        if xlim[0] is None:
            xlim[0] = x0
        if xlim[1] is None:
            xlim[1] = x1
        pl.xlim(xlim)
    else:
        pl.xlim([x0, x1])

    for p in plots:
        ptype = p[0]
        kwargs = {}
        if ptype=='plot':
            x, y = p[1], p[2]
            if len(p)>=4:
                kwargs = p[3]
            pl.plot(x, y, **kwargs)
        elif ptype=='bar':
            x, y = p[1], p[2]
            if len(p)>=4:
                kwargs = p[3]
            pl.bar(x, y, **kwargs)
        elif ptype in ('fill_between', 'fillbetween', 'fill'):
            x, y_l, y_h = p[1], p[2], p[3]
            if len(p)>=5:
                kwargs = p[4]
            pl.fill_between(x, y_l, y_h, **kwargs)
        elif ptype in ('errorbar', 'error'):
            x, y = p[1], p[2]
            if len(p)>=4:
                kwargs = p[3]
            pl.errorbar(x, y, **kwargs)

    if title is not None:
        pl.title(title)
    if xlabel is not None:
        pl.xlabel(xlabel)
    if ylabel is not None:
        pl.ylabel(ylabel)
    if xticks is not None:
        pl.xticks(**xticks)
    if yticks is not None:
        pl.yticks(**yticks)

    make_axes_area_auto_adjustable(fig.axes[0])

    if style is None:
        fig.axes[0].grid(True, alpha=0.2)

    pl.legend()

    pl.tight_layout()

    if save is not None:
        pl.savefig(save, dpi=dpi)
    if show:
        pl.show()



def histogram(samples, min_x=None, max_x=None, n=10):
    if min_x is None:
        min_x = min(samples) * 1.0
    if max_x is None:
        max_x = max(samples) * 1.0

    d = (max_x - min_x) * 1.0 / n

    ys = []
    xs = []

    for i in range(n):
        cmin = i * d + min_x
        cmax = (i + 1) * d + min_x
        xs.append((cmin+cmax)*0.5)
        ys.append(
            len([s for s in samples if s>=cmin and s<cmax])
        )
        if i==n-1:
            ys[-1] = ys[-1] + len([s for s in samples if s==cmax])

    return xs, ys
