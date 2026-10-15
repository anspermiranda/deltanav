"""Static explanatory figures for the README (not data-dependent): the four outcomes of the see-through test.

usage: python tools/make_figures.py docs/img"""
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle

RED, GREY, PURPLE, BLUE = '#dc3c3c', '#8a8f99', '#855cc7', '#1473dc'


def camera(ax, x, y):
    ax.add_patch(Polygon([[x, y], [x - 0.35, y + 0.22], [x - 0.35, y - 0.22]], closed=True, fc='#2b2f38', ec='none'))
    ax.text(x - 0.2, y - 0.3, 'camera of\nthe other video', ha='center', va='top', fontsize=8.5, color='#2b2f38')


def panel(ax, kind):
    ax.set_xlim(-0.7, 4.3); ax.set_ylim(-1.75, 1.3); ax.axis('off'); ax.set_aspect('equal')
    camera(ax, 0, 0)
    px = 2.3                                              # the surface point from the first video
    ax.add_patch(Rectangle((3.8, -1.0), 0.15, 2.0, fc='#c9ccd3', ec='none'))
    ax.text(3.88, 1.05, 'wall', ha='center', fontsize=8.5, color=GREY)
    if kind == 'free':
        ax.plot([0, 3.8], [0, 0], color=BLUE, lw=2)
        ax.plot(3.8, 0, 'x', color=BLUE, ms=10, mew=2.5)
        ax.add_patch(Circle((px, 0), 0.13, fc='white', ec=RED, lw=2.2, ls='--'))
        title, verdict, col = 'Seen through', 'depth lies beyond the point:\na vote for "gone"', RED
    elif kind == 'same':
        ax.plot([0, px - 0.13], [0, 0], color=BLUE, lw=2)
        ax.add_patch(Circle((px, 0), 0.13, fc=GREY, ec='#555', lw=1.5))
        ax.plot(px - 0.13, 0, 'x', color=BLUE, ms=10, mew=2.5)
        title, verdict, col = 'Same', 'depth matches the point:\na vote for "still there"', '#444'
    elif kind == 'occluded':
        ax.add_patch(Rectangle((1.2, -0.45), 0.25, 0.9, fc='#5d6370', ec='none'))
        ax.text(1.32, 0.55, 'something\nnearer', ha='center', fontsize=8.5, color='#5d6370')
        ax.plot([0, 1.2], [0, 0], color=BLUE, lw=2); ax.plot(1.2, 0, 'x', color=BLUE, ms=10, mew=2.5)
        ax.plot([1.45, px], [0, 0], color=BLUE, lw=1, ls=':')
        ax.add_patch(Circle((px, 0), 0.13, fc='white', ec=PURPLE, lw=2))
        title, verdict, col = 'Hidden', 'something in front:\nno vote', PURPLE
    else:
        ax.add_patch(Polygon([[0, 0], [2.6, 1.25], [3.4, 0.6]], closed=True, fc=(0.08, 0.45, 0.86, 0.10), ec='none'))
        ax.plot([0, 2.6], [0, 1.25], color=BLUE, lw=1, ls=':'); ax.plot([0, 3.4], [0, 0.6], color=BLUE, lw=1, ls=':')
        ax.text(2.2, 0.62, 'what the\nframe saw', fontsize=8, color=BLUE, ha='center', rotation=0)
        ax.add_patch(Circle((2.3, 0.0), 0.13, fc='white', ec=PURPLE, lw=2))
        title, verdict, col = 'Never in view', 'the frame looked elsewhere:\nno vote', PURPLE
    ax.text(px, -0.25, 'point from\nthe first video', ha='center', va='top', fontsize=8.5, color='#444')
    ax.set_title(title, fontsize=12, fontweight='bold', color=col, pad=2)
    ax.text(1.8, -1.7, verdict, ha='center', va='bottom', fontsize=10, color=col)


def visibility_test(path):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), dpi=140)
    for ax, k in zip(axes, ('free', 'same', 'occluded', 'never')):
        panel(ax, k)
    fig.text(0.5, 0.005, 'A point is called gone only with at least 3 see-through votes, 70% of the decisive votes, and '
             '25% of the frames that pointed there. Everything else that was not confirmed is "could not be checked".',
             ha='center', fontsize=10, color='#333')
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(path, bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'docs/img'
    os.makedirs(out, exist_ok=True)
    visibility_test(os.path.join(out, 'visibility_test.png'))
    print('wrote', os.path.join(out, 'visibility_test.png'))
