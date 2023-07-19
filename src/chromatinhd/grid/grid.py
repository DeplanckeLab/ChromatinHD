import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

active_fig = None


class Element:
    pos = None
    dim = None

    @property
    def width(self):
        return self.dim[0]

    @property
    def height(self):
        return self.dim[1]


TITLE_HEIGHT = 0.3

# AXIS_WIDTH = 0.2
# AXIS_HEIGHT = 0.2

AXIS_WIDTH = AXIS_HEIGHT = 0.0


class Ax(Element):
    ax2 = None
    insets = None

    def __init__(self, dim=None, pos=(0.0, 0.0)):
        global active_fig
        self.ax = mpl.figure.Axes(active_fig, [0, 0, 1, 1])
        self.dim = dim
        self.pos = pos

    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, value):
        if len(value) != 2:
            raise ValueError("dim must be a tuple of length 2")
        if value[0] <= 0 or value[1] <= 0:
            raise ValueError("dim must be positive")
        self._dim = value

    @property
    def height(self):
        h = self.dim[1]

        # add some extra height if we have a title
        if self.ax.get_title() != "":
            h += TITLE_HEIGHT
        if self.ax.axison:
            h += AXIS_HEIGHT
        return h

    @property
    def width(self):
        w = self.dim[0]
        if self.ax.axison:
            w += AXIS_WIDTH
        return w

    def align(self):
        pass

    def position(self, fig, pos=(0, 0)):
        fig_width, fig_height = fig.get_size_inches()
        width, height = self.dim
        x, y = self.pos[0] + pos[0], self.pos[1] + pos[1]

        axes = [self.ax]
        if self.ax2 is not None:
            axes.append(self.ax2)

        for ax in axes:
            ax.set_position(
                [
                    x / fig_width,
                    (fig_height - y - height) / fig_height,
                    width / fig_width,
                    height / fig_height,
                ]
            )

            fig.add_axes(ax)

        for inset, inset_position, inset_offset, inset_anchor in self.insets or []:
            inset.position(
                fig,
                pos=(
                    x
                    + (width - inset.dim[0]) * inset_anchor[0]
                    + (width) * inset_position[0]
                    + inset_offset[0],
                    y
                    + (height - inset.dim[1]) * inset_anchor[1]
                    + (height) * inset_position[1]
                    + inset_offset[1],
                ),
            )

    def add_twinx(self):
        global active_fig
        self.ax2 = mpl.figure.Axes(active_fig, [0, 0, 1, 1])
        self.ax2.xaxis.set_visible(False)
        self.ax2.patch.set_visible(False)
        self.ax2.yaxis.tick_right()
        self.ax2.yaxis.set_label_position("right")
        self.ax2.yaxis.set_offset_position("right")
        self.ax.yaxis.tick_left()
        return self.ax2

    def add_inset(self, inset, pos=(0, 0), offset=(0, 0), anchor=(0, 0)):
        if self.insets is None:
            self.insets = []
        self.insets.append([inset, pos, offset, anchor])
        return inset

    def __iter__(self):
        yield self
        yield self.ax


class Panel(Ax):
    pass


class Title(Panel):
    def __init__(self, label, dim=None):
        if dim is None:
            dim = (1, TITLE_HEIGHT)
        super().__init__(dim=dim)
        self.label = label
        self.ax.set_axis_off()
        self.ax.text(0.5, 0.5, label, ha="center", va="center", size="large")


class Wrap(Element):
    title = None

    def __init__(
        self,
        ncol=6,
        padding_width=0.5,
        padding_height=None,
        margin_height=0.5,
        margin_width=0.5,
    ):
        self.ncol = ncol
        self.padding_width = padding_width
        self.padding_height = (
            padding_height if padding_height is not None else padding_width
        )
        self.margin_width = margin_width
        self.margin_height = margin_height
        self.elements = []
        self.pos = (0, 0)

    def add(self, element):
        self.elements.append(element)
        return element

    def align(self):
        width = 0
        height = 0
        nrow = 1
        x = 0
        y = 0
        next_y = 0

        if self.title is not None:
            y += self.title.height

        for i, el in enumerate(self.elements):
            el.align()

            el.pos = (x, y)

            next_y = max(next_y, y + el.height + self.padding_height)
            height = max(height, next_y)

            width = max(width, x + el.width)

            if (self.ncol > 1) and ((i == 0) or (((i + 1) % (self.ncol)) != 0)):
                x += el.width + self.padding_width
            else:
                nrow += 1
                x = 0
                y = next_y

        if self.title is not None:
            self.title.dim = (width, self.title.dim[1])

        self.dim = (width, height)

    def set_title(self, label):
        if self.title is not None:
            try:
                self.title.ax.remove()
            except KeyError:
                pass
            except AttributeError:
                pass

        self.title = Title(label)

    def position(self, fig, pos=(0, 0)):
        pos = self.pos[0] + pos[0], self.pos[1] + pos[1]
        if self.title is not None:
            self.title.position(fig, pos)
        for el in self.elements:
            el.position(fig, pos)

    def __getitem__(self, key):
        return list(self.elements)[key]

    def get_bottom_left_corner(self):
        return self.elements[self.ncol * ((len(self.elements) % self.ncol) - 1)]


class WrapAutobreak(Wrap):
    title = None

    def __init__(
        self,
        max_width,
        max_n_row=-1,
        padding_width=0.5,
        padding_height=None,
        margin_height=0.5,
        margin_width=0.5,
    ):
        self.max_width = max_width
        self.max_n_row = max_n_row
        super().__init__(
            ncol=1,
            padding_width=padding_width,
            padding_height=padding_height,
            margin_height=margin_height,
            margin_width=margin_width,
        )

    def align(self):
        width = 0
        height = 0
        self.nrow = 1
        x = 0
        y = 0
        next_y = 0

        if self.title is not None:
            y += self.title.height
        for i, el in enumerate(self.elements):
            el.align()

            el.pos = (x, y)

            next_y = max(next_y, y + el.height + self.padding_height)
            height = max(height, next_y)

            width = max(width, x + el.width)

            x += el.width + self.padding_width

            if x > self.max_width:
                self.nrow += 1
                x = 0
                y = next_y

        if self.title is not None:
            self.title.dim = (width, self.title.dim[1])

        self.dim = (width, height)


class Grid(Element):
    title = None

    def __init__(
        self,
        nrow=1,
        ncol=1,
        padding_width=0.5,
        padding_height=None,
        margin_height=0.5,
        margin_width=0.5,
    ):
        self.padding_width = padding_width
        self.padding_height = (
            padding_height if padding_height is not None else padding_width
        )
        self.margin_width = margin_width
        self.margin_height = margin_height
        self.elements = [[None for _ in range(ncol)] for _ in range(nrow)]

        self.pos = (0, 0)

        self.nrow = nrow
        self.ncol = ncol

        self.paddings_height = [None] * (nrow)
        self.paddings_width = [None] * (ncol)

    def align(self):
        width = 0
        height = 0
        x = 0
        y = 0
        next_y = 0

        if self.title is not None:
            y += self.title.height

        widths = [0] * self.ncol
        heights = [0] * self.nrow

        assert len(self.paddings_height) == self.nrow, (
            len(self.paddings_height),
            self.nrow,
        )
        assert len(self.paddings_width) == self.ncol, (
            len(self.paddings_width),
            self.ncol,
        )

        for row, row_elements in enumerate(self.elements):
            for col, el in enumerate(row_elements):
                if el is not None:
                    el.align()
                    if el.width > widths[col]:
                        widths[col] = el.width
                    if el.height > heights[row]:
                        heights[row] = el.height

        for row, (row_elements, el_height) in enumerate(zip(self.elements, heights)):
            padding_height = self.paddings_height[min(row + 1, self.nrow - 1)]
            if padding_height is None:
                padding_height = self.padding_height

            x = 0
            for col, (el, el_width) in enumerate(zip(row_elements, widths)):
                if el is not None:
                    el.pos = (x, y)

                    next_y = max(next_y, y + el.height + padding_height)
                    height = max(height, next_y)

                    width = max(width, x + el.width)

                padding_width = self.paddings_width[min(col + 1, self.ncol - 1)]
                if padding_width is None:
                    padding_width = self.padding_width

                x += el_width + padding_width
            y += el_height + padding_height

        if self.title is not None:
            self.title.dim = (width, self.title.dim[1])

        self.dim = (width, height)

    def set_title(self, label):
        self.title = Title(label)

    def position(self, fig, pos=(0, 0)):
        pos = self.pos[0] + pos[0], self.pos[1] + pos[1]
        if self.title is not None:
            self.title.position(fig, pos)

        for row_elements in self.elements:
            for el in row_elements:
                if el is not None:
                    el.position(fig, pos)

    def __getitem__(self, index):
        return self.elements[index[0]][index[1]]

    def __setitem__(self, index, v):
        row = index[0]
        col = index[1]

        if not isinstance(row, int) or not isinstance(col, int):
            raise TypeError("row and col must be integers")

        if row >= (self.nrow):
            # add new row(s)
            for i in range(self.nrow, row + 1):
                self.elements.append([None for _ in range(self.ncol)])
            self.nrow = row + 1
            self.paddings_height.append(None)

        if col >= (self.ncol):
            # add new col(s)
            for i in range(self.ncol, col + 1):
                for row_ in self.elements:
                    row_.append(None)
            self.ncol = col + 1
            self.paddings_width.append(None)

        self.elements[row][col] = v

    def add_under(self, el, column=0, padding=None):
        if (self.nrow == 1) and self[0, 0] is None:
            row = 0
        else:
            row = self.nrow

        # get column index if column is a panel
        if "grid.Element" in column.__class__.__mro__.__repr__():
            try:
                column = (
                    np.array(self.elements).flatten().tolist().index(column) % self.ncol
                )
            except ValueError as e:
                raise ValueError(
                    "The panel specified as column was not found in the grid"
                ) from e
        self[row, column] = el
        if padding is not None:
            self.paddings_height[row] = padding
        return el

    def add_right(self, el, row=0, padding=None):
        if (self.ncol == 1) and (self[0, 0] is None):
            column = 0
        else:
            column = self.ncol

        # get column index if column is a panel
        if "grid.Element" in row.__class__.__mro__.__repr__():
            try:
                row = np.array(self.elements).flatten().tolist().index(row) // self.ncol
            except ValueError as e:
                raise ValueError(
                    "The panel specified as row was not found in the grid"
                ) from e

        self[row, column] = el
        if padding is not None:
            self.paddings_width[column] = padding
        return el


class _Figure(mpl.figure.Figure):
    main: Panel

    def __init__(self, main: Panel, *args, **kwargs):
        self.main = main
        global active_fig
        active_fig = self
        super().__init__(*args, **kwargs)

    def plot(self):
        self.main.align()
        self.set_size_inches(*self.main.dim)
        self.main.position(self)

    def set_tight_bounds(self):
        new_bounds = self.get_tightbbox().extents
        current_size = self.get_size_inches()
        new_size = new_bounds[2] - new_bounds[0], new_bounds[3] - new_bounds[1]

        self.set_figwidth(new_bounds[2] - new_bounds[0])
        self.set_figheight(new_bounds[3] - new_bounds[1])

        for ax in self.axes:
            new_bbox = ax.get_position()
            current_axis_bounds = ax.get_position().extents
            new_bbox = mpl.figure.Bbox(
                np.array(
                    [
                        (current_axis_bounds[0] - (new_bounds[0] / current_size[0]))
                        / ((new_bounds[2] - new_bounds[0]) / current_size[0]),
                        (current_axis_bounds[1] - (new_bounds[1] / current_size[1]))
                        / ((new_bounds[3] - new_bounds[1]) / current_size[1]),
                        (current_axis_bounds[2] - (new_bounds[0] / current_size[0]))
                        / ((new_bounds[2] - new_bounds[0]) / current_size[0]),
                        (current_axis_bounds[3] - (new_bounds[1] / current_size[1]))
                        / ((new_bounds[3] - new_bounds[1]) / current_size[1]),
                    ]
                ).reshape((2, 2))
            )
            ax.set_position(new_bbox)


def Figure(main, *args, **kwargs):
    return plt.figure(*args, main=main, **kwargs, FigureClass=_Figure)
