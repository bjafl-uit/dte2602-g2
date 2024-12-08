"""Color map for class data and functions for color conversion.

This module provides a color map for visualizing data with different classes. 
It provides functionality for converting between different color formats,
lightening colors, and returning colors based on a key or proportion 
of the colormap length. The module aslo provides color conversion functions.

Classes
-------
DiscreteColorMap
    Color map suitable for class data.

Methods
-------
convert_color
    Convert RGB(A) color to specified RGB format.
    Supported formats to convert to/from are:
    - int: integer RGB tuple
    - float: float RGB tuple
    - hex: hexadecimal RGB string

Notes
-----
- The color map and conversion function is utilized in the plot module for
    visualizing model performance.
"""
from typing import Optional, Union, Literal


class DiscreteColorMap:
    """Color map for visualizing different classes of data.

    Constants
    ---------
    TAB10_COLORS: list[str]
        Default colors from matplotlib tab10 colormap
        https://matplotlib.org/stable/users/explain/colors/colormaps.html
    """

    TAB10_COLORS = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    def __init__(self, colors: Optional[list[str]] = None) -> None:
        """Initialize color map."""
        if colors is None:
            colors = self.TAB10_COLORS
        self._colors = [_color_to_float(c) for c in colors]

    @property
    def colors(self) -> list[tuple[float, float, float]]:
        """Return list of RGB colors."""
        return self._colors

    @colors.setter
    def colors(self, colors: list[Union[str, tuple[int, float]]]) -> None:
        """Set list of RGB colors."""
        self._colors = [self._color_to_float(c) for c in colors]

    def get_color(
            self,
            key: int,
            lighten_factor: float = 0
    ) -> tuple[float, float, float]:
        """Return color based on key.

        Parameters
        ----------
        key: int
            Integer key
        lighten_factor: float
            Factor to lighten color by. 0.0 is no change, 1.0 is white.

        Returns
        -------
        color: tuple[float, float, float]
            RGB color tuple
        """
        color = self.colors[key]
        if lighten_factor > 0:
            color = self.lighten_color(color, lighten_factor)
        return color

    def lighten_color(
            self,
            color: Union[str, tuple[int | float]],
            lighten_factor: float
    ) -> tuple[float, float, float]:
        """Lighten color by factor.

        Parameters
        ----------
        color: str | tuple[int | float]
            Color to lighten
        lighten_factor: float
            Factor to lighten color by. 0.0 is no change, 1.0 is white.

        Returns
        -------
        lightened_color: tuple[float, float, float]
            Lightened color
        """
        color = _color_to_float(color)
        return tuple(min(1.0, c + (1 - c) * lighten_factor) for c in color)

    def __getitem__(self, key: int) -> tuple[float, float, float]:
        """Return color based on key.

        Parameters
        ----------
        key: int
            Integer key

        Returns
        -------
        color: tuple[float, float, float]
            RGB color tuple (1.0 is max value)
        """
        return self.colors[key]

    def __call__(self, value: float) -> tuple[float, float, float]:
        """Return color based on proportion of colormap length.

        Parameters
        ----------
        value: float
            Value between 0.0 and 1.0

        Returns
        -------
        color: tuple[float, float, float]
            RGB color tuple (1.0 is max value)
        """
        value = max(0.0, min(1.0, value))
        return self.colors[int(value * len(self.colors))]

    def __len__(self) -> int:
        """Return number of colors in color map."""
        return len(self.colors)

    def __iter__(self):
        """Return iterator over colors."""
        return iter(self.colors)


# Color conversion methods

def convert_color(
        color: Union[str, tuple[int | float]],
        convert_to: Literal['int', 'float', 'hex'] = 'float'
) -> Union[str, tuple[int | float]]:
    """Convert RGB(A) color to specified RGB format.

    Parameters
    ----------
    color: str | tuple[int | float]
        Color to convert
    convert_to: str
        Format to convert to

    Returns
    -------
    converted_color: str | tuple[int | float]
        Converted rgb color

    Notes
    -----
    - Alpha channel is ignored and removed in conversion.
    """
    if convert_to == 'int':
        return _color_to_int(color)
    elif convert_to == 'float':
        return _color_to_float(color)
    elif convert_to == 'hex':
        return _color_to_hex(color)


def _color_to_int(
        color: str | tuple[int | float]
) -> tuple[int, int, int]:
    """Convert color to integer rgb tuple."""
    if isinstance(color, str):
        return _hex_to_int(color)
    color = color[:3]
    if isinstance(color[0], float):
        return tuple(int(c * 255) for c in color)
    return color


def _color_to_float(
        color: str | tuple[int | float]
) -> tuple[float, float, float]:
    """Convert color to float format."""
    if isinstance(color, str):
        color = _hex_to_int(color)
    color = color[:3]
    if isinstance(color[0], int):
        return tuple(c / 255 for c in color)
    return color


def _color_to_hex(color: str | tuple[int | float]) -> str:
    """Convert color to hexadecimal rgb string."""
    if isinstance(color, tuple):
        color = color[:3]
        if isinstance(color[0], float):
            color = tuple(int(c * 255) for c in color)
        return "#{:02x}{:02x}{:02x}".format(*color)
    return color


def _hex_to_int(color: str) -> tuple[int, int, int]:
    """Convert hexadecimal color to rgb int tuple."""
    if color.startswith("#"):
        color = color[1:]
    return tuple(int(color[i:i+2], 16) for i in range(0, 6, 2))
