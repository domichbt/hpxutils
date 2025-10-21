"""Healpix plotting utilies."""

import warnings

import healpy as hp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .healpix_array import HealpixArray


def radec_to_density(
    ra: np.ndarray,
    dec: np.ndarray,
    nside: int,
    nest: bool = False,
    weights: np.ndarray = None,
    empty_value=0.0,
    return_dtype=float,
) -> HealpixArray:
    """
    From a RA, Dec catalog, return the HEALPix density map in the (nside, nest) parameters.

    Parameters
    ----------
    ra : np.ndarray
        Right ascensions.
    dec : np.ndarray
        Declinations.
    nside : int
        nside for the output map.
    nest : bool, optional
        Ordering for the output map, by default False
    weights : np.ndarray, optional
        Optional weights for the density computation, by default None
    empty_value : float, optional
        What to use for empty pixels, by default 0.0
    return_dtype : type, optional
        Type of the returned array, by default float (must be compatible with ``empty_value``).

    Returns
    -------
    HealpixArray
        HEALPix density map.
    """
    density = (
        np.bincount(
            hp.ang2pix(nside=nside, theta=ra, phi=dec, lonlat=True, nest=nest),
            minlength=hp.nside2npix(nside),
            weights=weights,
        )
        .astype(float)
        .astype(return_dtype)
    )
    density[density == 0.0] = empty_value
    return HealpixArray(density, nest=nest, bad=empty_value)


def hpcolormesh(
    healpix_map: np.ndarray,
    nest: bool,
    ax: matplotlib.axes.Axes | None = None,
    projection: str = "mollweide",
    mesh_size: int = 1000,
    **kwargs,
) -> tuple[matplotlib.axes.Axes, matplotlib.collections.QuadMesh]:
    """
    Create a HEALPix map plot from a HEALPix map.

    If an existing :py:class:`matplotlib.axes.Axes` **in the right projection already** is passed, the map will be plotted on it. Otherwise, a new Axes in the current figure is created, with the required ``projection``.

    Parameters
    ----------
    healpix_map : np.ndarray
        Map to plot.
    nest : bool
        `True` if the map is in nested ordering, `False` for ring ordering.
    ax : matplotlib.axes.Axes, optional
        Existing :py:class:`matplotlib.axes.Axes` **in the right projection already** to plot on ; if ``None``, a new one will be created. By default None.
    projection : str, optional
        Type of projection, by default "mollweide". See :py:fun:`matplotlib.projections.get_projection_names`. Default is Mollweide, can be set to None when passing an existing Axes object.
    mesh_size : int, optional
        Number of horizontal pixels to use for the colormesh, by default 1000
    **kwargs
        Additional arguments for :py:fun:`matplotlib.pyplot.pcolormesh` (colormap...).

    Returns
    -------
    matplotlib.axes.Axes
        Axes where the map was plotted.
    matplotlib.collections.QuadMesh
        QuadMesh returned by :py:fun:`matplotlib.pyplot.pcolormesh`.

    Notes
    -----
    By default, the returned Axes has its grid turned off and all labels removed (this emulates :py:fun:`healpy.mollview`'s behavior). This can be changed for example using
    .. code-block:: python

        ax.tick_params(axis='both', which='both', labelbottom=True, labeltop=False, labelleft=True, labelright=False)
        ax.grid(visible=True)

    Example
    -------
    Plot a basic healpix in a subfigure with some other plots:
    .. code-block:: python

        from downmock.plotting import hpcolormesh
        import numpy as np
        import matplotlib.pyplot as plt
        hpmap = np.arange(192)
        nest = True
        fig = plt.figure(layout="constrained", figsize=(10, 4))
        subfig1, subfig2 = fig.subfigures(1, 2, width_ratios=[2, 1])
        axmoll = subfig1.add_subplot(111, projection="mollweide")
        _, im = hpcolormesh(hpmap, nest=nest, ax=axmoll, projection=None) # set rasterize=True for neater pdf output
        axmoll.grid()
        axmoll.tick_params(axis='both', which='both', labelbottom=False, labeltop=False, labelleft=True, labelright=False)
        # can also use axmoll.axis('off') for drastic results
        ax1, ax2 = subfig2.subplots(2, 1)
        ax1.scatter(np.arange(10), np.cos(np.arange(10)))
        ax2.scatter(np.arange(10), np.sin(np.arange(10)))
        axmoll.set_title("Dummy Healpix map")
        subfig1.colorbar(im, ax=axmoll, location="bottom", shrink=0.5)
        subfig2.suptitle("Working subfigures")
        fig.suptitle("Really big title of my whole figure")
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 4), layout="constrained")
        ax = fig.add_subplot(111, projection=projection)
    else:
        if (projection is not None) and not isinstance(
            ax, matplotlib.projections.get_projection_class(projection)
        ):
            warnings.warn(
                "Provided projection is not the same as provided `Axes` projection. Ignoring argument "
                "`projection`. Match `projection` to your `Axes` or set it to `None` to avoid this warning.",
                stacklevel=2,
            )

    nside = hp.npix2nside(len(healpix_map))

    theta = np.linspace(np.pi, 0, mesh_size // 2)
    phi = np.linspace(-np.pi, np.pi, mesh_size)
    phi_mesh, theta_mesh = np.meshgrid(phi, theta)
    meshgrid_pixels = hp.ang2pix(nside=nside, phi=phi_mesh, theta=theta_mesh, nest=nest)

    # corresponding longitude and latitude
    longitude = np.radians(np.linspace(-180, 180, mesh_size))
    latitude = np.radians(np.linspace(-90, 90, mesh_size // 2))

    meshgrid_values = healpix_map[meshgrid_pixels]

    im = ax.pcolormesh(longitude, latitude, meshgrid_values, **kwargs)

    ax.tick_params(
        axis="both",
        which="both",
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False,
    )
    ax.grid(visible=False)

    return ax, im


def hpdensity(
    ra: np.ndarray,
    dec: np.ndarray,
    nside: int,
    nest: bool,
    weights: np.ndarray | None = None,
    cmap_name: str = "viridis",
    **kwargs,
) -> tuple[np.ndarray, matplotlib.axes.Axes, matplotlib.collections.QuadMesh]:
    """
    Plot a (HEALPix) density map based on RA, Dec coordinates and return the map.

    Parameters
    ----------
    ra : np.ndarray
        Right ascension in degrees.
    dec : np.ndarray
        Declination in degrees.
    nside : int
        Desired NSIDE of the output map.
    nest : bool
        Desired ordering of the output map, True for NESTED and False for RING.
    weights : np.ndarray
        Optional weights (same shape as the data).
    cmap_name : str, optional
        Name of the colormap to use, by default "viridis"

    Returns
    -------
    tuple[np.ndarray, matplotlib.axes.Axes, matplotlib.collections.QuadMesh]
        The density map, the Axes it was plotted on and the corresponding QuadMesh object.
    """
    density = radec_to_density(
        ra=ra,
        dec=dec,
        nside=nside,
        nest=nest,
        weights=weights,
        empty_value=np.nan,
        return_dtype=float,
    )
    cmap = plt.cm.get_cmap(cmap_name)
    cmap.set_bad("lightgray")
    ax, im = hpcolormesh(density, nest=nest, cmap=cmap, **kwargs)
    plt.colorbar(im, ax=ax, location="right")
    ax.tick_params(
        axis="both",
        which="both",
        labelbottom=False,
        labeltop=False,
        labelleft=True,
        labelright=False,
    )
    return density, ax, im


def density_summary(
    ra: np.ndarray,
    dec: np.ndarray,
    nside: int,
    nest: bool,
    weights: np.ndarray | None = None,
    cmap_name: str = "viridis",
) -> tuple[
    matplotlib.figure.Figure, matplotlib.axes.Axes, matplotlib.axes.Axes, np.ndarray
]:
    """
    Plot a (HEALPix) density map based on RA, Dec coordinates as well as a density histogram and return the map.

    Parameters
    ----------
    ra : np.ndarray
        Right ascension in degrees.
    dec : np.ndarray
        Declination in degrees.
    nside : int
        Desired NSIDE of the output map.
    nest : bool
        Desired ordering of the output map, True for NESTED and False for RING.
    weights : np.ndarray
        Optional weights (same shape as the data).
    cmap_name : str, optional
        Name of the colormap to use, by default "viridis"

    Returns
    -------
    tuple[ matplotlib.figure.Figure, matplotlib.axes.Axes, matplotlib.axes.Axes, np.ndarray ]
        The figure, mollweide Axes, histogram Axes and density map.
    """
    fig = plt.figure(figsize=(12, 4), layout="constrained")
    figmoll, fighist = fig.subfigures(1, 2, width_ratios=[2, 1])
    axmoll = figmoll.add_subplot(111, projection="mollweide")
    density_map, _, _ = hpdensity(
        ra=ra,
        dec=dec,
        nside=nside,
        nest=nest,
        weights=weights,
        cmap_name=cmap_name,
        ax=axmoll,
        rasterized=True,
    )

    axhist = fighist.add_subplot(111)
    axhist.hist(density_map)
    axhist.set_xlabel("Objects per pixel")

    return fig, axmoll, axhist, density_map
