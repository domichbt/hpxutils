"""Utility functions for dealing with Healpix maps."""

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import warnings

import healpy as hp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class HealpixArray(np.ndarray):
    """Healpix map stored in a numpy array."""

    def __new__(cls, healpix_array: np.ndarray, nest: bool, bad: float = hp.UNSEEN):
        """
        Healpix map stored in a numpy array.

        Parameters
        ----------
        healpix_array : np.ndarray
            Numpy array storing the HEALPix map.
        nest : bool
            True for nested ordering and False for ring ordering.
        bad : float, optional
            Value that describes a bad pixel, by default hp.UNSEEN.
        """
        obj = np.asarray(healpix_array).view(cls)
        obj.nest = nest
        obj.bad = bad
        obj.nside = hp.get_nside(obj)
        obj.npix = obj.shape[-1]
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.nest = getattr(obj, "nest", None)
        self.bad = getattr(obj, "bad", hp.UNSEEN)
        self.nside = getattr(obj, "nside", None)
        self.npix = getattr(obj, "npix", None)

    def where_bad(self) -> np.ndarray:
        """Boolean mask indicating where the bad values of the array are located."""
        if self.bad is np.nan:
            return np.isnan(self.view(np.ndarray))
        else:
            return np.equal(self.view(np.ndarray), self.bad)

    def plot(self, **kwargs):
        """
        Wrap :py:fun:`downmock.plotting.hpcolormesh` using the instance's internal map and ordering information.

        Returns
        -------
        matplotlib.axes.Axes
            Axes where the map was plotted.
        matplotlib.collections.QuadMesh
            QuadMesh returned by :py:fun:`matplotlib.pyplot.pcolormesh`.
        """
        return hpcolormesh(healpix_map=self.view(np.ndarray), nest=self.nest, **kwargs)

    def reorder(self, nest_out: bool, inplace: bool = True) -> Self:
        """
        Change the ordering of the :py:class:`HealpixArray` instance in place.

        Parameters
        ----------
        nest_out : bool
            ``True`` for nested, ``False`` for ring ordering. If the current order is already ``nest_out``, nothing is done.
        inplace: bool
            ``True`` for in-place operation, ``False`` to return a new reordered :py:class:`HealpixArray`. Default is ``True``.
        """
        if nest_out == self.nest:
            if inplace:
                pass
            else:
                return self.copy()
        else:
            new_array = hp.reorder(
                map_in=self.view(np.ndarray),
                inp="NEST" if self.nest else "RING",
                out="NEST" if nest_out else "RING",
            )
            if inplace:
                np.copyto(
                    dst=self,
                    src=new_array,
                )
                self.nest = nest_out
            else:
                return HealpixArray(
                    healpix_array=new_array,
                    nest=nest_out,
                    bad=self.bad,
                )

    def ud_grade(
        self,
        nside_out: int,
        nest_out: bool | None = None,
        pess: bool = False,
        power: float | None = None,
        dtype: type | None = None,
    ) -> Self:
        """
        Return a :py:class:`HealpixArray` with degraded or upgraded resolution using :py:fun:`healpy.pixelfunc.ud_grade`.

        Parameters
        ----------
        nside_out : int
            Output resolution.
        nest_out : bool | None, optional
            Output ordering, by default ``None`` for the same as the input.
        pess : bool, optional
            Pessimistic, by default ``False``. If ``True``, in degrading, reject pixels which contains a bad sub_pixel. Otherwise, estimate average with good pixels.
        power : float | None, optional
            If non-zero, divide the result by ``(nside_in/nside_out)**power`` Examples: ``power=-2`` keeps the sum of the map invariant (useful for hitmaps), ``power=2`` divides the mean by another factor of ``(nside_in/nside_out)**2`` (useful for variance maps).
        dtype : type | None, optional
            the type of the output map, by default None

        Returns
        -------
        HealpixArray
            A new :py:class:`HealpixArray` with the upgraded or degraded map.
        """
        if nest_out is None:
            nest_out = self.nest
        if nest_out == self.nest and nside_out == self.nside:
            return self.copy()
        else:
            new_array = hp.ud_grade(
                map_in=self.view(np.ndarray),
                nside_out=nside_out,
                pess=pess,
                order_in="NEST" if self.nest else "RING",
                order_out="NEST" if nest_out else "RING",
                power=power,
                dtype=dtype,
            )
            return HealpixArray(healpix_array=new_array, nest=nest_out, bad=self.bad)


class HealpixMask(HealpixArray):
    """Boolean mask stored in a Healpix array. Always stored in nested scheme.

    The class supports elementwise comparison via operators & and |, **even for different nsides**. If two different with different ``nside`` are compared, the returned :py:class:``HealpixMask`` will have the greater ``nside`` of the two.
    """

    nest = True
    order = "NEST"

    def __new__(cls, mask: np.ndarray, nest: bool):
        """
        Healpix boolean mask stored in a numpy array. Always stored in nested scheme. Does not support bad pixels.

        Parameters
        ----------
        healpix_array : np.ndarray
            Numpy array storing the HEALPix map.
        nest : bool
            Input ordering. ``True`` for nested ordering and ``False`` for ring ordering.
        """
        if not isinstance(mask.dtype, np.dtypes.BoolDType):
            raise TypeError("Cannot use a non-boolean mask.")
        obj = super().__new__(cls, healpix_array=mask, nest=nest, bad=False)
        if not nest:
            obj.reorder(nest_out=cls.nest, inplace=True)
        return obj

    def where_bad(self):
        return NotImplemented

    def as_order(self, nest: bool) -> np.ndarray:
        if nest == self.nest:
            return self.view(np.ndarray)
        else:
            return self.reorder(nest_out=nest, inplace=False).view(np.ndarray)

    def __and__(self, mask1: Self) -> Self:
        """
        Intersection (pixel-wise logical and) of two boolean Healpix masks. The returned mask has the highest resolution of the two input masks.

        Parameters
        ----------
        mask1 : downmock.contamination_map.HealpixMask
            HealPix mask.

        Returns
        -------
        Self
            New HealpixMask instance with the biggest nside of the two original masks, representing the intersection of both masks.
        """
        if mask1.nside == self.nside:
            return HealpixMask(
                mask1.view(np.ndarray) & self.view(np.ndarray), nest=self.nest
            )
        elif mask1.nside < self.nside:
            mask_sd, mask_hd = mask1, self
        else:
            mask_sd, mask_hd = self, mask1
        # Now upgrade the resolution of the SD mask
        upgraded_mask_sd = hp.ud_grade(
            mask_sd.view(np.ndarray),
            nside_out=mask_hd.nside,
            order_in=self.order,
            order_out=self.order,
        )
        return HealpixMask(upgraded_mask_sd & mask_hd.view(np.ndarray), nest=self.nest)

    def __or__(self, mask1: Self) -> Self:
        """
        Union (pixel-wise logical or) of two boolean Healpix masks. The returned mask has the highest resolution of the two input masks.

        Parameters
        ----------
        mask1 : downmock.contamination_map.HealpixMask
            HealPix mask.

        Returns
        -------
        Self
            New HealpixMask instance with the biggest nside of the two original masks, representing the union of both masks.
        """
        if mask1.nside == self.nside:
            return HealpixMask(
                mask1.view(np.ndarray) | self.view(np.ndarray), nest=self.nest
            )
        elif mask1.nside < self.nside:
            mask_sd, mask_hd = mask1, self
        else:
            mask_sd, mask_hd = self, mask1
        # Now upgrade the resolution of the SD mask
        upgraded_mask_sd = hp.ud_grade(
            mask_sd.view(np.ndarray),
            nside_out=mask_hd.nside,
            order_in=self.order,
            order_out=self.order,
        )
        return HealpixMask(upgraded_mask_sd | mask_hd.view(np.ndarray), nest=self.nest)


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
