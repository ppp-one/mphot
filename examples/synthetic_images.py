"""
Synthetic images of galaxies and nebulae from an mphot exposure plan.

The shape of the target comes from a survey image (CDS hips2fits service). The
brightness and the noise come from `mphot.get_exposure_extended`. This needs
an internet connection.
"""

import math
import urllib.parse
import urllib.request

import numpy as np
from scipy.signal import fftconvolve

HIPS2FITS = "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"


def survey_cutout(hips, ra, dec, n_pix, scale):
    """Get a square survey image, north up, as an array."""
    query = urllib.parse.urlencode(
        {
            "hips": hips,
            "width": n_pix,
            "height": n_pix,
            "fov": n_pix * scale / 3600,
            "ra": ra,
            "dec": dec,
            "projection": "TAN",
            "format": "fits",
        }
    )
    data = urllib.request.urlopen(f"{HIPS2FITS}?{query}", timeout=300).read()

    # A minimal FITS reader: 80-character header cards, then the image.
    header, offset = {}, 0
    while data[offset : offset + 3] != b"END":
        card = data[offset : offset + 80].decode("ascii", "replace")
        if card[8:10] == "= ":
            header[card[:8].strip()] = card[10:].split("/")[0].strip()
        offset += 80
    start = (offset // 2880 + 1) * 2880
    dtype = {-32: ">f4", -64: ">f8", 16: ">i2", 32: ">i4"}[int(header["BITPIX"])]
    image = np.frombuffer(data[start:], dtype=dtype, count=n_pix * n_pix)
    image = image.reshape(n_pix, n_pix).astype(float)
    image = image * float(header.get("BSCALE", 1)) + float(header.get("BZERO", 0))
    image = np.nan_to_num(image, nan=np.nanmedian(image))
    return image[::-1]  # FITS rows go from south to north


def gaussian_kernel(fwhm, scale):
    """A normalised Gaussian with a FWHM in arcsec, for pixels of `scale` arcsec."""
    sigma = fwhm / 2.355 / scale
    r = np.arange(-int(4 * sigma) - 1, int(4 * sigma) + 2)
    kernel = np.exp(-(r[:, None] ** 2 + r[None, :] ** 2) / (2 * sigma**2))
    return kernel / kernel.sum()


def simulate(
    target,
    plan,
    props,
    hips,
    survey_seeing,
    seeing,
    n_subs,
    bin_factor=3,
    fov=12.0,
    seed=1,
):
    """
    Make a synthetic stack of `n_subs` sub-exposures.

    The steps are:

    1. Get the survey image and remove its sky.
    2. Blur it to the site seeing. A survey with worse seeing stays as it is.
    3. Scale it so that its mean in the catalogue ellipse is the mean from mphot.
    4. Add the noise of the stack: Poisson noise of target, sky and dark
       current, and read noise.

    The sum of n Poisson values is one Poisson value of the sum, and the sum of
    n read noises is one Gaussian that is sqrt(n) wider. Thus one random draw
    makes the full stack. The survey image has its own noise. Where it is less
    than the noise of the stack, only the difference is added. Where it is
    more, the survey image is smoothed by up to 1.5".

    Args:
        target: A `mphot.Target`.
        plan (dict): The result of `mphot.get_exposure_extended`.
        props (dict): The instrument properties of the plan.
        hips (str): The survey, for example "CDS/P/SDSS9/r".
        survey_seeing (float): Seeing of the survey [arcsec].
        seeing (float): Seeing at the site [arcsec].
        n_subs (int): Number of sub-exposures.
        bin_factor (int): The image has pixels of bin_factor x bin_factor
            pixels of the plan. The plan can itself be binned.
        fov (float): Width of the image [arcmin].
        seed (int): Seed of the random numbers.

    Returns:
        tuple: The image [e/s per pixel of the plan, sky removed], and the
            mean surface brightness of the target in the same unit.
    """

    rng = np.random.default_rng(seed)
    b, d = plan["N_sky [e/pix/s]"], plan["N_dc [e/pix/s]"]
    rn, t = plan["N_rn [e_rms/pix]"], plan["t_sub [s]"]
    mean = plan["N_target [e/pix/s]"] * 10 ** (
        0.4 * (plan["mu_target [mag/arcsec2]"] - plan["mu_mean [mag/arcsec2]"])
    )
    k2 = bin_factor**2
    scale = bin_factor * plan['plate_scale ["/pix]']  # arcsec per image pixel
    # A binned plan has pixels of detector_k2 detector pixels, which saturate.
    detector_k2 = plan["pixel_binning"] ** 2

    # Get all of the catalogue ellipse and some sky around it.
    size = max(fov, 1.2 * target.major_axis)
    n_pix = round(size * 60 / scale)
    image = survey_cutout(hips, target.ra, target.dec, n_pix, scale)

    y, x = (np.indices(image.shape) - (n_pix - 1) / 2) * scale
    pa = math.radians(target.position_angle or 0.0)
    u = -x * math.sin(pa) + y * math.cos(pa)
    v = x * math.cos(pa) + y * math.sin(pa)
    inside = (u / (30 * target.major_axis)) ** 2 + (
        v / (30 * target.minor_axis)
    ) ** 2 <= 1
    sky = ~inside & (np.hypot(x, y) > 0.45 * size * 60)
    values = image[sky]
    image -= np.median(values[np.abs(values - np.median(values)) < 3 * values.std()])

    if seeing > survey_seeing:
        extra = math.sqrt(seeing**2 - survey_seeing**2)
        image = fftconvolve(image, gaussian_kernel(extra, scale), mode="same")
    image *= mean / image[inside].mean()

    def white_noise(img):
        """Pixel noise from the differences between adjacent sky pixels."""
        diff = (img[:, 1:] - img[:, :-1])[sky[:, 1:]]
        return 1.4826 * np.median(np.abs(diff - np.median(diff))) / math.sqrt(2)

    def stack_noise(rate):
        variance = n_subs * k2 * ((np.clip(rate, 0, None) + b + d) * t + rn**2)
        return np.sqrt(variance) / (n_subs * t * k2)

    # Smooth the survey image only where its noise is too high.
    widths = [0.0, 0.75, 1.5]
    versions = [image] + [
        fftconvolve(image, gaussian_kernel(w, scale), mode="same") for w in widths[1:]
    ]
    sigmas = [white_noise(v) for v in versions]
    choice = np.full(image.shape, len(widths) - 1)
    for i in reversed(range(len(widths))):
        choice[sigmas[i] <= 0.8 * stack_noise(versions[i])] = i
    image = np.choose(choice, versions)
    survey_sigma = np.choose(choice, sigmas)

    added = np.sqrt(np.clip(stack_noise(image) ** 2 - survey_sigma**2, 0, None))
    result = image + rng.normal(size=image.shape) * added
    full = props["well_depth"] * detector_k2
    saturated = (image + b + d) * t > full
    result[saturated] = full / t - b - d

    half = round(fov * 60 / scale / 2)
    c = n_pix // 2
    return result[c - half : c + half, c - half : c + half], mean


def show(ax, image, mean, title, caption=""):
    """Show an image with the same stretch for all targets, relative to the mean."""
    lo, soft, hi = -0.5, 0.25, 25.0
    shown = np.arcsinh((np.clip(image / mean, lo, hi) - lo) / soft) / np.arcsinh(
        (hi - lo) / soft
    )
    ax.imshow(shown, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, loc="left", fontsize=11, weight="bold")
    ax.text(
        0, -0.03, caption, transform=ax.transAxes, va="top", fontsize=9, color="#52514e"
    )
