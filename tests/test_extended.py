import json
import math

import pytest

import mphot
from mphot import extended
from mphot.extended import get_exposure_extended
from mphot.targets import (
    Target,
    TargetNotFound,
    UnsupportedTarget,
    resolve_target,
)

# The instrument presets of the web demo.
CCD = {
    "plate_scale": 0.35,
    "N_dc": 0.2,
    "N_rn": 6.328,
    "well_depth": 64000,
    "well_fill": 0.7,
    "read_time": 10.5,
    "r0": 0.5,
    "r1": 0.14,
}
INGAAS = {
    **CCD,
    "plate_scale": 0.31,
    "N_dc": 110,
    "N_rn": 90,
    "well_depth": 56000,
    "read_time": 0.1,
}
CMOS = {
    "plate_scale": 0.9,
    "N_dc": 0.1,
    "N_rn": 8.47,
    "well_depth": 1048560,
    "well_fill": 0.7,
    "read_time": 5,
    "r0": 0.254,
    "r1": 0.099,
}


@pytest.fixture(scope="module")
def systems():
    names = {}
    for key, efficiency, band in (
        ("ccd_r", "speculoos_Andor_iKon-L-936_-60", "r"),
        ("ingaas_j", "speculoos_PIRT_1280SciCam_-60", "J"),
        ("cmos_halpha", "generic_IMX461", "halpha"),
        ("cmos_luminance", "generic_IMX461", "luminance"),
    ):
        names[key], _ = mphot.generate_system_response(
            f"resources/systems/{efficiency}.csv", f"resources/filters/{band}.csv"
        )
    return names


# ----------------------------------------------------------------------------
# Names


@pytest.mark.parametrize(
    "name", ["M51", "m 51", "Messier 51", "NGC 5194", "ngc5194", "Whirlpool"]
)
def test_resolve_target_accepts_common_spellings(name):
    target = resolve_target(name)
    assert target.name == "NGC5194"
    assert target.kind == "galaxy"
    assert target.label == "M51 (NGC 5194, Whirlpool Galaxy)"


def test_resolve_target_follows_a_duplicate_entry():
    # OpenNGC lists IC 11 as a duplicate of NGC 281.
    assert resolve_target("IC 11").name == "NGC0281"


def test_resolve_target_reports_an_ambiguous_common_name():
    with pytest.raises(TargetNotFound, match="several objects"):
        resolve_target("Orion")
    assert resolve_target("Orion Nebula").name == "NGC1976"


@pytest.mark.parametrize(
    "name, reason",
    [
        ("M102", "NGC 5866"),
        ("M45", "open cluster"),
        ("M13", "globular cluster"),
        ("Horsehead", "dark nebula"),
    ],
)
def test_resolve_target_refuses_what_it_cannot_model(name, reason):
    with pytest.raises(UnsupportedTarget, match=reason):
        resolve_target(name)


def test_resolve_target_reports_an_unknown_name():
    with pytest.raises(TargetNotFound):
        resolve_target("NGC 99999")


def test_galaxy_shape_comes_from_the_catalogue():
    m51 = resolve_target("M51")
    assert (m51.major_axis, m51.minor_axis, m51.position_angle) == (13.71, 11.67, 163.0)
    # OpenNGC has no position angle for the Ring Nebula.
    assert resolve_target("M57").position_angle is None


def test_nebulae_carry_line_data_and_no_stellar_infrared_magnitudes():
    ring = resolve_target("M57")
    assert ring.kind == "planetary nebula"
    assert ring.halpha_source == "Acker et al. 1992 Hbeta"
    assert ring.halpha_flux > 0
    # OpenNGC's J for M57 is the central star, so it must not be used.
    assert "J" not in ring.magnitudes

    dumbbell = resolve_target("M27")
    assert dumbbell.line_ratios["oiii_5007"] == pytest.approx(1106 / 262, rel=1e-3)


# ----------------------------------------------------------------------------
# Physics


def _target(**kwargs) -> Target:
    base = {
        "name": "TEST",
        "label": "Test",
        "kind": "galaxy",
        "object_type": "G",
        "ra": 0.0,
        "dec": 0.0,
        "major_axis": 2.0,
        "minor_axis": 1.0,
    }
    return Target(**{**base, **kwargs})


def test_continuum_in_its_own_catalogue_band_gives_back_the_catalogue_magnitude():
    target = _target(magnitudes={"J": 10.0})
    band = extended._catalogue_band("J")

    source = extended._continuum(target, band, band_centre=1.235)

    mu = 10.0 + 2.5 * math.log10(target.area)
    expected = 10 ** (-0.4 * mu) * extended._vega_in_band("J")
    assert source["rate"] == pytest.approx(expected, rel=1e-9)


def test_emission_turns_the_halpha_flux_into_photons():
    flux = 1e-12  # erg/s/cm2
    # [NII] is inside the top hat below, so its ratio is set to 0.
    target = _target(
        kind="emission nebula",
        object_type="HII",
        halpha_flux=flux,
        halpha_source="test",
        line_ratios={"nii_6584": 0.0},
    )
    lam = extended._wavelengths()
    response = ((lam > 0.650) & (lam < 0.662)).astype(float)

    source = extended._emission(target, response, 0.656, 0.0)

    photon_energy = 6.62607015e-34 * 2.99792458e8 / 0.65628e-6
    expected = flux * 1e-3 / target.area / photon_energy
    assert source["rate"] == pytest.approx(expected, rel=1e-9)
    assert list(source["lines"]) == ["Halpha"]


def test_extinction_curve():
    assert extended.extinction_curve(0.549) == pytest.approx(1.0, abs=0.01)
    # Cardelli et al. give A(K)/A(V) of about 0.11.
    assert extended.extinction_curve(2.2) == pytest.approx(0.11, abs=0.01)


def test_disk_peak_offset_of_m51():
    # For M51, B = 8.61 over 13.71' x 11.67' is 22.75 mag/arcsec2, and an
    # exponential disk with that mean inside the 25 mag isophote has its
    # centre 2.55 mag brighter.
    assert extended._disk_peak_offset(resolve_target("M51")) == pytest.approx(
        2.55, abs=0.01
    )


def test_single_magnitude_uses_the_default_spectrum():
    target = _target(magnitudes={"B": 12.0})
    source = extended._continuum(target, extended._catalogue_band("V"), 0.545)
    # 4990 K is the grid temperature closest to the 5000 K default.
    assert source["spectrum"] == "4990 K stellar spectrum (assumed)"
    assert any("only one magnitude" in w for w in source["warnings"])


# ----------------------------------------------------------------------------
# Exposure plans


def test_galaxy_plan(systems):
    props = {**CCD, "name": systems["ccd_r"]}
    sky = {"pwv": 2.5, "airmass": 1.1, "seeing": 1.35}
    result = get_exposure_extended("M51", props, sky, snr=10)

    json.dumps(result)
    assert result["kind"] == "galaxy"
    assert result["airmass"] == 1.1

    # The sub-exposure is the longer of two minimums: background-limited (sky
    # and dark variance = 10 x read noise^2), and readout at most 10% of the
    # time. Here the 10.5 s readout needs the longer time.
    b, d = result["N_sky [e/pix/s]"], result["N_dc [e/pix/s]"]
    background_limited = 10 * CCD["N_rn"] ** 2 / (b + d)
    readout_limited = CCD["read_time"] * 9
    assert result["t_background [s]"] == pytest.approx(background_limited)
    assert result["t_sub [s]"] == pytest.approx(
        max(background_limited, readout_limited)
    )
    assert result["t_sub set by"] == "readout"
    assert result["exposure [s]"] == pytest.approx(result["subs"] * result["t_sub [s]"])
    # Adding one more sub-exposure would overshoot, so the plan just reaches it.
    assert result["SNR per sub"] * math.sqrt(result["subs"]) >= 10 * 0.95


def test_one_magnitude_fainter_needs_about_six_times_longer(systems):
    props = {**CCD, "name": systems["ccd_r"]}
    sky = {"pwv": 2.5, "airmass": 1.1}
    common = {"snr": 50, "flat_error": 0.0, "peak_offset": 0.0}

    # Well below the sky, where the time scales as 10**(0.8 * mag).
    fainter = get_exposure_extended("M51", props, sky, mu_offset=4, **common)
    faintest = get_exposure_extended("M51", props, sky, mu_offset=5, **common)

    ratio = faintest["exposure [s]"] / fainter["exposure [s]"]
    assert 10**0.8 * 0.9 < ratio < 10**0.8 * 1.02


def test_flat_field_errors_limit_the_snr(systems):
    props = {**INGAAS, "name": systems["ingaas_j"]}
    result = get_exposure_extended(
        "M51", props, {"pwv": 2.5, "airmass": 1.1}, snr=10, flat_error=0.01
    )

    # M51's mean is 4 mag below the J sky, so 1% flats allow only SNR ~2.
    assert result["SNR limit"] < 10
    assert result["subs"] is None
    assert result["exposure [s]"] is None
    assert any("is not possible" in w for w in result["warnings"])

    # There is no limit by default.
    assert (
        get_exposure_extended("M51", props, {"pwv": 2.5, "airmass": 1.1})["SNR limit"]
        is None
    )
    json.dumps(result)


def test_airmass_outside_the_sky_model_is_refused(systems):
    # At sea level, airmass 2.5 is airmass 3.4 at Paranal, beyond the grid.
    props = {**CCD, "name": systems["ccd_r"]}
    with pytest.raises(ValueError, match="outside mphot's sky model"):
        get_exposure_extended("M51", props, {"pwv": 2.5, "airmass": 2.5}, h=0)


def test_nebula_in_a_narrowband_filter(systems):
    props = {**CMOS, "name": systems["cmos_halpha"]}
    result = get_exposure_extended("M57", props, {"pwv": 2.5, "airmass": 1.1})

    lines = result["line counts [e/pix/s]"]
    assert max(lines, key=lines.get) == "Halpha"
    # A 3 nm filter centred on Halpha lets through almost none of [OIII].
    assert "[OIII] 5007" not in lines
    assert result["subs"] >= 1


def test_one_sub_exposure_is_only_as_long_as_the_snr_needs(systems):
    # M57 is bright in Halpha. One exposure, shorter than the background-limited
    # time, gives SNR 10.
    props = {**CMOS, "name": systems["cmos_halpha"]}
    result = get_exposure_extended("M57", props, {"pwv": 2.5, "airmass": 1.1}, snr=10)

    assert result["subs"] == 1
    assert result["t_sub [s]"] < result["t_background [s]"]
    assert result["t_sub set by"] == "SNR"
    assert result["SNR per sub"] == pytest.approx(10, rel=1e-6)


def test_band_between_catalogue_bands_is_called_interpolation(systems):
    # The J band of the InGaAs camera lies between V and K, but past B and V
    # when those are all the target has.
    props = {**INGAAS, "name": systems["ingaas_j"]}
    target = _target(magnitudes={"V": 10.0, "K": 8.0}, major_axis=5.0, minor_axis=5.0)
    result = get_exposure_extended(target, props, {"pwv": 2.5, "airmass": 1.2})
    assert any("between the catalogue bands V and K" in w for w in result["warnings"])

    target = _target(magnitudes={"B": 10.5, "V": 10.0}, major_axis=5.0, minor_axis=5.0)
    result = get_exposure_extended(target, props, {"pwv": 2.5, "airmass": 1.2})
    assert any("The error can be up to 1 mag" in w for w in result["warnings"])


def test_snr_close_to_the_flat_field_limit_is_explained(systems):
    props = {**CCD, "name": systems["ccd_r"]}
    sky = {"pwv": 2.5, "airmass": 1.1}
    limit = get_exposure_extended("M51", props, sky, flat_error=0.01)["SNR limit"]

    result = get_exposure_extended("M51", props, sky, snr=0.9 * limit, flat_error=0.01)
    assert result["subs"] is not None
    assert any("times longer" in w for w in result["warnings"])


def test_saturation_magnitude_of_stars(systems):
    props = {**CCD, "name": systems["ccd_r"]}

    def magnitude(seeing):
        sky = {"pwv": 2.5, "airmass": 1.1, "seeing": seeing}
        return get_exposure_extended("M51", props, sky)["saturation magnitude [mag]"]

    # In better seeing, the light of a star is in fewer pixels, so fainter
    # stars fill the well to well_fill.
    assert magnitude(1.0) > magnitude(2.0)
    no_seeing = get_exposure_extended("M51", props, {"pwv": 2.5, "airmass": 1.1})
    assert no_seeing["saturation magnitude [mag]"] is None


def test_nebula_in_a_broad_filter_matches_its_visual_magnitude(systems):
    # M42's lines come from the Halpha map and typical line ratios. Spread over
    # the catalogue ellipse, its V = 4.0 gives 21.96 mag/arcsec2. The two are
    # independent, so they need only agree roughly.
    props = {**CMOS, "name": systems["cmos_luminance"]}
    result = get_exposure_extended("M42", props, {"pwv": 2.5, "airmass": 1.1})
    assert result["mu_mean [mag/arcsec2]"] == pytest.approx(21.96, abs=0.5)


def test_high_dark_current_keeps_the_readout_below_ten_percent(systems):
    # The dark current is 7 times the sky, so the background-limited time is
    # only a few seconds. The readout limit makes the sub-exposures longer.
    props = {**CCD, "name": systems["ccd_r"], "N_dc": 100}
    result = get_exposure_extended("M51", props, {"pwv": 2.5, "airmass": 1.1})

    assert result["N_dc [e/pix/s]"] > result["N_sky [e/pix/s]"]
    assert result["t_background [s]"] < 5
    assert result["t_sub set by"] == "readout"
    readout = CCD["read_time"] / (result["t_sub [s]"] + CCD["read_time"])
    assert readout == pytest.approx(0.1)


def test_high_read_noise_gives_one_exposure_and_a_note(systems):
    # With 100 e read noise, a background-limited sub-exposure is almost two
    # hours. One shorter exposure gives the SNR.
    props = {**CCD, "name": systems["ccd_r"], "N_rn": 100}
    result = get_exposure_extended("M51", props, {"pwv": 2.5, "airmass": 1.1})
    assert result["subs"] == 1
    assert result["t_sub set by"] == "SNR"
    assert any(w.startswith("One exposure of") for w in result["warnings"])

    # A longest exposure makes it a stack.
    props["max_exp"] = 300
    result = get_exposure_extended("M51", props, {"pwv": 2.5, "airmass": 1.1})
    assert result["subs"] > 1
    assert result["t_sub set by"] == "max_exp"


# ----------------------------------------------------------------------------
# Pixel binning and sky brightness


def test_digital_binning_keeps_the_precision_of_a_star(systems):
    # Digital binning only adds detector pixels after the readout. The
    # aperture holds the same pixels, so the precision does not change.
    props = {**CCD, "name": systems["ccd_r"]}
    sky = {"pwv": 2.5, "airmass": 1.1, "seeing": 1.35}
    image, _, parts = mphot.get_precision(props, sky, 5800, 50)
    binned, _, binned_parts = mphot.get_precision(
        {**props, "pixel_binning": 4}, sky, 5800, 50
    )

    assert binned["All"] == pytest.approx(image["All"], rel=1e-9)
    assert binned_parts["t [s]"] == pytest.approx(parts["t [s]"])
    assert binned_parts["N_sky [e/pix/s]"] == pytest.approx(
        16 * parts["N_sky [e/pix/s]"]
    )
    assert binned_parts['plate_scale ["/pix]'] == pytest.approx(4 * CCD["plate_scale"])


def test_on_chip_binning_reads_once(systems):
    props = {**CCD, "name": systems["ccd_r"]}
    sky = {"pwv": 2.5, "airmass": 1.1, "seeing": 1.35}
    digital, _, _ = mphot.get_precision({**props, "pixel_binning": 4}, sky, 5800, 50)
    on_chip, _, _ = mphot.get_precision(
        {**props, "pixel_binning": 4, "pixel_binning_type": "on-chip"}, sky, 5800, 50
    )
    # 16 readouts become 1, so the read noise falls by a factor of 4.
    assert on_chip["Read noise"] == pytest.approx(digital["Read noise"] / 4)

    with pytest.raises(ValueError, match="pixel_binning_type"):
        mphot.get_precision({**props, "pixel_binning_type": "average"}, sky, 5800, 50)


def test_sky_factor_scales_the_sky(systems):
    props = {**CCD, "name": systems["ccd_r"]}
    sky = {"pwv": 2.5, "airmass": 1.1, "seeing": 1.35}
    _, _, dark_site = mphot.get_precision(props, sky, 5800, 50)
    _, _, bright_site = mphot.get_precision(props, {**sky, "sky_factor": 5}, 5800, 50)
    assert bright_site["N_sky [e/pix/s]"] == pytest.approx(
        5 * dark_site["N_sky [e/pix/s]"]
    )

    plan = get_exposure_extended("M51", props, sky)
    brighter = get_exposure_extended("M51", props, {**sky, "sky_factor": 5})
    assert brighter["N_sky [e/pix/s]"] == pytest.approx(5 * plan["N_sky [e/pix/s]"])


def test_binning_of_an_extended_source(systems):
    props = {**CCD, "name": systems["ccd_r"]}
    sky = {"pwv": 2.5, "airmass": 1.1, "seeing": 1.35}
    # SNR 100 needs a stack with and without binning. At SNR 10, one binned
    # exposure is sufficient.
    plan = get_exposure_extended("M51", props, sky, snr=100)
    binned = get_exposure_extended("M51", {**props, "pixel_binning": 4}, sky, snr=100)

    # Digital binning changes neither the sub-exposure nor when the detector
    # pixels fill, but one image pixel collects 16 times the light.
    assert binned["t_sub [s]"] == pytest.approx(plan["t_sub [s]"])
    assert binned["t_saturation [s]"] == pytest.approx(plan["t_saturation [s]"])
    assert binned["saturation magnitude [mag]"] == pytest.approx(
        plan["saturation magnitude [mag]"]
    )
    assert binned["SNR per sub"] == pytest.approx(4 * plan["SNR per sub"])
