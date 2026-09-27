"""Messier, NGC and IC objects, looked up by name."""

import math
import re
from dataclasses import dataclass, field
from functools import cache

import numpy as np
import pandas as pd

from mphot.paths import TARGETS_DIR

CATALOGUE_FILE = "openngc.csv"
"""The NGC, IC and addendum objects of OpenNGC, trimmed by
``resources/targets/build_targets.py``. OpenNGC is released under CC-BY-SA-4.0
(https://github.com/mattiaverga/OpenNGC)."""

NEBULA_LINES_FILE = "nebula_lines.csv"
"""Observed Halpha flux and line ratios of nebulae, from the same script."""

KINDS = {
    "G": "galaxy",
    "GPair": "galaxy",
    "GTrpl": "galaxy",
    "GGroup": "galaxy",
    "PN": "planetary nebula",
    "HII": "emission nebula",
    "EmN": "emission nebula",
    "Cl+N": "emission nebula",
    "Neb": "emission nebula",
    "SNR": "supernova remnant",
    "RfN": "reflection nebula",
}
"""The kind of source mphot models for each OpenNGC object type."""

_POINT_SOURCE_HINT = "Its stars are point sources: use get_precision for them."

UNSUPPORTED_TYPES = {
    "OCl": f"is an open cluster. {_POINT_SOURCE_HINT}",
    "*Ass": f"is a stellar association. {_POINT_SOURCE_HINT}",
    "*": "is a star. Use get_precision.",
    "**": "is a double star. Use get_precision.",
    "Nova": "is a nova. Use get_precision.",
    "GCl": "is a globular cluster. Globular clusters are not supported yet.",
    "DrkN": "is a dark nebula. It is seen in absorption, which mphot does not model.",
    "Other": "is not a galaxy or a nebula.",
}
"""Why each OpenNGC object type that mphot cannot model is refused."""

BANDS = ("B", "V", "J", "H", "K")
"""Catalogue bands of OpenNGC, in the order of their columns."""

_MESSIER = re.compile(r"^M(?:ESSIER)?(\d{1,3})$")
_NGC_IC = re.compile(r"^(NGC|IC)(\d{1,4})(.*)$")


class TargetNotFound(ValueError):
    """Raised when no Messier, NGC or IC object matches a name."""


class UnsupportedTarget(ValueError):
    """Raised when an object exists but mphot cannot model it."""


@dataclass(frozen=True)
class Target:
    """
    A galaxy or nebula from the catalogue.

    Attributes:
        name (str): OpenNGC name, such as "NGC5194".
        label (str): Name to show, such as "M51 (NGC 5194, Whirlpool Galaxy)".
        kind (str): One of the values of `KINDS`.
        object_type (str): OpenNGC object type, such as "G" or "PN".
        ra (float): Right ascension [deg].
        dec (float): Declination [deg].
        major_axis (float): Major axis [arcmin]. NaN if the catalogue has none.
        minor_axis (float): Minor axis [arcmin]. Equal to the major axis if the
            catalogue has none.
        position_angle (float | None): Position angle of the major axis
            [deg], from north through east. None if the catalogue has none,
            as for most nebulae.
        magnitudes (dict): Total Vega magnitudes by band ("B", "V", "J", "H",
            "K"), only for the bands the catalogue has. Nebulae have no J, H
            and K, because OpenNGC gives those of a star in the nebula.
        radial_velocity (float): Radial velocity [km/s]. 0 if unknown.
        halpha_flux (float | None): Total observed Halpha flux of a nebula
            [erg/s/cm2].
        halpha_source (str | None): Where `halpha_flux` comes from.
        line_ratios (dict): Measured observed line ratios to Halpha, by line key
            (such as "oiii_5007"). Empty if none are known.
        c_hbeta (float | None): Logarithmic extinction at Hbeta, if known.
    """

    name: str
    label: str
    kind: str
    object_type: str
    ra: float
    dec: float
    major_axis: float
    minor_axis: float
    position_angle: float | None = None
    magnitudes: dict = field(default_factory=dict)
    radial_velocity: float = 0.0
    halpha_flux: float | None = None
    halpha_source: str | None = None
    line_ratios: dict = field(default_factory=dict)
    c_hbeta: float | None = None

    @property
    def area(self) -> float:
        """Area of the catalogue ellipse [arcsec2]. NaN without a size."""
        return math.pi / 4 * (self.major_axis * 60) * (self.minor_axis * 60)


@cache
def _catalogue() -> pd.DataFrame:
    table = pd.read_csv(
        TARGETS_DIR / CATALOGUE_FILE,
        dtype={"ngc": str, "ic": str, "common_names": str},
    )
    table["key"] = table["name"].str.replace(" ", "").str.upper()
    return table


@cache
def _nebula_lines() -> pd.DataFrame:
    return pd.read_csv(TARGETS_DIR / NEBULA_LINES_FILE).set_index("name")


@cache
def _common_names() -> dict[str, list[str]]:
    """Map each lower-case common name to the catalogue names that carry it."""
    names: dict[str, list[str]] = {}
    table = _catalogue()
    for name, common in zip(table["name"], table["common_names"], strict=True):
        if isinstance(common, str):
            for alias in common.split(","):
                names.setdefault(alias.strip().lower(), []).append(name)
    return names


def display_name(name: str) -> str:
    """'NGC5194' -> 'NGC 5194', 'IC0434' -> 'IC 434'. Other names pass through."""
    match = _NGC_IC.match(name.replace(" ", ""))
    if match:
        prefix, number, rest = match.groups()
        rest = f" {rest}" if rest.startswith("NED") else rest
        return f"{prefix} {int(number)}{rest}"
    return name


def _row(key: str) -> pd.Series | None:
    table = _catalogue()
    rows = table[table["key"] == key]
    return None if rows.empty else rows.iloc[0]


def _messier_row(number: int) -> pd.Series:
    if number == 102:
        raise UnsupportedTarget(
            "M102 has no agreed identification. NGC 5866 is the usual "
            "candidate: enter 'NGC 5866' to use it."
        )
    if not 1 <= number <= 110:
        raise TargetNotFound(f"There is no Messier object M{number}.")
    table = _catalogue()
    rows = table[table["messier"] == number]
    main = rows[rows["type"] != "Dup"]
    return (main if not main.empty else rows).iloc[0]


def _lookup(text: str) -> pd.Series:
    """Find the catalogue row of a name, before duplicates are followed."""
    key = re.sub(r"[\s_\-]+", "", text).upper()

    match = _MESSIER.match(key)
    if match:
        return _messier_row(int(match.group(1)))

    match = _NGC_IC.match(key)
    if match:
        prefix, number, rest = match.groups()
        row = _row(f"{prefix}{int(number):04d}{rest}")
        if row is not None:
            return row
        raise TargetNotFound(f"There is no {display_name(key)} in the catalogue.")

    row = _row(key)
    if row is not None:
        return row

    # Common names: an exact match first, then names that contain the text.
    common = _common_names()
    wanted = " ".join(text.lower().split())
    if wanted in common:
        return _row(common[wanted][0].replace(" ", "").upper())
    if len(wanted) >= 3:
        hits = {alias: names for alias, names in common.items() if wanted in alias}
        objects = {name for names in hits.values() for name in names}
        if len(objects) == 1:
            return _row(objects.pop().replace(" ", "").upper())
        if objects:
            options = ", ".join(
                f"{alias.title()} ({display_name(names[0])})"
                for alias, names in sorted(hits.items())[:6]
            )
            raise TargetNotFound(f"'{text}' matches several objects: {options}.")

    raise TargetNotFound(
        f"No Messier, NGC or IC object matches '{text}'. Enter a name such as "
        "'M51', 'NGC 5194' or 'Whirlpool Galaxy'."
    )


def _follow_duplicate(row: pd.Series) -> pd.Series:
    """Return the main entry of an object that OpenNGC lists twice."""
    for _ in range(4):
        if row["type"] != "Dup":
            return row
        if isinstance(row["ngc"], str):
            row = _lookup(f"NGC{row['ngc']}")
        elif isinstance(row["ic"], str):
            row = _lookup(f"IC{row['ic']}")
        elif pd.notna(row["messier"]):
            row = _messier_row(int(row["messier"]))
        else:
            break
    raise TargetNotFound(f"{display_name(row['name'])} has no main catalogue entry.")


def _label(row: pd.Series) -> str:
    parts = [display_name(row["name"])]
    if isinstance(row["common_names"], str):
        parts.append(row["common_names"].split(",")[0].strip())
    if pd.notna(row["messier"]):
        return f"M{int(row['messier'])} ({', '.join(parts)})"
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]} ({parts[1]})"


def _finite(value) -> float | None:
    return float(value) if value is not None and np.isfinite(value) else None


def resolve_target(name: str) -> Target:
    """
    Look up a galaxy or nebula by its Messier, NGC or IC name.

    Names are matched without regard to case or spaces, so "M51", "m 51",
    "Messier 51", "NGC 5194" and "ngc5194" all give the Whirlpool Galaxy. Common
    names such as "Whirlpool Galaxy" or "Ring Nebula" also work, and so does a
    unique part of one, such as "Whirlpool".

    Args:
        name (str): Name of the object.

    Returns:
        Target: The object, with the catalogue data mphot uses.

    Raises:
        TargetNotFound: If no object matches, or if a common name matches more
            than one object.
        UnsupportedTarget: If the object is not a galaxy or nebula mphot can
            model, such as an open cluster.
    """

    row = _follow_duplicate(_lookup(name))
    label = _label(row)

    if row["type"] not in KINDS:
        reason = UNSUPPORTED_TYPES.get(row["type"], "is not a galaxy or a nebula.")
        raise UnsupportedTarget(f"{label} {reason}")

    major = float(row["major_axis"])
    minor = float(row["minor_axis"]) if np.isfinite(row["minor_axis"]) else major

    magnitudes = {
        band: float(row[band.lower()])
        for band in BANDS
        if np.isfinite(row[band.lower()])
    }

    lines = _nebula_lines()
    halpha_flux = halpha_source = c_hbeta = None
    line_ratios = {}
    if row["name"] in lines.index:
        entry = lines.loc[row["name"]]
        halpha_flux = _finite(entry["halpha_flux"])
        halpha_source = entry["halpha_source"]
        c_hbeta = _finite(entry["c_hbeta"])
        line_ratios = {
            key: float(entry[key])
            for key in (
                "hbeta",
                "oiii_5007",
                "hei_5876",
                "nii_6584",
                "sii_6717",
                "sii_6731",
            )
            if np.isfinite(entry[key])
        }

    velocity = row["radial_velocity"]
    return Target(
        name=row["name"],
        label=label,
        kind=KINDS[row["type"]],
        object_type=row["type"],
        ra=float(row["ra"]),
        dec=float(row["dec"]),
        major_axis=major,
        minor_axis=minor,
        position_angle=_finite(row["position_angle"]),
        magnitudes=magnitudes,
        radial_velocity=float(velocity) if np.isfinite(velocity) else 0.0,
        halpha_flux=halpha_flux,
        halpha_source=halpha_source,
        line_ratios=line_ratios,
        c_hbeta=c_hbeta,
    )


def target_suggestions() -> list[tuple[str, str]]:
    """
    Names to offer in a search box, each with a short description.

    Every Messier object mphot can model comes first, in order, then every
    other galaxy or nebula with a common name.

    Returns:
        list: (name to enter, description) pairs, such as
            ("M51", "Whirlpool Galaxy, NGC 5194, galaxy").
    """

    table = _catalogue()
    table = table[table["type"].isin(list(KINDS))]

    suggestions = []
    for _, row in table[table["messier"].notna()].sort_values("messier").iterrows():
        parts = [display_name(row["name"]), KINDS[row["type"]]]
        if isinstance(row["common_names"], str):
            parts.insert(0, row["common_names"].split(",")[0].strip())
        suggestions.append((f"M{int(row['messier'])}", ", ".join(parts)))

    named = table[table["messier"].isna() & table["common_names"].notna()]
    for _, row in named.iterrows():
        common = row["common_names"].split(",")[0].strip()
        kind = KINDS[row["type"]]
        suggestions.append((common, f"{display_name(row['name'])}, {kind}"))
    return suggestions
