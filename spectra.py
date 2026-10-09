"""Hydrogen-like Bohr transitions; trying to correspond wavelength to colour"""
RYD_EV = 13.6057
HC_EV_NM = 1239.84

def transition(n_hi, n_lo):
    """Photon energy (eV) and wavelength (nm) of hydrogen-like elements electron jump"""
    dE = RYD_EV * (1.0 / n_lo**2 - 1.0 / n_hi**2)
    return dE, HC_EV_NM / dE

def region(nm):
    return "UV" if nm < 380 else "IR" if nm > 750 else "visible"

def wavelength_to_bgr(nm):
    """Approximating visible colour to wavelengths that we get"""
    if 380 <= nm < 440: r, g, b = (440 - nm) / 60, 0.0, 1.0
    elif 440 <= nm < 490: r, g, b = 0.0, (nm - 440) / 50, 1.0
    elif 490 <= nm < 510: r, g, b = 0.0, 1.0, (510 - nm) / 20
    elif 510 <= nm < 580: r, g, b = (nm - 510) / 70, 1.0, 0.0
    elif 580 <= nm < 645: r, g, b = 1.0, (645 - nm) / 65, 0.0
    elif 645 <= nm <= 750: r, g, b = 1.0, 0.0, 0.0
    else:
        return None
    if nm < 420: f = 0.3 + 0.7 * (nm - 380) / 40
    elif nm > 700: f = 0.3 + 0.7 * (750 - nm) / 50
    else: f = 1.0
    return tuple(int(255 * (v * f) ** 0.8) for v in (b, g, r))
