"""Shared code for atom_based.ipynb and hybrid.ipynb.

This matching helper is extracted from the previously supplied overall-rate
notebooks. It keeps their all-outcome correction conventions, ordered level-2
routes, common-clock rate formula, and robust optimizers. Use this helper with
these two notebooks; an earlier merge-only helper lacks the added rate/search API.

No standalone sanity-check routines or automatically executed tests are included.
All source .dill expressions are read from the user's chosen directory unchanged.
"""

from functools import lru_cache
from itertools import product
import numpy as np

C = 0.2                       # km / microsecond
NUM_FREQ = 100
POSTPROCESSING = "one_way"     # "one_way", "ad" (forced), or "best" (optional AD)
FINAL_POOLING = "route"        # "route" or "class"; see the policy cell

_I = np.eye(2, dtype=complex)
_X = np.array([[0., 1.], [1., 0.]], dtype=complex)
_Z = np.diag([1., -1.]).astype(complex)
_H = (_X + _Z) / np.sqrt(2.0)
_S = np.diag([1., 1j])


def kron_all(*ops):
    out = np.array([1.], dtype=complex)
    for op in ops:
        out = np.kron(out, op)
    return out


H3 = kron_all(_H, _H, _H)
_GHZ_BASE = kron_all(_I, _I, _Z) @ H3
_W_KET = np.zeros(8, dtype=complex)
_W_KET[[1, 2, 4]] = 1.0 / np.sqrt(3.0)
_G_KET = np.zeros(8, dtype=complex)
_G_KET[[0, 7]] = 1.0 / np.sqrt(2.0)
_W_IDEAL = np.outer(_W_KET, _W_KET.conj())
_G_IDEAL = np.outer(_G_KET, _G_KET.conj())
_PAIRS = ((4, 8), (2, 3), (1, 6))  # BC, AB, AC repeater stations
_ENDS = (0, 5, 7)                  # end users A, B, C
_BITS9 = ((np.arange(512)[:, None] >> np.arange(8, -1, -1)) & 1)
_END_ROWS = _BITS9[:, _ENDS] @ np.array([4, 2, 1])
_EYE4 = np.eye(4, dtype=complex)


def _real_scalar(value, name, *, atol=1e-12):
    a = np.asarray(value, dtype=complex)
    if a.size != 1:
        raise ValueError(f"{name} must be scalar; got {a.shape}.")
    z = complex(a.reshape(-1)[0])
    if not np.isfinite(z):
        raise FloatingPointError(f"{name} is not finite: {z!r}.")
    if abs(z.imag) > atol:
        raise FloatingPointError(f"{name} is not real: {z!r}.")
    return float(z.real)


def _probability(value, name, *, atol=1e-12):
    """Repair boundary roundoff only; never discard a small positive probability."""
    p = _real_scalar(value, name, atol=atol)
    if p < -atol or p > 1.0 + atol:
        raise ValueError(f"{name} must be in [0, 1]; got {p!r}.")
    return min(1.0, max(0.0, p))


def _density_matrix(value, *, atol=1e-10):
    dm = np.asarray(value, dtype=complex)
    if dm.shape != (8, 8):
        raise ValueError(f"Expected an 8 x 8 density matrix; got {dm.shape}.")
    if not np.isfinite(dm).all():
        raise FloatingPointError("The density matrix is not finite.")
    if not np.allclose(dm, dm.conj().T, rtol=0, atol=atol):
        raise ValueError("The density matrix is not Hermitian.")
    if abs(np.trace(dm) - 1.0) > atol:
        raise ValueError("The density matrix must have unit trace.")
    if np.linalg.eigvalsh((dm + dm.conj().T) / 2).min() < -atol:
        raise ValueError("The density matrix is not positive semidefinite.")
    return dm


def _condition(raw, name="branch"):
    raw = np.asarray(raw, dtype=complex)
    p = _probability(np.trace(raw), f"{name} probability")
    if p == 0.0:
        return None, 0.0
    # Only remove floating-point anti-Hermitian roundoff, not physical noise.
    rho = (raw + raw.conj().T) / (2.0 * p)
    return _density_matrix(rho), p


def normalize(dm):
    rho, p = _condition(dm)
    if p == 0.0:
        raise ValueError("Cannot normalize an impossible (zero-trace) branch.")
    return rho


def get_bin_entropy(p):
    p = _probability(p, "binary-entropy argument")
    if p == 0.0 or p == 1.0:
        return 0.0
    return float(-p * np.log2(p) - (1 - p) * np.log1p(-p) / np.log(2.0))


_QOBS = {
    "W": (kron_all(_X, _X, _I), kron_all(_X, _I, _X),
          -kron_all(_Z, _Z, _Z)),
    "GHZ": (kron_all(_Z, _Z, _I), kron_all(_Z, _I, _Z),
            kron_all(_X, _X, _X)),
}


def get_qbers(dm, protocol="W"):
    """Return (Q_AB, Q_AC, Q_phase) in the protocol's correct bases."""
    if protocol not in _QOBS:
        raise ValueError("protocol must be 'W' or 'GHZ'.")
    return tuple(_probability((1 - np.einsum("ij,ji->", dm, op)) / 2,
                              f"{protocol} QBER {i}")
                 for i, op in enumerate(_QOBS[protocol]))


def get_phase_QBER(dm):
    return get_qbers(dm, "W")[2]


def get_bit_QBER(dm):
    return max(get_qbers(dm, "W")[:2])


def get_sk_fraction(dm, protocol="W"):
    """Signed one-way expression; the *rate* clips it per output class/route.

    max(h(Q_AB), h(Q_AC)), not h(max(Q_AB,Q_AC)), is required when
    either QBER exceeds 1/2. They coincide in the usual low-error regime.
    """
    qab, qac, phase = get_qbers(dm, protocol)
    return 1.0 - get_bin_entropy(phase) - max(
        get_bin_entropy(qab), get_bin_entropy(qac))


def get_sk_fraction_GHZ(dm):
    return get_sk_fraction(dm, "GHZ")


def get_ad_state(dm, protocol="W"):
    """One repetition-code AD round, expressed in the computational key frame.

    Complete accepted target outcomes: 000 and 111. This is an exact
    two-copy density-matrix formula, NOT an assumption of Bell/GHZ diagonality.
    Elementwise products below intentionally do not conjugate the second state.
    The returned state's key formula is the GHZ/computational-key formula.
    """
    if protocol not in ("W", "GHZ"):
        raise ValueError("protocol must be 'W' or 'GHZ'.")
    sigma = H3 @ dm @ H3.conj().T if protocol == "W" else dm
    complement = np.arange(8) ^ 7
    raw = sigma * sigma + sigma * sigma[np.ix_(complement, complement)]
    return _condition(raw, "AD")


def effective_secret_fraction(dm, protocol="W", postprocessing="one_way"):
    """Secret bits per originally distributed copy, including the AD copy cost."""
    if postprocessing not in ("one_way", "ad", "best"):
        raise ValueError("postprocessing must be 'one_way', 'ad', or 'best'.")
    f0 = max(0.0, get_sk_fraction(dm, protocol))
    if postprocessing == "one_way":
        return f0
    rho_ad, p_ad = get_ad_state(dm, protocol)
    f_ad = (p_ad / 2.0 * max(0.0, get_sk_fraction(rho_ad, "GHZ"))
            if p_ad > 0 else 0.0)
    return max(f0, f_ad) if postprocessing == "best" else f_ad


def get_loss_ratio(distance_km, loss_db_per_km=0.3):
    return -np.expm1(-np.log(10.0) * loss_db_per_km * np.asarray(distance_km) / 10.0)


def _multiplexed_click_probability(p_single, num_freq):
    p_single = _probability(p_single, "single-mode click probability")
    if (isinstance(num_freq, (bool, np.bool_))
            or not isinstance(num_freq, (int, np.integer)) or num_freq < 0):
        raise ValueError("NUM_FREQ must be a nonnegative integer.")
    if num_freq == 0 or p_single == 0.0:
        return 0.0
    if p_single == 1.0:
        return 1.0
    return _probability(-np.expm1(int(num_freq) * np.log1p(-p_single)),
                        "multiplexed click probability")


def _bell(parity, sign):
    """Bell bra: (00 +/- 11)/sqrt(2), or (01 +/- 10)/sqrt(2)."""
    v = np.zeros(4, dtype=complex)
    v[parity] = 1.0 / np.sqrt(2.0)
    v[3 - parity] = (-1.0) ** sign / np.sqrt(2.0)
    return v


def _projection(measurements):
    """8 x 512 map for disjoint local bras; endpoints stay in (l1,u2,r3)."""
    used = [q for inds, _ in measurements for q in inds]
    if sorted(used) != sorted(set(range(9)) - set(_ENDS)):
        raise ValueError("The measurement bras must cover each repeater qubit once.")
    amplitudes = np.ones(512, dtype=complex)
    for inds, bra in measurements:
        index = _BITS9[:, inds] @ (1 << np.arange(len(inds) - 1, -1, -1))
        amplitudes *= np.asarray(bra)[index]
    k = np.zeros((8, 512), dtype=complex)
    k[_END_ROWS, np.arange(512)] = amplitudes
    return k


@lru_cache(None)
def _paulis():
    return tuple(kron_all(*[(_X if x else _I) @ (_Z if z else _I)
                             for x, z in zip(xs, zs)])
                 for xs in product((0, 1), repeat=3)
                 for zs in product((0, 1), repeat=3))


def _w_correction(ideal_vector):
    """Fix a Pauli rule on ideal inputs once; never fit corrections to noisy states."""
    p = float(np.vdot(ideal_vector, ideal_vector).real)
    if p <= 0.0:
        raise ValueError("Cannot infer a correction from an impossible ideal branch.")
    for u in _paulis():
        if abs(abs(np.vdot(_W_KET, u @ ideal_vector)) ** 2 / p - 1) < 1e-12:
            return u
    raise RuntimeError("No W-branch Pauli correction was found.")


def _edge(a, b):
    for pair in _PAIRS:
        if {pair[0] // 3, pair[1] // 3} == {a, b}:
            return pair
    raise ValueError("Invalid child pair.")


def _node_qubit(child, neighbor):
    return next(q for q in _edge(child, neighbor) if q // 3 == child)


@lru_cache(None)
def merge_instrument(route):
    """Return (record, corrected Kraus matrix) for every accepted sub-outcome.

    WWW_W: 24 Kraus terms for one-even/two-odd (the even pair is traced),
           and 12 for two-even/one-odd with a predetermined 00 postselection.
    WWW_GHZ: all eight all-even Bell sign records, corrected separately.
    GGG: all 64 Bell records, including noisy parity-inconsistent records.
    Mixed routes: every teleportation/Bell/X outcome; W-to-Bell 0 projections
           remain probabilistic and are INCLUDED in the Kraus matrices.
    """
    out = []
    if route == "WWW_W":
        ideal = kron_all(_W_KET, _W_KET, _W_KET)
        for even in range(3):
            odd = [j for j in range(3) if j != even]
            for signs in product((0, 1), repeat=2):
                common = [(_PAIRS[j], _bell(1, s)) for j, s in zip(odd, signs)]
                ks = [_projection([(_PAIRS[even], _EYE4[v])] + common)
                      for v in (0, 3)]
                u = _w_correction(ks[0] @ ideal)
                # The even-pair value is traced, not used in the correction.
                out.extend((("one_even", even, signs, v), u @ k)
                           for v, k in zip((0, 3), ks))
        for odd in range(3):
            zero = (odd + 2) % 3  # cyclic, predetermined choice of the 00 pair
            other = 3 - odd - zero
            for so, se in product((0, 1), repeat=2):
                k = _projection([(_PAIRS[zero], _EYE4[0]),
                                 (_PAIRS[odd], _bell(1, so)),
                                 (_PAIRS[other], _bell(0, se))])
                u = _w_correction(k @ ideal)
                out.append((("two_even", odd, so, se), u @ k))
    elif route == "WWW_GHZ":
        for s_bc, s_ab, s_ac in product((0, 1), repeat=3):
            signs = (s_bc, s_ab, s_ac)
            k = _projection([(pair, _bell(0, s)) for pair, s in zip(_PAIRS, signs)])
            exponents = (s_bc - s_ab - s_ac,
                         s_ac - s_bc - s_ab,
                         s_ab - s_bc - s_ac)
            phases = kron_all(*[np.diag([1., (1j) ** (n % 4)]) for n in exponents])
            u = kron_all(_I, _I, _Z) @ H3 @ phases
            out.append((signs, u @ k))
    elif route == "GGG":
        for pars in product((0, 1), repeat=3):
            for signs in product((0, 1), repeat=3):
                k = _projection([(pair, _bell(p, s))
                                 for pair, p, s in zip(_PAIRS, pars, signs)])
                # Anchor the Pauli frame at A. Ignore the redundant BC parity
                # when it disagrees under noise, but retain this noisy outcome.
                u = kron_all(_Z if sum(signs) % 2 else _I,
                             _X if pars[1] else _I, _X if pars[2] else _I)
                out.append(((pars, signs), u @ k))
    elif len(route) == 3 and set(route) <= {"G", "W"} and route.count("G") == 1:
        g = route.index("G")
        ws = [i for i in range(3) if i != g]
        base = [((_node_qubit(w, ws[1 - i]),), np.array([1., 0.]))
                for i, w in enumerate(ws)]
        for p0, s0, p1, s1 in product((0, 1), repeat=4):
            records = ((p0, s0), (p1, s1))
            measures = base + [(_edge(g, w), _bell(p, s))
                               for w, (p, s) in zip(ws, records)]
            k = _projection(measures)
            local = [_I, _I, _I]
            for w, (p, s) in zip(ws, records):
                # W gives Psi+, so include its known extra endpoint X.
                local[w] = (_X if p ^ 1 else _I) @ (_Z if s else _I)
            out.append((records, kron_all(*local) @ k))
    elif len(route) == 3 and set(route) <= {"G", "W"} and route.count("G") == 2:
        w = route.index("W")
        a, b = (w + 1) % 3, (w + 2) % 3  # deterministic teleportation direction
        for pg, sg, pt, st, sx in product((0, 1), repeat=5):
            measures = [((_node_qubit(w, b),), np.array([1., 0.])),
                        ((_node_qubit(b, w),), np.array([1., (-1.) ** sx]) / np.sqrt(2)),
                        (_edge(a, b), _bell(pg, sg)), (_edge(a, w), _bell(pt, st))]
            k = _projection(measures)
            local = [_I, _I, _I]
            local[a] = _Z if sg ^ sx else _I
            local[b] = _X if pg else _I
            local[w] = (_X if pt ^ 1 else _I) @ (_Z if st else _I)
            out.append(((pg, sg, pt, st, sx), kron_all(*local) @ k))
    else:
        raise ValueError(f"Unknown merge route: {route!r}.")
    for _, k in out:
        k.setflags(write=False)
    return tuple(out)


@lru_cache(None)
def _compiled_terms(route):
    """Compile the full Kraus sum once, not a representative state.

    A product input has matrix entries rho0[a] * rho1[b] * rho2[c].
    Identical constant monomials from ALL corrected outcomes are combined.
    The 1e-13 tolerance here removes only fixed Clifford-circuit cancellation
    roundoff; it is independent of states, source brightness, and probabilities.
    """
    keys, coefficients = [], []
    for _, k in merge_instrument(route):
        if route == "WWW_GHZ":
            # Every correction is U_base D_m. Factor the SAME U_base out
            # of the sum, while retaining every different D_m and Kraus map.
            # This is exact for arbitrary inputs, not a symmetry assumption.
            k = _GHZ_BASE.conj().T @ k
        r, c = np.nonzero(np.abs(k) > 1e-14)
        v = k[r, c]
        output = r[:, None] * 8 + r[None, :]
        a = (c[:, None] // 64) * 8 + c[None, :] // 64
        b = ((c[:, None] // 8) % 8) * 8 + (c[None, :] // 8) % 8
        e = (c[:, None] % 8) * 8 + c[None, :] % 8
        key = (((output * 64 + a) * 64 + b) * 64 + e).ravel()
        keys.append(key)
        coefficients.append((v[:, None] * v[None, :].conj()).ravel())
    keys, coefficients = np.concatenate(keys), np.concatenate(coefficients)
    sort = np.argsort(keys)
    keys, coefficients = keys[sort], coefficients[sort]
    starts = np.r_[0, np.flatnonzero(np.diff(keys)) + 1]
    unique = keys[starts]
    weights = np.add.reduceat(coefficients, starts)
    mask = np.abs(weights) > 1e-13
    unique, weights = unique[mask], weights[mask]
    c = unique % 64
    b = (unique // 64) % 64
    a = (unique // (64 ** 2)) % 64
    o = unique // (64 ** 3)
    for item in (o, a, b, c, weights):
        item.setflags(write=False)
    return o, a, b, c, weights


def merge_raw(route, rho0, rho1=None, rho2=None):
    """Full corrected *unnormalized* output, with no small-probability cutoff."""
    if rho1 is None:
        rho1 = rho0
    if rho2 is None:
        rho2 = rho0
    o, a, b, c, w = _compiled_terms(route)
    values = (w * np.asarray(rho0).ravel()[a] * np.asarray(rho1).ravel()[b]
              * np.asarray(rho2).ravel()[c])
    raw = (np.bincount(o, weights=values.real, minlength=64)
           + 1j * np.bincount(o, weights=values.imag, minlength=64)).reshape(8, 8)
    if route == "WWW_GHZ":
        raw = _GHZ_BASE @ raw @ _GHZ_BASE.conj().T
    return raw


def merge_W(dm):
    return _condition(merge_raw("WWW_W", dm), "WWW -> W")


def merge_GHZ(dm):
    return _condition(merge_raw("WWW_GHZ", dm), "WWW -> GHZ")


def get_final_state_G_G_G(dm):
    return _condition(merge_raw("GGG", dm), "GGG -> GHZ")[0]


def _make_branches(dm0, dm_w=None, dm_g=None, pi_w=None, pi_g=None):
    """Yield (route, protocol, probability per current merge cycle, state).

    At level 2 the eight ORDERED child compositions are enumerated explicitly.
    This avoids assuming a noisy state is symmetric under exchanging endpoints.
    """
    if dm_w is None and dm_g is None and pi_w is None:
        for route, protocol in (("WWW_W", "W"), ("WWW_GHZ", "GHZ")):
            state, p = _condition(merge_raw(route, dm0), route)
            if p > 0:
                yield route, protocol, p, state
        return
    states = {"W": dm_w, "G": dm_g}
    weights = {"W": pi_w, "G": pi_g}
    for types in product(("W", "G"), repeat=3):
        composition_probability = float(np.prod([weights[t] for t in types]))
        if composition_probability == 0.0:
            continue
        inputs = [states[t] for t in types]
        if any(rho is None for rho in inputs):
            raise RuntimeError("Positive composition probability has no input state.")
        pattern = "".join(types)
        routes = (("WWW_W", "W"), ("WWW_GHZ", "GHZ")) if pattern == "WWW" else ((pattern, "GHZ"),)
        for route, protocol in routes:
            state, a = _condition(merge_raw(route, *inputs), route)
            probability = composition_probability * a
            if probability > 0:
                yield route, protocol, probability, state


def _validate_policy(num_level, postprocessing, pooling):
    if (isinstance(num_level, (bool, np.bool_))
            or not isinstance(num_level, (int, np.integer))
            or num_level not in (0, 1, 2)):
        raise ValueError("This implementation supports num_level = 0, 1, or 2 only.")
    if postprocessing not in ("one_way", "ad", "best"):
        raise ValueError("postprocessing must be 'one_way', 'ad', or 'best'.")
    if pooling not in ("route", "class"):
        raise ValueError("pooling must be 'route' or 'class'.")


def _zero_rates(num_level, postprocessing, pooling):
    return {"total_bps": 0.0, "W_bps": 0.0, "GHZ_bps": 0.0,
            "p_accept": 0.0, "p_W": 0.0, "p_GHZ": 0.0,
            "pi_W": 0.0, "pi_GHZ": 0.0,
            "generation_rate_per_us": 0.0,
            "generation_time_us": float("inf"), "round_time_us": float("inf"),
            "cycle_time_bound_us": float("inf"), "level": int(num_level),
            "postprocessing": postprocessing, "pooling": pooling,
            "routes": [], "level1": None}


def _finish_rates(branches, child_rate, preparation_factor, t_merge, t_meas,
                  num_level, postprocessing, pooling, return_states):
    """Common-clock renewal reward, evaluated with reciprocal times for stability."""
    p_accept = _probability(sum(b[2] for b in branches), "total merge acceptance")
    if p_accept == 0.0 or child_rate == 0.0:
        return _zero_rates(num_level, postprocessing, pooling)
    # C = preparation_factor / child_rate + t_merge is the cycle-time bound.
    # denominator = child_rate * (C + p_accept * t_meas).
    denominator = preparation_factor + (t_merge + p_accept * t_meas) * child_rate
    cycle_flux_bps = 1e6 * child_rate / denominator
    generation_rate = child_rate * p_accept / (preparation_factor + t_merge * child_rate)
    p_by_class = {kind: sum(b[2] for b in branches if b[1] == kind)
                  for kind in ("W", "GHZ")}
    if pooling == "class":
        used = []
        for kind in ("W", "GHZ"):
            p = p_by_class[kind]
            if p == 0.0:
                continue
            rho = sum((prob * state for _, k, prob, state in branches if k == kind),
                      start=np.zeros((8, 8), dtype=complex)) / p
            used.append((kind + "_pooled", kind, p, _density_matrix(rho)))
    else:
        used = branches
    rates = {"W": 0.0, "GHZ": 0.0}
    rows = []
    for route, kind, prob, rho in used:
        fraction = effective_secret_fraction(rho, kind, postprocessing)
        rate = cycle_flux_bps * prob * fraction
        rates[kind] += rate
        qab, qac, phase = get_qbers(rho, kind)
        row = {"route": route, "protocol": kind, "probability": prob,
               "conditional_weight": prob / p_accept,
               "secret_fraction_per_copy": fraction, "rate_bps": rate,
               "Q_AB": qab, "Q_AC": qac, "Q_phase": phase}
        if return_states:
            row["state"] = rho.copy()
        rows.append(row)
    inverse_generation_rate = (1.0 / generation_rate if generation_rate > 0 else float("inf"))
    result = {"total_bps": float(rates["W"] + rates["GHZ"]),
              "W_bps": float(rates["W"]), "GHZ_bps": float(rates["GHZ"]),
              "p_accept": p_accept, "p_W": p_by_class["W"], "p_GHZ": p_by_class["GHZ"],
              "pi_W": p_by_class["W"] / p_accept, "pi_GHZ": p_by_class["GHZ"] / p_accept,
              "generation_rate_per_us": generation_rate,
              "generation_time_us": inverse_generation_rate,
              "round_time_us": inverse_generation_rate + t_meas,
              "cycle_time_bound_us": preparation_factor / child_rate + t_merge,
              "level": int(num_level), "postprocessing": postprocessing, "pooling": pooling,
              "routes": rows, "level1": None}
    return result


def overall_network_rates(dm0, elementary_rate_per_us, t_merge_1, t_merge_2,
                          num_level, *, t_meas=100.0, postprocessing="one_way",
                          pooling="route", return_states=False):
    """Accepted children are stored regardless of W/GHZ type; failures restart.

    H3=11/6 is retained at level 1 (fixed-duration geometric EL attempts).
    The distribution-free factor-3 time upper bound is retained at level 2.
    Perfect memories, no label-dependent timing selection, independent children.
    Reported rates use these conservative time bounds, not an optimal scheduler.
    """
    _validate_policy(num_level, postprocessing, pooling)
    r0 = _real_scalar(elementary_rate_per_us, "elementary generation rate")
    if r0 < 0:
        raise ValueError("The elementary generation rate cannot be negative.")
    for label, duration in (("t_merge_1", t_merge_1), ("t_merge_2", t_merge_2), ("t_meas", t_meas)):
        if _real_scalar(duration, label) < 0:
            raise ValueError(f"{label} cannot be negative.")
    if r0 == 0:
        return _zero_rates(num_level, postprocessing, pooling)
    dm0 = _density_matrix(dm0)
    if num_level == 0:
        return _finish_rates([("EL_W", "W", 1.0, dm0)], r0, 1.0, 0.0, t_meas,
                             num_level, postprocessing, pooling, return_states)
    first = list(_make_branches(dm0))
    p1 = _probability(sum(b[2] for b in first), "level-1 total acceptance")
    if p1 == 0:
        return _zero_rates(num_level, postprocessing, pooling)
    pw = sum(b[2] for b in first if b[1] == "W")
    pg = sum(b[2] for b in first if b[1] == "GHZ")
    r1 = r0 * p1 / (11.0 / 6.0 + t_merge_1 * r0)
    metadata = {"p_W": pw, "p_GHZ": pg, "p_accept": p1,
                "pi_W": pw / p1, "pi_GHZ": pg / p1,
                "generation_rate_per_us": r1}
    if num_level == 1:
        result = _finish_rates(first, r0, 11.0 / 6.0, t_merge_1, t_meas,
                               num_level, postprocessing, pooling, return_states)
    else:
        if r1 == 0.0:  # Only machine underflow can cause this when r0,p1>0.
            return _zero_rates(num_level, postprocessing, pooling)
        dm_w = next((b[3] for b in first if b[1] == "W"), None)
        dm_g = next((b[3] for b in first if b[1] == "GHZ"), None)
        second = list(_make_branches(dm0, dm_w, dm_g, pw / p1, pg / p1))
        result = _finish_rates(second, r1, 3.0, t_merge_2, t_meas,
                               num_level, postprocessing, pooling, return_states)
    result["level1"] = metadata
    return result


# Robust atom-based optimization
import warnings

import numpy as np
from scipy.optimize import OptimizeResult, minimize_scalar


def optimize_q(
    rate_function,
    d,
    p_l_val,
    num_level,
    *,
    q_bounds=(0.0, 1.0),
    previous_q=None,
    extra_q=(),
    n_log=201,
    n_linear=101,
    log_min=1e-8,
    grid_refinements=1,
    local_tol=1e-9,
    maxiter=300,
    warn_no_positive=True,
):
    """Maximize a deterministic, finite, real-valued key-rate calculation.

    q_bounds preserves the physical search interval, including both endpoints.
    log_min is the first logarithmic *sample*, NOT a new physical lower bound.
    The linear grid still spans q_bounds; local searches can sample below
    log_min. No positive rate or success probability is thresholded away.

    Defaults use 201 log + 101 linear samples, then double both resolutions.
    Local searches refine every detected grid peak and the best grid point.
    previous_q and extra_q are reevaluated at the CURRENT distance and loss.
    No interpolation of rates or monotonicity constraint is used.

    Returns a scipy.optimize.OptimizeResult. x=[best_q], fun=-best_signed_rate.
    positive_rate_found=False means no positive value was found, NOT a proof
    that the physical optimum is zero. success also checks local convergence;
    it is never a global-optimality certificate. Model exceptions propagate.
    sampled_q/sampled_rates contain all actual function evaluations.
    """
    bounds_array = np.asarray(q_bounds, dtype=float)
    if bounds_array.shape != (2,) or not np.isfinite(bounds_array).all():
        raise ValueError("q_bounds must contain two finite numbers.")
    lo, hi = map(float, bounds_array)
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError("Require 0 <= q_bounds[0] < q_bounds[1] <= 1.")
    for name, value, minimum in (
        ("n_log", n_log, 3), ("n_linear", n_linear, 3),
        ("grid_refinements", grid_refinements, 0), ("maxiter", maxiter, 1),
    ):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or value < minimum):
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    log_min, local_tol = float(log_min), float(local_tol)
    if not np.isfinite(log_min) or not 0 < log_min < hi:
        raise ValueError("log_min must be positive and below the upper q bound.")
    if not np.isfinite(local_tol) or not 0 < local_tol < 1:
        raise ValueError("local_tol must lie strictly between zero and one.")

    hints = list(extra_q)
    if previous_q is not None:
        previous_q = float(previous_q)
        if np.isnan(previous_q):
            previous_q = None  # No useful optimum at the preceding distance.
        elif not np.isfinite(previous_q) or not lo <= previous_q <= hi:
            raise ValueError("previous_q must be within q_bounds, None, or NaN.")
        else:
            # Include the hint itself and neighboring scales, never as a bound.
            hints.extend(previous_q * f for f in (0.5, 0.8, 1.0, 1.25, 2.0)
                         if lo <= previous_q * f <= hi)
    hints = np.asarray(hints, dtype=float)
    if hints.ndim != 1 or not np.isfinite(hints).all():
        raise ValueError("extra_q must be a finite one-dimensional sequence.")
    if np.any(hints < lo) or np.any(hints > hi):
        raise ValueError("Every extra_q candidate must lie in q_bounds.")

    cache = {}

    def evaluate(q_value):
        q_value = float(q_value)
        if not lo <= q_value <= hi:
            raise ValueError("Optimizer tried a point outside q_bounds.")
        if q_value not in cache:
            value = np.asarray(rate_function([q_value], d, p_l_val, num_level))
            if value.size != 1:
                raise ValueError("The key-rate function must return one scalar.")
            scalar = complex(value.item())
            if (not np.isfinite(scalar.real) or not np.isfinite(scalar.imag)
                    or scalar.imag != 0.0):
                raise FloatingPointError(
                    f"Nonfinite/nonreal key rate {scalar!r} at q={q_value!r}, "
                    f"distance={d!r}, p_l={p_l_val!r}, level={num_level!r}."
                )
            cache[q_value] = float(scalar.real)
        return cache[q_value]

    local_results = []
    history = []
    for refinement in range(grid_refinements + 1):
        scale = 2 ** refinement
        grid = np.unique(np.concatenate((
            np.linspace(lo, hi, (n_linear - 1) * scale + 1),
            np.geomspace(max(lo, log_min), hi, (n_log - 1) * scale + 1),
            hints,
        )))
        rates = np.array([evaluate(q_value) for q_value in grid])
        best_index = int(np.argmax(rates))
        peaks = {best_index}
        for j in range(1, len(grid) - 1):
            if (rates[j] >= rates[j - 1] and rates[j] >= rates[j + 1]
                    and (rates[j] > rates[j - 1] or rates[j] > rates[j + 1])):
                peaks.add(j)

        for j in sorted(peaks):
            a, b = float(grid[max(0, j - 1)]), float(grid[min(len(grid) - 1, j + 1)])
            # Avoid wasting a local search on a completely flat window.
            if evaluate(a) == evaluate(grid[j]) == evaluate(b):
                continue
            # Normalize the local coordinate so tolerances scale with the
            # bracket width, not the absolute magnitude of q or the rate.
            local = minimize_scalar(
                lambda u: -evaluate(a + float(u) * (b - a)),
                bounds=(0.0, 1.0), method="bounded",
                options={"xatol": local_tol, "maxiter": maxiter},
            )
            evaluate(a + float(local.x) * (b - a))
            local_results.append(local)

        current_q = max(cache, key=cache.get)
        history.append({
            "grid_refinement": refinement,
            "grid_size": len(grid),
            "grid_best_q": float(grid[best_index]),
            "grid_best_rate": float(rates[best_index]),
            "best_q": current_q,
            "best_rate": cache[current_q],
        })

    best_q = max(cache, key=cache.get)
    best_rate = cache[best_q]
    positive = best_rate > 0.0
    failed = sum(not bool(local.success) for local in local_results)
    if not positive:
        message = (
            "No positive rate found at this sampling resolution; this is NOT "
            "a certified zero. Increase grid density or inspect the rate function."
        )
        status = 2
        if warn_no_positive:
            warnings.warn(
                f"{message} distance={d!r}, p_l={p_l_val!r}, level={num_level!r}.",
                RuntimeWarning, stacklevel=2,
            )
    elif failed:
        message = "Positive candidate retained, but some local refinements did not converge."
        status = 1
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    else:
        message = "Positive rate found by grid and local refinement; not a certified global maximum."
        status = 0
    samples = sorted(cache)
    return OptimizeResult(
        x=np.array([best_q]), fun=-best_rate, success=(status == 0),
        status=status, message=message, nfev=len(cache),
        positive_rate_found=positive, local_refinements_failed=failed,
        grid_best_q=history[-1]["grid_best_q"],
        grid_best_rate=history[-1]["grid_best_rate"],
        search_history=history,
        sampled_q=np.array(samples),
        sampled_rates=np.array([cache[q_value] for q_value in samples]),
    )

# Robust hybrid optimization
import warnings
from itertools import product

import numpy as np
from scipy.optimize import OptimizeResult, minimize, minimize_scalar



def _integer(value, name, minimum):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite.")
    return value


class _Axis:
    """Shifted logarithm maps an entire closed interval to [0, 1]."""

    def __init__(self, lower, upper, decades):
        self.lower, self.upper = float(lower), float(upper)
        self.span = self.upper - self.lower
        self.scale = self.span * 10.0**(-decades)
        if self.scale <= 0 or not np.isfinite(self.scale):
            raise ValueError("Bounds are too small for the requested log_decades.")
        self.length = np.log1p(self.span / self.scale)

    def physical(self, u):
        u = np.clip(np.asarray(u, float), 0.0, 1.0)
        x = self.lower + self.scale * np.expm1(u * self.length)
        x = np.clip(x, self.lower, self.upper)
        # Exact physical bounds, not approximate exp/log round trips.
        return np.where(u == 0.0, self.lower,
                        np.where(u == 1.0, self.upper, x))

    def unit(self, x):
        return np.clip(np.log1p((np.asarray(x) - self.lower) / self.scale)
                       / self.length, 0.0, 1.0)

    def grid(self, n_log, n_linear):
        return np.unique(np.concatenate((
            self.physical(np.linspace(0.0, 1.0, n_log)),
            np.linspace(self.lower, self.upper, n_linear),
        )))


def _neighbors(grid, x):
    """Immediate strictly lower/upper grid neighbors; preserve edge points."""
    left = int(np.searchsorted(grid, x, side="left")) - 1
    right = int(np.searchsorted(grid, x, side="right"))
    return (float(grid[left]) if left >= 0 else float(x),
            float(grid[right]) if right < len(grid) else float(x))


def _peaks_2d(values):
    """Interior 8-neighbor maxima, excluding exactly flat neighborhoods."""
    center = values[1:-1, 1:-1]
    ge = np.ones(center.shape, dtype=bool)
    strict = np.zeros(center.shape, dtype=bool)
    for di, dj in product((-1, 0, 1), repeat=2):
        if di == dj == 0:
            continue
        other = values[1+di:values.shape[0]-1+di,
                       1+dj:values.shape[1]-1+dj]
        ge &= center >= other
        strict |= center > other
    return [(int(i)+1, int(j)+1) for i, j in np.argwhere(ge & strict)]


def _peaks_1d(values):
    peaks = [i for i in range(1, len(values)-1)
             if values[i] >= values[i-1] and values[i] >= values[i+1]
             and (values[i] > values[i-1] or values[i] > values[i+1])]
    peaks.append(int(np.argmax(values)))
    return sorted(set(peaks), key=lambda i: (-float(values[i]), i))


def optimize_hybrid(
    rate_function,
    d,
    p_l_val,
    num_level,
    *,
    bounds=((0.0, 0.99), (0.0, 0.3)),
    previous_x=None,
    extra_points=(),
    n_log=25,
    n_linear=13,
    log_decades=6.0,
    grid_refinements=1,
    max_starts=8,
    edge_starts=3,
    local_tol=1e-8,
    local_maxfev=600,
    refinement_rtol=1e-4,
    warn=True,
):
    """Return the best actually evaluated (q, lambda), plus search diagnostics.

    All rate_function evaluations have the original signature. bounds must be
    a nondegenerate nonnegative rectangle (q upper bound <= 1). Both exact
    endpoints of both parameters are evaluated. For the uploaded notebook the
    bounds are ((0, 0.99), (0, 0.3)), NOT the older q <= 0.5 interval.

    Each grid axis is the union of n_linear linear samples and n_log shifted-
    logarithmic samples. log_decades sets sampling emphasis, NOT a physical
    lower bound; zero and the interval below the first positive sample remain
    in the local search domain. Each refinement doubles each grid resolution.
    With defaults there are about 36 x 36, then 72 x 72 grid points, followed by
    at most max_starts joint searches and edge_starts searches per edge/stage.
    Repeated exact parameter pairs use a per-call cache.

    previous_x and extra_points are physical (q, lambda) pairs inside bounds.
    A previous solution and nearby pairs are evaluated at the CURRENT distance.
    NaN hints are not accepted: use None when no preceding positive result exists.

    x is always the best evaluated pair and fun = -best_signed_rate. No positive
    rate found is reported explicitly; it is not proof of a zero physical optimum.
    success means a positive candidate with no local termination failures, not
    global optimality. refinement_stable only compares the last two finite search
    passes. Model exceptions/nonfinite values propagate rather than becoming zero.
    sampled_parameters and sampled_rates retain all actual model evaluations.
    """
    if not callable(rate_function):
        raise TypeError("rate_function must be callable.")
    box = np.asarray(bounds, dtype=float)
    if (box.shape != (2, 2) or not np.isfinite(box).all()
            or np.any(box[:, 0] < 0) or np.any(box[:, 0] >= box[:, 1])
            or box[0, 1] > 1.0):
        raise ValueError("bounds must be two finite increasing nonnegative pairs; q <= 1.")
    n_log = _integer(n_log, "n_log", 3)
    n_linear = _integer(n_linear, "n_linear", 3)
    grid_refinements = _integer(grid_refinements, "grid_refinements", 0)
    max_starts = _integer(max_starts, "max_starts", 1)
    edge_starts = _integer(edge_starts, "edge_starts", 1)
    local_maxfev = _integer(local_maxfev, "local_maxfev", 3)
    local_tol = _positive(local_tol, "local_tol")
    refinement_rtol = _positive(refinement_rtol, "refinement_rtol")
    log_decades = _positive(log_decades, "log_decades")
    if log_decades > 14 or local_tol >= 1:
        raise ValueError("Require log_decades <= 14 and local_tol < 1.")
    axes = [_Axis(*row, log_decades) for row in box]
    lower, upper = box[:, 0], box[:, 1]

    def point(value, name):
        x = np.asarray(value, dtype=float)
        if (x.shape != (2,) or not np.isfinite(x).all()
                or np.any(x < lower) or np.any(x > upper)):
            raise ValueError(f"{name} must be a finite (q, lambda) pair inside bounds.")
        return x.copy()

    hints = [point(x, "extra_points entry") for x in extra_points]
    if previous_x is not None:
        previous = point(previous_x, "previous_x")
        hints.insert(0, previous)
        for factors in product((0.8, 1.0, 1.25), repeat=2):
            hints.append(np.clip(previous * factors, lower, upper))

    cache = {}
    best_x, best_rate = None, -np.inf
    local_records, history = [], []

    def evaluate(x):
        nonlocal best_x, best_rate
        x = point(x, "candidate")
        key = tuple(float(v) for v in x)
        if key not in cache:
            try:
                value = np.asarray(rate_function(x.copy(), d, p_l_val, num_level))
            except Exception as exc:
                # Preserve the exception and its type; add reproducible context only.
                if hasattr(exc, "add_note"):
                    exc.add_note(f"At (q, lambda)={key}, distance={d!r}, "
                                 f"p_l={p_l_val!r}, level={num_level!r}.")
                raise
            if value.size != 1:
                raise ValueError("The key-rate function must return one scalar.")
            z = complex(value.item())
            if not np.isfinite(z.real) or not np.isfinite(z.imag) or z.imag != 0:
                raise FloatingPointError(
                    f"Nonfinite/nonreal rate {z!r} at {key}, distance={d!r}, "
                    f"p_l={p_l_val!r}, level={num_level!r}.")
            rate = float(z.real)
            cache[key] = rate
            if rate > best_rate:
                best_x, best_rate = np.array(key), rate
        return cache[key]

    def physical(u):
        return np.array([float(a.physical(v)) for a, v in zip(axes, u)])

    def unit(x):
        return np.array([float(a.unit(v)) for a, v in zip(axes, x)])

    def record_local(result, kind, stage):
        local_records.append(dict(
            kind=kind, stage=stage, success=bool(result.success),
            nfev=int(result.nfev), message=str(result.message),
        ))

    for hint in hints:
        evaluate(hint)

    for stage in range(grid_refinements + 1):
        factor = 2**stage
        grids = [a.grid((n_log-1)*factor+1, (n_linear-1)*factor+1) for a in axes]
        values = np.array([[evaluate((q, lam)) for lam in grids[1]] for q in grids[0]])
        grid_max = np.unravel_index(int(np.argmax(values)), values.shape)
        grid_best = float(values[grid_max])
        sampled_peak_indices = _peaks_2d(values)
        # A zero edge must not exclude the best negative interior seed.
        interior_max = np.unravel_index(int(np.argmax(values[1:-1, 1:-1])),
                                       values[1:-1, 1:-1].shape)
        sampled_peak_indices.append((int(interior_max[0])+1, int(interior_max[1])+1))
        sampled_peak_indices.append(tuple(map(int, grid_max)))
        ranked_indices = sorted(set(sampled_peak_indices),
                                key=lambda ij: (-float(values[ij]), ij))
        candidates = [best_x.copy()]
        candidates += sorted(hints, key=lambda x: -evaluate(x))[:2]
        candidates += [np.array([grids[0][i], grids[1][j]]) for i, j in ranked_indices]

        # Distinct seeds in shifted-log coordinates, not eight adjacent starts
        # at the same peak. Every grid/hint point is still retained in cache.
        seeds = []
        for x in candidates:
            u = unit(x)
            if not any(np.max(np.abs(u-unit(s))) < 0.015 for s in seeds):
                seeds.append(x.copy())
            if len(seeds) >= max_starts:
                break

        for start in seeds:
            neighbors = [_neighbors(g, x) for g, x in zip(grids, start)]
            probes = [evaluate(v) for v in product(
                (neighbors[0][0], start[0], neighbors[0][1]),
                (neighbors[1][0], start[1], neighbors[1][1]),
            )]
            # Do not pretend a local solver explores a flat, uninformative patch.
            if min(probes) == max(probes):
                continue
            u0 = unit(start)
            simplex = np.tile(u0, (3, 1))
            for j in range(2):
                choices = []
                for coordinate in neighbors[j]:
                    if coordinate != start[j]:
                        probe = start.copy()
                        probe[j] = coordinate
                        choices.append((evaluate(probe), float(axes[j].unit(coordinate))))
                target = max(choices, key=lambda v: v[0])[1]
                simplex[j+1, j] = (u0[j] + target) / 2.0
            # Rescale this local objective, not the physical rate. This avoids
            # a fixed bps tolerance suppressing refinement of small-rate peaks.
            amplitude = max(max(abs(v) for v in probes), np.finfo(float).tiny)
            local = minimize(
                lambda u: -evaluate(physical(u)) / amplitude,
                u0, method="Nelder-Mead", bounds=((0.0, 1.0), (0.0, 1.0)),
                options=dict(initial_simplex=simplex, xatol=local_tol,
                             fatol=1e-10, maxfev=local_maxfev),
            )
            evaluate(physical(local.x))
            record_local(local, "joint", stage)

        # Independent boundary optimization: a clipped 2-D simplex alone does
        # not provide a reliable search along an entire edge of the rectangle.
        for fixed, end in product((0, 1), (0, 1)):
            moving = 1 - fixed
            edge_values = values[0 if end == 0 else -1, :] if fixed == 0 else values[:, 0 if end == 0 else -1]
            if np.min(edge_values) == np.max(edge_values):
                continue
            edge_grid = grids[moving]
            for index in _peaks_1d(edge_values)[:edge_starts]:
                i0, i1 = max(0, index-1), min(len(edge_grid)-1, index+1)
                if np.min(edge_values[i0:i1+1]) == np.max(edge_values[i0:i1+1]):
                    continue
                ua = float(axes[moving].unit(edge_grid[i0]))
                ub = float(axes[moving].unit(edge_grid[i1]))
                amplitude = max(float(np.max(np.abs(edge_values[i0:i1+1]))),
                                np.finfo(float).tiny)

                def edge_point(t):
                    x = lower.copy()
                    x[fixed] = box[fixed, end]
                    x[moving] = float(axes[moving].physical(ua + t*(ub-ua)))
                    return x

                local = minimize_scalar(
                    lambda t: -evaluate(edge_point(float(t))) / amplitude,
                    method="bounded", bounds=(0.0, 1.0),
                    options=dict(xatol=local_tol, maxiter=local_maxfev),
                )
                evaluate(edge_point(float(local.x)))
                record_local(local, f"edge_{fixed}_{end}", stage)

        history.append(dict(
            stage=stage, grid_shape=tuple(values.shape), grid_best_rate=grid_best,
            grid_best_x=[float(grids[0][grid_max[0]]), float(grids[1][grid_max[1]])],
            best_rate=best_rate, best_x=best_x.tolist(), nfev=len(cache),
        ))

    positive = best_rate > 0.0
    failures = sum(not item["success"] for item in local_records)
    improvement = np.nan
    stable = False
    if positive and len(history) >= 2:
        prev_rate = history[-2]["best_rate"]
        # This difference is a diagnostic, not a certified error estimate.
        improvement = abs(best_rate-prev_rate) / max(abs(best_rate), abs(prev_rate))
        stable = bool(improvement <= refinement_rtol)
    if not positive:
        status = 2
        message = "No positive rate found; this is NOT a certified physical zero."
    elif failures:
        status = 1
        message = "Positive candidate retained; some local searches hit their limits."
    else:
        status = 0
        message = "Positive candidate retained; no global-optimality certificate."
    if warn and (not positive or failures):
        warnings.warn(f"{message} d={d!r}, p_l={p_l_val!r}, level={num_level!r}.",
                      RuntimeWarning, stacklevel=2)
    if warn and positive and grid_refinements > 0 and not stable:
        warnings.warn(
            f"A denser pass improved the best rate by a relative {improvement:.3g}; "
            f"check higher grid density. d={d!r}, p_l={p_l_val!r}, level={num_level!r}.",
            RuntimeWarning, stacklevel=2)
    samples = np.array(list(cache))
    return OptimizeResult(
        x=best_x.copy(), fun=-best_rate, success=(status == 0), status=status,
        message=message, nfev=len(cache), positive_rate_found=positive,
        local_refinements_failed=failures, local_searches=local_records,
        grid_best_rate=history[-1]["grid_best_rate"], search_history=history,
        refinement_relative_gain=improvement, refinement_stable=stable,
        near_upper_bound=tuple(bool(v) for v in (upper-best_x <= 1e-6*(upper-lower))),
        sampled_parameters=samples,
        sampled_rates=np.array(list(cache.values()), dtype=float),
    )


def hybrid_diagnostic_record(result, d, p_l_val, num_level):
    """One CSV-compatible row. Records the actual best candidate, even if <= 0."""
    return dict(
        distance_km=float(d), p_l=float(p_l_val), level=int(num_level),
        best_q=float(result.x[0]), best_lambda=float(result.x[1]),
        best_signed_rate_bps=float(-result.fun),
        positive_rate_found=bool(result.positive_rate_found),
        search_success=bool(result.success), grid_best_rate_bps=float(result.grid_best_rate),
        refinement_stable=bool(result.refinement_stable),
        refinement_relative_gain=float(result.refinement_relative_gain),
        q_near_upper_bound=bool(result.near_upper_bound[0]),
        lambda_near_upper_bound=bool(result.near_upper_bound[1]),
        nfev=int(result.nfev), local_refinements_failed=int(result.local_refinements_failed),
        grid_stages=len(result.search_history), message=str(result.message),
    )

# Short source-loading API used by both notebooks. Serialized expressions are
# supplied by the user; no source expressions, hash checks, or fallback models
# are bundled here.
from pathlib import Path
import pandas as pd
import sympy as sp
try:
    import dill as _serializer
except ImportError:
    import pickle as _serializer


def load_source_function(path, argument_names):
    """Load a trusted serialized SymPy expression and return a NumPy function."""
    with Path(path).open("rb") as stream:
        expression = _serializer.load(stream)
    if not hasattr(expression, "free_symbols"):
        expression = sp.sympify(expression)
    symbols = {str(symbol): symbol for symbol in expression.free_symbols}
    args = [symbols.get(name, sp.Symbol(name)) for name in argument_names]
    return sp.lambdify(args, expression, modules="numpy", cse=True)


probability = _probability
empty_rates = _zero_rates
multiplexed_click_probability = _multiplexed_click_probability


# Shared CSV export. No assertion cells, model-test runs, or hash manifests.
def _write_csv(frame, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def run_rate_sweeps(components, optimize, architecture, *,
                    distances=None, losses=(0.01, 0.1, 0.5), levels=(0, 1, 2),
                    output_dir=None, postprocessing="one_way", pooling="route",
                    verbose=True):
    """Optimize the common total; save W/GHZ at those same source settings.

    Returns (tables_by_loss, optimizer_results, route_rates), retaining the
    earlier notebooks' return convention. Only the main rate CSVs are exported.
    Each completed point is checkpointed; previous-distance optima are hints,
    not copied rates or restrictions on the search interval.
    """
    distances = np.asarray(np.linspace(0.0, 500.0, 20)
                           if distances is None else distances, dtype=float)
    levels = tuple(levels)
    folder = (Path(output_dir) if output_dir is not None
              else Path(f"overall_rates_{postprocessing}_{pooling}"))
    folder.mkdir(parents=True, exist_ok=True)
    tables, diagnostics, route_rows = {}, [], []
    for loss in losses:
        loss = float(loss)
        frame = pd.DataFrame({"distance_km": distances})
        for level in levels:
            names = ["keyrate_bps", "keyrate_W_bps", "keyrate_GHZ_bps", "opt_qs",
                     "p_accept", "generation_time_us", "round_time_us"]
            if architecture == "hybrid":
                names.append("opt_lambdas")
            for name in names:
                frame[f"{name}_level_{level}"] = np.nan
        suffix = format(loss, ".12g").replace(".", "_")
        path = folder / f"{architecture}_p_l_{suffix}.csv"
        for level in levels:
            previous = None
            for index, distance in enumerate(distances):
                result = optimize(float(distance), loss, level, previous)
                parts = components(result.x, float(distance), loss, level)
                for column, key in (("keyrate_bps", "total_bps"),
                                    ("keyrate_W_bps", "W_bps"),
                                    ("keyrate_GHZ_bps", "GHZ_bps"),
                                    ("p_accept", "p_accept"),
                                    ("generation_time_us", "generation_time_us"),
                                    ("round_time_us", "round_time_us")):
                    frame.loc[index, f"{column}_level_{level}"] = parts[key]
                frame.loc[index, f"opt_qs_level_{level}"] = (
                    float(result.x[0]) if result.positive_rate_found else np.nan)
                if architecture == "hybrid":
                    frame.loc[index, f"opt_lambdas_level_{level}"] = (
                        float(result.x[1]) if result.positive_rate_found else np.nan)
                previous = result.x.copy() if result.positive_rate_found else None
                diagnostics.append(dict(
                    distance_km=float(distance), p_l=loss, level=level,
                    q=float(result.x[0]),
                    **({"lambda": float(result.x[1])} if architecture == "hybrid" else {}),
                    total_bps=parts["total_bps"], W_bps=parts["W_bps"],
                    GHZ_bps=parts["GHZ_bps"], nfev=int(result.nfev),
                    positive_rate_found=bool(result.positive_rate_found),
                    search_success=bool(result.success), message=str(result.message)))
                for route in parts["routes"]:
                    route_rows.append(dict(distance_km=float(distance), p_l=loss,
                        level=level, q=float(result.x[0]),
                        **({"lambda": float(result.x[1])} if architecture == "hybrid" else {}),
                        postprocessing=postprocessing, pooling=pooling, **route))
                _write_csv(frame, path)
                if verbose:
                    print(f"{architecture}: loss={loss:g}, level={level}, "
                          f"d={distance:g} km, total={parts['total_bps']:.6g} bps")
        tables[loss] = frame
    return tables, pd.DataFrame(diagnostics), pd.DataFrame(route_rows)
