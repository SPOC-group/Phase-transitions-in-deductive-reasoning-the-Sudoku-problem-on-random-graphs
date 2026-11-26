
from functools import cached_property
from math import exp, log, sqrt,comb
from pathlib import Path

from scipy.integrate import quad
from scipy.optimize import brentq

from csv import DictWriter
from math import comb
DATA = Path(__file__).resolve().parent / "data"


def coverage(m, degree, colors):
    counts = {k: int(k == 0) for k in range(m + 1)}
    coefficients = {}
    for k in range(degree + 1):
        coefficients[k] = counts[colors] * comb(degree, k) * m**(degree - k)
        counts = {j: j * counts[j] + (m - j + 1) * counts.get(j - 1, 0)
                  for j in range(m + 1)}
    return coefficients


def evaluate(coefficients, x, z):
    degree = len(coefficients) - 1
    return sum(c * x**k * z**(degree - k)
               for k, c in coefficients.items() if c)


def derivative(coefficients):
    degree = len(coefficients) - 1
    return {k: (k + 1) * coefficients[k + 1] - (degree - k) * coefficients[k]
            for k in range(degree)}


def write_csv(path, rows):
    rows = iter(rows)
    first = next(rows)
    with path.open("w", newline="") as file:
        writer = DictWriter(file, fieldnames=first.keys())
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(rows)

class InfinityTheory:
    def __init__(self, q, d):
        self.q, self.d, self.m = q, d, q - 1
        m, n = self.m, d - 1
        self.scale = m**n

        f = coverage(m, n, m)
        missing = coverage(m, n, m - 1)
        h = {k: missing[k] // m for k in missing}
        g = {k: f[k] + h[k] for k in f}
        W = coverage(m, d, m)

        self.f = lambda r: evaluate(f, r, 1 - r)
        self.h = lambda r: evaluate(h, r, 1 - r)
        self.W = lambda r: evaluate(W, r, 1 - r)
        fp, Wp, gp = derivative(f), derivative(W), derivative(g)
        fpp = derivative(fp)
        self.fp = lambda r: evaluate(fp, r, 1 - r)
        self.fpp = lambda r: evaluate(fpp, r, 1 - r)
        self.gp = lambda r: evaluate(gp, r, 1 - r)
        self.Wp = lambda r: evaluate(Wp, r, 1 - r)

    def chi(self, r):
        return (1 - r) * self.fp(r) / self.h(r)

    def fast_e_mu(self, r):
        return (r * self.h(r) / (1 - r) - self.f(r)) / self.scale

    def slow_e_mu(self, r):
        return (r * self.fp(r) - self.f(r)) / self.scale

    @cached_property
    def transition(self):
        if self.d == self.q:
            return None, 0.0, 0.0

        contact = brentq(lambda r: self.chi(r) - 1, 1e-10, 1)
        if self.fpp(contact) >= 0:
            return log(self.slow_e_mu(contact)), contact, contact

        inflection = brentq(self.fpp, 1e-10, contact, xtol=1e-14)

        def tangent(rs):
            return brentq(lambda r: self.f(rs) - self.f(r) - self.fp(r) * (rs - r),
                          0, inflection, xtol=1e-14)

        rs = brentq(lambda r: self.h(r) - (1 - r) * self.fp(tangent(r)),
                    contact, 1, xtol=1e-14)
        rm = tangent(rs)
        return log(self.slow_e_mu(rm)), rm, rs

    def fast(self, r):
        e_mu = self.fast_e_mu(r)
        hint = e_mu * self.m**self.d
        Z_fac = self.q * (hint + self.W(r))
        Z_var = self.q * self.m * (2 * r - r*r)
        mu = log(e_mu)
        phi = log(Z_fac) - self.d / 2 * log(Z_var)
        rho = self.q * hint / Z_fac
        return dict(q=self.q, d=self.d, branch="fast", mu=mu, rho=rho, phi=phi,
                    s=phi - mu * rho, chi=self.chi(r), seriality=0.0)

    def fast_r(self, mu):
        mu_c, _, rs = self.transition
        if mu == mu_c:
            return rs
        e_mu = exp(mu)
        return brentq(lambda r: r * self.h(r) - (1 - r) * (self.f(r) + self.scale * e_mu),
                      rs, 1, xtol=1e-14)

    def fields(self, r):
        fp = self.fp(r)
        eta = fp**(-1 / (self.d - 2))
        eta_prime = -eta * self.fpp(r) / ((self.d - 2) * fp)
        return eta, r * eta, eta_prime, eta + r * eta_prime

    def slow(self, r_minus):
        _, rm, rs = self.transition
        eta0, xi0, _, _ = self.fields(r_minus)
        eta_m, _, _, _ = self.fields(rm)
        d = self.d

        def edge(r):
            eta, _, _, xi_prime = self.fields(r)
            return eta * xi_prime

        def site(r):
            eta, _, _, xi_prime = self.fields(r)
            return eta**(d - 1) * xi_prime * self.Wp(r)

        e_mu = self.slow_e_mu(r_minus)
        mu = log(e_mu)
        hint = e_mu * (self.m * eta0)**d
        ze = (2 * (eta0 * xi0 + quad(edge, r_minus, rm)[0] + eta_m**2 * (rs - rm))
              - (eta_m * rs)**2)
        zi = (hint + eta0**d * self.W(r_minus) + quad(site, r_minus, rm)[0]
              + eta_m**d * (self.W(rs) - self.W(rm)))
        rho = hint / zi
        phi = log(self.q * zi) - d / 2 * log(self.q * self.m * ze)
        branch = "marginal" if rm == rs else "mixed"
        A0 = xi0 * (eta0 + eta0**(d - 1) * self.h(r_minus)) / ze
        return dict(q=self.q, d=d, branch=branch, mu=mu, rho=rho, phi=phi,
                    s=phi - mu * rho, chi=self.fp(rs) / self.fp(rm), A0=A0, ze=ze)

    def slow_r(self, mu):
        mu_c, rm, _ = self.transition
        if mu == mu_c:
            return rm
        e_mu = exp(mu)
        return brentq(lambda r: self.slow_e_mu(r) - e_mu, 0, rm, xtol=1e-14)

    def profile(self, r_minus, point, steps):
        _, rm, _ = self.transition
        eta, xi, ep, xp = self.fields(r_minus)
        rows = {0: dict(m=0., cumulative=point["A0"])}
        speed = sqrt(max(0., -ep * xp))
        mass = xp * (eta - xi + eta**(self.d - 1) *
                     (self.h(r_minus) + r_minus * self.gp(r_minus))) / point["ze"]
        step = (rm - r_minus) / steps
        time, cumulative = 0., point["A0"]

        for i in range(1, steps + 1):
            r = r_minus + i * step
            eta, xi, ep, xp = self.fields(r)
            next_speed = sqrt(max(0., -ep * xp))
            next_mass = xp * (eta - xi + eta**(self.d - 1) *
                              (self.h(r) + r * self.gp(r))) / point["ze"]
            time += step * (speed + next_speed) / 2
            cumulative += step * (mass + next_mass) / 2
            rows[i] = dict(m=time, cumulative=cumulative)
            speed, mass = next_speed, next_mass

        for row in rows.values():
            row["m"] /= time
        return rows

    def point(self, mu, steps=4000):
        mu_c, _, _ = self.transition
        if mu_c is None or mu >= mu_c:
            return self.fast(self.fast_r(mu))

        r = self.slow_r(mu)
        point = self.slow(r)
        profile = self.profile(r, point, steps)
        point["seriality"] = sum(
            (profile[i]["m"] - profile[i - 1]["m"])
            * (1 - (profile[i]["cumulative"] + profile[i - 1]["cumulative"]) / 2)
            for i in range(1, steps + 1)
        )
        return point

    def minimum(self):
        mu_c, rm, rs = self.transition
        if (self.q, self.d) == (3, 3):
            # The r=0 limit has e^mu (q-1)^d ~ 4r^2 and W ~ 12r^2.
            return dict(rho=4 / (4 + 12), mu="-inf")
        if mu_c is None:
            r = brentq(lambda r: self.fast(r)["s"], 1e-7, 0.999999, xtol=1e-14)
            return self.fast(r)
        if self.fast(rs)["s"] < 0:
            r = brentq(lambda r: self.fast(r)["s"], rs, 0.999999, xtol=1e-14)
            return self.fast(r)
        r = brentq(lambda r: self.slow(r)["s"], rm * 1e-5, rm, xtol=1e-14)
        return self.slow(r)


def thresholds():
    # Table I, with (9,20) from the text.
    cases = {3: range(3, 6), 4: range(4, 10), 5: range(5, 14), 9: range(20, 21)}
    for q, degrees in cases.items():
        for d in degrees:
            theory = InfinityTheory(q, d)
            mu_c, rm, rs = theory.transition
            minimum = theory.minimum()
            yield dict(q=q, d=d, branch=("fast" if mu_c is None else
                                         "marginal" if rm == rs else "mixed"),
                       mu_c=mu_c, rho_c=theory.fast(rs)["rho"] if mu_c is not None else None,
                       r_m=rm if mu_c is not None else None,
                       r_s=rs if mu_c is not None else None,
                       mu_min=minimum["mu"], rho_min=minimum["rho"])


def chi_curves():
    for q, d in ((3, 3), (3, 4), (3, 5), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8)):
        theory = InfinityTheory(q, d)
        mu_c, _, rs = theory.transition
        low = theory.fast(rs)["rho"] if mu_c is not None else theory.minimum()["rho"]
        yield dict(q=q, d=d, rho=1., chi=0.)
        for i in range(99, int(100 * low), -1):
            rho = i / 100
            r = brentq(lambda r: theory.fast(r)["rho"] - rho, max(rs, 1e-9), 1 - 1e-10)
            yield dict(q=q, d=d, rho=rho, chi=theory.chi(r))
        if mu_c is not None:
            yield dict(q=q, d=d, rho=low, chi=theory.chi(rs))


def seriality_curves():
    widths = {(3, 3): 0, (3, 4): 9, (4, 4): 0, (4, 5): 9,
              (4, 6): 10, (4, 7): 9, (4, 8): 8}
    for (q, d), width in widths.items():
        theory = InfinityTheory(q, d)
        mu_c, rm, rs = theory.transition
        if mu_c is not None:
            for i in range(500):
                mu = mu_c - width + i * (width - 1e-5) / 499
                point = theory.point(mu)
                yield {k: point[k] for k in ("q", "d", "branch", "mu", "rho", "seriality")}
            left = theory.slow(rm)
            rho_c = theory.fast(rs)["rho"]
            yield dict(q=q, d=d, branch=left["branch"], mu=mu_c,
                       rho=rho_c, seriality=max(0., 1 - left["A0"]))
            yield dict(q=q, d=d, branch="fast", mu=mu_c, rho=rho_c, seriality=0.)
        else:
            low = theory.minimum()
            yield dict(q=q, d=d, branch="fast", mu=low["mu"], rho=low["rho"], seriality=0.)
        yield dict(q=q, d=d, branch="fast", mu="inf", rho=1., seriality=0.)


def entropy_curve():
    theory = InfinityTheory(4, 7)
    mu_c, _, rs = theory.transition
    lower = theory.minimum()["mu"]
    for i in range(401):
        mu = lower + i * (mu_c - lower) / 401
        point = theory.slow(theory.slow_r(mu))
        yield {k: point[k] for k in ("q", "d", "branch", "mu", "rho", "phi", "s")}
    for i in range(401):
        mu = mu_c + i * (3 - mu_c) / 400
        point = theory.fast(rs if i == 0 else theory.fast_r(mu))
        yield {k: point[k] for k in ("q", "d", "branch", "mu", "rho", "phi", "s")}


def cumulative_curves():
    # Figure 4 fixes rho. Figures 13 and 15 fix mu.
    for q, d, fixed_mu, fixed_rho, figure in (
        (4, 5, None, 0.3175, 4), (4, 7, None, 0.1775, 4),
        (4, 5, -3, None, 13), (4, 8, -4, None, 15)
    ):
        theory = InfinityTheory(q, d)
        _, rm, _ = theory.transition
        if fixed_mu is None:
            r = brentq(lambda r: theory.slow(r)["rho"] - fixed_rho, rm * 1e-5, rm)
        else:
            r = theory.slow_r(fixed_mu)
        point = theory.slow(r)
        label = {k: point[k] for k in ("q", "d", "branch", "mu", "rho")}
        yield dict(figure=figure, **label, m=0., cumulative=0.)
        profile = theory.profile(r, point, 10000)
        j = 0
        for i in range(201):
            m = i / 200
            while profile[j + 1]["m"] < m:
                j += 1
            a, b = profile[j], profile[j + 1]
            fraction = (m - a["m"]) / (b["m"] - a["m"])
            cumulative = a["cumulative"] + fraction * (b["cumulative"] - a["cumulative"])
            yield dict(figure=figure, **label, m=m, cumulative=min(1., cumulative))
        if point["branch"] == "mixed":
            yield dict(figure=figure, **label, m=1., cumulative=1.)


def phase_plane():
    for q, d, mu, figure in ((4, 5, -1, 12), (4, 5, -3, 13), (4, 8, -4, 15)):
        theory = InfinityTheory(q, d)
        label = dict(q=q, d=d, mu=mu, figure=figure)
        if figure == 12:
            r = theory.fast_r(mu)
            Y = ((1 - r) / theory.h(r))**(1 / (d - 2))
            X = r * Y
            yield dict(**label, segment="initial", X=exp(mu) * ((q-1) * Y)**(d-1), Y=Y)
        else:
            _, rm, rs = theory.transition
            r = theory.slow_r(mu)
            Y0, _, _, _ = theory.fields(r)
            yield dict(**label, segment="initial", X=exp(mu) * ((q-1) * Y0)**(d-1), Y=Y0)
            for i in range(201):
                Y, X, _, _ = theory.fields(r + (rm - r) * i / 200)
                yield dict(**label, segment="slow", X=X, Y=Y)
            Y, _, _, _ = theory.fields(rm)
            X = rs * Y
        yield dict(**label, segment="saddle", X=X, Y=Y)
        yield dict(**label, segment="terminal", X=X, Y=X)


def main():
    DATA.mkdir(exist_ok=True)
    write_csv(DATA / "thresholds.csv", thresholds())
    write_csv(DATA / "fast_chi.csv", chi_curves())
    write_csv(DATA / "cumulative_infinity.csv", cumulative_curves())
    write_csv(DATA / "phase_plane_infinity.csv", phase_plane())
    write_csv(DATA / "entropy_infinity.csv", entropy_curve())
    write_csv(DATA / "seriality_infinity.csv", seriality_curves())


if __name__ == "__main__":
    main()
