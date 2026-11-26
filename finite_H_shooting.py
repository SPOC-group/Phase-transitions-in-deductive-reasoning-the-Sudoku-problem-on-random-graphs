
from math import exp, log,comb
from pathlib import Path

import mpmath as mp
from scipy.linalg import solve_banded
from scipy.optimize import root

from infinite_H import InfinityTheory
from csv import DictWriter



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


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def continuation_guesses(q, d, mu, horizons):
    # it is better to start at small H and use the  solut at the previous H to initialize the next one. At each H of interest, we use the resulting
    # X_t and Y_t to initialize the shooting problem.
    # We use the H -> infinity theory only for Figure 13 to get x_m
    
    m, n = q - 1, d - 1
    f_coef = coverage(m, n, m)
    h_all = coverage(m, n, m - 1)
    h_coef = {k: c // m for k, c in h_all.items()}
    e_mu = exp(mu)

    def fields(message, H):
        X = {-1: 0.}
        for t in range(H + 1):
            X[t] = X[t - 1] + message["L", t]
        Y = {H: X[H]}
        suffix = 0.
        for t in range(H - 1, -1, -1):
            Y[t] = X[t] + message["B", t + 1] + suffix
            suffix += message.get(("E", t + 1), 0.)
        return X, Y

    def update(message, H):
        X, Y = fields(message, H)
        new = {("L", 0): e_mu * (m * Y[0])**n}
        for t in range(1, H + 1):
            x, previous, y = X[t - 1], X[t - 2], Y[t]
            L = evaluate(f_coef, x, y - x) - evaluate(f_coef, previous, y - previous)
            B = L + evaluate(h_coef, x, y - x)
            new["L", t] = L
            new["B", t] = B
            if t > 1:
                new["E", t] = B - evaluate(h_coef, previous, y - previous)
        scale = sum(new.values())
        return {key: value / scale for key, value in new.items()}, scale

    H = min(5, min(horizons))
    message = {
        (sector, t): 1.
        for sector, first in {"L": 0, "B": 1, "E": 2}.items()
        for t in range(first, H + 1)
    }
    size = len(message)
    message = {key: value / size for key, value in message.items()}
    for _ in range(20000):
        new, _ = update(message, H)
        error = max(abs(new[key] - message[key]) for key in message)
        message = {key: .8 * message[key] + .2 * new[key] for key in message}
        if error < 1e-14:
            break

    for target in horizons:
        while H < target:
            step = 1 if H < 19 else 5 if H < 99 else 10
            next_H = min(H + step, target)
            guess = {}
            for sector, first in {"L": 0, "B": 1, "E": 2}.items():
                for t in range(first, next_H + 1):
                    position = t * H / next_H
                    j = int(position)
                    a = message.get((sector, j), 0.)
                    b = message.get((sector, j + 1), 0.)
                    guess[sector, t] = a + (position - j) * (b - a)
            total = sum(guess.values())
            guess = {key: value / total for key, value in guess.items()}

            def residual(v):
                trial = dict(zip(guess, v.tolist()))
                proposed, _ = update(trial, next_H)
                return tuple(proposed[key] - trial[key] for key in trial)

            fit = root(residual, tuple(guess.values()), options={"xtol": 1e-10})
            message = dict(zip(guess, fit.x.tolist()))
            H = next_H

        _, scale = update(message, H)
        factor = scale**(-1 / (n - 1))
        X, Y = fields({key: value * factor for key, value in message.items()}, H)
        u, v = refine_shooting_guess(X, Y, H, m, n, mu, f_coef, h_coef)
        yield H, u, v


def refine_shooting_guess(X, Y, H, m, n, mu, f_coef, h_coef):
    """Refine the continued fields before extracting the first two values."""
    with mp.workdps(65):
        fields = {2*t + i: mp.mpf(field[t])
                  for t in range(H + 1) for i, field in enumerate((X, Y))}
        exp_mu = mp.exp(mu)

        def x(t):
            return fields.get(2*t, 0)

        def y(t):
            return fields[2*t + 1]

        def f(a, b):
            return evaluate(f_coef, a, b - a)

        def h(a, b):
            return evaluate(h_coef, a, b - a)

        def equation(i):
            t = i // 2
            if i % 2 == 0:
                if t == 0:
                    return x(0) - exp_mu * (m * y(0))**n
                return x(t) - x(t-1) - f(x(t-1), y(t)) + f(x(t-2), y(t))
            if t == H:
                return y(H) - x(H)
            if t == H-1:
                return y(t) - y(t+1) - h(x(t), y(t+1))
            return y(t) - y(t+1) - h(x(t), y(t+1)) + h(x(t), y(t+2))

        size, width = len(fields), 4
        delta = mp.mpf("1e-22")
        residual = {i: equation(i) for i in fields}
        bands = {k: {} for k in range(2*width + 1)}
        for j in fields:
            fields[j] += delta
            for i in range(max(0, j-width), min(size, j+width+1)):
                bands[width + i - j][j] = float((equation(i) - residual[i]) / delta)
            fields[j] -= delta

        jacobian = tuple(tuple(band.get(j, 0.) for j in fields)
                         for band in bands.values())
        for _ in range(4):
            rhs = tuple(-float(equation(i)) for i in fields)
            correction = solve_banded((width, width), jacobian, rhs)
            for j, value in enumerate(correction):
                fields[j] += float(value)

        r1 = x(0) / y(0)
        a2 = (y(0) - y(1)) / (y(0) - x(0))
        return mp.log(r1 / (1 - r1)), mp.log(a2 / (1 - a2))


def shoot(q, d, mu, H, u0, v0):
    m, n = q - 1, d - 1
    f_coef = coverage(m, n, m)
    h_all = coverage(m, n, m - 1)
    h_coef = {k: h_all[k] // m for k in h_all}  # One specified color is absent.
    h_y_coef = {k: (n - k) * h_coef[k] for k in range(n)}
    W_coef = coverage(m, d, m)

    def f(x, y):
        return evaluate(f_coef, x, y - x)

    def h(x, y):
        return evaluate(h_coef, x, y - x)

    def W(x, y):
        return evaluate(W_coef, x, y - x)

    def trajectory(u, v):
        r1 = 1 / (1 + mp.exp(-u))
        a2 = 1 / (1 + mp.exp(-v))
        Y = {0: (r1 / (e_mu * m**n))**(mp.mpf(1) / (n - 1))}
        X = {-1: mp.mpf(0), 0: r1 * Y[0]}
        Y[1] = Y[0] - a2 * (Y[0] - X[0])

        for t in range(1, H + 1):
            if t > 1:
                x, y = X[t - 2], Y[t - 1]
                drop = Y[t - 2] - y
                target = h(x, y) - drop

                def slope(y_new):
                    return evaluate(h_y_coef, x, y_new - x)

                Y[t] = mp.findroot(lambda y_new: h(x, y_new) - target,
                                   y - drop / slope(y), solver="newton",
                                   df=slope, tol=mp.mpf("1e-78"))
            X[t] = X[t - 1] + f(X[t - 1], Y[t]) - f(X[t - 2], Y[t])
        return X, Y

    def residual(u, v):
        X, Y = trajectory(u, v)
        R1 = (Y[H] - X[H]) / max(Y[H], X[H])
        R2 = mp.log((Y[H - 1] - Y[H]) / h(X[H - 1], Y[H]))
        return R1, R2

    def jacobian(u, v):
        step = mp.mpf("1e-27")
        R1, R2 = residual(u, v)
        Ru1, Ru2 = residual(u + step, v)
        Rv1, Rv2 = residual(u, v + step)
        return mp.matrix(((Ru1 - R1, Rv1 - R1), (Ru2 - R2, Rv2 - R2))) / step

    with mp.workdps(90):
        e_mu = mp.exp(mu)
        u, v = mp.findroot(residual, (mp.mpf(u0), mp.mpf(v0)),
                           J=jacobian, tol=mp.mpf("1e-45"))
        X, Y = trajectory(u, v)
        weights = {0: e_mu * (m * Y[0])**d}
        for t in range(1, H + 1):
            weights[t] = W(X[t - 1], Y[t]) - W(X[t - 2], Y[t])
        Z_fac = q * sum(weights.values())
        Z_var = q * m * (2 * sum(Y[t] * (X[t] - X[t - 1]) for t in range(H + 1))
                         - X[H]**2)
        rho = q * weights[0] / Z_fac
        phi = mp.log(Z_fac) - d / 2 * mp.log(Z_var)
        s = phi - mu * rho
        rows = {}
        cumulative = 0
        for t in range(H + 1):
            P_t = q * weights[t] / Z_fac
            cumulative += P_t
            rows[t] = dict(q=q, d=d, mu=mu, H=H, t=t, m=t / H,
                           X=float(X[t]), Y=float(Y[t]), P_t=float(P_t),
                           cumulative=float(cumulative), rho=float(rho),
                           phi=float(phi), s=float(s))
        return rows


def crossing(rows, field, threshold, direction):
    t = next(t for t in rows if direction * (rows[t][field] - threshold) > 0)
    y0, y1 = rows[t - 1][field], rows[t][field]
    return t - 1 + (threshold - y0) / (y1 - y0)


def main():
    theory = InfinityTheory(4, 8)
    _, rm, _ = theory.transition
    ym, xm, _, _ = theory.fields(rm)
    DATA.mkdir(exist_ok=True)
    trajectories, duration = {}, {}

    cases = {
        (4, 5, -1.): (50, 100, 200),
        (4, 5, -3.): (50, 100, 200),
        (4, 8, -3.): range(50, 201, 10),
        (4, 8, -4.): (50, *range(100, 201, 10)),
    }
    for (q, d, mu), figure_labels in cases.items():
        horizons = {label - 1: label for label in figure_labels}
        for H, u, v in continuation_guesses(q, d, mu, horizons):
            H_figure = horizons[H]
            rows = shoot(q, d, mu, H, u, v)
            for t, p in rows.items():
                p.update(H_figure=H_figure, m_figure=(t + 1) / H_figure)
            trajectories[q, d, mu, H] = rows
            if (q, d) == (4, 8) and (mu == -3 or (mu == -4 and H_figure >= 100)):
                tx = crossing(rows, "X", xm, 1)
                ty = crossing(rows, "Y", ym, -1)
                duration[mu, H] = dict(
                    q=q, d=d, mu=mu, H=H, H_figure=H_figure,
                    log_H=log(H), log_H_figure=log(H_figure), x_m=xm, y_m=ym,
                    H_minus_H2=H - tx,
                    H_minus_H2_XY=H - max(tx, ty),
                    count_XY=sum(p["X"] > xm and p["Y"] < ym for p in rows.values())
                )

    write_csv(DATA / "shooting_trajectories.csv",
              (p for rows in trajectories.values() for p in rows.values()))
    write_csv(DATA / "mixed_final_window.csv", duration.values())


if __name__ == "__main__":
    main()
