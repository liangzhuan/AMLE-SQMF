import numpy as np
from scipy.linalg import eigh, svd


def pca_projection(X, y, d):
    c = X.mean(axis=1)
    U = svd(X - c[:, None], full_matrices=False)[0][:, :d]
    return c + U @ (U.T @ (y - c)), dict(iterations=1, converged=True)


def sphere_projection(X, y, d):
    c = X.mean(axis=1)
    W = svd(X - c[:, None], full_matrices=False)[0][:, :d + 1]
    Z = (X - c[:, None]).T @ W
    design = np.column_stack((Z, np.ones(X.shape[1])))
    coefficients, _, rank, _ = np.linalg.lstsq(
        design, -np.sum(Z ** 2, axis=1), rcond=None,
    )
    center = -0.5 * coefficients[:-1]
    radius_squared = float(center @ center - coefficients[-1])
    delta = W.T @ (y - c) - center
    distance = np.linalg.norm(delta)
    if rank < d + 2 or not np.isfinite(radius_squared) or radius_squared <= 1e-12:
        raise ValueError("The local sphere is not identifiable.")
    if not np.isfinite(distance) or distance < 1e-12:
        raise ValueError("The query is at the local sphere center.")
    prediction = c + W @ (center + np.sqrt(radius_squared) * delta / distance)
    return prediction, dict(iterations=1, converged=True)


def quadratic_projection(X, y, d):
    mean = X.mean(axis=1)
    centered = X - mean[:, None]
    scale = float(np.median(np.linalg.norm(centered, axis=0)))
    if not np.isfinite(scale) or scale < 1e-10:
        scale = float(np.sqrt(np.mean(centered ** 2)))
    if not np.isfinite(scale) or scale < 1e-10:
        scale = 1.0
    data = centered / scale
    query = (y - mean) / scale
    c = data.mean(axis=1)
    Z = data - c[:, None]
    W = svd(Z, full_matrices=False)[0][:, :d + 1]
    U, v = W[:, :d], W[:, d]
    T = U.T @ Z
    ii, jj = np.triu_indices(d)
    features = (T[ii] * T[jj]).T
    theta = np.linalg.solve(
        features.T @ features + 1e-5 * np.eye(len(ii)),
        features.T @ (v @ Z),
    )
    A = np.zeros((d, d))
    A[ii, jj] = 0.5 * theta
    A += A.T
    tau = U.T @ (query - c)

    def evaluate(value):
        prediction = c + U @ value + v * (value @ A @ value)
        residual = prediction - query
        loss = float(np.linalg.norm(residual))
        gradient = U.T @ residual + 2.0 * (A @ value) * (v @ residual)
        gradient = gradient / loss if loss > 1e-15 else np.zeros(d)
        return prediction, loss, gradient

    converged = False
    iterations = 0
    for iteration in range(180):
        _, loss, gradient = evaluate(tau)
        norm_squared = float(gradient @ gradient)
        if norm_squared <= 1e-16:
            converged = True
            break
        step = 0.2
        for _ in range(20):
            candidate = tau - step * gradient
            _, candidate_loss, _ = evaluate(candidate)
            if np.isfinite(candidate_loss) and candidate_loss <= loss - 1e-4 * step * norm_squared:
                tau = candidate
                iterations = iteration + 1
                break
            step *= 0.5
        else:
            break
    prediction, _, gradient = evaluate(tau)
    converged = converged or np.linalg.norm(gradient) <= 1e-8
    return mean + scale * prediction, dict(iterations=iterations, converged=bool(converged))


def affine_coordinates(X, y, d):
    c = X.mean(axis=1)
    centered = X - c[:, None]
    Z = np.column_stack((centered, y - c))
    U, singular, _ = svd(Z, full_matrices=False)
    cutoff = max(Z.shape) * np.finfo(float).eps * singular[0]
    rank = int(np.sum(singular > cutoff))
    if rank <= d:
        raise ValueError("The affine span must exceed the tangent dimension.")
    basis = U[:, :rank]
    return c, basis, basis.T @ centered, basis.T @ (y - c)


def bandwidth(data, x, k):
    distances = np.sum((data - x[:, None]) ** 2, axis=0)
    order = np.argsort(distances)
    h = float(np.sqrt(np.max(distances[order[1:k + 1]])))
    if not np.isfinite(h) or h <= 0:
        raise ValueError("The kernel bandwidth must be positive.")
    return h, order, distances


def tangent_basis(columns, d):
    U, singular, _ = svd(columns, full_matrices=False)
    if len(singular) < d or singular[d - 1] <= 1e-12 * max(1.0, singular[0]):
        raise ValueError("The local tangent is rank deficient.")
    return U[:, :d]


def mfit_projection(data, query, d, k):
    pair_distances = np.sum((data[:, :, None] - data[:, None, :]) ** 2, axis=0)
    cache = {}
    completions = 0
    x = query.copy()
    converged = False
    for iteration in range(49):
        h, _, distances = bandwidth(data, x, k)
        positive = np.maximum(1.0 - distances / h ** 2, 0.0)
        positive[positive <= 64 * np.finfo(float).eps] = 0.0
        weights = positive ** 0.1
        if weights.sum() <= 0:
            raise ValueError("MFIT has no positive kernel weights.")
        weights /= weights.sum()
        C = np.zeros((data.shape[0], data.shape[0]))
        direction = np.zeros(data.shape[0])
        for i in np.flatnonzero(positive > 0):
            ids = np.flatnonzero((pair_distances[:, i] > 0) & (pair_distances[:, i] <= h ** 2))
            key = (int(i), ids.tobytes())
            if key not in cache:
                support = ids.copy()
                try:
                    tangent = tangent_basis(data[:, support] - data[:, i, None], d)
                except ValueError:
                    for j in np.argsort(pair_distances[:, i]):
                        if pair_distances[j, i] <= 0 or j in support:
                            continue
                        support = np.append(support, j)
                        try:
                            tangent = tangent_basis(data[:, support] - data[:, i, None], d)
                            break
                        except ValueError:
                            continue
                    else:
                        raise ValueError("MFIT cannot identify a rank-d local tangent.")
                cache[key] = tangent, len(support) > len(ids)
            tangent, completed = cache[key]
            completions += int(completed)
            C += weights[i] * (tangent @ tangent.T)
            delta = data[:, i] - x
            direction += weights[i] * (delta - tangent @ (tangent.T @ delta))
        tangent = eigh(C)[1][:, -d:]
        candidate = x + 0.25 * (direction - tangent @ (tangent.T @ direction))
        movement = float(np.linalg.norm(candidate - x))
        x = candidate
        if movement <= 1e-7:
            converged = True
            break
    return x, dict(iterations=iteration + 1, converged=converged,
                   movement=movement, support_completions=completions)


def logkde_projection(data, query, d, k):
    x = query.copy()
    converged = False
    for iteration in range(49):
        h, order, distances = bandwidth(data, x, k)
        ids = order[:k]
        weights = np.exp(-distances[ids] / h ** 2)
        mean = data[:, ids] @ weights / weights.sum()
        centered = data - mean[:, None]
        weights_fit = np.exp(-np.sum(centered ** 2, axis=0) / h ** 2)
        tangent = tangent_basis(centered * np.sqrt(weights_fit)[None, :], d)
        delta = mean - x
        candidate = x + delta - tangent @ (tangent.T @ delta)
        movement = float(np.linalg.norm(candidate - x))
        x = candidate
        if movement <= 1e-7:
            converged = True
            break
    return x, dict(iterations=iteration + 1, converged=converged, movement=movement)


def project_baseline(method, X, y, d=3):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.size or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Expected finite X(D, n) and y(D).")
    if d < 1 or min(X.shape[0], X.shape[1] - 1) < d + 1:
        raise ValueError("Require d >= 1, D >= d+1, and n >= d+2.")
    if method == "LPCA":
        prediction, diagnostic = pca_projection(X, y, d)
    elif method == "SPH":
        prediction, diagnostic = sphere_projection(X, y, d)
    elif method == "Q-LS":
        prediction, diagnostic = quadratic_projection(X, y, d)
    elif method in ("MFIT", "LOG-KDE"):
        c, basis, data, query = affine_coordinates(X, y, d)
        solver = mfit_projection if method == "MFIT" else logkde_projection
        result, diagnostic = solver(data, query, d, X.shape[1] - 1)
        prediction = c + basis @ result
        diagnostic["affine_rank"] = basis.shape[1]
    else:
        raise ValueError(f"Unknown baseline: {method}")
    if prediction.shape != y.shape or not np.isfinite(prediction).all():
        raise ValueError(f"{method} produced a nonfinite reconstruction.")
    return prediction, diagnostic
