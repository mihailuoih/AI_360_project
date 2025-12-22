import math


class BasePool:
    """Базовый интерфейс пула без комиссий."""

    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)

    def spot_price(self) -> float:
        raise NotImplementedError

    def quote_out(self, dx: float) -> float:
        """Пользователь отдаёт dx базового актива, получает dy кэша."""
        raise NotImplementedError

    def quote_in(self, dy: float) -> float:
        """Пользователь отдаёт dy кэша, получает dx базового актива."""
        raise NotImplementedError

    def apply_out(self, dx: float) -> float:
        dy = self.quote_out(dx)
        if dy <= 0:
            return 0.0
        self.x += dx
        self.y -= dy
        return dy

    def apply_in(self, dy: float) -> float:
        dx = self.quote_in(dy)
        if dx <= 0:
            return 0.0
        self.x -= dx
        self.y += dy
        return dx

    def curve_points(self, n: int = 50) -> tuple:
        """Возвращает массивы (x_points, y_points) для отрисовки теоретической кривой."""
        raise NotImplementedError


class ConstantProductPool(BasePool):
    """Классический AMM x*y = k без комиссий."""

    def spot_price(self) -> float:
        return self.y / self.x if self.x > 0 else float("inf")

    def quote_out(self, dx: float) -> float:
        if dx <= 0 or self.x <= 0 or self.y <= 0:
            return 0.0
        k = self.x * self.y
        x_new = self.x + dx
        y_new = k / x_new
        dy = self.y - y_new
        return max(dy, 0.0)

    def quote_in(self, dy: float) -> float:
        if dy <= 0 or self.x <= 0 or self.y <= 0:
            return 0.0
        k = self.x * self.y
        y_new = self.y + dy
        x_new = k / y_new
        dx = self.x - x_new
        return max(dx, 0.0)

    def curve_points(self, n: int = 50) -> tuple:
        if self.x <= 0 or self.y <= 0:
            return [], []
        k = self.x * self.y
        xs = []
        ys = []
        x_max = self.x * 2
        step = x_max / n
        for i in range(1, n + 1):
            x_val = i * step
            y_val = k / x_val
            xs.append(x_val)
            ys.append(y_val)
        return xs, ys


class HFPool(BasePool):
    """
    HFMM (Curve CryptoInvariant) для двух активов без комиссии.
    Использует инвариант F(x,D)=0 с параметрами A и gamma.
    """

    def __init__(self, x: float, y: float, A: float = 100.0, gamma: float = 1e-3):
        super().__init__(x, y)
        self.A = A
        self.gamma = gamma

    def _K0(self, x: float, y: float, D: float) -> float:
        if D == 0:
            return 0.0
        return (x * y * (2 ** 2)) / (D ** 2)

    def _K(self, x: float, y: float, D: float) -> float:
        K0 = self._K0(x, y, D)
        denom = (self.gamma + 1 - K0)
        if denom == 0:
            return 0.0
        return self.A * K0 * (self.gamma ** 2) / (denom ** 2)

    def _F(self, x: float, y: float, D: float) -> float:
        K = self._K(x, y, D)
        return K * D * (x + y) + x * y - K * (D ** 2) - (D / 2) ** 2

    def _solve_D(self, x: float, y: float) -> float:
        """
        Решение F(x, y, D) = 0 по D бинарным поиском.
        """
        lo = max(x, y, 1e-12)
        hi = max(x + y, 1.0)
        f_lo = self._F(x, y, lo)
        f_hi = self._F(x, y, hi)

        expand = 0
        while f_lo * f_hi > 0 and expand < 100:
            hi *= 2
            f_hi = self._F(x, y, hi)
            expand += 1

        if f_lo * f_hi > 0:
            return max(hi, 1e-12)

        for _ in range(80):
            mid = 0.5 * (lo + hi)
            f_mid = self._F(x, y, mid)
            if abs(f_mid) < 1e-12:
                return max(mid, 1e-12)
            if f_lo * f_mid < 0:
                hi = mid
                f_hi = f_mid
            else:
                lo = mid
                f_lo = f_mid
        return max(0.5 * (lo + hi), 1e-12)

    def _solve_y(self, x_new: float, D: float, y_init: float) -> float:
        """
        Решение F(x_new, y, D) = 0 по y бинарным поиском.
        """
        lo = 1e-12
        hi = max(y_init * 2, 1.0)
        f_lo = self._F(x_new, lo, D)
        f_hi = self._F(x_new, hi, D)

        expand = 0
        while f_lo * f_hi > 0 and expand < 100:
            hi *= 2
            f_hi = self._F(x_new, hi, D)
            expand += 1

        if f_lo * f_hi > 0:
            return max(y_init, 1e-12)

        for _ in range(80):
            mid = 0.5 * (lo + hi)
            f_mid = self._F(x_new, mid, D)
            if abs(f_mid) < 1e-12:
                return max(mid, 1e-12)
            if f_lo * f_mid < 0:
                hi = mid
                f_hi = f_mid
            else:
                lo = mid
                f_lo = f_mid
        return max(0.5 * (lo + hi), 1e-12)

    def _solve_x(self, y_new: float, D: float, x_init: float) -> float:
        """
        Решение F(x, y_new, D) = 0 по x бинарным поиском.
        """
        lo = 1e-12
        hi = max(x_init * 2, 1.0)
        f_lo = self._F(lo, y_new, D)
        f_hi = self._F(hi, y_new, D)

        expand = 0
        while f_lo * f_hi > 0 and expand < 100:
            hi *= 2
            f_hi = self._F(hi, y_new, D)
            expand += 1

        if f_lo * f_hi > 0:
            return max(x_init, 1e-12)

        for _ in range(80):
            mid = 0.5 * (lo + hi)
            f_mid = self._F(mid, y_new, D)
            if abs(f_mid) < 1e-12:
                return max(mid, 1e-12)
            if f_lo * f_mid < 0:
                hi = mid
                f_hi = f_mid
            else:
                lo = mid
                f_lo = f_mid
        return max(0.5 * (lo + hi), 1e-12)

    def spot_price(self) -> float:
        dx = min(self.x * 1e-6, 1.0)
        if dx <= 0:
            return float("inf")
        dy = self.quote_out(dx)
        return dy / dx if dx > 0 else float("inf")

    def quote_out(self, dx: float) -> float:
        if dx <= 0:
            return 0.0
        x, y = float(self.x), float(self.y)
        D = self._solve_D(x, y)
        x_new = x + dx
        y_new = self._solve_y(x_new, D, y)
        dy = y - y_new
        if dy <= 0 or dy >= y:
            return 0.0
        return dy

    def quote_in(self, dy: float) -> float:
        if dy <= 0:
            return 0.0
        x, y = float(self.x), float(self.y)
        D = self._solve_D(x, y)
        y_new = y + dy
        x_new = self._solve_x(y_new, D, x)
        dx = x - x_new
        if dx <= 0 or dx >= x:
            return 0.0
        return dx

    def curve_points(self, n: int = 50) -> tuple:
        if self.x <= 0 or self.y <= 0:
            return [], []
        D = self._solve_D(self.x, self.y)
        xs = []
        ys = []
        x_max = self.x * 2
        step = x_max / n
        for i in range(1, n + 1):
            x_val = i * step
            y_val = self._solve_y(x_val, D, self.y)
            xs.append(x_val)
            ys.append(max(y_val, 0.0))
        return xs, ys
