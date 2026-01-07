# =================================  TESTY  ===================================
# Testy do tego pliku obejmują jedynie weryfikację poprawności wyników dla
# prawidłowych danych wejściowych - obsługa niepoprawych danych wejściowych
# nie jest ani wymagana ani sprawdzana. W razie potrzeby lub chęci można ją 
# wykonać w dowolny sposób we własnym zakresie.
# =============================================================================
import numpy as np

def chebyshev_nodes(n: int = 10) -> np.ndarray | None:
    if not isinstance(n, int) or n<=0:
        return None
    k = np.arange(n)
    if n == 1:
        return np.array([0.0])
    x = np.cos(k*np.pi/(n-1))
    return x
print (chebyshev_nodes(3))


def bar_cheb_weights(n: int = 10) -> np.ndarray | None:
    """Funkcja tworząca wektor wag dla węzłów Czebyszewa wymiaru (n,).

    Args:
        n (int): Liczba wag węzłów Czebyszewa.
    
    Returns:
        (np.ndarray): Wektor wag dla węzłów Czebyszewa (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    j = np.arange(n)
    sigj = np.ones(n)
    sigj[0] = 1/2
    sigj[-1] = 1/2
    return ((-1)**j)*sigj
print(bar_cheb_weights(3))



def barycentric_inte(
    xi: np.ndarray, yi: np.ndarray, wi: np.ndarray, x: np.ndarray
) -> np.ndarray | None:
    """Funkcja przeprowadza interpolację metodą barycentryczną dla zadanych 
    węzłów xi i wartości funkcji interpolowanej yi używając wag wi. Zwraca 
    wyliczone wartości funkcji interpolującej dla argumentów x w postaci 
    wektora (n,).

    Args:
        xi (np.ndarray): Wektor węzłów interpolacji (m,).
        yi (np.ndarray): Wektor wartości funkcji interpolowanej w węzłach (m,).
        wi (np.ndarray): Wektor wag interpolacji (m,).
        x (np.ndarray): Wektor argumentów dla funkcji interpolującej (n,).
    
    Returns:
        (np.ndarray): Wektor wartości funkcji interpolującej (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not all(isinstance(arg, np.ndarray) for arg in [xi, yi, wi, x]):
        return None
    if xi.shape != yi.shape or xi.shape != wi.shape:
        return None
    
    diff = x[:, None] - xi
    eps = 1e-15
    mask = np.abs(diff) < eps
    diff[mask] = 1.0
    kernels = wi / diff
    
    numerator = np.sum(kernels * yi, axis=1)
    denominator = np.sum(kernels, axis=1)
    result = numerator / denominator

    x_indices, node_indices = np.where(mask)
    result[x_indices] = yi[node_indices]

    return result



def L_inf(
    xr: int | float | list | np.ndarray, x: int | float | list | np.ndarray
) -> float | None:
    """Funkcja obliczająca normę L-nieskończoność. Powinna działać zarówno na 
    wartościach skalarnych, listach, jak i wektorach biblioteki numpy.

    Args:
        xr (int | float | list | np.ndarray): Wartość dokładna w postaci 
            skalara, listy lub wektora (n,).
        x (int | float | list | np.ndarray): Wartość przybliżona w postaci 
            skalara, listy lub wektora (n,).

    Returns:
        (float): Wartość normy L-nieskończoność.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """

    xr_arr = np.asarray(xr)
    x_arr = np.asarray(x)

    if xr_arr.shape != x_arr.shape:
        return None

    return float(np.max(np.abs(xr_arr - x_arr)))
