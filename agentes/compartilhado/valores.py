from decimal import Decimal, InvalidOperation
import re


VALOR_MAXIMO = Decimal("999999999999.99")


def normalizar_valor_monetario(valor: object) -> Decimal | None:
    multiplicador = Decimal("1")
    if isinstance(valor, bool):
        return None
    if isinstance(valor, (int, float, Decimal)):
        texto = str(valor)
    elif isinstance(valor, str):
        texto = valor.strip().lower().replace("r$", "")
        texto = re.sub(r"\b(?:real|reais)\b", "", texto).strip()
        correspondencia_mil = re.fullmatch(r"(.+?)\s*(?:mil|k)", texto)
        if correspondencia_mil:
            texto = correspondencia_mil.group(1)
            multiplicador = Decimal("1000")
        texto = texto.replace(" ", "")
    else:
        return None

    if not texto:
        return None
    if "," in texto and "." in texto:
        if texto.rfind(",") > texto.rfind("."):
            texto = texto.replace(".", "").replace(",", ".")
        else:
            texto = texto.replace(",", "")
    elif "," in texto:
        texto = texto.replace(",", ".")
    elif re.fullmatch(r"[+-]?\d{1,3}(?:\.\d{3})+", texto):
        texto = texto.replace(".", "")

    if not re.fullmatch(r"[+-]?\d+(?:\.\d{1,2})?", texto):
        return None

    try:
        resultado = Decimal(texto) * multiplicador
    except InvalidOperation:
        return None
    if not resultado.is_finite() or abs(resultado) > VALOR_MAXIMO:
        return None
    try:
        return resultado.quantize(Decimal("0.01"))
    except InvalidOperation:
        return None
