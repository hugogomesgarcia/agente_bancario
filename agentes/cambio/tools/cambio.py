import json
import logging
import os
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, localcontext
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from google.adk.tools import ToolContext


logger = logging.getLogger(__name__)

ETAPA_CAMBIO = "etapa_cambio"
AGUARDANDO_ESCLARECIMENTO = "aguardando_esclarecimento_cambio"
AGUARDANDO_PROXIMA_ACAO = "aguardando_proxima_acao_cambio"
RESULTADO_CAMBIO = "resultado_cambio"
INTERPRETACAO_PARCIAL = "interpretacao_parcial_cambio"
TENTATIVAS_INTERPRETACAO = "tentativas_interpretacao_cambio"

URL_COTACAO = "https://economia.awesomeapi.com.br/json/last/{par}"
TIMEOUT_SEGUNDOS = 5


def _codigo_ativo(valor: str | None) -> str | None:
    if not isinstance(valor, str):
        return None
    codigo = valor.strip().upper()
    return codigo if re.fullmatch(r"[A-Z0-9]{2,10}", codigo) else None


def _decimal_positivo(valor) -> Decimal | None:
    try:
        numero = Decimal(str(valor))
    except (InvalidOperation, TypeError, ValueError):
        return None
    if not numero.is_finite() or numero <= 0:
        return None
    return numero


def _decimal_texto(valor: Decimal) -> str:
    texto = format(valor, "f")
    return texto.rstrip("0").rstrip(".") if "." in texto else texto


def _data_utc(timestamp) -> str | None:
    try:
        instante = int(str(timestamp))
        if instante > 10_000_000_000:
            instante //= 1000
        return datetime.fromtimestamp(instante, timezone.utc).strftime(
            "%d/%m/%Y às %H:%M UTC"
        )
    except (ValueError, TypeError, OverflowError, OSError):
        return None


def _erro(codigo: str) -> dict:
    return {"sucesso": False, "erro": codigo}


def _buscar_par(ativo_base: str, ativo_destino: str) -> dict:
    token = os.getenv("AWESOMEAPI_TOKEN", "").strip()
    if not token:
        return _erro("token_nao_configurado")

    par = f"{ativo_base}-{ativo_destino}"
    requisicao = Request(
        URL_COTACAO.format(par=par),
        headers={
            "Accept": "application/json",
            "User-Agent": "Banco-Agil/1.0",
            "x-api-key": token,
        },
    )
    try:
        with urlopen(requisicao, timeout=TIMEOUT_SEGUNDOS) as resposta:
            dados = json.loads(resposta.read().decode("utf-8"))
    except HTTPError as erro_http:
        codigo_http = erro_http.code
        erro_http.close()
        if codigo_http == 404:
            return _erro("par_nao_disponivel")
        if codigo_http in {401, 403}:
            return _erro("token_invalido")
        if codigo_http == 429:
            return _erro("limite_api_excedido")
        logger.warning("AwesomeAPI respondeu com HTTP %s", codigo_http)
        return _erro("api_indisponivel")
    except (URLError, TimeoutError, OSError):
        logger.warning("Não foi possível acessar a AwesomeAPI")
        return _erro("api_indisponivel")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _erro("resposta_api_invalida")

    if not isinstance(dados, dict):
        return _erro("resposta_api_invalida")
    cotacao = dados.get(f"{ativo_base}{ativo_destino}")
    if not isinstance(cotacao, dict):
        return _erro("resposta_api_invalida")
    if cotacao.get("code") != ativo_base or cotacao.get("codein") != ativo_destino:
        return _erro("resposta_api_invalida")

    compra = _decimal_positivo(cotacao.get("bid"))
    venda = _decimal_positivo(cotacao.get("ask"))
    consultado_em = _data_utc(cotacao.get("timestamp"))
    nome = cotacao.get("name")
    if (
        compra is None
        or venda is None
        or consultado_em is None
        or not isinstance(nome, str)
        or not nome.strip()
    ):
        return _erro("resposta_api_invalida")

    return {
        "sucesso": True,
        "ativo_base": ativo_base,
        "ativo_destino": ativo_destino,
        "nome_par": nome.strip(),
        "compra": _decimal_texto(compra),
        "venda": _decimal_texto(venda),
        "consultado_em": consultado_em,
        "fonte": "AwesomeAPI",
        "derivada_par_inverso": False,
        "derivada_par_cruzado": False,
    }


def _buscar_par_ou_inverso(ativo_base: str, ativo_destino: str) -> dict:
    resultado = _buscar_par(ativo_base, ativo_destino)
    if resultado.get("erro") != "par_nao_disponivel":
        return resultado

    inversa = _buscar_par(ativo_destino, ativo_base)
    if not inversa.get("sucesso"):
        if inversa.get("erro") == "par_nao_disponivel":
            return resultado
        return inversa

    compra_inversa = Decimal(inversa["compra"])
    venda_inversa = Decimal(inversa["venda"])
    with localcontext() as contexto:
        contexto.prec = 16
        compra = Decimal(1) / venda_inversa
        venda = Decimal(1) / compra_inversa
    return {
        "sucesso": True,
        "ativo_base": ativo_base,
        "ativo_destino": ativo_destino,
        "nome_par": f"{ativo_base}/{ativo_destino}",
        "compra": _decimal_texto(compra),
        "venda": _decimal_texto(venda),
        "consultado_em": inversa["consultado_em"],
        "fonte": "AwesomeAPI",
        "derivada_par_inverso": True,
        "derivada_par_cruzado": False,
    }


def _consultar_cotacao(ativo_base: str, ativo_destino: str) -> dict:
    resultado = _buscar_par_ou_inverso(ativo_base, ativo_destino)
    if resultado.get("sucesso") or resultado.get("erro") != "par_nao_disponivel":
        return resultado
    if "BRL" in {ativo_base, ativo_destino}:
        resultado.update(
            {"ativo_base": ativo_base, "ativo_destino": ativo_destino}
        )
        return resultado

    base_brl = _buscar_par_ou_inverso(ativo_base, "BRL")
    destino_brl = _buscar_par_ou_inverso(ativo_destino, "BRL")
    for referencia in (base_brl, destino_brl):
        if not referencia.get("sucesso"):
            if referencia.get("erro") != "par_nao_disponivel":
                return referencia
            resultado.update(
                {"ativo_base": ativo_base, "ativo_destino": ativo_destino}
            )
            return resultado

    with localcontext() as contexto:
        contexto.prec = 16
        compra = Decimal(base_brl["compra"]) / Decimal(destino_brl["venda"])
        venda = Decimal(base_brl["venda"]) / Decimal(destino_brl["compra"])
    horarios = [base_brl["consultado_em"], destino_brl["consultado_em"]]
    consultado_em = horarios[0] if horarios[0] == horarios[1] else " e ".join(horarios)
    return {
        "sucesso": True,
        "ativo_base": ativo_base,
        "ativo_destino": ativo_destino,
        "nome_par": f"{ativo_base}/{ativo_destino}",
        "compra": _decimal_texto(compra),
        "venda": _decimal_texto(venda),
        "consultado_em": consultado_em,
        "fonte": "AwesomeAPI",
        "derivada_par_inverso": False,
        "derivada_par_cruzado": True,
        "pares_referencia": [f"{ativo_base}/BRL", f"{ativo_destino}/BRL"],
    }


def processar_solicitacao_cambio(
    status: str,
    destino_explicito: bool,
    tool_context: ToolContext,
    ativo_base: str | None = None,
    ativo_destino: str | None = None,
    quantidade_base: float | None = None,
    evidencia_base: str | None = None,
    evidencia_destino: str | None = None,
    evidencia_quantidade: str | None = None,
    pergunta_esclarecimento: str | None = None,
) -> dict:
    """Interpreta toda a fala em um par ou registra uma pergunta de esclarecimento.

    Use status "resolvido" somente quando o ativo base estiver claro. Use status
    "precisa_esclarecimento" se faltar informação ou houver ambiguidade. A
    moeda de destino é BRL apenas quando nenhuma foi expressa pelo cliente.
    """
    if status == "precisa_esclarecimento":
        pergunta = (
            pergunta_esclarecimento.strip()
            if isinstance(pergunta_esclarecimento, str)
            else ""
        )
        if not pergunta:
            resultado = _erro("pergunta_esclarecimento_ausente")
        else:
            parcial = {
                "ativo_base": _codigo_ativo(ativo_base),
                "ativo_destino": _codigo_ativo(ativo_destino),
                "destino_explicito": destino_explicito,
                "quantidade_base": quantidade_base,
            }
            tool_context.state[INTERPRETACAO_PARCIAL] = parcial
            resultado = {
                "sucesso": True,
                "tipo": "esclarecimento",
                "pergunta": pergunta,
                "interpretacao_parcial": parcial,
            }
        tool_context.state[RESULTADO_CAMBIO] = resultado
        return resultado

    if status != "resolvido":
        resultado = _erro("status_invalido")
        tool_context.state[RESULTADO_CAMBIO] = resultado
        return resultado

    base = _codigo_ativo(ativo_base)
    destino = _codigo_ativo(ativo_destino)
    evidencia_base_valida = isinstance(evidencia_base, str) and bool(
        evidencia_base.strip()
    )
    evidencia_destino_valida = isinstance(evidencia_destino, str) and bool(
        evidencia_destino.strip()
    )
    quantidade = (
        _decimal_positivo(quantidade_base)
        if quantidade_base is not None
        else None
    )
    evidencia_quantidade_valida = isinstance(
        evidencia_quantidade, str
    ) and bool(evidencia_quantidade.strip())
    if base is None or not evidencia_base_valida:
        resultado = _erro("ativo_base_invalido")
    elif destino_explicito and (destino is None or not evidencia_destino_valida):
        resultado = _erro("ativo_destino_explicito_invalido")
    elif not destino_explicito and (
        destino not in {None, "BRL"} or evidencia_destino_valida
    ):
        resultado = _erro("destino_padrao_inconsistente")
    elif (quantidade_base is None) != (evidencia_quantidade is None):
        resultado = _erro("quantidade_invalida")
    elif quantidade_base is not None and (
        quantidade is None or not evidencia_quantidade_valida
    ):
        resultado = _erro("quantidade_invalida")
    else:
        destino = destino if destino_explicito else "BRL"
        if base == destino:
            resultado = _erro("ativos_iguais")
        else:
            resultado = _consultar_cotacao(base, destino)
            resultado["evidencia_base"] = evidencia_base.strip()
            resultado["evidencia_destino"] = (
                evidencia_destino.strip() if evidencia_destino_valida else None
            )
            resultado["destino_explicito"] = destino_explicito
            if resultado.get("sucesso") and quantidade is not None:
                with localcontext() as contexto:
                    contexto.prec = 24
                    total_compra = quantidade * Decimal(resultado["compra"])
                    total_venda = quantidade * Decimal(resultado["venda"])
                resultado.update(
                    {
                        "quantidade_base": _decimal_texto(quantidade),
                        "evidencia_quantidade": evidencia_quantidade.strip(),
                        "total_compra": _decimal_texto(total_compra),
                        "total_venda": _decimal_texto(total_venda),
                    }
                )

    tool_context.state[RESULTADO_CAMBIO] = resultado
    if resultado.get("sucesso"):
        tool_context.state[INTERPRETACAO_PARCIAL] = None
    return resultado
