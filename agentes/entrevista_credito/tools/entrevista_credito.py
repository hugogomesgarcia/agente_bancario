from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
import re

from google.adk.tools import ToolContext

from agentes.compartilhado.dados_csv import (
    DadosCsvIndisponiveis,
    bloquear_csv,
    ler_csv,
    preparar_csv,
    substituir_csv,
)


ROOT = Path(__file__).resolve().parents[3]
CLIENTES_CSV = ROOT / "csv" / "local" / "clientes.csv"
COLUNA_CPF = "CPF"
COLUNA_SCORE = "Score"
DADOS_ENTREVISTA = "dados_entrevista_credito"
RESULTADO_ENTREVISTA = "resultado_entrevista_credito"
RETORNO_ENTREVISTA = "retorno_entrevista_credito"

PESOS_EMPREGO = {
    "formal": Decimal("300"),
    "autonomo": Decimal("200"),
    "desempregado": Decimal("0"),
}
PESOS_DEPENDENTES = {
    0: Decimal("100"),
    1: Decimal("80"),
    2: Decimal("60"),
}
PESOS_DIVIDAS = {True: Decimal("-100"), False: Decimal("100")}


def _normalizar_cpf(cpf: object) -> str:
    return re.sub(r"\D", "", str(cpf))


def normalizar_numero_dependentes(valor: object) -> int | None:
    try:
        numero = Decimal(str(valor))
    except (InvalidOperation, TypeError, ValueError):
        return None
    if (
        isinstance(valor, bool)
        or not numero.is_finite()
        or numero != numero.to_integral_value()
        or not 0 <= numero <= 1000
    ):
        return None
    return int(numero)


def calcular_score_credito(
    renda_mensal: object,
    tipo_emprego: str,
    despesas_fixas: object,
    numero_dependentes: object,
    tem_dividas: object,
) -> int:
    try:
        renda = Decimal(str(renda_mensal))
        despesas = Decimal(str(despesas_fixas))
    except (InvalidOperation, TypeError, ValueError) as erro:
        raise ValueError("Dados financeiros inválidos.") from erro

    dependentes = normalizar_numero_dependentes(numero_dependentes)
    if (
        not renda.is_finite()
        or not despesas.is_finite()
        or renda < 0
        or despesas < 0
        or dependentes is None
        or tipo_emprego not in PESOS_EMPREGO
        or not isinstance(tem_dividas, bool)
    ):
        raise ValueError("Dados financeiros inválidos.")

    peso_dependentes = PESOS_DEPENDENTES.get(dependentes, Decimal("30"))
    score = (
        (renda / (despesas + 1)) * Decimal("30")
        + PESOS_EMPREGO[tipo_emprego]
        + peso_dependentes
        + PESOS_DIVIDAS[tem_dividas]
    )
    arredondado = int(score.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return min(1000, max(0, arredondado))


def concluir_entrevista_credito(tool_context: ToolContext) -> dict:
    """Calcula e persiste o score do cliente autenticado após a entrevista."""
    if tool_context.state.get("cliente_autenticado") is not True:
        resultado = {"sucesso": False, "erro": "cliente_nao_autenticado"}
        tool_context.state[RESULTADO_ENTREVISTA] = resultado
        return resultado

    cpf = _normalizar_cpf(tool_context.state.get("cpf_cliente", ""))
    dados = tool_context.state.get(DADOS_ENTREVISTA)
    if len(cpf) != 11 or not isinstance(dados, dict):
        resultado = {"sucesso": False, "erro": "dados_incompletos"}
        tool_context.state[RESULTADO_ENTREVISTA] = resultado
        return resultado

    campos_necessarios = {
        "renda_mensal",
        "tipo_emprego",
        "despesas_fixas",
        "numero_dependentes",
        "tem_dividas",
    }
    if not campos_necessarios.issubset(dados):
        resultado = {"sucesso": False, "erro": "dados_incompletos"}
        tool_context.state[RESULTADO_ENTREVISTA] = resultado
        return resultado

    try:
        score = calcular_score_credito(
            **{chave: dados[chave] for chave in campos_necessarios}
        )
    except ValueError:
        resultado = {"sucesso": False, "erro": "dados_invalidos"}
        tool_context.state[RESULTADO_ENTREVISTA] = resultado
        return resultado

    invocation_id = getattr(tool_context, "invocation_id", None)
    anterior = tool_context.state.get(RESULTADO_ENTREVISTA)
    if (
        invocation_id
        and isinstance(anterior, dict)
        and anterior.get("sucesso") is True
        and anterior.get("invocation_id") == invocation_id
        and anterior.get("score_atualizado") == score
    ):
        return anterior

    temporario = None
    try:
        with bloquear_csv(CLIENTES_CSV.parent):
            if (CLIENTES_CSV.parent / ".transacao_credito.json").exists():
                raise DadosCsvIndisponiveis
            campos, clientes = ler_csv(
                CLIENTES_CSV, {COLUNA_CPF, COLUNA_SCORE}
            )
            indices = [
                indice
                for indice, cliente in enumerate(clientes)
                if _normalizar_cpf(cliente.get(COLUNA_CPF, "")) == cpf
            ]
            if len(indices) != 1:
                resultado = {
                    "sucesso": False,
                    "erro": "perfil_credito_indisponivel",
                }
                tool_context.state[RESULTADO_ENTREVISTA] = resultado
                return resultado
            clientes_atualizados = [dict(cliente) for cliente in clientes]
            clientes_atualizados[indices[0]][COLUNA_SCORE] = str(score)
            temporario = preparar_csv(
                CLIENTES_CSV, campos, clientes_atualizados
            )
            substituir_csv(temporario, CLIENTES_CSV)
    except DadosCsvIndisponiveis:
        resultado = {"sucesso": False, "erro": "base_clientes_indisponivel"}
        tool_context.state[RESULTADO_ENTREVISTA] = resultado
        return resultado
    finally:
        if temporario is not None:
            temporario.unlink(missing_ok=True)

    resultado = {
        "sucesso": True,
        "score_atualizado": score,
        "invocation_id": invocation_id,
    }
    tool_context.state[RESULTADO_ENTREVISTA] = resultado
    tool_context.state[RETORNO_ENTREVISTA] = {"score_atualizado": score}
    return resultado
