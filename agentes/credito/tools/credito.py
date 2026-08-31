from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
import logging
import os
from pathlib import Path
import re
import tempfile

from google.adk.tools import ToolContext

from agentes.compartilhado.dados_csv import (
    DadosCsvIndisponiveis,
    bloquear_csv,
    ler_csv,
    preparar_csv,
    substituir_csv,
)
from agentes.compartilhado.valores import normalizar_valor_monetario


logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
CLIENTES_CSV = ROOT / "csv" / "local" / "clientes.csv"
SCORE_LIMITE_CSV = ROOT / "csv" / "local" / "score_limite.csv"
SOLICITACOES_CSV = (
    ROOT / "csv" / "local" / "solicitacoes_aumento_limite.csv"
)

COLUNA_CPF = "CPF"
COLUNA_SCORE = "Score"
COLUNA_LIMITE = "Limite de Crédito"
COLUNAS_SOLICITACAO = [
    "cpf_cliente",
    "data_hora_solicitacao",
    "limite_atual",
    "novo_limite_solicitado",
    "status_pedido",
]
RESULTADO_CREDITO = "resultado_credito"
ETAPA_CREDITO = "etapa_credito"
AGUARDANDO_PROXIMA_ACAO = "aguardando_proxima_acao"
AGUARDANDO_DECISAO_ENTREVISTA = "aguardando_decisao_entrevista"
AGUARDANDO_NOVO_LIMITE = "aguardando_novo_limite"
PENDENCIA_REANALISE = "pendencia_reanalise_credito"

BaseCreditoIndisponivel = DadosCsvIndisponiveis


class PerfilCreditoIndisponivel(BaseCreditoIndisponivel):
    pass


def _bloquear_csv(diretorio: Path | None = None):
    return bloquear_csv(diretorio or CLIENTES_CSV.parent)


def _normalizar_cpf(cpf: object) -> str:
    return re.sub(r"\D", "", str(cpf))


def _formatar_decimal(valor: Decimal) -> str:
    return f"{valor.quantize(Decimal('0.01')):.2f}"


def _ler_csv(
    caminho: Path, colunas_obrigatorias: set[str]
) -> tuple[list[str], list[dict[str, str]]]:
    return ler_csv(caminho, colunas_obrigatorias)


def _preparar_csv(
    caminho: Path, campos: list[str], linhas: list[dict[str, object]]
) -> Path:
    return preparar_csv(caminho, campos, linhas)


def _substituir_csv(temporario: Path, destino: Path) -> None:
    substituir_csv(temporario, destino)


def _caminho_transacao() -> Path:
    return CLIENTES_CSV.parent / ".transacao_credito.json"


def _escrever_transacao(dados: dict) -> None:
    caminho = _caminho_transacao()
    temporario = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=caminho.parent,
            prefix=f".{caminho.name}.",
            suffix=".tmp",
            delete=False,
        ) as arquivo:
            temporario = Path(arquivo.name)
            json.dump(dados, arquivo, ensure_ascii=False)
            arquivo.flush()
            os.fsync(arquivo.fileno())
        os.replace(temporario, caminho)
    except (OSError, TypeError, ValueError) as erro:
        if temporario is not None:
            temporario.unlink(missing_ok=True)
        raise BaseCreditoIndisponivel from erro


def _restaurar_transacao_sem_lock() -> None:
    caminho = _caminho_transacao()
    if not caminho.exists():
        return
    temporarios: list[Path] = []
    try:
        with caminho.open("r", encoding="utf-8") as arquivo:
            dados = json.load(arquivo)
        campos_clientes = dados["campos_clientes"]
        clientes = dados["clientes"]
        campos_solicitacoes = dados["campos_solicitacoes"]
        solicitacoes = dados["solicitacoes"]
        campos_validos = all(
            isinstance(campos, list)
            and campos
            and all(isinstance(campo, str) for campo in campos)
            for campos in (campos_clientes, campos_solicitacoes)
        )
        linhas_validas = all(
            isinstance(linhas, list)
            and all(
                isinstance(linha, dict)
                and all(isinstance(chave, str) for chave in linha)
                for linha in linhas
            )
            for linhas in (clientes, solicitacoes)
        )
        if not campos_validos or not linhas_validas:
            raise ValueError

        temporario_clientes = _preparar_csv(
            CLIENTES_CSV, campos_clientes, clientes
        )
        temporarios.append(temporario_clientes)
        temporario_solicitacoes = _preparar_csv(
            SOLICITACOES_CSV, campos_solicitacoes, solicitacoes
        )
        temporarios.append(temporario_solicitacoes)
        _substituir_csv(temporario_clientes, CLIENTES_CSV)
        _substituir_csv(temporario_solicitacoes, SOLICITACOES_CSV)
        caminho.unlink()
    except (
        BaseCreditoIndisponivel,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as erro:
        raise BaseCreditoIndisponivel from erro
    finally:
        for temporario in temporarios:
            temporario.unlink(missing_ok=True)


def recuperar_transacao_pendente() -> bool:
    """Restaura o estado anterior de uma aprovação interrompida."""
    try:
        with _bloquear_csv():
            _restaurar_transacao_sem_lock()
    except BaseCreditoIndisponivel:
        logger.exception("Falha ao recuperar transação de crédito pendente")
        return False
    return True


def migrar_base_clientes(caminho_padrao: Path, caminho_local: Path) -> None:
    """Acrescenta dados de crédito ao CSV local sem sobrescrever cadastros."""
    colunas_credito = {COLUNA_SCORE, COLUNA_LIMITE}
    temporario = None
    try:
        with _bloquear_csv(caminho_local.parent):
            campos_padrao, clientes_padrao = _ler_csv(
                caminho_padrao, {COLUNA_CPF, *colunas_credito}
            )
            campos_locais, clientes_locais = _ler_csv(
                caminho_local, {COLUNA_CPF}
            )
            faltantes = [
                coluna
                for coluna in campos_padrao
                if coluna in colunas_credito and coluna not in campos_locais
            ]
            if not faltantes:
                return

            padrao_por_cpf = {
                _normalizar_cpf(cliente.get(COLUNA_CPF, "")): cliente
                for cliente in clientes_padrao
            }
            for cliente in clientes_locais:
                cliente_padrao = padrao_por_cpf.get(
                    _normalizar_cpf(cliente.get(COLUNA_CPF, "")), {}
                )
                for coluna in faltantes:
                    cliente[coluna] = cliente_padrao.get(coluna, "")

            temporario = _preparar_csv(
                caminho_local, [*campos_locais, *faltantes], clientes_locais
            )
            _substituir_csv(temporario, caminho_local)
    except BaseCreditoIndisponivel:
        logger.warning("Não foi possível migrar a base local de clientes")
    finally:
        if temporario is not None:
            temporario.unlink(missing_ok=True)


def _registrar_resultado(tool_context: ToolContext, resultado: dict) -> dict:
    resultado_completo = {
        **resultado,
        "invocation_id": getattr(tool_context, "invocation_id", None),
    }
    tool_context.state[RESULTADO_CREDITO] = resultado_completo
    if resultado.get("sucesso") and resultado.get("tipo") in {
        "consulta_limite",
        "aumento_limite",
    }:
        if (
            resultado.get("tipo") == "aumento_limite"
            and resultado.get("status") == "rejeitado"
        ):
            tool_context.state[ETAPA_CREDITO] = AGUARDANDO_DECISAO_ENTREVISTA
            tool_context.state[PENDENCIA_REANALISE] = {
                "novo_limite_solicitado": resultado[
                    "novo_limite_solicitado"
                ]
            }
        else:
            tool_context.state[ETAPA_CREDITO] = AGUARDANDO_PROXIMA_ACAO
            tool_context.state[PENDENCIA_REANALISE] = None
    return resultado_completo


def _validar_cliente_autenticado(tool_context: ToolContext) -> str | None:
    if tool_context.state.get("cliente_autenticado") is not True:
        return None
    cpf = _normalizar_cpf(tool_context.state.get("cpf_cliente", ""))
    return cpf if len(cpf) == 11 else None


def _buscar_cliente(
    cpf: str, clientes: list[dict[str, str]]
) -> tuple[int, dict[str, str]]:
    encontrados = [
        (indice, cliente)
        for indice, cliente in enumerate(clientes)
        if _normalizar_cpf(cliente.get(COLUNA_CPF, "")) == cpf
    ]
    if len(encontrados) != 1:
        raise PerfilCreditoIndisponivel
    return encontrados[0]


def _dados_credito_cliente(cliente: dict[str, str]) -> tuple[int, Decimal]:
    try:
        score_decimal = Decimal(str(cliente[COLUNA_SCORE]).strip())
        limite = normalizar_valor_monetario(cliente[COLUNA_LIMITE])
    except (KeyError, InvalidOperation) as erro:
        raise PerfilCreditoIndisponivel from erro
    if (
        limite is None
        or limite < 0
        or not score_decimal.is_finite()
        or score_decimal != score_decimal.to_integral_value()
    ):
        raise PerfilCreditoIndisponivel
    score = int(score_decimal)
    if not 0 <= score <= 1000:
        raise PerfilCreditoIndisponivel
    return score, limite


def _limite_permitido(score: int) -> Decimal:
    _, faixas = _ler_csv(
        SCORE_LIMITE_CSV,
        {"score_minimo", "score_maximo", "limite_maximo"},
    )
    intervalos = []
    for faixa in faixas:
        try:
            minimo = int(faixa["score_minimo"])
            maximo = int(faixa["score_maximo"])
            limite = normalizar_valor_monetario(faixa["limite_maximo"])
        except (KeyError, TypeError, ValueError) as erro:
            raise BaseCreditoIndisponivel from erro
        if (
            minimo < 0
            or maximo > 1000
            or minimo > maximo
            or limite is None
            or limite < 0
        ):
            raise BaseCreditoIndisponivel
        intervalos.append((minimo, maximo, limite))

    ordenados = sorted(intervalos)
    if any(
        atual[0] <= anterior[1]
        for anterior, atual in zip(ordenados, ordenados[1:])
    ):
        raise BaseCreditoIndisponivel
    correspondentes = [
        limite for minimo, maximo, limite in ordenados if minimo <= score <= maximo
    ]
    if len(correspondentes) != 1:
        raise BaseCreditoIndisponivel
    return correspondentes[0]


def consultar_limite_credito(tool_context: ToolContext) -> dict:
    """Consulta o limite atual do cliente já autenticado pela triagem."""
    cpf = _validar_cliente_autenticado(tool_context)
    if cpf is None:
        return _registrar_resultado(
            tool_context,
            {"sucesso": False, "erro": "cliente_nao_autenticado"},
        )
    try:
        with _bloquear_csv():
            _restaurar_transacao_sem_lock()
            _, clientes = _ler_csv(
                CLIENTES_CSV, {COLUNA_CPF, COLUNA_SCORE, COLUNA_LIMITE}
            )
            _, cliente = _buscar_cliente(cpf, clientes)
            _, limite = _dados_credito_cliente(cliente)
    except PerfilCreditoIndisponivel:
        return _registrar_resultado(
            tool_context,
            {"sucesso": False, "erro": "perfil_credito_indisponivel"},
        )
    except BaseCreditoIndisponivel:
        return _registrar_resultado(
            tool_context,
            {"sucesso": False, "erro": "base_credito_indisponivel"},
        )
    return _registrar_resultado(
        tool_context,
        {
            "sucesso": True,
            "tipo": "consulta_limite",
            "limite_atual": _formatar_decimal(limite),
        },
    )


def consultar_score_credito(tool_context: ToolContext) -> dict:
    """Consulta o score atual do cliente já autenticado pela triagem."""
    cpf = _validar_cliente_autenticado(tool_context)
    if cpf is None:
        return _registrar_resultado(
            tool_context, {"sucesso": False, "erro": "cliente_nao_autenticado"}
        )
    try:
        with _bloquear_csv():
            _restaurar_transacao_sem_lock()
            _, clientes = _ler_csv(
                CLIENTES_CSV, {COLUNA_CPF, COLUNA_SCORE, COLUNA_LIMITE}
            )
            _, cliente = _buscar_cliente(cpf, clientes)
            score, _ = _dados_credito_cliente(cliente)
    except PerfilCreditoIndisponivel:
        return _registrar_resultado(
            tool_context, {"sucesso": False, "erro": "perfil_credito_indisponivel"}
        )
    except BaseCreditoIndisponivel:
        return _registrar_resultado(
            tool_context, {"sucesso": False, "erro": "base_credito_indisponivel"}
        )
    return _registrar_resultado(
        tool_context, {"sucesso": True, "tipo": "consulta_score", "score_atual": score}
    )


def _gravar_solicitacao(
    campos_clientes: list[str],
    clientes_anteriores: list[dict[str, str]],
    clientes_atualizados: list[dict[str, str]],
    campos_solicitacoes: list[str],
    solicitacoes_anteriores: list[dict[str, str]],
    solicitacoes_atualizadas: list[dict[str, str]],
    aprovado: bool,
) -> None:
    temporarios: list[Path] = []
    try:
        if aprovado:
            temporario_clientes = _preparar_csv(
                CLIENTES_CSV, campos_clientes, clientes_atualizados
            )
            temporarios.append(temporario_clientes)
        temporario_solicitacoes = _preparar_csv(
            SOLICITACOES_CSV, campos_solicitacoes, solicitacoes_atualizadas
        )
        temporarios.append(temporario_solicitacoes)

        if not aprovado:
            _substituir_csv(temporario_solicitacoes, SOLICITACOES_CSV)
            return

        _escrever_transacao(
            {
                "campos_clientes": campos_clientes,
                "clientes": clientes_anteriores,
                "campos_solicitacoes": campos_solicitacoes,
                "solicitacoes": solicitacoes_anteriores,
            }
        )
        try:
            _substituir_csv(temporario_clientes, CLIENTES_CSV)
            _substituir_csv(temporario_solicitacoes, SOLICITACOES_CSV)
            try:
                _caminho_transacao().unlink()
            except OSError as erro:
                raise BaseCreditoIndisponivel from erro
        except BaseCreditoIndisponivel:
            try:
                _restaurar_transacao_sem_lock()
            except BaseCreditoIndisponivel:
                logger.critical(
                    "A transação de crédito exige recuperação no próximo acesso",
                    exc_info=True,
                )
            raise
    finally:
        for temporario in temporarios:
            temporario.unlink(missing_ok=True)


def solicitar_aumento_limite(
    novo_limite_solicitado: str, tool_context: ToolContext
) -> dict:
    """Avalia e registra um novo limite total solicitado pelo cliente."""
    cpf = _validar_cliente_autenticado(tool_context)
    if cpf is None:
        return _registrar_resultado(
            tool_context,
            {"sucesso": False, "erro": "cliente_nao_autenticado"},
        )
    novo_limite = normalizar_valor_monetario(novo_limite_solicitado)
    if novo_limite is None or novo_limite <= 0:
        return _registrar_resultado(
            tool_context,
            {"sucesso": False, "erro": "valor_invalido"},
        )

    invocation_id = getattr(tool_context, "invocation_id", None)
    resultado_anterior = tool_context.state.get(RESULTADO_CREDITO)
    if (
        invocation_id
        and isinstance(resultado_anterior, dict)
        and resultado_anterior.get("invocation_id") == invocation_id
        and resultado_anterior.get("novo_limite_solicitado")
        == _formatar_decimal(novo_limite)
    ):
        return resultado_anterior

    try:
        with _bloquear_csv():
            _restaurar_transacao_sem_lock()
            campos_clientes, clientes = _ler_csv(
                CLIENTES_CSV, {COLUNA_CPF, COLUNA_SCORE, COLUNA_LIMITE}
            )
            indice_cliente, cliente = _buscar_cliente(cpf, clientes)
            score, limite_atual = _dados_credito_cliente(cliente)
            if novo_limite <= limite_atual:
                return _registrar_resultado(
                    tool_context,
                    {
                        "sucesso": False,
                        "erro": "valor_nao_representa_aumento",
                        "limite_atual": _formatar_decimal(limite_atual),
                    },
                )
            limite_maximo = _limite_permitido(score)
            campos_solicitacoes, solicitacoes = _ler_csv(
                SOLICITACOES_CSV, set(COLUNAS_SOLICITACAO)
            )

            aprovado = novo_limite <= limite_maximo
            status = "aprovado" if aprovado else "rejeitado"
            solicitacoes_atualizadas = [*solicitacoes]
            solicitacoes_atualizadas.append(
                {
                    "cpf_cliente": cpf,
                    "data_hora_solicitacao": datetime.now(timezone.utc).isoformat(),
                    "limite_atual": _formatar_decimal(limite_atual),
                    "novo_limite_solicitado": _formatar_decimal(novo_limite),
                    "status_pedido": status,
                }
            )
            clientes_atualizados = [dict(linha) for linha in clientes]
            if aprovado:
                clientes_atualizados[indice_cliente][COLUNA_LIMITE] = (
                    _formatar_decimal(novo_limite)
                )

            _gravar_solicitacao(
                campos_clientes,
                clientes,
                clientes_atualizados,
                campos_solicitacoes,
                solicitacoes,
                solicitacoes_atualizadas,
                aprovado,
            )
    except PerfilCreditoIndisponivel:
        return _registrar_resultado(
            tool_context,
            {"sucesso": False, "erro": "perfil_credito_indisponivel"},
        )
    except BaseCreditoIndisponivel:
        logger.exception("Falha ao processar solicitação de aumento de limite")
        return _registrar_resultado(
            tool_context,
            {"sucesso": False, "erro": "base_credito_indisponivel"},
        )

    return _registrar_resultado(
        tool_context,
        {
            "sucesso": True,
            "tipo": "aumento_limite",
            "status": status,
            "score_atual": score,
            "limite_atual": _formatar_decimal(limite_atual),
            "novo_limite_solicitado": _formatar_decimal(novo_limite),
            "limite_maximo_score": _formatar_decimal(limite_maximo),
        },
    )
