import csv
from datetime import date, datetime
from pathlib import Path
import re

from google.adk.tools import ToolContext

ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = ROOT / "csv" / "local" / "clientes.csv"

MESES = {
    "janeiro": 1,
    "fevereiro": 2,
    "marco": 3,
    "abril": 4,
    "maio": 5,
    "junho": 6,
    "julho": 7,
    "agosto": 8,
    "setembro": 9,
    "outubro": 10,
    "novembro": 11,
    "dezembro": 12,
}


def normalizar_cpf(cpf: str) -> str:
    return re.sub(r"\D", "", str(cpf))


def cpf_valido(cpf: str) -> bool:
    if len(cpf) != 11 or cpf == cpf[0] * 11:
        return False

    for indice in (9, 10):
        soma = sum(
            int(digito) * peso
            for digito, peso in zip(cpf[:indice], range(indice + 1, 1, -1))
        )
        digito_verificador = (soma * 10 % 11) % 10
        if int(cpf[indice]) != digito_verificador:
            return False

    return True


def normalizar_data_nascimento(data_nascimento: str) -> date | None:
    """Converte formatos usuais de data para comparação com o cadastro."""
    if not isinstance(data_nascimento, str):
        return None

    texto = data_nascimento.strip().lower()
    for formato in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(texto, formato).date()
        except ValueError:
            pass

    correspondencia = re.fullmatch(
        r"(\d{1,2})\s+(?:de|do)\s+([a-zç]+|\d{1,2})\s+(?:de\s+)?(\d{4}|\d{2})",
        texto,
    )
    if not correspondencia:
        return None

    dia, mes, ano = correspondencia.groups()
    mes = int(mes) if mes.isdigit() else MESES.get(mes.replace("ç", "c"))
    if mes is None:
        return None
    if len(ano) == 2:
        ano = f"19{ano}" if int(ano) >= 30 else f"20{ano}"

    try:
        return datetime(int(ano), mes, int(dia)).date()
    except ValueError:
        return None


def _registrar_falha(tool_context: ToolContext) -> dict:
    tentativas = min(3, int(tool_context.state.get("tentativas_falhas", 0)) + 1)
    tool_context.state["tentativas_falhas"] = tentativas
    return {
        "tentativas_restantes": max(0, 3 - tentativas),
        "encerrar_atendimento": tentativas >= 3,
    }


def _limite_de_tentativas_atingido(tool_context: ToolContext) -> bool:
    return int(tool_context.state.get("tentativas_falhas", 0)) >= 3


def _buscar_cliente(cpf: str) -> tuple[dict | None, str | None]:
    try:
        with CSV_PATH.open("r", encoding="utf-8", newline="") as arquivo:
            leitor = csv.DictReader(arquivo, skipinitialspace=True)
            if not leitor.fieldnames or "CPF" not in leitor.fieldnames:
                return None, "A base de clientes está indisponível no momento."

            for cliente in leitor:
                if normalizar_cpf(cliente.get("CPF", "")) == cpf:
                    return cliente, None
    except (OSError, UnicodeDecodeError, csv.Error):
        return None, "A base de clientes está indisponível no momento."

    return None, None


def consultar_cliente(cpf: str, tool_context: ToolContext) -> dict:
    """Valida um CPF e verifica se ele está cadastrado na base local."""
    if _limite_de_tentativas_atingido(tool_context):
        return {
            "cpf_valido": False,
            "encontrado": False,
            "mensagem": "Limite de tentativas de autenticação atingido.",
            "tentativas_restantes": 0,
            "encerrar_atendimento": True,
        }

    cpf_normalizado = normalizar_cpf(cpf)
    if not cpf_valido(cpf_normalizado):
        return {
            "cpf_valido": False,
            "encontrado": False,
            "mensagem": "CPF inválido.",
            **_registrar_falha(tool_context),
        }

    cliente, erro = _buscar_cliente(cpf_normalizado)
    if erro:
        return {"cpf_valido": True, "encontrado": False, "erro": erro}
    if cliente is None:
        return {
            "cpf_valido": True,
            "encontrado": False,
            "mensagem": "CPF não encontrado na base de clientes.",
            **_registrar_falha(tool_context),
        }

    tool_context.state["cpf_em_validacao"] = cpf_normalizado
    return {
        "cpf_valido": True,
        "encontrado": True,
        "mensagem": "CPF localizado. Solicite a data de nascimento para concluir a autenticação.",
    }


def autenticar_cliente(cpf: str, data_nascimento: str, tool_context: ToolContext) -> dict:
    """Compara a data de nascimento informada com o cadastro do CPF."""
    if _limite_de_tentativas_atingido(tool_context):
        return {
            "autenticado": False,
            "mensagem": "Limite de tentativas de autenticação atingido.",
            "tentativas_restantes": 0,
            "encerrar_atendimento": True,
        }

    cpf_normalizado = normalizar_cpf(cpf)
    if not cpf_valido(cpf_normalizado):
        return {
            "autenticado": False,
            "mensagem": "Não foi possível autenticar os dados informados.",
            **_registrar_falha(tool_context),
        }

    data_informada = normalizar_data_nascimento(data_nascimento)
    if data_informada is None:
        return {
            "autenticado": False,
            "mensagem": "Não foi possível autenticar os dados informados.",
            **_registrar_falha(tool_context),
        }

    cliente, erro = _buscar_cliente(cpf_normalizado)
    if erro:
        return {"autenticado": False, "erro": erro}
    if cliente is None:
        return {
            "autenticado": False,
            "mensagem": "Não foi possível autenticar os dados informados.",
            **_registrar_falha(tool_context),
        }

    try:
        data_cadastrada = normalizar_data_nascimento(cliente["Data de Nascimento"])
    except KeyError:
        data_cadastrada = None
    if data_cadastrada is None:
        return {"autenticado": False, "erro": "A base de clientes está indisponível no momento."}

    if data_informada != data_cadastrada:
        return {
            "autenticado": False,
            "mensagem": "Não foi possível autenticar os dados informados.",
            **_registrar_falha(tool_context),
        }

    tool_context.state["tentativas_falhas"] = 0
    tool_context.state["cliente_autenticado"] = True
    tool_context.state["cpf_cliente"] = cpf_normalizado
    return {"autenticado": True, "mensagem": "Cliente autenticado com sucesso."}
