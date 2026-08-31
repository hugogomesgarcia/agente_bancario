import re

from google.adk.models import LlmResponse
from google.adk.tools import ToolContext, exit_loop
from google.genai import types

from .estado import OPCOES_RESPOSTA


ATENDIMENTO_ENCERRADO = "atendimento_encerrado"
AGUARDANDO_CONFIRMACAO_ENCERRAMENTO = "aguardando_confirmacao_encerramento"
OPCOES_CONFIRMACAO_ENCERRAMENTO = ["Sim, encerrar", "Não, continuar"]


def _resposta(texto: str) -> LlmResponse:
    return LlmResponse(
        content=types.Content(
            role="model", parts=[types.Part.from_text(text=texto)]
        )
    )


def solicitar_confirmacao_encerramento(tool_context: ToolContext) -> dict:
    """Pede confirmação antes de encerrar um atendimento identificado pelo modelo."""
    if tool_context.state.get(ATENDIMENTO_ENCERRADO):
        return {"sucesso": False, "erro": "atendimento_ja_encerrado"}
    tool_context.state[AGUARDANDO_CONFIRMACAO_ENCERRAMENTO] = True
    tool_context.state[OPCOES_RESPOSTA] = OPCOES_CONFIRMACAO_ENCERRAMENTO
    return {"sucesso": True, "confirmacao_solicitada": True}


def tratar_confirmacao_encerramento(callback_context, texto: str, *, limpar=None):
    """Executa somente a confirmação explícita de um encerramento já solicitado."""
    if not callback_context.state.get(AGUARDANDO_CONFIRMACAO_ENCERRAMENTO):
        return None

    normalizado = re.sub(r"[^a-záàâãéêíóôõúç ]", "", texto.lower()).strip()
    if normalizado in {"sim", "sim encerrar", "encerrar"}:
        callback_context.state[AGUARDANDO_CONFIRMACAO_ENCERRAMENTO] = None
        callback_context.state[OPCOES_RESPOSTA] = None
        callback_context.state[ATENDIMENTO_ENCERRADO] = True
        if limpar is not None:
            limpar(callback_context)
        exit_loop(callback_context)
        return _resposta("Atendimento encerrado. Obrigado por falar com o Banco Ágil!")

    if normalizado in {"não", "nao", "não continuar", "nao continuar"}:
        callback_context.state[AGUARDANDO_CONFIRMACAO_ENCERRAMENTO] = None
        callback_context.state[OPCOES_RESPOSTA] = None
        return _resposta("Tudo bem. Vamos continuar de onde paramos.")

    callback_context.state[OPCOES_RESPOSTA] = OPCOES_CONFIRMACAO_ENCERRAMENTO
    return _resposta("Posso fechar seu atendimento então?")
