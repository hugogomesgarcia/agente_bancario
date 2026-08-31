import logging
import os
import re

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools import ToolContext, exit_loop
from google.genai import types

from agentes.compartilhado.estado import OPCOES_RESPOSTA
from agentes.compartilhado.encerramento import (
    ATENDIMENTO_ENCERRADO,
    solicitar_confirmacao_encerramento,
    tratar_confirmacao_encerramento,
)
from agentes.compartilhado.valores import normalizar_valor_monetario
from .tools.entrevista_credito import (
    DADOS_ENTREVISTA,
    concluir_entrevista_credito,
    normalizar_numero_dependentes,
)


logger = logging.getLogger(__name__)

ETAPA_ENTREVISTA = "etapa_entrevista_credito"
AGUARDANDO_RENDA = "aguardando_renda_mensal"
AGUARDANDO_EMPREGO = "aguardando_tipo_emprego"
AGUARDANDO_DESPESAS = "aguardando_despesas_fixas"
AGUARDANDO_DEPENDENTES = "aguardando_numero_dependentes"
AGUARDANDO_DIVIDAS = "aguardando_dividas_ativas"
VALOR_MONETARIO_INTERPRETADO = "valor_monetario_interpretado"
TENTATIVAS_INTERPRETACAO_VALOR = "tentativas_interpretacao_valor"
REGISTRO_ENTREVISTA_INTERPRETADO = "registro_entrevista_interpretado"
CONCLUIR_ENTREVISTA = "concluir_entrevista_credito"

PROXIMAS_ETAPAS = {
    AGUARDANDO_RENDA: AGUARDANDO_EMPREGO,
    AGUARDANDO_EMPREGO: AGUARDANDO_DESPESAS,
    AGUARDANDO_DESPESAS: AGUARDANDO_DEPENDENTES,
    AGUARDANDO_DEPENDENTES: AGUARDANDO_DIVIDAS,
    AGUARDANDO_DIVIDAS: CONCLUIR_ENTREVISTA,
}


def _resposta(texto: str) -> LlmResponse:
    return LlmResponse(
        content=types.Content(
            role="model", parts=[types.Part.from_text(text=texto)]
        )
    )


def _mensagem_usuario_atual(callback_context: CallbackContext) -> str:
    conteudo = callback_context.user_content
    if conteudo is None:
        return ""
    return "".join(parte.text or "" for parte in conteudo.parts or []).strip()


def _normalizar_texto(texto: str) -> str:
    return re.sub(r"[^a-záàâãéêíóôõúç+ ]", "", texto.lower()).strip()


def _dados(callback_context: CallbackContext) -> dict:
    dados = callback_context.state.get(DADOS_ENTREVISTA)
    if not isinstance(dados, dict):
        dados = {}
        callback_context.state[DADOS_ENTREVISTA] = dados
    return dados


def registrar_resposta_entrevista(
    tool_context: ToolContext,
    renda_mensal: str | None = None,
    tipo_emprego: str | None = None,
    despesas_fixas: str | None = None,
    numero_dependentes: int | None = None,
    tem_dividas: bool | None = None,
) -> dict:
    """Valida e registra a resposta interpretada pelo modelo para a etapa atual."""
    etapa = tool_context.state.get(ETAPA_ENTREVISTA)
    campos_por_etapa = {
        AGUARDANDO_RENDA: ("renda_mensal", renda_mensal),
        AGUARDANDO_EMPREGO: ("tipo_emprego", tipo_emprego),
        AGUARDANDO_DESPESAS: ("despesas_fixas", despesas_fixas),
        AGUARDANDO_DEPENDENTES: ("numero_dependentes", numero_dependentes),
        AGUARDANDO_DIVIDAS: ("tem_dividas", tem_dividas),
    }
    campo = campos_por_etapa.get(etapa)
    if campo is None or campo[1] is None:
        resultado = {"sucesso": False, "erro": "campo_ausente_ou_etapa_invalida"}
        tool_context.state[REGISTRO_ENTREVISTA_INTERPRETADO] = resultado
        return resultado

    nome, valor = campo
    if nome in {"renda_mensal", "despesas_fixas"}:
        valor_normalizado = normalizar_valor_monetario(valor)
        if valor_normalizado is None or valor_normalizado < 0:
            resultado = {"sucesso": False, "erro": "valor_invalido"}
            tool_context.state[REGISTRO_ENTREVISTA_INTERPRETADO] = resultado
            return resultado
        valor = f"{valor_normalizado:.2f}"
    elif nome == "tipo_emprego":
        if valor not in {"formal", "autonomo", "desempregado"}:
            resultado = {"sucesso": False, "erro": "tipo_emprego_invalido"}
            tool_context.state[REGISTRO_ENTREVISTA_INTERPRETADO] = resultado
            return resultado
    elif nome == "numero_dependentes":
        valor = normalizar_numero_dependentes(valor)
        if valor is None:
            resultado = {"sucesso": False, "erro": "numero_dependentes_invalido"}
            tool_context.state[REGISTRO_ENTREVISTA_INTERPRETADO] = resultado
            return resultado
    elif not isinstance(valor, bool):
        resultado = {"sucesso": False, "erro": "dividas_invalido"}
        tool_context.state[REGISTRO_ENTREVISTA_INTERPRETADO] = resultado
        return resultado

    resultado = {"sucesso": True, "campo_registrado": nome, "valor": valor}
    tool_context.state[REGISTRO_ENTREVISTA_INTERPRETADO] = resultado
    return resultado


def _resposta_apos_registro_interpretado(callback_context: CallbackContext) -> LlmResponse:
    etapa = callback_context.state.get(ETAPA_ENTREVISTA)
    if etapa == AGUARDANDO_EMPREGO:
        callback_context.state[OPCOES_RESPOSTA] = ["Formal", "Autônomo", "Desempregado"]
        return _resposta("Qual é o seu tipo de emprego: formal, autônomo ou desempregado?")
    if etapa == AGUARDANDO_DESPESAS:
        return _resposta("Qual é o total das suas despesas fixas mensais?")
    if etapa == AGUARDANDO_DEPENDENTES:
        return _resposta("Quantos dependentes você possui?")
    if etapa == AGUARDANDO_DIVIDAS:
        callback_context.state[OPCOES_RESPOSTA] = ["Sim", "Não"]
        return _resposta("Você possui dívidas ativas? Responda sim ou não.")
    resultado = concluir_entrevista_credito(callback_context)
    if not resultado.get("sucesso"):
        callback_context.state[ETAPA_ENTREVISTA] = AGUARDANDO_DIVIDAS
        callback_context.state[OPCOES_RESPOSTA] = ["Sim", "Não"]
        return _resposta(
            "Não foi possível atualizar seu score agora. Responda novamente se "
            "possui dívidas ativas para tentar outra vez."
        )
    callback_context.state[DADOS_ENTREVISTA] = None
    callback_context.state[ETAPA_ENTREVISTA] = None
    callback_context.actions.transfer_to_agent = "credito"
    return _resposta(
        "Entrevista concluída. Seu score foi atualizado para "
        f"{resultado['score_atualizado']}."
    )


def confirmar_valor_monetario_informado(
    valor: str, tool_context: ToolContext
) -> dict:
    """Valida valor de renda ou despesa; use apenas número ou forma como 12k/12 mil."""
    valor_normalizado = normalizar_valor_monetario(valor)
    resultado = (
        {"sucesso": True, "valor_normalizado": f"{valor_normalizado:.2f}"}
        if valor_normalizado is not None and valor_normalizado >= 0
        else {"sucesso": False, "erro": "valor_invalido"}
    )
    tool_context.state[VALOR_MONETARIO_INTERPRETADO] = resultado
    return resultado


def _requisicao_contem_retorno_de_ferramenta(llm_request: LlmRequest) -> bool:
    return any(
        parte.function_response
        for conteudo in llm_request.contents[-1:]
        for parte in conteudo.parts or []
    )


def _pedir_interpretacao_de_valor(llm_request: LlmRequest) -> None:
    llm_request.append_instructions(
        [
            "A resposta atual informa um valor financeiro em linguagem natural. "
            "Extraia somente o valor que o cliente declarou e chame "
            "confirmar_valor_monetario_informado com o número normalizado. Por "
            "exemplo, sete mil reais e 7k devem ser enviados como 7000. Não estime "
            "nem invente números. Se não houver um valor identificável, peça que o "
            "cliente informe renda ou despesas com um valor numérico."
        ]
    )


def interceptar_entrevista(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    callback_context.state[OPCOES_RESPOSTA] = None
    if callback_context.state.get(ATENDIMENTO_ENCERRADO):
        exit_loop(callback_context)
        return _resposta("Este atendimento já foi encerrado.")

    texto = _mensagem_usuario_atual(callback_context)
    confirmacao = tratar_confirmacao_encerramento(
        callback_context,
        texto,
        limpar=lambda contexto: contexto.state.update(
            {ETAPA_ENTREVISTA: None, DADOS_ENTREVISTA: None}
        ),
    )
    if confirmacao is not None:
        return confirmacao

    if callback_context.state.get("cliente_autenticado") is not True:
        callback_context.actions.transfer_to_agent = "triagem"
        return _resposta(
            "Precisamos concluir sua autenticação antes da entrevista de crédito."
        )

    etapa = callback_context.state.get(ETAPA_ENTREVISTA)
    if etapa is None:
        callback_context.state[DADOS_ENTREVISTA] = {}
        callback_context.state[ETAPA_ENTREVISTA] = AGUARDANDO_RENDA
        return _resposta(
            "Vamos iniciar sua entrevista de crédito. Qual é a sua renda mensal?"
        )

    if _requisicao_contem_retorno_de_ferramenta(llm_request):
        registro = callback_context.state.get(REGISTRO_ENTREVISTA_INTERPRETADO)
        callback_context.state[REGISTRO_ENTREVISTA_INTERPRETADO] = None
        if isinstance(registro, dict):
            if registro.get("sucesso"):
                dados = dict(_dados(callback_context))
                dados[registro["campo_registrado"]] = registro["valor"]
                callback_context.state[DADOS_ENTREVISTA] = dados
                callback_context.state[ETAPA_ENTREVISTA] = PROXIMAS_ETAPAS[etapa]
                return _resposta_apos_registro_interpretado(callback_context)
            return _resposta(
                "Não consegui validar essa resposta. Informe o dado solicitado "
                "novamente."
            )

    valor_interpretado = None
    if _requisicao_contem_retorno_de_ferramenta(llm_request):
        resultado = callback_context.state.get(VALOR_MONETARIO_INTERPRETADO)
        callback_context.state[VALOR_MONETARIO_INTERPRETADO] = None
        if not isinstance(resultado, dict) or not resultado.get("sucesso"):
            tentativas = int(
                callback_context.state.get(TENTATIVAS_INTERPRETACAO_VALOR, 0)
            )
            if tentativas == 0:
                callback_context.state[TENTATIVAS_INTERPRETACAO_VALOR] = 1
                llm_request.append_instructions(
                    [
                        "A ferramenta rejeitou o formato enviado. Se a mensagem do "
                        "cliente contiver um valor claro, chame-a novamente uma única "
                        "vez com somente o número decimal normalizado, por exemplo "
                        "12k deve ser 12000. Se não houver valor claro, peça o dado "
                        "ao cliente sem estimar."
                    ]
                )
                return None
            callback_context.state[TENTATIVAS_INTERPRETACAO_VALOR] = 0
            return _resposta(
                "Informe sua renda ou despesas mensais usando um valor numérico "
                "igual ou maior que zero."
            )
        callback_context.state[TENTATIVAS_INTERPRETACAO_VALOR] = 0
        valor_interpretado = resultado["valor_normalizado"]

    dados = _dados(callback_context)
    if etapa == AGUARDANDO_RENDA:
        renda = normalizar_valor_monetario(valor_interpretado or texto)
        if renda is None or renda < 0:
            _pedir_interpretacao_de_valor(llm_request)
            return None
        callback_context.state[TENTATIVAS_INTERPRETACAO_VALOR] = 0
        dados["renda_mensal"] = f"{renda:.2f}"
        callback_context.state[DADOS_ENTREVISTA] = dados
        callback_context.state[ETAPA_ENTREVISTA] = AGUARDANDO_EMPREGO
        callback_context.state[OPCOES_RESPOSTA] = [
            "Formal",
            "Autônomo",
            "Desempregado",
        ]
        return _resposta(
            "Qual é o seu tipo de emprego: formal, autônomo ou desempregado?"
        )

    if etapa == AGUARDANDO_EMPREGO:
        emprego = {
            "formal": "formal",
            "autônomo": "autonomo",
            "autonomo": "autonomo",
            "desempregado": "desempregado",
        }.get(_normalizar_texto(texto))
        if emprego is None:
            llm_request.append_instructions(
                [
                    "Interprete o tipo de emprego informado e chame "
                    "registrar_resposta_entrevista com tipo_emprego igual a "
                    "formal, autonomo ou desempregado. Não deduza uma categoria "
                    "se o cliente não a informou."
                ]
            )
            return None
        dados["tipo_emprego"] = emprego
        callback_context.state[DADOS_ENTREVISTA] = dados
        callback_context.state[ETAPA_ENTREVISTA] = AGUARDANDO_DESPESAS
        return _resposta("Qual é o total das suas despesas fixas mensais?")

    if etapa == AGUARDANDO_DESPESAS:
        despesas = normalizar_valor_monetario(valor_interpretado or texto)
        if despesas is None or despesas < 0:
            _pedir_interpretacao_de_valor(llm_request)
            return None
        callback_context.state[TENTATIVAS_INTERPRETACAO_VALOR] = 0
        dados["despesas_fixas"] = f"{despesas:.2f}"
        callback_context.state[DADOS_ENTREVISTA] = dados
        callback_context.state[ETAPA_ENTREVISTA] = AGUARDANDO_DEPENDENTES
        return _resposta("Quantos dependentes você possui?")

    if etapa == AGUARDANDO_DEPENDENTES:
        dependentes = (
            normalizar_numero_dependentes(texto)
            if texto.isdigit() and len(texto) <= 4
            else None
        )
        if dependentes is None:
            llm_request.append_instructions(
                [
                    "Interprete quantos dependentes financeiros o cliente "
                    "informou e chame registrar_resposta_entrevista com "
                    "numero_dependentes inteiro entre 0 e 1000. Não conte "
                    "pessoas que o cliente não disse serem dependentes."
                ]
            )
            return None
        dados["numero_dependentes"] = dependentes
        callback_context.state[DADOS_ENTREVISTA] = dados
        callback_context.state[ETAPA_ENTREVISTA] = AGUARDANDO_DIVIDAS
        callback_context.state[OPCOES_RESPOSTA] = ["Sim", "Não"]
        return _resposta("Você possui dívidas ativas? Responda sim ou não.")

    if etapa == AGUARDANDO_DIVIDAS:
        tem_dividas = {"sim": True, "não": False, "nao": False}.get(_normalizar_texto(texto))
        if tem_dividas is None:
            llm_request.append_instructions(
                [
                    "Interprete se o cliente declarou possuir dívidas ativas e "
                    "chame registrar_resposta_entrevista com tem_dividas booleano. "
                    "Não trate uma negativa sobre encerrar como resposta sobre "
                    "dívidas."
                ]
            )
            return None
        dados["tem_dividas"] = tem_dividas
        callback_context.state[DADOS_ENTREVISTA] = dados
        resultado = concluir_entrevista_credito(callback_context)
        if not resultado.get("sucesso"):
            callback_context.state[OPCOES_RESPOSTA] = ["Sim", "Não"]
            return _resposta(
                "Não foi possível atualizar seu score agora. Responda novamente "
                "se possui dívidas ativas para tentar outra vez."
            )
        callback_context.state[ETAPA_ENTREVISTA] = None
        callback_context.state[DADOS_ENTREVISTA] = None
        callback_context.actions.transfer_to_agent = "credito"
        return _resposta(
            f"Entrevista concluída. Seu score foi atualizado para "
            f"{resultado['score_atualizado']}."
        )

    callback_context.state[ETAPA_ENTREVISTA] = None
    callback_context.state[DADOS_ENTREVISTA] = None
    return _resposta("Vamos reiniciar a entrevista de crédito.")


def restringir_resposta_livre(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> LlmResponse | None:
    partes = llm_response.content.parts if llm_response.content else []
    chamadas = [parte for parte in partes or [] if parte.function_call]
    if chamadas:
        if len(chamadas) == len(partes or []):
            return None
        return LlmResponse(content=types.Content(role="model", parts=chamadas))
    return _resposta(
        "Não consegui validar essa resposta. Informe novamente o dado solicitado "
        "ou use uma das opções disponíveis."
    )


def tratar_erro_do_modelo(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
    error: Exception,
) -> LlmResponse:
    logger.error("Falha no modelo da entrevista de crédito: %s", type(error).__name__)
    return _resposta(
        "Não consegui interpretar essa resposta agora. Informe novamente o dado "
        "solicitado ou use uma das opções disponíveis."
    )


agente_entrevista_credito = Agent(
    name="entrevista_credito",
    model=os.getenv("GOOGLE_MODEL", "gemini-3.5-flash-lite"),
    description=(
        "Conduz entrevista financeira, recalcula o score e retorna o cliente "
        "ao atendimento de crédito."
    ),
    instruction=(
        "A entrevista é conduzida por um fluxo local determinístico. Não produza "
        "respostas livres nem altere dados sem a ferramenta disponível. Para texto "
        "livre de emprego, dependentes ou dívidas, interprete o campo atual e chame "
        "registrar_resposta_entrevista; ela valida antes de alterar o estado. Se o cliente "
        "parecer querer encerrar todo o atendimento, chame somente "
        "solicitar_confirmacao_encerramento; ela não encerra a sessão."
    ),
    tools=[
        confirmar_valor_monetario_informado,
        registrar_resposta_entrevista,
        solicitar_confirmacao_encerramento,
    ],
    before_model_callback=interceptar_entrevista,
    after_model_callback=restringir_resposta_livre,
    on_model_error_callback=tratar_erro_do_modelo,
)
