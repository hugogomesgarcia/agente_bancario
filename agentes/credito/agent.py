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
from .tools.credito import (
    AGUARDANDO_DECISAO_ENTREVISTA,
    AGUARDANDO_NOVO_LIMITE,
    AGUARDANDO_PROXIMA_ACAO,
    ETAPA_CREDITO,
    PENDENCIA_REANALISE,
    RESULTADO_CREDITO,
    consultar_limite_credito,
    consultar_score_credito,
    solicitar_aumento_limite,
)
from agentes.entrevista_credito.tools.entrevista_credito import (
    RETORNO_ENTREVISTA,
)


logger = logging.getLogger(__name__)

AGUARDANDO_RETENTATIVA_REANALISE = "aguardando_retentativa_reanalise"
ACAO_ENTREVISTA_CREDITO = "acao_entrevista_credito"


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


def _requisicao_contem_retorno_de_ferramenta(llm_request: LlmRequest) -> bool:
    return any(
        parte.function_response
        for conteudo in llm_request.contents[-1:]
        for parte in conteudo.parts or []
    )


def _ferramentas_retornadas(llm_request: LlmRequest) -> set[str]:
    return {
        parte.function_response.name
        for conteudo in llm_request.contents[-1:]
        for parte in conteudo.parts or []
        if parte.function_response and parte.function_response.name
    }


def _formatar_moeda(valor: str) -> str:
    inteiro, centavos = valor.split(".")
    grupos = []
    while inteiro:
        grupos.insert(0, inteiro[-3:])
        inteiro = inteiro[:-3]
    return f"R$ {'.'.join(grupos)},{centavos}"


def aceitar_entrevista_credito(tool_context: ToolContext) -> dict:
    """Registra o aceite da entrevista somente após uma solicitação rejeitada."""
    if tool_context.state.get(ETAPA_CREDITO) != AGUARDANDO_DECISAO_ENTREVISTA:
        return {"sucesso": False, "erro": "entrevista_nao_pendente"}
    tool_context.state[ACAO_ENTREVISTA_CREDITO] = "aceita"
    return {"sucesso": True, "acao": "aceita"}


def recusar_entrevista_credito(tool_context: ToolContext) -> dict:
    """Registra a recusa da entrevista sem encerrar o atendimento."""
    if tool_context.state.get(ETAPA_CREDITO) != AGUARDANDO_DECISAO_ENTREVISTA:
        return {"sucesso": False, "erro": "entrevista_nao_pendente"}
    tool_context.state[ACAO_ENTREVISTA_CREDITO] = "recusada"
    return {"sucesso": True, "acao": "recusada"}


def _resposta_do_resultado(resultado: dict, *, apos_entrevista: bool = False) -> str:
    if not resultado.get("sucesso"):
        erro = resultado.get("erro")
        if erro == "valor_invalido":
            return (
                "Informe o novo limite total desejado usando um valor numérico "
                "positivo."
            )
        if erro == "valor_nao_representa_aumento":
            return (
                "O novo limite precisa ser maior que seu limite atual de "
                f"{_formatar_moeda(resultado['limite_atual'])}. Informe outro valor."
            )
        if erro == "cliente_nao_autenticado":
            return "Precisamos concluir sua autenticação antes da consulta de crédito."
        if erro == "perfil_credito_indisponivel":
            return (
                "Seu cadastro não possui os dados de crédito necessários para "
                "esta consulta. Posso ajudar com outro assunto ou encerrar."
            )
        return (
            "Não foi possível consultar ou atualizar os dados de crédito agora. "
            "Você pode tentar novamente ou informar outro assunto."
        )

    if resultado["tipo"] == "consulta_limite":
        return (
            f"Seu limite atual é {_formatar_moeda(resultado['limite_atual'])}. "
            "Posso ajudar com outro assunto? Se preferir, diga que deseja encerrar."
        )

    if resultado["tipo"] == "consulta_score":
        return (
            f"Seu score de crédito atual é {resultado['score_atual']}. "
            "Posso consultar seu limite ou analisar uma solicitação de aumento."
        )

    limite_atual = _formatar_moeda(resultado["limite_atual"])
    novo_limite = _formatar_moeda(resultado["novo_limite_solicitado"])
    if resultado["status"] == "aprovado":
        return (
            f"Sua solicitação foi aprovada e o limite foi atualizado de "
            f"{limite_atual} para {novo_limite}. Posso ajudar com outro assunto? "
            "Se preferir, diga que deseja encerrar."
        )
    if apos_entrevista:
        return (
            f"Após a atualização do score, sua solicitação do novo limite de "
            f"{novo_limite} continua incompatível com a faixa permitida e foi "
            "rejeitada. Posso ajudar com outro assunto ou encerrar."
        )
    return (
        f"Sua solicitação do novo limite de {novo_limite} foi rejeitada porque "
        "o valor não é compatível com seu score atual. Posso realizar uma "
        "entrevista de crédito para tentar reajustar esse score. Você deseja "
        "continuar com a entrevista?"
    )


def _normalizar_resposta(texto: str) -> str:
    return re.sub(r"[^a-záàâãéêíóôõúç ]", "", texto.lower()).strip()


def _solicitou_consulta_score(texto: str) -> bool:
    normalizado = _normalizar_resposta(texto)
    return bool(
        "score" in normalizado
        and not re.search(
            r"\b(?:aument\w*|elev\w*|ampli\w*|subir)\b", normalizado
        )
        and re.search(
            r"\b(?:qual|quanto|consult\w*|saber|ver|informe|informar)\b"
            r"|\bmeu score\b",
            normalizado,
        )
    )


def _solicitou_consulta_limite(texto: str) -> bool:
    normalizado = _normalizar_resposta(texto)
    return bool(
        "limite" in normalizado
        and not re.search(
            r"\b(?:aument\w*|elev\w*|ampli\w*|subir)\b"
            r"|\b(?:mais limite|limite maior)\b",
            normalizado,
        )
        and re.search(
            r"\b(?:qual|quanto|consult\w*|saber|ver|informe|informar)\b"
            r"|\blimite atual\b|\bmeu limite\b",
            normalizado,
        )
    )


def _solicitou_aumento_limite(texto: str) -> bool:
    normalizado = _normalizar_resposta(texto)
    if "score" in normalizado:
        return False
    negou_aumento = re.search(
        r"\b(?:não|nao)\s+(?:quero|desejo|vou)\s+"
        r"(?:aument\w*|elev\w*|ampli\w*|subir)\b",
        normalizado,
    )
    return bool(
        re.search(
            r"\b(?:aument\w*|elev\w*|ampli\w*|subir)\b"
            r"|\b(?:mais limite|limite maior)\b",
            normalizado,
        )
        and not negou_aumento
    )


def _solicitou_aumento_sem_valor(texto: str) -> bool:
    normalizado = _normalizar_resposta(texto)
    return bool(
        _solicitou_aumento_limite(texto)
        and not re.search(r"\d", texto)
        and not re.search(r"\b(?:mil|k)\b", normalizado)
    )


def _solicitou_entrevista_credito(texto: str) -> bool:
    normalizado = _normalizar_resposta(texto)
    return bool(
        "entrevista" in normalizado
        or (
            "score" in normalizado
            and re.search(
                r"\b(?:aument\w*|atualiz\w*|elev\w*|melhor\w*|recalcul\w*)\b",
                normalizado,
            )
        )
    )


def _recusou_entrevista(texto: str) -> bool:
    normalizado = _normalizar_resposta(texto)
    if re.match(r"^(?:não|nao)\b", normalizado):
        return True
    if "entrevista" not in normalizado:
        return False
    return bool(
        re.search(
            r"\b(?:não|nao)\s+(?:quero|desejo|vou|farei|aceito)\b"
            r"|\b(?:não|nao)\s+tenho interesse\b"
            r"|\b(?:prefiro\s+(?:não|nao)|dispenso|recuso)\b",
            normalizado,
        )
    )


def _aceitou_entrevista(texto: str) -> bool:
    normalizado = _normalizar_resposta(texto)
    if normalizado == "sim":
        return True
    return "entrevista" in normalizado and bool(
        re.search(r"\b(?:quero|desejo|aceito|vamos|pode)\b", normalizado)
    )


def _solicitou_assunto_sem_suporte(texto: str) -> bool:
    normalizado = _normalizar_resposta(texto)
    return bool(
        re.search(
            r"\b(?:câmbio|cambio|cotação|cotacao|dólar|dolar|euro|pix|boleto|"
            r"empréstimo|emprestimo|financiamento|investimento|seguro|saldo|"
            r"extrato|fatura|saque|depósito|deposito|senha|transferência|"
            r"transferencia)\b"
            r"|\boutro assunto\b",
            normalizado,
        )
    )


def _responder_consulta_credito(
    callback_context: CallbackContext, texto: str
) -> LlmResponse | None:
    if _solicitou_consulta_score(texto):
        resultado = consultar_score_credito(callback_context)
    elif _solicitou_consulta_limite(texto):
        resultado = consultar_limite_credito(callback_context)
    else:
        return None
    callback_context.state[ETAPA_CREDITO] = AGUARDANDO_PROXIMA_ACAO
    return _responder_resultado_credito(callback_context, resultado)


def _responder_resultado_credito(
    callback_context: CallbackContext,
    resultado: dict,
    *,
    apos_entrevista: bool = False,
) -> LlmResponse:
    if resultado.get("status") == "rejeitado" and not apos_entrevista:
        callback_context.state[OPCOES_RESPOSTA] = ["Sim", "Não"]
    if resultado.get("erro") in {
        "valor_invalido",
        "valor_nao_representa_aumento",
    }:
        callback_context.state[ETAPA_CREDITO] = AGUARDANDO_NOVO_LIMITE
    return _resposta(
        _resposta_do_resultado(resultado, apos_entrevista=apos_entrevista)
    )


def _solicitou_nova_tentativa(texto: str) -> bool:
    return _normalizar_resposta(texto) in {
        "sim",
        "tentar",
        "tentar novamente",
        "pode tentar",
        "pode tentar novamente",
    }


def _reanalisar_pendencia(callback_context: CallbackContext) -> LlmResponse:
    pendencia = callback_context.state.get(PENDENCIA_REANALISE)
    if not isinstance(pendencia, dict) or not pendencia.get(
        "novo_limite_solicitado"
    ):
        callback_context.state[ETAPA_CREDITO] = None
        return _resposta(
            "A entrevista atualizou seu score, mas não há solicitação de aumento "
            "pendente para reanalisar. Você deseja consultar seu limite atual ou "
            "solicitar um aumento?"
        )

    resultado = solicitar_aumento_limite(
        pendencia["novo_limite_solicitado"], callback_context
    )
    if not resultado.get("sucesso"):
        callback_context.state[ETAPA_CREDITO] = AGUARDANDO_RETENTATIVA_REANALISE
        callback_context.state[OPCOES_RESPOSTA] = ["Sim", "Não"]
        return _resposta(
            _resposta_do_resultado(resultado)
            + " Diga que deseja tentar novamente para reanalisar o mesmo limite."
        )

    callback_context.state[PENDENCIA_REANALISE] = None
    callback_context.state[ETAPA_CREDITO] = AGUARDANDO_PROXIMA_ACAO
    return _resposta(_resposta_do_resultado(resultado, apos_entrevista=True))


def interceptar_fluxo_credito(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    callback_context.state[OPCOES_RESPOSTA] = None
    if callback_context.state.get(ATENDIMENTO_ENCERRADO):
        exit_loop(callback_context)
        return _resposta("Este atendimento já foi encerrado.")

    texto = _mensagem_usuario_atual(callback_context)
    confirmacao = tratar_confirmacao_encerramento(callback_context, texto)
    if confirmacao is not None:
        return confirmacao

    if callback_context.state.get("cliente_autenticado") is not True:
        callback_context.actions.transfer_to_agent = "triagem"
        return _resposta(
            "Precisamos concluir sua autenticação antes da consulta de crédito."
        )

    retorno_entrevista = callback_context.state.get(RETORNO_ENTREVISTA)
    if isinstance(retorno_entrevista, dict):
        callback_context.state[RETORNO_ENTREVISTA] = None
        return _reanalisar_pendencia(callback_context)

    if _requisicao_contem_retorno_de_ferramenta(llm_request):
        acao_entrevista = callback_context.state.get(ACAO_ENTREVISTA_CREDITO)
        callback_context.state[ACAO_ENTREVISTA_CREDITO] = None
        if acao_entrevista == "aceita":
            callback_context.state[ETAPA_CREDITO] = None
            callback_context.actions.transfer_to_agent = "entrevista_credito"
            return _resposta("")
        if acao_entrevista == "recusada":
            callback_context.state[PENDENCIA_REANALISE] = None
            callback_context.state[ETAPA_CREDITO] = AGUARDANDO_PROXIMA_ACAO
            if _ferramentas_retornadas(llm_request) & {
                "consultar_limite_credito",
                "consultar_score_credito",
            }:
                resultado = callback_context.state.get(RESULTADO_CREDITO)
                if isinstance(resultado, dict):
                    return _responder_resultado_credito(
                        callback_context, resultado
                    )
            return _resposta("Tudo bem. Posso ajudar com outro assunto?")

    if (
        callback_context.state.get(ETAPA_CREDITO)
        == AGUARDANDO_RETENTATIVA_REANALISE
    ):
        if _solicitou_nova_tentativa(texto):
            return _reanalisar_pendencia(callback_context)
        if _normalizar_resposta(texto) in {"não", "nao"}:
            callback_context.state[PENDENCIA_REANALISE] = None
            callback_context.state[ETAPA_CREDITO] = AGUARDANDO_PROXIMA_ACAO
            return _resposta(
                "Tudo bem. Posso ajudar com outro assunto ou encerrar o atendimento."
            )
        callback_context.state[OPCOES_RESPOSTA] = ["Sim", "Não"]
        return _resposta(
            "Você deseja tentar novamente a análise do mesmo limite? Responda sim ou não."
        )

    if (
        callback_context.state.get(ETAPA_CREDITO)
        == AGUARDANDO_DECISAO_ENTREVISTA
        and not _requisicao_contem_retorno_de_ferramenta(llm_request)
    ):
        if _aceitou_entrevista(texto) and not _recusou_entrevista(texto):
            callback_context.state[ETAPA_CREDITO] = None
            callback_context.actions.transfer_to_agent = "entrevista_credito"
            return _resposta("")
        if _recusou_entrevista(texto):
            callback_context.state[PENDENCIA_REANALISE] = None
            callback_context.state[ETAPA_CREDITO] = AGUARDANDO_PROXIMA_ACAO
            resposta_consulta = _responder_consulta_credito(
                callback_context, texto
            )
            if resposta_consulta is not None:
                return resposta_consulta
            if _solicitou_assunto_sem_suporte(texto):
                callback_context.actions.transfer_to_agent = "triagem"
                return _resposta("")
            if _solicitou_aumento_limite(texto):
                callback_context.state[ETAPA_CREDITO] = AGUARDANDO_NOVO_LIMITE
                llm_request.append_instructions(
                    [
                        "A entrevista foi recusada. Interprete somente o novo "
                        "pedido de aumento e chame solicitar_aumento_limite se "
                        "houver um valor total claro. Caso contrário, peça o novo "
                        "limite total."
                    ]
                )
                return None
            return _resposta(
                "Tudo bem. Posso ajudar com outro assunto ou encerrar o atendimento."
            )
        llm_request.append_instructions([
            "A entrevista de crédito está pendente. Interprete a mensagem inteira: "
            "chame aceitar_entrevista_credito para aceite ou "
            "recusar_entrevista_credito para recusa. Se a mensagem trouxer outro "
            "pedido de crédito, execute primeiro a decisão sobre a entrevista."
        ])
        return None

    if _solicitou_entrevista_credito(texto):
        callback_context.actions.transfer_to_agent = "entrevista_credito"
        return _resposta("")

    resposta_consulta = _responder_consulta_credito(callback_context, texto)
    if resposta_consulta is not None:
        return resposta_consulta

    if _solicitou_assunto_sem_suporte(texto):
        callback_context.actions.transfer_to_agent = "triagem"
        return _resposta("")

    if _solicitou_aumento_sem_valor(texto):
        callback_context.state[ETAPA_CREDITO] = AGUARDANDO_NOVO_LIMITE
        return _resposta(
            "Qual é o novo limite total desejado? Informe um valor numérico."
        )

    if callback_context.state.get(ETAPA_CREDITO) == AGUARDANDO_NOVO_LIMITE:
        if normalizar_valor_monetario(texto) is not None:
            resultado = solicitar_aumento_limite(texto, callback_context)
            return _responder_resultado_credito(callback_context, resultado)
        llm_request.append_instructions([
            "O cliente precisa informar o novo limite total. Interprete o texto e, "
            "se houver um valor claro, chame solicitar_aumento_limite com ele. Não "
            "escolha entre vários valores nem invente um valor."
        ])
        return None

    if (
        callback_context.state.get(ETAPA_CREDITO) == AGUARDANDO_PROXIMA_ACAO
        and not _requisicao_contem_retorno_de_ferramenta(llm_request)
    ):
        if normalizar_valor_monetario(texto) is not None:
            resultado = solicitar_aumento_limite(texto, callback_context)
            return _responder_resultado_credito(callback_context, resultado)
        return None

    if _requisicao_contem_retorno_de_ferramenta(llm_request):
        resultado = callback_context.state.get(RESULTADO_CREDITO)
        if isinstance(resultado, dict):
            return _responder_resultado_credito(callback_context, resultado)

    return None


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
        "Você deseja consultar seu limite atual ou solicitar um aumento? "
        "Para solicitar um aumento, informe o novo limite total desejado."
    )


def tratar_erro_do_modelo(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
    error: Exception,
) -> LlmResponse:
    logger.error("Falha no modelo do agente de crédito: %s", type(error).__name__)
    return _resposta(
        "Não consegui interpretar a solicitação agora. Você pode consultar seu "
        "limite atual ou informar o novo limite total desejado."
    )


agente_credito = Agent(
    name="credito",
    model=os.getenv("GOOGLE_MODEL", "gemini-3.5-flash-lite"),
    description=(
        "Consulta limite de crédito e avalia solicitações de aumento para "
        "clientes já autenticados."
    ),
    instruction="""
Você é exclusivamente o agente de crédito do Banco Ágil. O cliente já foi
autenticado pela triagem. Nunca solicite CPF ou data de nascimento e nunca
receba CPF como argumento de ferramenta.

Para consultar o limite atual, chame consultar_limite_credito. Para aumento,
chame solicitar_aumento_limite somente quando houver um novo limite total
numericamente claro; passe exatamente o valor informado. Se o cliente ainda não
informou o valor, peça o novo limite total. As ferramentas são as únicas fontes
de limite, aprovação, rejeição e atualização. Não antecipe nem invente esses
resultados.

Para assuntos sem suporte, informe que só pode atender consultas, pedidos de
aumento e a entrevista de crédito. Depois de uma consulta ou decisão, o fluxo
local perguntará sobre outra necessidade. Quando uma solicitação for rejeitada,
o fluxo local oferecerá a entrevista e cuidará do encaminhamento.

Se o cliente parecer querer encerrar todo o atendimento, chame
solicitar_confirmacao_encerramento. Essa ferramenta apenas pede confirmação;
nunca encerre a sessão diretamente.
""",
    tools=[
        consultar_limite_credito,
        consultar_score_credito,
        solicitar_aumento_limite,
        aceitar_entrevista_credito,
        recusar_entrevista_credito,
        solicitar_confirmacao_encerramento,
    ],
    before_model_callback=interceptar_fluxo_credito,
    after_model_callback=restringir_resposta_livre,
    on_model_error_callback=tratar_erro_do_modelo,
    disallow_transfer_to_peers=True,
)
