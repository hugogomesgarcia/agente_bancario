import os
import re

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest
from google.adk.models import LlmResponse
from google.adk.tools import exit_loop
from google.genai import types

from agentes.compartilhado.encerramento import (
    ATENDIMENTO_ENCERRADO,
    solicitar_confirmacao_encerramento,
    tratar_confirmacao_encerramento,
)
from agentes.compartilhado.transferencia import transferir_silenciosamente
from agentes.compartilhado.valores import normalizar_valor_monetario
from agentes.credito.tools.credito import AGUARDANDO_PROXIMA_ACAO, ETAPA_CREDITO
from . import guardrail
from .tools.consultar_client import (
    autenticar_cliente,
    consultar_cliente,
    normalizar_data_nascimento,
)

ETAPA_AUTENTICACAO = "etapa_autenticacao"
AGUARDANDO_CPF = "aguardando_cpf"
AGUARDANDO_DATA_NASCIMENTO = "aguardando_data_nascimento"


def encerrar_apos_limite_de_tentativas(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    if int(callback_context.state.get("tentativas_falhas", 0)) < 3:
        return None

    return _encerrar_atendimento(
        callback_context,
        "Não foi possível autenticar seus dados após três tentativas. "
        "Por segurança, este atendimento será encerrado.",
    )


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


def _encerrar_atendimento(
    callback_context: CallbackContext, mensagem: str
) -> LlmResponse:
    callback_context.state[ATENDIMENTO_ENCERRADO] = True
    exit_loop(callback_context)
    return _resposta(mensagem)


def _cpf_informado(texto: str) -> str | None:
    texto_limpo = texto.strip()
    if re.fullmatch(r"\d{11}", texto_limpo):
        return texto_limpo
    correspondencia = re.search(
        r"(?<!\d)\d{3}\.\d{3}\.\d{3}-\d{2}(?!\d)", texto
    )
    if correspondencia:
        return correspondencia.group()
    correspondencia = re.search(
        r"\bcpf\b\D{0,20}(\d{11})(?!\d)", texto, re.IGNORECASE
    )
    return correspondencia.group(1) if correspondencia else None


def _data_nascimento_informada(texto: str) -> str | None:
    if normalizar_data_nascimento(texto) is not None:
        return texto
    correspondencia = re.search(
        r"(?<!\d)(?:\d{1,2}[/-]\d{1,2}[/-]\d{4}|\d{4}-\d{1,2}-\d{1,2})(?!\d)",
        texto,
    )
    if correspondencia and normalizar_data_nascimento(correspondencia.group()) is not None:
        return correspondencia.group()
    correspondencia = re.search(
        r"\d{1,2}\s+(?:de|do)\s+(?:[a-zç]+|\d{1,2})\s+(?:de\s+)?(?:\d{4}|\d{2})",
        texto.lower(),
    )
    if correspondencia and normalizar_data_nascimento(correspondencia.group()) is not None:
        return correspondencia.group()
    return None


def _solicitacao_explicita_de_encerramento(texto: str) -> bool:
    texto_normalizado = re.sub(r"\s+", " ", texto.lower()).strip()
    if re.search(
        r"\b(?:não|nao)\s+(?:(?:quero|desejo|pode|podemos|vamos)\s+)?"
        r"(?:encerrar|finalizar|cancelar|parar|sair)\b",
        texto_normalizado,
    ):
        return False

    return bool(
        re.search(
            r"\b(?:(?:quero|desejo|gostaria de|pode|podemos|vamos)\s+"
            r"(?:encerrar|finalizar)(?:\s+(?:o\s+)?"
            r"(?:atendimento|conversa|sessão))?|"
            r"(?:quero|desejo|gostaria de|pode|podemos|vamos)\s+"
            r"(?:cancelar|parar)\s+(?:o\s+)?(?:atendimento|conversa|sessão)|"
            r"(?:quero|desejo|gostaria de|pode|podemos|vamos)\s+sair|"
            r"(?:encerre|finalize)(?:\s+(?:o\s+)?"
            r"(?:atendimento|conversa|sessão))?|"
            r"(?:cancele|pare)\s+(?:o\s+)?(?:atendimento|conversa|sessão)|"
            r"(?:encerrar|finalizar|cancelar|parar|sair)\s+(?:o\s+)?"
            r"(?:atendimento|conversa|sessão)|"
            r"(?:não|nao)\s+(?:quero|desejo)\s+(?:mais\s+)?continuar)\b",
            texto_normalizado,
        )
    )


def _transferir_para_especialista_se_aplicavel(
    callback_context: CallbackContext, mensagem_usuario: str
) -> LlmResponse | None:
    if not (
        callback_context.state.get("cliente_autenticado") is True
        and callback_context.state.get(guardrail.CONTEXTO_GUARDRAIL)
        == guardrail.POS_AUTENTICACAO
    ):
        return None
    if (
        callback_context.state.get(ETAPA_CREDITO) == AGUARDANDO_PROXIMA_ACAO
        and normalizar_valor_monetario(mensagem_usuario) is not None
    ):
        return transferir_silenciosamente(callback_context, "credito")
    return None


def _preparar_resposta_do_modelo(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
    contexto: str,
    instrucoes: str,
) -> None:
    callback_context.state[guardrail.CONTEXTO_GUARDRAIL] = contexto
    llm_request.append_instructions([instrucoes])


def _requisicao_contem_retorno_de_ferramenta(llm_request: LlmRequest) -> bool:
    return any(
        parte.function_response
        for conteudo in llm_request.contents[-1:]
        for parte in conteudo.parts or []
    )


def interceptar_autenticacao_local(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    if callback_context.state.get(ATENDIMENTO_ENCERRADO):
        return _encerrar_atendimento(
            callback_context, "Este atendimento já foi encerrado."
        )

    resposta_encerramento = encerrar_apos_limite_de_tentativas(
        callback_context, llm_request
    )
    if resposta_encerramento:
        return resposta_encerramento

    texto = _mensagem_usuario_atual(callback_context)
    confirmacao = tratar_confirmacao_encerramento(callback_context, texto)
    if confirmacao is not None:
        return confirmacao

    if (
        not callback_context.state.get("cliente_autenticado")
        and _solicitacao_explicita_de_encerramento(texto)
    ):
        solicitar_confirmacao_encerramento(callback_context)
        return _resposta("Você deseja encerrar este atendimento?")

    if callback_context.state.get("cliente_autenticado"):
        if not _requisicao_contem_retorno_de_ferramenta(llm_request):
            callback_context.state[guardrail.CONTEXTO_GUARDRAIL] = (
                guardrail.POS_AUTENTICACAO
            )
            transferencia = _transferir_para_especialista_se_aplicavel(
                callback_context, texto
            )
            if transferencia is not None:
                return transferencia
        _preparar_resposta_do_modelo(
            callback_context,
            llm_request,
            guardrail.POS_AUTENTICACAO,
            """
O cliente está autenticado. Interprete a mensagem inteira e escolha a ação
correta. Para consultas de limite, score ou aumento de crédito, transfira para
credito com transfer_to_agent; não produza texto antes ou depois. Para entrevista
de crédito, atualização, recálculo ou melhoria do score, transfira para
entrevista_credito. Para câmbio, cotação de moedas ou criptomoedas, transfira para
cambio. Não interprete o par na triagem e não produza texto antes ou depois da
transferência.

Não assuma o papel do especialista, não responda à questão bancária, não peça
valor, renda ou outros dados operacionais, não invente informações e não afirme
que houve consulta, pedido, aprovação, encaminhamento ou operação. Para qualquer
outro assunto sem fonte ou ferramenta, diga de forma natural que não consegue
responder com segurança. Nunca prometa usar uma ferramenta em uma resposta
futura: use-a neste turno ou não alegue seu uso.

Se o cliente parecer querer encerrar todo o atendimento, chame
solicitar_confirmacao_encerramento. Ela apenas pede confirmação: nunca encerre
nem transfira o atendimento por esse motivo.
""",
        )
        return None

    etapa = callback_context.state.get(ETAPA_AUTENTICACAO)
    if etapa is None:
        callback_context.state[ETAPA_AUTENTICACAO] = AGUARDANDO_CPF
        return _resposta(
            "Olá! Seja bem-vindo ao Banco Ágil. Para começarmos, informe o seu CPF."
        )

    if etapa == AGUARDANDO_CPF:
        cpf = _cpf_informado(texto)
        if cpf is None:
            return _resposta("Olá! Para continuarmos, informe seu CPF.")

        resultado = consultar_cliente(cpf, callback_context)
        if resultado.get("encerrar_atendimento"):
            return encerrar_apos_limite_de_tentativas(callback_context, llm_request)
        if resultado.get("erro"):
            return _resposta(resultado["erro"])
        if resultado["encontrado"]:
            callback_context.state[ETAPA_AUTENTICACAO] = AGUARDANDO_DATA_NASCIMENTO
            return _resposta("CPF localizado. Informe sua data de nascimento.")
        return _resposta("Não foi possível validar o CPF informado. Tente novamente.")

    if etapa == AGUARDANDO_DATA_NASCIMENTO:
        data_nascimento = _data_nascimento_informada(texto)
        if data_nascimento is None:
            return _resposta("Informe sua data de nascimento para continuarmos.")

        resultado = autenticar_cliente(
            callback_context.state["cpf_em_validacao"], data_nascimento, callback_context
        )
        if resultado.get("encerrar_atendimento"):
            return encerrar_apos_limite_de_tentativas(callback_context, llm_request)
        if resultado.get("erro"):
            return _resposta(resultado["erro"])
        if resultado["autenticado"]:
            callback_context.state[ETAPA_AUTENTICACAO] = None
            callback_context.state["cpf_em_validacao"] = None
            return _resposta("Autenticação realizada com sucesso! Como posso ajudar você hoje?")
        return _resposta(
            "Não foi possível autenticar os dados informados. Tente novamente."
        )

    callback_context.state[ETAPA_AUTENTICACAO] = None
    return _resposta("Informe seu CPF para continuarmos o atendimento.")


agente_triagem = Agent(
    name="triagem",
    model=os.getenv("GOOGLE_MODEL", "gemini-3.5-flash-lite"),
    description="Agente responsável pela triagem de clientes do banco.",
    instruction="""
Você é exclusivamente o agente de triagem do Banco Ágil. A autenticação é
conduzida localmente. Nunca revele dados cadastrais. Uma ferramenta só comprova
o que o resultado dela declara e não alegue consulta ou operação sem evidência.
Crédito, entrevista de crédito e câmbio estão disponíveis por transferência.
Nunca assuma as funções dos especialistas. Siga também as instruções restritas
adicionadas a cada turno. Para uma intenção de encerramento, use somente
solicitar_confirmacao_encerramento; a sessão só será fechada após confirmação.
""",
    tools=[solicitar_confirmacao_encerramento],
    before_model_callback=interceptar_autenticacao_local,
    after_model_callback=guardrail.aplicar_guardrail_de_resposta,
    on_model_error_callback=guardrail.tratar_erro_do_modelo,
)
