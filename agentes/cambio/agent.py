import logging
import os

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools import exit_loop
from google.genai import types

from agentes.compartilhado.encerramento import (
    AGUARDANDO_CONFIRMACAO_ENCERRAMENTO,
    ATENDIMENTO_ENCERRADO,
    solicitar_confirmacao_encerramento,
    tratar_confirmacao_encerramento,
)
from agentes.compartilhado.estado import OPCOES_RESPOSTA
from .tools.cambio import (
    AGUARDANDO_ESCLARECIMENTO,
    AGUARDANDO_PROXIMA_ACAO,
    ETAPA_CAMBIO,
    INTERPRETACAO_PARCIAL,
    RESULTADO_CAMBIO,
    TENTATIVAS_INTERPRETACAO,
    processar_solicitacao_cambio,
)


logger = logging.getLogger(__name__)


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


def _formatar_decimal(valor: str) -> str:
    inteiro, separador, fracao = valor.partition(".")
    sinal = ""
    if inteiro.startswith("-"):
        sinal, inteiro = "-", inteiro[1:]
    grupos = []
    while inteiro:
        grupos.append(inteiro[-3:])
        inteiro = inteiro[:-3]
    inteiro_formatado = ".".join(reversed(grupos)) or "0"
    return sinal + inteiro_formatado + ("," + fracao if separador else "")


def _resposta_cotacao(resultado: dict) -> str:
    base = resultado["ativo_base"]
    destino = resultado["ativo_destino"]
    texto = (
        f"Segundo a {resultado['fonte']}, a cotação de {resultado['nome_par']} "
        f"atualizada em {resultado['consultado_em']} é: para 1 {base}, compra "
        f"{destino} {_formatar_decimal(resultado['compra'])} e venda {destino} "
        f"{_formatar_decimal(resultado['venda'])}."
    )
    if resultado.get("derivada_par_inverso"):
        texto += " Essa cotação foi calculada a partir do par inverso disponível."
    if resultado.get("derivada_par_cruzado"):
        referencias = " e ".join(resultado["pares_referencia"])
        texto += (
            " Essa cotação foi calculada a partir dos pares de referência "
            f"{referencias} disponíveis na fonte."
        )
    if resultado.get("quantidade_base"):
        quantidade = _formatar_decimal(resultado["quantidade_base"])
        total_compra = _formatar_decimal(resultado["total_compra"])
        total_venda = _formatar_decimal(resultado["total_venda"])
        texto += (
            f" Para {quantidade} {base}, o valor correspondente é {destino} "
            f"{total_compra} pela cotação de compra e {destino} {total_venda} "
            "pela cotação de venda."
        )
    return (
        texto
        + " Os valores são informativos e não representam uma oferta ou operação "
        "do Banco Ágil. Posso consultar outra cotação ou ajudar com outro assunto?"
    )


def _resposta_erro(resultado: dict) -> str:
    erro = resultado.get("erro")
    if erro == "quantidade_invalida":
        return (
            "Não consegui identificar a quantidade a converter. Informe um valor "
            "numérico positivo e as moedas desejadas."
        )
    if erro in {
        "ativo_base_invalido",
        "ativo_destino_explicito_invalido",
        "destino_padrao_inconsistente",
        "status_invalido",
        "pergunta_esclarecimento_ausente",
    }:
        return "Não consegui identificar o par com segurança. Informe as moedas ou criptomoedas que deseja comparar."
    if erro == "ativos_iguais":
        return "As duas pontas da cotação são iguais. Com qual outra moeda ou criptomoeda você deseja comparar?"
    if erro == "par_nao_disponivel":
        return "A AwesomeAPI não disponibiliza esse par de cotação. Você pode informar outro par para consulta."
    if erro == "token_nao_configurado":
        return "A consulta de câmbio está temporariamente indisponível por uma falha de configuração."
    if erro == "token_invalido":
        return "A consulta de câmbio está temporariamente indisponível porque a autenticação da fonte falhou."
    if erro == "limite_api_excedido":
        return "O limite temporário da fonte de cotações foi atingido. Tente novamente mais tarde."
    return "Não foi possível consultar a cotação agora. Você pode tentar novamente ou informar outro assunto."


def interceptar_fluxo_cambio(
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
            {
                ETAPA_CAMBIO: None,
                INTERPRETACAO_PARCIAL: None,
                RESULTADO_CAMBIO: None,
            }
        ),
    )
    if confirmacao is not None:
        return confirmacao

    if callback_context.state.get("cliente_autenticado") is not True:
        callback_context.actions.transfer_to_agent = "triagem"
        return _resposta(
            "Precisamos concluir sua autenticação antes da consulta de câmbio."
        )

    etapa = callback_context.state.get(ETAPA_CAMBIO)
    if etapa == AGUARDANDO_PROXIMA_ACAO and texto.casefold() == "nova cotação":
        callback_context.state[ETAPA_CAMBIO] = AGUARDANDO_ESCLARECIMENTO
        callback_context.state[INTERPRETACAO_PARCIAL] = None
        return _resposta(
            "Claro. Quais moedas ou criptomoedas você deseja comparar? Se "
            "quiser converter um valor, informe também a quantidade."
        )
    if (
        etapa == AGUARDANDO_PROXIMA_ACAO
        and texto.casefold() == "encerrar atendimento"
    ):
        solicitar_confirmacao_encerramento(callback_context)
        return _resposta("Você deseja encerrar este atendimento?")

    if _requisicao_contem_retorno_de_ferramenta(llm_request):
        if callback_context.state.get(AGUARDANDO_CONFIRMACAO_ENCERRAMENTO):
            return _resposta("Você deseja encerrar este atendimento?")

        resultado = callback_context.state.get(RESULTADO_CAMBIO)
        if isinstance(resultado, dict):
            if resultado.get("sucesso") and resultado.get("tipo") == "esclarecimento":
                callback_context.state[ETAPA_CAMBIO] = AGUARDANDO_ESCLARECIMENTO
                callback_context.state[TENTATIVAS_INTERPRETACAO] = 0
                return _resposta(resultado["pergunta"])
            if resultado.get("sucesso"):
                callback_context.state[ETAPA_CAMBIO] = AGUARDANDO_PROXIMA_ACAO
                callback_context.state[TENTATIVAS_INTERPRETACAO] = 0
                callback_context.state[OPCOES_RESPOSTA] = [
                    "Nova cotação",
                    "Encerrar atendimento",
                ]
                return _resposta(_resposta_cotacao(resultado))

            if resultado.get("erro") in {
                "ativo_base_invalido",
                "ativo_destino_explicito_invalido",
                "destino_padrao_inconsistente",
                "quantidade_invalida",
                "status_invalido",
                "pergunta_esclarecimento_ausente",
            }:
                tentativas = int(
                    callback_context.state.get(TENTATIVAS_INTERPRETACAO, 0)
                )
                if tentativas == 0:
                    callback_context.state[TENTATIVAS_INTERPRETACAO] = 1
                    llm_request.append_instructions(
                        [
                            "A chamada anterior violou o contrato da ferramenta. "
                            "Releia a mensagem completa e tente uma única vez: não "
                            "ignore destino ou quantidade explícitos e peça "
                            "esclarecimento se o pedido não estiver claro."
                        ]
                    )
                    return None
            callback_context.state[ETAPA_CAMBIO] = AGUARDANDO_PROXIMA_ACAO
            return _resposta(_resposta_erro(resultado))

    parcial = callback_context.state.get(INTERPRETACAO_PARCIAL)
    if callback_context.state.get(ETAPA_CAMBIO) == AGUARDANDO_ESCLARECIMENTO:
        llm_request.append_instructions(
            [
                "A mensagem atual responde à pergunta de esclarecimento feita no "
                "turno anterior. Interprete-a junto com todo o histórico e com a "
                f"interpretação parcial registrada: {parcial!r}."
            ]
        )
    elif etapa == AGUARDANDO_PROXIMA_ACAO:
        llm_request.append_instructions(
            [
                "A cotação anterior terminou. Interprete a necessidade atual sem "
                "reutilizar silenciosamente um ativo anterior; use o histórico "
                "somente quando o cliente fizer uma referência contextual clara."
            ]
        )
    callback_context.state[RESULTADO_CAMBIO] = None
    return None


def filtrar_resposta_mista(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> LlmResponse | None:
    partes = llm_response.content.parts if llm_response.content else []
    chamadas = [parte for parte in partes or [] if parte.function_call]
    if chamadas:
        if len(chamadas) == len(partes or []):
            return None
        return LlmResponse(content=types.Content(role="model", parts=chamadas))
    if not partes:
        return _resposta(
            "Não consegui responder agora. Você pode explicar novamente como "
            "posso ajudar com câmbio?"
        )
    return None


def tratar_erro_do_modelo(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
    error: Exception,
) -> LlmResponse:
    logger.error("Falha no modelo do agente de câmbio: %s", type(error).__name__)
    return _resposta(
        "Não consegui responder agora. Você pode tentar novamente ou explicar "
        "como posso ajudar com câmbio."
    )


agente_cambio = Agent(
    name="cambio",
    model=os.getenv("GOOGLE_MODEL", "gemini-3.5-flash-lite"),
    description=(
        "Interpreta solicitações de câmbio em linguagem natural e consulta "
        "cotações atuais de moedas e criptomoedas para clientes autenticados."
    ),
    instruction="""
Você é exclusivamente o agente de câmbio do Banco Ágil. Interprete a mensagem
inteira e o contexto da conversa semanticamente; não use correspondências
parciais. Para toda consulta ou dúvida de par, chame somente
processar_solicitacao_cambio e não produza texto antes ou depois da chamada.

Use status resolvido quando identificar com segurança o ativo base. Informe os
códigos internacionais em ativo_base e ativo_destino. Moedas de países podem
ser mencionadas pelo nome, país, gentílico ou de forma casual; criptomoedas
podem aparecer por nome ou símbolo. Registre em evidencia_base e
evidencia_destino os trechos exatos que sustentam cada ponta.

O destino padrão é BRL somente se o cliente não expressar nenhum destino. Se
houver destino explícito, marque destino_explicito como true e preserve-o. Leia
com especial atenção expressões como "em", "para", "contra", "cotado em" e
equivalentes. Exemplos: "moeda da China" é CNY/BRL; "moeda da China em euro" é
CNY/EUR; "euro na moeda chinesa" é EUR/CNY; "real em dólar" é BRL/USD;
"bitcoin" é BTC/BRL; "ethereum em euro" é ETH/EUR.

Quando o cliente informar uma quantidade a converter, preencha quantidade_base
com o valor numérico na moeda base e evidencia_quantidade com o trecho exato que
o sustenta. Não preencha esses campos em uma consulta apenas da taxa. Exemplos:
"quantos reais dão 10000 ienes" é quantidade_base 10000, JPY/BRL; "quanto valem
200 dólares em euros" é quantidade_base 200, USD/EUR. Nunca faça o cálculo.

Use status precisa_esclarecimento e formule pergunta_esclarecimento quando
faltar o ativo base, quando somente o destino estiver claro ou quando um termo
for genuinamente ambíguo, como "peso" sem país ou "sol" sem distinguir PEN de
SOL. Não adivinhe. Em respostas a uma pergunta anterior, use o histórico para
completar o par.

Não trate a simples menção a uma moeda ou criptomoeda como pedido de cotação. Se
o cliente pedir opinião, recomendação ou perguntar se deve comprar, vender ou
investir, responda naturalmente que o Banco Ágil não oferece recomendação de
investimento e que a decisão depende do perfil e dos riscos do cliente. Não dê
conselho personalizado, não tente identificar um par e não chame a ferramenta só
porque um ativo foi mencionado. Você pode oferecer uma cotação atual caso o
cliente queira consultá-la.

Nunca produza valores de cotação. A ferramenta e a resposta local são as únicas
fontes financeiras. Para outro assunto bancário, transfira silenciosamente para
triagem. Para intenção de encerrar todo o atendimento, chame somente
solicitar_confirmacao_encerramento.
""",
    tools=[processar_solicitacao_cambio, solicitar_confirmacao_encerramento],
    before_model_callback=interceptar_fluxo_cambio,
    after_model_callback=filtrar_resposta_mista,
    on_model_error_callback=tratar_erro_do_modelo,
    generate_content_config=types.GenerateContentConfig(temperature=0),
    disallow_transfer_to_peers=True,
)
