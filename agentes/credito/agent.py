import json
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
from agentes.compartilhado.guardrail import (
    PerfilRevisao,
    PerguntaPendente,
    criar_revisao_de_resposta,
    criar_tratamento_de_erro,
    extrair_fatos,
)
from agentes.compartilhado.modelo import configuracao_geracao, criar_modelo
from agentes.compartilhado.transferencia import transferir_silenciosamente
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


AGUARDANDO_RETENTATIVA_REANALISE = "aguardando_retentativa_reanalise"
ACAO_ENTREVISTA_CREDITO = "acao_entrevista_credito"
FERRAMENTAS_DE_RESULTADO = {
    "consultar_limite_credito",
    "consultar_score_credito",
    "solicitar_aumento_limite",
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


def _requisicao_contem_retorno_de_ferramenta(llm_request: LlmRequest) -> bool:
    return any(
        parte.function_response
        for conteudo in llm_request.contents[-1:]
        for parte in conteudo.parts or []
    )


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


def _resposta_do_resultado(
    resultado: dict, *, apos_entrevista: bool = False, fechamento: bool = True
) -> str:
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

    oferta = " Posso ajudar com outro assunto? Se preferir, diga que deseja encerrar."
    if resultado["tipo"] == "consulta_limite":
        return (
            f"Seu limite atual é {_formatar_moeda(resultado['limite_atual'])}."
            + (oferta if fechamento else "")
        )

    if resultado["tipo"] == "consulta_score":
        return f"Seu score de crédito atual é {resultado['score_atual']}." + (
            " Posso consultar seu limite ou analisar uma solicitação de aumento."
            if fechamento
            else ""
        )

    limite_atual = _formatar_moeda(resultado["limite_atual"])
    novo_limite = _formatar_moeda(resultado["novo_limite_solicitado"])
    if resultado["status"] == "aprovado":
        return (
            f"Sua solicitação foi aprovada e o limite foi atualizado de "
            f"{limite_atual} para {novo_limite}." + (oferta if fechamento else "")
        )
    score_atual = resultado["score_atual"]
    limite_maximo = _formatar_moeda(resultado["limite_maximo_score"])
    if apos_entrevista:
        return (
            f"Após a atualização, seu score é {score_atual} e permite um limite "
            f"total de até {limite_maximo}. Por isso, sua solicitação de "
            f"{novo_limite} foi rejeitada. Posso ajudar com outro assunto ou "
            "encerrar."
        )
    return (
        f"Seu score atual é {score_atual} e permite um limite total de até "
        f"{limite_maximo}. Por isso, sua solicitação de {novo_limite} foi "
        "rejeitada. Posso realizar uma entrevista de crédito para tentar "
        "reajustar esse score. Você deseja continuar com a entrevista?"
    )


def _normalizar_resposta(texto: str) -> str:
    return re.sub(r"[^a-záàâãéêíóôõúç ]", "", texto.lower()).strip()


# Atalhos locais só tratam mensagens cujas palavras pertencem inteiramente ao
# vocabulário da intenção. Qualquer outra palavra ("libras", "isso", "vale")
# indica um pedido composto ou fora do roteiro, que segue para o modelo.
PALAVRAS_PEDIDO = frozenset(
    """
    a o as os um uma de do da dos das no na e é eu me meu minha meus minhas
    por favor oi olá ola pra para então entao agora ok só so também tambem você
    voce vc pode poderia consegue quero queria gostaria preciso desejo saber ver
    consultar verificar checar conferir mostrar mostre informe informar diga
    dizer qual quanto está esta tá ta atual atualmente hoje mas porém porem
    """.split()
)
PALAVRAS_LIMITE = frozenset(
    "limite crédito credito cartão cartao disponível disponivel total valor tenho".split()
)
PALAVRAS_SCORE = frozenset("score crédito credito".split())
PALAVRAS_AUMENTO = frozenset(
    """
    aumentar aumento aumente elevar eleve ampliar amplie subir suba mais maior
    solicitar pedir fazer novo que falei disse já ja limite crédito credito
    cartão cartao
    """.split()
)
PALAVRAS_ENTREVISTA = frozenset(
    """
    entrevista crédito credito fazer realizar iniciar começar comecar como posso
    melhorar aumentar atualizar recalcular elevar score
    """.split()
)
PALAVRAS_ACEITE = frozenset(
    "sim claro aceito vamos pode ser fazer entrevista quero ok beleza bora".split()
)
PALAVRAS_RECUSA = frozenset(
    """
    não nao quero desejo vou farei aceito prefiro fazer a entrevista obrigado
    obrigada agora dispenso recuso tenho interesse por enquanto valeu
    """.split()
)
PALAVRAS_NEGACAO = frozenset({"não", "nao", "dispenso", "recuso"})


def _palavras_dentro_de(texto: str, vocabulario: frozenset[str]) -> bool:
    palavras = _normalizar_resposta(texto).split()
    return bool(palavras) and all(palavra in vocabulario for palavra in palavras)


def _solicitou_consulta_score(texto: str) -> bool:
    return "score" in _normalizar_resposta(texto).split() and _palavras_dentro_de(
        texto, PALAVRAS_PEDIDO | PALAVRAS_SCORE
    )


def _solicitou_consulta_limite(texto: str) -> bool:
    return "limite" in _normalizar_resposta(texto).split() and _palavras_dentro_de(
        texto, PALAVRAS_PEDIDO | PALAVRAS_LIMITE
    )


def _solicitou_aumento_limite(texto: str) -> bool:
    # Aceita "elevar/ampliar/subir o limite", mas não "não quero aumentar".
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
    return bool(
        _solicitou_aumento_limite(texto)
        and not re.search(r"\d", texto)
        and _palavras_dentro_de(texto, PALAVRAS_PEDIDO | PALAVRAS_AUMENTO)
    )


def _solicitou_entrevista_credito(texto: str) -> bool:
    palavras = set(_normalizar_resposta(texto).split())
    return _palavras_dentro_de(texto, PALAVRAS_PEDIDO | PALAVRAS_ENTREVISTA) and (
        "entrevista" in palavras
        or (
            "score" in palavras
            and bool(
                palavras
                & {"aumentar", "atualizar", "elevar", "melhorar", "recalcular"}
            )
        )
    )


def _aceitou_entrevista(texto: str) -> bool:
    palavras = set(_normalizar_resposta(texto).split())
    return _palavras_dentro_de(texto, PALAVRAS_PEDIDO | PALAVRAS_ACEITE) and bool(
        palavras & {"sim", "claro", "aceito", "vamos", "bora"}
        or {"quero", "entrevista"} <= palavras
        or {"pode", "ser"} <= palavras
    )


def _recusa_e_restante(texto: str) -> tuple[bool, str]:
    """Identifica uma recusa no início da mensagem e devolve o pedido restante."""
    primeira, *restante = re.split(
        r"[,;.!?]|\b(?:mas|porém|porem)\b", texto, maxsplit=1, flags=re.IGNORECASE
    )
    palavras = set(_normalizar_resposta(primeira).split())
    if not (
        _palavras_dentro_de(primeira, PALAVRAS_RECUSA)
        and palavras & PALAVRAS_NEGACAO
    ):
        return False, texto
    restante = "".join(restante).strip()
    if not _normalizar_resposta(restante) or _palavras_dentro_de(
        restante, PALAVRAS_RECUSA | PALAVRAS_PEDIDO
    ):
        restante = ""
    return True, restante


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


def _responder_resultados_credito(
    callback_context: CallbackContext, resultados: list[dict]
) -> LlmResponse:
    """Responde a todas as ferramentas do turno; a decisão de aumento fica por último."""
    unicos = {
        json.dumps(resultado, sort_keys=True, default=str): resultado
        for resultado in resultados
    }
    ordenados = sorted(
        unicos.values(),
        key=lambda resultado: resultado.get("tipo") == "aumento_limite",
    )
    textos = [
        _resposta_do_resultado(resultado, fechamento=False)
        for resultado in ordenados[:-1]
    ]
    final = _responder_resultado_credito(callback_context, ordenados[-1])
    return _resposta(" ".join([*textos, final.content.parts[0].text]))


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
        return _responder_retorno_de_ferramenta(callback_context, llm_request)

    etapa = callback_context.state.get(ETAPA_CREDITO)
    if etapa == AGUARDANDO_RETENTATIVA_REANALISE:
        if _solicitou_nova_tentativa(texto):
            return _reanalisar_pendencia(callback_context)
        if _normalizar_resposta(texto) in {"não", "nao"}:
            callback_context.state[PENDENCIA_REANALISE] = None
            callback_context.state[ETAPA_CREDITO] = AGUARDANDO_PROXIMA_ACAO
            return _resposta(
                "Tudo bem. Posso ajudar com outro assunto ou encerrar o atendimento."
            )
        llm_request.append_instructions(
            [
                "Uma reanálise do limite falhou e aguarda a decisão do cliente "
                "de tentar novamente. Se a mensagem trouxer outro assunto, "
                "atenda-o conforme suas instruções; não chame "
                "solicitar_aumento_limite para essa reanálise."
            ]
        )
        return None

    if etapa == AGUARDANDO_DECISAO_ENTREVISTA:
        if _aceitou_entrevista(texto):
            callback_context.state[ETAPA_CREDITO] = None
            return transferir_silenciosamente(
                callback_context, "entrevista_credito"
            )
        recusou, restante = _recusa_e_restante(texto)
        if recusou:
            callback_context.state[PENDENCIA_REANALISE] = None
            callback_context.state[ETAPA_CREDITO] = AGUARDANDO_PROXIMA_ACAO
            if not restante:
                return _resposta(
                    "Tudo bem. Posso ajudar com outro assunto ou encerrar o "
                    "atendimento."
                )
            resposta_consulta = _responder_consulta_credito(
                callback_context, restante
            )
            if resposta_consulta is not None:
                return resposta_consulta
            if _solicitou_aumento_limite(restante):
                callback_context.state[ETAPA_CREDITO] = AGUARDANDO_NOVO_LIMITE
                llm_request.append_instructions(
                    [
                        "A entrevista foi recusada. Interprete somente o novo "
                        "pedido de aumento e chame solicitar_aumento_limite se "
                        "houver um valor total claro. Caso contrário, não peça o "
                        "valor: o sistema pedirá o novo limite total."
                    ]
                )
                return None
            llm_request.append_instructions(
                [
                    "O cliente recusou a entrevista de crédito e a recusa já foi "
                    "registrada; não chame ferramentas sobre a entrevista. Atenda "
                    f"somente o restante da mensagem: {restante!r}."
                ]
            )
            return None
        llm_request.append_instructions([
            "A entrevista de crédito foi oferecida e aguarda decisão. Interprete a "
            "mensagem inteira: chame aceitar_entrevista_credito para aceite ou "
            "recusar_entrevista_credito para recusa. Se a mensagem trouxer outro "
            "pedido de crédito, execute primeiro a decisão sobre a entrevista. Se "
            "for uma dúvida ou comentário sem decisão, responda brevemente sem "
            "ferramentas; o sistema repetirá a pergunta sobre a entrevista."
        ])
        return None

    if _solicitou_entrevista_credito(texto):
        return transferir_silenciosamente(
            callback_context, "entrevista_credito"
        )

    resposta_consulta = _responder_consulta_credito(callback_context, texto)
    if resposta_consulta is not None:
        return resposta_consulta

    if _solicitou_aumento_sem_valor(texto):
        callback_context.state[ETAPA_CREDITO] = AGUARDANDO_NOVO_LIMITE
        return _resposta(
            "Qual é o novo limite total desejado? Informe um valor numérico."
        )

    if etapa in {AGUARDANDO_NOVO_LIMITE, AGUARDANDO_PROXIMA_ACAO} and (
        normalizar_valor_monetario(texto) is not None
    ):
        resultado = solicitar_aumento_limite(texto, callback_context)
        return _responder_resultado_credito(callback_context, resultado)

    if etapa == AGUARDANDO_NOVO_LIMITE:
        llm_request.append_instructions([
            "O cliente precisa informar o novo limite total. Interprete o texto e, "
            "se houver um valor claro, chame solicitar_aumento_limite com ele. Não "
            "escolha entre vários valores nem invente um valor. Se não houver "
            "valor, não o peça: o sistema pedirá o novo limite total."
        ])
    return None


def _responder_retorno_de_ferramenta(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    acao_entrevista = callback_context.state.get(ACAO_ENTREVISTA_CREDITO)
    callback_context.state[ACAO_ENTREVISTA_CREDITO] = None
    if acao_entrevista == "aceita":
        callback_context.state[ETAPA_CREDITO] = None
        return transferir_silenciosamente(callback_context, "entrevista_credito")
    if acao_entrevista == "recusada":
        callback_context.state[PENDENCIA_REANALISE] = None
        callback_context.state[ETAPA_CREDITO] = AGUARDANDO_PROXIMA_ACAO

    resultados = [
        parte.function_response.response
        for conteudo in llm_request.contents[-1:]
        for parte in conteudo.parts or []
        if parte.function_response
        and parte.function_response.name in FERRAMENTAS_DE_RESULTADO
        and isinstance(parte.function_response.response, dict)
    ]
    if resultados:
        return _responder_resultados_credito(callback_context, resultados)
    if acao_entrevista == "recusada":
        llm_request.append_instructions(
            [
                "A recusa da entrevista foi registrada. Se a mensagem do cliente "
                "trouxer outro pedido, atenda-o agora conforme suas instruções; "
                "caso contrário, confirme a recusa em uma frase e pergunte se pode "
                "ajudar com outro assunto."
            ]
        )
    return None


def _escopo(estado) -> str:
    return (
        "Agente de crédito com cliente autenticado. Consulta limite e score e "
        "avalia pedidos de aumento somente por ferramentas; após uma rejeição, "
        "oferece a entrevista de crédito. Pode responder com naturalidade a "
        "cumprimentos, agradecimentos e conversa breve, e explicar em termos "
        "gerais que o limite máximo depende do score e que a entrevista pode "
        "atualizar o score. Outros assuntos são transferidos para a triagem."
    )


def _pergunta_pendente(estado) -> PerguntaPendente | None:
    etapa = estado.get(ETAPA_CREDITO)
    if etapa == AGUARDANDO_NOVO_LIMITE:
        return PerguntaPendente(
            "Qual é o novo limite total desejado? Informe um valor numérico.",
            termos=("limite", "valor"),
        )
    if etapa == AGUARDANDO_DECISAO_ENTREVISTA:
        return PerguntaPendente(
            "Você deseja fazer a entrevista de crédito para tentar reajustar seu "
            "score?",
            ("Sim", "Não"),
            ("entrevista",),
        )
    if etapa == AGUARDANDO_RETENTATIVA_REANALISE:
        return PerguntaPendente(
            "Você deseja tentar novamente a análise do mesmo limite?",
            ("Sim", "Não"),
            ("novamente", "reanálise", "análise"),
        )
    return None


def _resposta_segura(estado) -> str:
    if _pergunta_pendente(estado) is not None:
        return "Desculpe, não consegui entender."
    return (
        "Desculpe, não consegui responder isso com segurança. Posso consultar "
        "seu limite ou score, ou analisar um pedido de aumento de limite."
    )


def _fatos_verificados(estado) -> dict:
    return {
        **extrair_fatos(
            estado,
            RESULTADO_CREDITO,
            (
                "limite_atual",
                "novo_limite_solicitado",
                "limite_maximo_score",
                "score_atual",
                "status",
            ),
        ),
        **extrair_fatos(estado, PENDENCIA_REANALISE, ("novo_limite_solicitado",)),
    }


PERFIL_REVISAO = PerfilRevisao(
    agente="credito",
    escopo=_escopo,
    resposta_segura=_resposta_segura,
    fatos_verificados=_fatos_verificados,
    pergunta_pendente=_pergunta_pendente,
)
revisar_resposta = criar_revisao_de_resposta(PERFIL_REVISAO)


agente_credito = Agent(
    name="credito",
    model=criar_modelo(),
    mode="chat",
    generate_content_config=configuracao_geracao(),
    description=(
        "Consulta limite e score de crédito e avalia solicitações de aumento "
        "para clientes já autenticados."
    ),
    instruction="""
Você é o agente de crédito do Banco Ágil. Fale em português do Brasil, com tom
cordial, natural e breve. O cliente já foi autenticado pela triagem. Nunca
solicite CPF ou data de nascimento e nunca receba CPF como argumento de
ferramenta.

Para consultar o limite atual, chame consultar_limite_credito; para o score,
consultar_score_credito. Para aumento, chame solicitar_aumento_limite somente
quando houver um novo limite total numericamente claro; passe exatamente o valor
informado. Se a mensagem trouxer mais de um pedido de crédito, como consultar o
limite e pedir um aumento, chame todas as ferramentas necessárias no mesmo turno.
As ferramentas são as únicas fontes de limite, score, aprovação,
rejeição e atualização: não antecipe, estime nem invente esses resultados. Não
escreva texto junto com uma chamada de ferramenta; o sistema formata o resultado.

Interprete a mensagem inteira com o histórico. Para qualquer pedido que não seja
consulta de limite ou score, aumento de limite ou entrevista de crédito, use
transfer_to_agent para triagem sem produzir texto. Isso inclui câmbio, cotações
e conversões de valores para outras moedas, mesmo quando envolverem o limite do
cliente (por exemplo, "quanto vale meu limite em libras"), e qualquer outro
produto ou assunto. A triagem encaminhará ao especialista adequado.

Você pode responder em texto, de forma breve, a cumprimentos, agradecimentos e
dúvidas gerais sobre como funciona a análise de crédito: o limite máximo depende
do score e a entrevista de crédito pode atualizar o score. Quando uma pergunta
estiver pendente, o sistema a acrescentará ao final da sua resposta; não a
repita. Depois de uma consulta ou decisão, o fluxo local perguntará sobre outra
necessidade. Quando uma solicitação for rejeitada, o fluxo local oferecerá a
entrevista e cuidará do encaminhamento.

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
    after_model_callback=revisar_resposta,
    on_model_error_callback=criar_tratamento_de_erro(PERFIL_REVISAO),
    disallow_transfer_to_peers=True,
)
