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
from agentes.compartilhado.valores import normalizar_valor_monetario
from .tools.entrevista_credito import (
    DADOS_ENTREVISTA,
    concluir_entrevista_credito,
    normalizar_numero_dependentes,
)

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
CAMPOS_MONETARIOS = {
    AGUARDANDO_RENDA: "renda_mensal",
    AGUARDANDO_DESPESAS: "despesas_fixas",
}
PERGUNTAS = {
    AGUARDANDO_RENDA: PerguntaPendente(
        "Qual é a sua renda mensal?", termos=("renda",)
    ),
    AGUARDANDO_EMPREGO: PerguntaPendente(
        "Qual é o seu tipo de emprego: formal, autônomo ou desempregado?",
        ("Formal", "Autônomo", "Desempregado"),
        ("emprego", "trabalho"),
    ),
    AGUARDANDO_DESPESAS: PerguntaPendente(
        "Qual é o total das suas despesas fixas mensais?", termos=("despesa",)
    ),
    AGUARDANDO_DEPENDENTES: PerguntaPendente(
        "Quantos dependentes você possui?", termos=("dependente",)
    ),
    AGUARDANDO_DIVIDAS: PerguntaPendente(
        "Você possui dívidas ativas? Responda sim ou não.",
        ("Sim", "Não"),
        ("dívida", "divida"),
    ),
}
INSTRUCAO_FORA_DA_PERGUNTA = (
    "Se a mensagem não responder à pergunta atual (dúvida, comentário ou "
    "conversa), não chame ferramentas: responda em uma ou duas frases; o sistema "
    "repetirá a pergunta atual. Para outro assunto fora da entrevista, use "
    "transfer_to_agent para triagem; a entrevista ficará pausada."
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


def _perguntar(callback_context: CallbackContext, etapa: str) -> LlmResponse:
    pergunta = PERGUNTAS[etapa]
    if pergunta.opcoes:
        callback_context.state[OPCOES_RESPOSTA] = list(pergunta.opcoes)
    return _resposta(pergunta.texto)


def _concluir(callback_context: CallbackContext) -> LlmResponse:
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


def _avancar(callback_context: CallbackContext, campo: str, valor) -> LlmResponse:
    dados = dict(_dados(callback_context))
    dados[campo] = valor
    callback_context.state[DADOS_ENTREVISTA] = dados
    proxima = PROXIMAS_ETAPAS[callback_context.state.get(ETAPA_ENTREVISTA)]
    if proxima == CONCLUIR_ENTREVISTA:
        return _concluir(callback_context)
    callback_context.state[ETAPA_ENTREVISTA] = proxima
    return _perguntar(callback_context, proxima)


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


def _ferramentas_retornadas(llm_request: LlmRequest) -> set[str]:
    return {
        parte.function_response.name
        for conteudo in llm_request.contents[-1:]
        for parte in conteudo.parts or []
        if parte.function_response and parte.function_response.name
    }


def _pedir_interpretacao_de_valor(llm_request: LlmRequest) -> None:
    llm_request.append_instructions(
        [
            "Se a resposta atual informar um valor financeiro em linguagem "
            "natural, extraia somente o valor que o cliente declarou e chame "
            "confirmar_valor_monetario_informado com o número normalizado. Por "
            "exemplo, sete mil reais e 7k devem ser enviados como 7000. Não estime "
            "nem invente números. " + INSTRUCAO_FORA_DA_PERGUNTA
        ]
    )


def _interpretar_campo(llm_request: LlmRequest, instrucao: str) -> None:
    llm_request.append_instructions([f"{instrucao} {INSTRUCAO_FORA_DA_PERGUNTA}"])


def _responder_retorno_de_ferramenta(
    callback_context: CallbackContext, llm_request: LlmRequest, etapa: str
) -> LlmResponse | None:
    ferramentas = _ferramentas_retornadas(llm_request)

    if "registrar_resposta_entrevista" in ferramentas:
        registro = callback_context.state.get(REGISTRO_ENTREVISTA_INTERPRETADO)
        callback_context.state[REGISTRO_ENTREVISTA_INTERPRETADO] = None
        if isinstance(registro, dict) and registro.get("sucesso"):
            return _avancar(
                callback_context, registro["campo_registrado"], registro["valor"]
            )
        return _resposta(
            "Não consegui validar essa resposta. "
            + PERGUNTAS[etapa].texto
        )

    if "confirmar_valor_monetario_informado" not in ferramentas:
        return None
    resultado = callback_context.state.get(VALOR_MONETARIO_INTERPRETADO)
    callback_context.state[VALOR_MONETARIO_INTERPRETADO] = None
    if isinstance(resultado, dict) and resultado.get("sucesso"):
        callback_context.state[TENTATIVAS_INTERPRETACAO_VALOR] = 0
        return _avancar(
            callback_context,
            CAMPOS_MONETARIOS[etapa],
            resultado["valor_normalizado"],
        )
    if int(callback_context.state.get(TENTATIVAS_INTERPRETACAO_VALOR, 0)) == 0:
        callback_context.state[TENTATIVAS_INTERPRETACAO_VALOR] = 1
        llm_request.append_instructions(
            [
                "A ferramenta rejeitou o formato enviado. Se a mensagem do "
                "cliente contiver um valor claro, chame-a novamente uma única "
                "vez com somente o número decimal normalizado, por exemplo "
                "12k deve ser 12000. Se não houver valor claro, não chame "
                "ferramentas e não estime; o sistema repetirá a pergunta."
            ]
        )
        return None
    callback_context.state[TENTATIVAS_INTERPRETACAO_VALOR] = 0
    return _resposta(
        "Informe sua renda ou despesas mensais usando um valor numérico "
        "igual ou maior que zero."
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
            "Vamos iniciar sua entrevista de crédito. "
            + PERGUNTAS[AGUARDANDO_RENDA].texto
        )
    if etapa not in PERGUNTAS:
        callback_context.state[ETAPA_ENTREVISTA] = None
        callback_context.state[DADOS_ENTREVISTA] = None
        return _resposta("Vamos reiniciar a entrevista de crédito.")

    if any(
        parte.function_response
        for conteudo in llm_request.contents[-1:]
        for parte in conteudo.parts or []
    ):
        return _responder_retorno_de_ferramenta(
            callback_context, llm_request, etapa
        )

    if etapa in CAMPOS_MONETARIOS:
        valor = normalizar_valor_monetario(texto)
        if valor is None or valor < 0:
            _pedir_interpretacao_de_valor(llm_request)
            return None
        callback_context.state[TENTATIVAS_INTERPRETACAO_VALOR] = 0
        return _avancar(callback_context, CAMPOS_MONETARIOS[etapa], f"{valor:.2f}")

    if etapa == AGUARDANDO_EMPREGO:
        emprego = {
            "formal": "formal",
            "autônomo": "autonomo",
            "autonomo": "autonomo",
            "desempregado": "desempregado",
        }.get(_normalizar_texto(texto))
        if emprego is None:
            _interpretar_campo(
                llm_request,
                "Interprete o tipo de emprego informado e chame "
                "registrar_resposta_entrevista com tipo_emprego igual a "
                "formal, autonomo ou desempregado. Não deduza uma categoria "
                "se o cliente não a informou.",
            )
            return None
        return _avancar(callback_context, "tipo_emprego", emprego)

    if etapa == AGUARDANDO_DEPENDENTES:
        dependentes = (
            normalizar_numero_dependentes(texto)
            if texto.isdigit() and len(texto) <= 4
            else None
        )
        if dependentes is None:
            _interpretar_campo(
                llm_request,
                "Interprete quantos dependentes financeiros o cliente "
                "informou e chame registrar_resposta_entrevista com "
                "numero_dependentes inteiro entre 0 e 1000. Não conte "
                "pessoas que o cliente não disse serem dependentes.",
            )
            return None
        return _avancar(callback_context, "numero_dependentes", dependentes)

    tem_dividas = {"sim": True, "não": False, "nao": False}.get(
        _normalizar_texto(texto)
    )
    if tem_dividas is None:
        _interpretar_campo(
            llm_request,
            "Interprete se o cliente declarou possuir dívidas ativas e "
            "chame registrar_resposta_entrevista com tem_dividas booleano. "
            "Não trate uma negativa sobre encerrar como resposta sobre "
            "dívidas.",
        )
        return None
    return _avancar(callback_context, "tem_dividas", tem_dividas)


def _pergunta_pendente(estado) -> PerguntaPendente | None:
    return PERGUNTAS.get(estado.get(ETAPA_ENTREVISTA))


PERFIL_REVISAO = PerfilRevisao(
    agente="entrevista_credito",
    escopo=lambda estado: (
        "Entrevista de crédito em andamento. Coleta, uma por vez, renda mensal, "
        "tipo de emprego, despesas fixas, número de dependentes e dívidas ativas "
        "para recalcular o score do cliente. Pode responder com brevidade a "
        "cumprimentos, comentários e dúvidas sobre a entrevista, como o motivo "
        "das perguntas ou quantas faltam. Não calcula, estima nem promete score, "
        "limite ou aprovação."
    ),
    resposta_segura=lambda estado: "Não consegui entender sua resposta.",
    fatos_verificados=lambda estado: extrair_fatos(
        estado,
        DADOS_ENTREVISTA,
        (
            "renda_mensal",
            "tipo_emprego",
            "despesas_fixas",
            "numero_dependentes",
            "tem_dividas",
        ),
    ),
    pergunta_pendente=_pergunta_pendente,
)


agente_entrevista_credito = Agent(
    name="entrevista_credito",
    model=criar_modelo(),
    mode="chat",
    generate_content_config=configuracao_geracao(),
    description=(
        "Conduz entrevista financeira, recalcula o score e retorna o cliente "
        "ao atendimento de crédito."
    ),
    instruction=(
        "Você conduz a entrevista de crédito do Banco Ágil em português do Brasil, "
        "com tom cordial e breve. O fluxo local faz as perguntas e altera os "
        "dados; você só interpreta respostas livres. Para texto livre de emprego, "
        "dependentes ou dívidas, interprete o campo atual e chame "
        "registrar_resposta_entrevista; ela valida antes de alterar o estado. "
        "Para valores de renda ou despesas, use "
        "confirmar_valor_monetario_informado. Se o cliente fizer uma dúvida ou "
        "comentário em vez de responder, responda brevemente sem ferramentas e "
        "sem repetir a pergunta: o sistema a acrescentará. As perguntas servem "
        "para recalcular o score; não calcule nem prometa score, limite ou "
        "aprovação. Para outro assunto fora da entrevista, transfira para "
        "triagem. Se o cliente parecer querer encerrar todo o atendimento, chame "
        "somente solicitar_confirmacao_encerramento; ela não encerra a sessão."
    ),
    tools=[
        confirmar_valor_monetario_informado,
        registrar_resposta_entrevista,
        solicitar_confirmacao_encerramento,
    ],
    before_model_callback=interceptar_entrevista,
    after_model_callback=criar_revisao_de_resposta(PERFIL_REVISAO),
    on_model_error_callback=criar_tratamento_de_erro(PERFIL_REVISAO),
    disallow_transfer_to_peers=True,
)
