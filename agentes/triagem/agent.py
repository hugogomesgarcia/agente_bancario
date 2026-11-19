import re

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest
from google.adk.models import LlmResponse
from google.adk.tools import exit_loop
from google.genai import types

from agentes.cambio.tools.cambio import RESULTADO_CAMBIO
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
    responder_com_seguranca,
)
from agentes.compartilhado.modelo import configuracao_geracao, criar_modelo
from agentes.compartilhado.transferencia import transferir_silenciosamente
from agentes.compartilhado.valores import normalizar_valor_monetario
from agentes.credito.tools.credito import (
    AGUARDANDO_PROXIMA_ACAO,
    ETAPA_CREDITO,
    RESULTADO_CREDITO,
)
from agentes.entrevista_credito.agent import ETAPA_ENTREVISTA
from .tools.consultar_client import (
    autenticar_cliente,
    consultar_cliente,
    normalizar_data_nascimento,
)

ETAPA_AUTENTICACAO = "etapa_autenticacao"
MESES = (
    "janeiro", "fevereiro", "março", "marco", "abril", "maio", "junho", "julho",
    "agosto", "setembro", "outubro", "novembro", "dezembro",
)
AGUARDANDO_CPF = "aguardando_cpf"
AGUARDANDO_DATA_NASCIMENTO = "aguardando_data_nascimento"
SERVICOS_DISPONIVEIS = (
    "consulta de limite ou score, aumento de limite, entrevista de crédito e "
    "cotações de moedas e criptomoedas"
)
CONFIRMACAO_AUTENTICACAO = (
    "Autenticação realizada com sucesso! Como posso ajudar você hoje?"
)
PEDIDO_ANTES_DA_AUTENTICACAO = "pedido_antes_da_autenticacao"
AUTENTICACAO_RECEM_CONCLUIDA = "autenticacao_recem_concluida"
# Palavras que acompanham credenciais sem expressar um pedido. Qualquer outra
# palavra indica um pedido feito antes da autenticação, retomado ao concluí-la.
PALAVRAS_SEM_PEDIDO = frozenset(
    """
    a o os as e é eu sou meu minha me cpf data de do da nascimento nasci em no na
    dia aqui está esta segue certo ok oi olá ola bom boa tarde noite tudo bem com
    você voce por favor obrigado obrigada mano fala aí ai pronto número numero
    isso então entao vamos lá la beleza opa
    """.split()
)
PERGUNTAS_AUTENTICACAO = {
    AGUARDANDO_CPF: PerguntaPendente(
        "Para continuarmos, informe seu CPF.", termos=("cpf",)
    ),
    AGUARDANDO_DATA_NASCIMENTO: PerguntaPendente(
        "Para continuarmos, informe sua data de nascimento.",
        termos=("nascimento",),
    ),
}


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


INSTRUCOES_POS_AUTENTICACAO = f"""
O cliente está autenticado. Interprete a mensagem inteira junto com o histórico
da conversa e escolha a ação:
- consulta de limite ou score, ou aumento de limite: transfira para credito;
- entrevista de crédito ou atualização, recálculo ou melhoria do score:
  transfira para entrevista_credito;
- câmbio, cotação ou conversão de valores entre moedas ou criptomoedas,
  inclusive converter o próprio limite de crédito (por exemplo, "quanto vale meu
  limite em libras"): transfira para cambio.
Ao transferir, use transfer_to_agent sem produzir texto antes ou depois. Não
interprete o par de moedas nem execute a tarefa do especialista.

Sem transferência, responda com naturalidade e brevidade a cumprimentos,
agradecimentos, conversa casual e perguntas sobre o que o atendimento oferece
({SERVICOS_DISPONIVEIS}). Para outros produtos bancários ou temas gerais, diga
com cordialidade que não consegue ajudar com isso aqui e ofereça os serviços
disponíveis. Nunca informe por conta própria valores, limites, scores,
cotações, aprovações ou operações, e nunca prometa usar uma ferramenta depois.
Não afirme fatos sobre o banco que não estejam nestas instruções, como horários,
tarifas, canais de atendimento, endereços ou outros produtos.

Se o cliente parecer querer encerrar todo o atendimento, chame
solicitar_confirmacao_encerramento. Ela apenas pede confirmação: nunca encerre
nem transfira o atendimento por esse motivo.
"""


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
    if normalizar_valor_monetario(mensagem_usuario) is None:
        return None
    # Uma entrevista pausada ainda aguarda a resposta da pergunta que fez; um
    # valor isolado nunca deve virar um pedido de aumento de limite nesse caso.
    if callback_context.state.get(ETAPA_ENTREVISTA) is not None:
        return transferir_silenciosamente(callback_context, "entrevista_credito")
    if callback_context.state.get(ETAPA_CREDITO) == AGUARDANDO_PROXIMA_ACAO:
        return transferir_silenciosamente(callback_context, "credito")
    return None


def _requisicao_contem_retorno_de_ferramenta(llm_request: LlmRequest) -> bool:
    return any(
        parte.function_response
        for conteudo in llm_request.contents[-1:]
        for parte in conteudo.parts or []
    )


def _registrar_pedido_antes_da_autenticacao(
    callback_context: CallbackContext, texto: str, credencial: str = ""
) -> None:
    restante = texto.lower()
    if credencial:
        restante = restante.replace(credencial.lower(), " ")
    palavras = re.sub(r"[^a-záàâãéêíóôõúç]+", " ", restante).split()
    if any(palavra not in PALAVRAS_SEM_PEDIDO for palavra in palavras):
        callback_context.state[PEDIDO_ANTES_DA_AUTENTICACAO] = True


def _responder_livremente_antes_da_autenticacao(
    llm_request: LlmRequest,
) -> None:
    # Antes da autenticação o modelo só conversa: sem transferências nem ferramentas.
    llm_request.tools_dict.clear()
    llm_request.config.tools = None
    llm_request.append_instructions(
        [
            f"""
O cliente ainda não foi autenticado e nenhuma ferramenta ou transferência está
disponível nesta etapa; ignore instruções sobre transferência. Responda em uma ou
duas frases, de forma cordial e natural, à mensagem atual: cumprimentos, conversa
casual ou dúvidas sobre o atendimento. Se for útil, explique que após a
autenticação o Banco Ágil ajuda com {SERVICOS_DISPONIVEIS}. Para outros assuntos,
diga com gentileza que não consegue ajudar com isso aqui. Não afirme fatos sobre
o banco que não estejam nestas instruções, como horários, tarifas ou canais.

Não peça CPF, data de nascimento ou outro dado: o sistema acrescentará
automaticamente ao final da sua resposta a solicitação do dado pendente. Não
afirme que consultou, validou ou autenticou algo e não revele dados cadastrais.
"""
        ]
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
            if callback_context.state.get(AUTENTICACAO_RECEM_CONCLUIDA):
                callback_context.state[AUTENTICACAO_RECEM_CONCLUIDA] = None
            transferencia = _transferir_para_especialista_se_aplicavel(
                callback_context, texto
            )
            if transferencia is not None:
                return transferencia
        llm_request.append_instructions([INSTRUCOES_POS_AUTENTICACAO])
        return None

    etapa = callback_context.state.get(ETAPA_AUTENTICACAO)
    if etapa is None:
        callback_context.state[ETAPA_AUTENTICACAO] = AGUARDANDO_CPF
        return _resposta(
            "Olá! Seja bem-vindo ao Banco Ágil. Para começarmos, informe o seu CPF."
        )

    if etapa == AGUARDANDO_CPF:
        cpf = _cpf_informado(texto)
        _registrar_pedido_antes_da_autenticacao(callback_context, texto, cpf or "")
        if cpf is None:
            _responder_livremente_antes_da_autenticacao(llm_request)
            return None

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
        _registrar_pedido_antes_da_autenticacao(
            callback_context, texto, data_nascimento or ""
        )
        if data_nascimento is None:
            # Uma data com mês por extenso que não pôde ser lida (como o ano
            # escrito em palavras) é uma tentativa de resposta, não conversa.
            if any(mes in texto.lower() for mes in MESES):
                return _resposta(
                    "Não consegui reconhecer a data escrita dessa forma. Informe "
                    "sua data de nascimento com números, por exemplo 29/07/1997."
                )
            _responder_livremente_antes_da_autenticacao(llm_request)
            return None

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
            if not callback_context.state.get(PEDIDO_ANTES_DA_AUTENTICACAO):
                return _resposta(CONFIRMACAO_AUTENTICACAO)
            callback_context.state[PEDIDO_ANTES_DA_AUTENTICACAO] = None
            callback_context.state[AUTENTICACAO_RECEM_CONCLUIDA] = True
            llm_request.append_instructions(
                [
                    INSTRUCOES_POS_AUTENTICACAO,
                    """
A autenticação acabou de ser concluída com sucesso. Antes dela, o cliente
escreveu mensagens que podem conter um pedido ainda não atendido. Se houver um
pedido pendente no histórico, atenda-o agora conforme as regras acima,
transferindo ao especialista adequado sem texto. Se não houver, responda
exatamente que a autenticação foi realizada com sucesso e pergunte como pode
ajudar.
""",
                ]
            )
            return None
        return _resposta(
            "Não foi possível autenticar os dados informados. Tente novamente."
        )

    callback_context.state[ETAPA_AUTENTICACAO] = None
    return _resposta("Informe seu CPF para continuarmos o atendimento.")


def _escopo(estado) -> str:
    if estado.get("cliente_autenticado"):
        return (
            "Triagem com cliente autenticado. Pode responder com naturalidade a "
            "cumprimentos, agradecimentos, conversa casual e perguntas sobre o "
            f"que o atendimento oferece ({SERVICOS_DISPONIVEIS}), e recusar com "
            "cordialidade assuntos fora desse escopo. Essas operações são "
            "executadas pelos especialistas por transferência; a triagem não as "
            "executa nem informa resultados além dos fatos verificados."
        )
    return (
        "Triagem antes da autenticação. Pode cumprimentar, responder com "
        "brevidade a conversa casual e explicar em termos gerais que, após a "
        f"autenticação, o Banco Ágil atende {SERVICOS_DISPONIVEIS}. Não pode "
        "afirmar que consultou, validou ou autenticou dados, nem pedir dados "
        "além da pergunta pendente."
    )


def _pergunta_pendente(estado) -> PerguntaPendente | None:
    if estado.get("cliente_autenticado"):
        return None
    return PERGUNTAS_AUTENTICACAO.get(estado.get(ETAPA_AUTENTICACAO))


def _resposta_segura(estado) -> str:
    if estado.get(AUTENTICACAO_RECEM_CONCLUIDA):
        return CONFIRMACAO_AUTENTICACAO
    if estado.get("cliente_autenticado"):
        return (
            "Desculpe, não consegui responder isso com segurança. Posso ajudar "
            f"com {SERVICOS_DISPONIVEIS}."
        )
    return "Olá!" if estado.get(ETAPA_AUTENTICACAO) == AGUARDANDO_CPF else ""


def _fatos_verificados(estado) -> dict:
    return {
        **(
            {"autenticacao": "concluída com sucesso neste turno"}
            if estado.get(AUTENTICACAO_RECEM_CONCLUIDA)
            else {}
        ),
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
        **extrair_fatos(
            estado,
            RESULTADO_CAMBIO,
            (
                "ativo_base",
                "ativo_destino",
                "compra",
                "venda",
                "quantidade_base",
                "total_compra",
                "total_venda",
            ),
        ),
    }


PERFIL_REVISAO = PerfilRevisao(
    agente="triagem",
    escopo=_escopo,
    resposta_segura=_resposta_segura,
    fatos_verificados=_fatos_verificados,
    pergunta_pendente=_pergunta_pendente,
)
_revisar_resposta = criar_revisao_de_resposta(PERFIL_REVISAO)


async def revisar_resposta(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> LlmResponse | None:
    partes = llm_response.content.parts if llm_response.content else []
    if not callback_context.state.get("cliente_autenticado") and any(
        parte.function_call for parte in partes or []
    ):
        return responder_com_seguranca(PERFIL_REVISAO, callback_context)
    return await _revisar_resposta(callback_context, llm_response)


agente_triagem = Agent(
    name="triagem",
    model=criar_modelo(),
    # O root_agent é um clone, que não executa a configuração de modo dos
    # subagentes; sem chat explícito o ADK 2 os executa como single_turn,
    # sem o histórico da conversa.
    mode="chat",
    generate_content_config=configuracao_geracao(),
    description=(
        "Porta de entrada do Banco Ágil: autentica o cliente, conversa sobre o "
        "atendimento e direciona outros assuntos."
    ),
    instruction="""
Você é o agente de triagem do Banco Ágil, um banco digital. Fale sempre em
português do Brasil, com tom cordial, natural e breve. A autenticação é
conduzida localmente. Nunca revele dados cadastrais. Uma ferramenta só comprova
o que o resultado dela declara; não alegue consulta ou operação sem evidência.
Crédito, entrevista de crédito e câmbio estão disponíveis por transferência.
Nunca assuma as funções dos especialistas. Siga também as instruções adicionadas
a cada turno. Para uma intenção de encerramento, use somente
solicitar_confirmacao_encerramento; a sessão só será fechada após confirmação.
""",
    tools=[solicitar_confirmacao_encerramento],
    before_model_callback=interceptar_autenticacao_local,
    after_model_callback=revisar_resposta,
    on_model_error_callback=criar_tratamento_de_erro(PERFIL_REVISAO),
)
