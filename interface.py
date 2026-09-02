import logging

import streamlit as st

from agentes.agent import root_agent
from aplicacao.servico_atendimento import ServicoAtendimento


logger = logging.getLogger(__name__)


MENSAGEM_INICIAL = "Iniciar atendimento."
MENSAGEM_ERRO = (
    "Não foi possível processar sua mensagem agora. "
    "Por favor, tente novamente em alguns instantes."
)
CABECALHO_CHAT = """
<header class="cabecalho-banco">
    <div class="marca-banco">
        <div class="icone-banco" aria-hidden="true">
            <svg viewBox="0 0 24 24">
                <path d="M12 3 2 8v2h20V8L12 3Zm-6 9v6H3v3h18v-3h-3v-6h-2v6h-3v-6h-2v6H8v-6H6Z"/>
            </svg>
        </div>
        <div>
            <p class="nome-instituicao">Banco Ágil</p>
            <h1>Atendimento digital</h1>
        </div>
    </div>
    <div class="seguranca-atendimento">
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M17 8h-1V6a4 4 0 0 0-8 0v2H7a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-9a2 2 0 0 0-2-2Zm-7-2a2 2 0 0 1 4 0v2h-4V6Zm7 13H7v-9h10v9Z"/>
        </svg>
        <div>
            <strong>Ambiente seguro</strong>
            <span><i class="status-online"></i>Atendimento online</span>
        </div>
    </div>
</header>
"""
ROLAGEM_HISTORICO = """
<script>
const historico = document.querySelector('.st-key-historico-chat');
if (historico) {
    const distanciaDoFim = () =>
        historico.scrollHeight - historico.scrollTop - historico.clientHeight;
    const atualizarAcompanhamento = () => {
        historico.dataset.acompanharRolagem =
            distanciaDoFim() < 24 ? 'true' : 'false';
    };
    if (!historico.dataset.rolagemConfigurada) {
        historico.dataset.rolagemConfigurada = 'true';
        historico.dataset.acompanharRolagem = 'true';
        historico.addEventListener('scroll', atualizarAcompanhamento);
    }

    const agendarRolagem = () => {
        if (
            historico.dataset.acompanharRolagem === 'false' ||
            historico.rolagemPendente
        ) {
            return;
        }
        historico.rolagemPendente = true;
        requestAnimationFrame(() => requestAnimationFrame(() => {
            historico.rolagemPendente = false;
            if (historico.dataset.acompanharRolagem !== 'false') {
                historico.scrollTop = historico.scrollHeight;
            }
        }));
    };

    historico.observadorRolagem?.disconnect();
    historico.observadorRolagem = new MutationObserver(agendarRolagem);
    historico.observadorRolagem.observe(historico, {
        childList: true,
        subtree: true,
    });
    agendarRolagem();
}
</script>
"""
INDICADOR_DIGITACAO = """
<div class="indicador-digitacao" role="status" aria-label="Banco Ágil está digitando">
    <span></span><span></span><span></span>
</div>
"""


def _escapar_moeda_para_markdown(texto: str) -> str:
    return texto.replace("$", r"\$")


def _adicionar_respostas(
    mensagens: tuple[str, ...], opcoes_resposta: tuple[str, ...] = ()
) -> None:
    for indice, texto in enumerate(mensagens):
        mensagem = {"autor": "assistente", "texto": texto}
        if indice == len(mensagens) - 1 and opcoes_resposta:
            mensagem["opcoes_resposta"] = list(opcoes_resposta)
        st.session_state.mensagens.append(mensagem)


def _selecionar_opcao(indice_mensagem: int, opcao: str) -> None:
    mensagem = dict(st.session_state.mensagens[indice_mensagem])
    mensagem["opcao_selecionada"] = opcao
    st.session_state.mensagens[indice_mensagem] = mensagem
    st.session_state.resposta_rapida_selecionada = opcao


def _agendar_mensagem(texto: str) -> None:
    st.session_state.mensagens.append({"autor": "cliente", "texto": texto})
    st.session_state.mensagem_pendente = texto


def _iniciar_atendimento() -> None:
    st.session_state.servico_atendimento = ServicoAtendimento(root_agent)
    st.session_state.mensagens = []
    st.session_state.atendimento_encerrado = False
    st.session_state.pop("mensagem_pendente", None)
    st.session_state.pop("resposta_rapida_selecionada", None)

    try:
        resultado = st.session_state.servico_atendimento.enviar_mensagem(
            MENSAGEM_INICIAL
        )
    except Exception:
        logger.exception("Falha ao iniciar o atendimento")
        st.session_state.mensagens.append(
            {"autor": "assistente", "texto": MENSAGEM_ERRO}
        )
        return

    _adicionar_respostas(resultado.mensagens, resultado.opcoes_resposta)
    st.session_state.atendimento_encerrado = resultado.encerrado


def _reiniciar_atendimento() -> None:
    st.session_state.servico_atendimento.fechar()
    _iniciar_atendimento()


def _exibir_mensagem(mensagem: dict, indice: int) -> None:
    if mensagem["autor"] == "cliente":
        with st.container(key=f"mensagem-cliente-{indice}"):
            with st.chat_message("Cliente", avatar=":material/person:"):
                st.markdown(_escapar_moeda_para_markdown(mensagem["texto"]))
        return

    with st.container(key=f"mensagem-assistente-{indice}"):
        with st.chat_message("Banco Ágil", avatar=":material/account_balance:"):
            st.markdown(_escapar_moeda_para_markdown(mensagem["texto"]))
            opcao_selecionada = mensagem.get("opcao_selecionada")
            opcoes = (
                [opcao_selecionada]
                if opcao_selecionada
                else mensagem.get("opcoes_resposta", [])
            )
            if opcoes:
                with st.container(key=f"opcoes-mensagem-{indice}"):
                    colunas = st.columns(len(opcoes))
                    for indice_opcao, (coluna, opcao) in enumerate(
                        zip(colunas, opcoes)
                    ):
                        with coluna:
                            st.button(
                                opcao,
                                key=f"opcao-mensagem-{indice}-{indice_opcao}",
                                disabled=bool(opcao_selecionada),
                                on_click=_selecionar_opcao,
                                args=(indice, opcao),
                                use_container_width=True,
                            )


def _rolar_historico_se_necessario() -> None:
    st.html(ROLAGEM_HISTORICO, unsafe_allow_javascript=True)


def _exibir_indicador_digitacao() -> None:
    with st.container(key="mensagem-assistente-digitando"):
        with st.chat_message("Banco Ágil", avatar=":material/account_balance:"):
            st.markdown(INDICADOR_DIGITACAO, unsafe_allow_html=True)


def _processar_mensagem_pendente(texto: str) -> None:
    try:
        resultado = st.session_state.servico_atendimento.enviar_mensagem(texto)
    except Exception:
        logger.exception("Falha ao processar mensagem do atendimento")
        respostas = (MENSAGEM_ERRO,)
        atendimento_encerrado = False
        opcoes_resposta = ()
    else:
        respostas = resultado.mensagens
        atendimento_encerrado = resultado.encerrado
        opcoes_resposta = resultado.opcoes_resposta

    _adicionar_respostas(respostas, opcoes_resposta)
    st.session_state.atendimento_encerrado = atendimento_encerrado
    st.session_state.pop("mensagem_pendente", None)


st.set_page_config(
    page_title="Banco Ágil | Atendimento",
    page_icon=":material/account_balance:",
    layout="centered",
)

st.markdown(
    """
    <style>
    :root {
        color-scheme: light;
        --azul-banco: #0b3158;
        --azul-profundo: #071f38;
        --azul-interacao: #124a7d;
        --dourado-banco: #b4945a;
        --fundo-pagina: #e9edf0;
        --fundo-conversa: #f4f5f6;
        --borda: #cdd3d8;
        --texto: #172b3d;
    }
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: var(--fundo-pagina);
        color: var(--texto);
    }
    [data-testid="stHeader"], #MainMenu, footer,
    [data-testid="stDecoration"] {
        display: none;
    }
    .stMainBlockContainer {
        align-items: center !important;
        box-sizing: border-box !important;
        display: flex !important;
        max-width: 980px;
        min-height: 100vh !important;
        padding: 2rem 1rem;
    }
    .stMainBlockContainer > [data-testid="stVerticalBlock"] {
        width: 100%;
    }
    .st-key-janela-atendimento {
        background: #ffffff;
        border: 1px solid #bcc5cc;
        border-radius: 4px;
        border-top: 4px solid var(--dourado-banco);
        box-shadow: 0 12px 32px rgba(18, 38, 55, 0.12);
        margin: 0 auto;
        max-width: 820px;
        overflow: hidden;
    }
    .cabecalho-banco {
        align-items: center;
        background: var(--azul-banco);
        border-bottom: 1px solid var(--azul-profundo);
        color: #ffffff;
        display: flex;
        justify-content: space-between;
        margin: 0;
        padding: 1.25rem 1.5rem;
    }
    .marca-banco {
        align-items: center;
        display: flex;
        gap: 0.9rem;
    }
    .icone-banco {
        align-items: center;
        border: 1px solid rgba(255, 255, 255, 0.55);
        border-radius: 3px;
        display: flex;
        height: 42px;
        justify-content: center;
        width: 42px;
    }
    .icone-banco svg {
        fill: #ffffff;
        height: 22px;
        width: 22px;
    }
    .cabecalho-banco .nome-instituicao {
        color: #d8c49f;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        line-height: 1;
        margin: 0 0 0.35rem;
        text-transform: uppercase;
    }
    .cabecalho-banco h1 {
        color: #ffffff;
        font-size: 1.12rem;
        font-weight: 650;
        letter-spacing: 0.01em;
        line-height: 1.2;
        margin: 0;
    }
    .seguranca-atendimento {
        align-items: center;
        display: flex;
        gap: 0.6rem;
    }
    .seguranca-atendimento > svg {
        fill: #d8c49f;
        height: 19px;
        width: 19px;
    }
    .seguranca-atendimento strong,
    .seguranca-atendimento span {
        display: block;
        line-height: 1.25;
    }
    .seguranca-atendimento strong {
        color: #ffffff;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .seguranca-atendimento span {
        color: #c3cfda;
        font-size: 0.68rem;
        margin-top: 0.2rem;
    }
    .status-online {
        background: #56bb8a;
        border-radius: 50%;
        display: inline-block;
        height: 6px;
        margin-right: 0.35rem;
        vertical-align: 1px;
        width: 6px;
    }
    .st-key-historico-chat {
        background: var(--fundo-conversa);
        padding: 1.5rem;
        scrollbar-color: #aeb7be transparent;
        scrollbar-width: thin;
    }
    [data-testid="stChatMessage"] {
        align-items: flex-end !important;
        background: transparent;
        display: flex !important;
        gap: 10px !important;
        justify-content: flex-start !important;
        margin: 0;
        padding: 0;
        width: 100%;
    }
    [data-testid^="stChatMessageAvatar"] {
        align-items: center;
        background: var(--azul-banco) !important;
        border: 0;
        border-radius: 3px !important;
        color: #ffffff !important;
        display: flex !important;
        flex: 0 0 36px !important;
        height: 36px !important;
        justify-content: center;
        width: 36px !important;
    }
    [data-testid^="stChatMessageAvatar"] * {
        color: #ffffff !important;
        fill: #ffffff !important;
    }
    [data-testid="stChatMessageContent"] {
        background: #ffffff;
        border: 1px solid #d5dade;
        border-left: 3px solid var(--dourado-banco);
        border-radius: 3px;
        color: var(--texto);
        flex: 0 1 auto;
        font-size: 0.92rem;
        height: auto !important;
        line-height: 1.5;
        margin: 0 !important;
        max-width: min(75%, 560px);
        padding: 10px 13px;
        word-break: break-word;
        width: fit-content !important;
    }
    [data-testid="stChatMessageContent"] > [data-testid="stVerticalBlock"] {
        height: auto !important;
    }
    [data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] {
        margin-bottom: 0 !important;
    }
    [data-testid="stChatMessageContent"] p {
        margin: 0;
    }
    [class*="st-key-mensagem-"] {
        margin-bottom: 1rem;
    }
    [class*="st-key-mensagem-cliente-"] [data-testid="stChatMessage"] {
        flex-direction: row-reverse;
        justify-content: flex-start !important;
    }
    [class*="st-key-mensagem-cliente-"] [data-testid^="stChatMessageAvatar"] {
        background: #647381 !important;
    }
    [class*="st-key-mensagem-cliente-"] [data-testid="stChatMessageContent"] {
        background: var(--azul-interacao);
        border-color: var(--azul-interacao);
        border-left-width: 1px;
        border-radius: 3px;
        color: #ffffff;
    }
    .st-key-mensagem-assistente-digitando .indicador-digitacao {
        align-items: center;
        display: flex;
        gap: 5px;
        height: 20px;
        padding: 0 2px;
    }
    .st-key-mensagem-assistente-digitando .indicador-digitacao span {
        animation: pulso-digitacao 1.15s infinite ease-in-out;
        background: var(--azul-interacao);
        border-radius: 50%;
        height: 7px;
        opacity: 0.35;
        width: 7px;
    }
    .st-key-mensagem-assistente-digitando .indicador-digitacao span:nth-child(2) {
        animation-delay: 0.16s;
    }
    .st-key-mensagem-assistente-digitando .indicador-digitacao span:nth-child(3) {
        animation-delay: 0.32s;
    }
    @keyframes pulso-digitacao {
        0%, 60%, 100% { transform: translateY(0); opacity: 0.35; }
        30% { transform: translateY(-4px); opacity: 1; }
    }
    .st-key-area-envio {
        background: #ffffff;
        border-top: 1px solid var(--borda);
        padding: 1rem 1.5rem 1.2rem;
    }
    [class*="st-key-opcoes-mensagem-"] {
        margin-top: 0.7rem;
    }
    [class*="st-key-opcao-mensagem-"] button {
        background: #ffffff;
        border: 1px solid var(--azul-interacao);
        border-radius: 3px;
        color: var(--azul-banco);
        font-size: 0.8rem;
        font-weight: 650;
        min-height: 34px;
    }
    [class*="st-key-opcao-mensagem-"] button:hover {
        background: #edf2f6;
        border-color: var(--azul-profundo);
        color: var(--azul-profundo);
    }
    [class*="st-key-opcao-mensagem-"] button:disabled {
        background: #e4e8ed;
        border-color: #d2d8e0;
        color: #68758a;
        opacity: 1;
    }
    [data-testid="stChatInput"] {
        background: transparent;
        border: 0;
        border-radius: 0;
        box-shadow: none;
        display: block;
        min-height: 0;
    }
    [data-testid="stChatInput"] > div {
        align-items: center;
        background: #ffffff;
        border: 1px solid #aeb8c0;
        border-radius: 3px;
        min-height: 54px;
        width: 100%;
    }
    [data-testid="stChatInput"]:focus-within > div {
        border-color: var(--azul-interacao);
        box-shadow: 0 0 0 2px rgba(18, 74, 125, 0.12);
    }
    [data-testid="stChatInput"] textarea {
        align-self: center;
        color: var(--texto);
    }
    [data-testid="stChatInputSubmitButton"] {
        align-self: center;
        background: var(--azul-banco);
        border-radius: 2px;
        color: #ffffff;
        margin-right: 0.4rem;
    }
    [data-testid="stAlert"] {
        background: #edf2f6;
        border: 1px solid #cbd5dd;
        border-radius: 3px;
        color: var(--texto);
    }
    .stButton > button[kind="primary"] {
        background: var(--azul-banco);
        border-color: var(--azul-banco);
        border-radius: 3px;
        min-height: 46px;
    }
    @media (max-width: 640px) {
        .stMainBlockContainer {
            align-items: flex-start !important;
            padding: 0;
        }
        .st-key-janela-atendimento {
            border-left: 0;
            border-radius: 0;
            border-right: 0;
            max-width: none;
        }
        .cabecalho-banco {
            padding: 1rem 0.9rem;
        }
        .seguranca-atendimento strong {
            font-size: 0.7rem;
        }
        .seguranca-atendimento span {
            font-size: 0.63rem;
        }
        .st-key-historico-chat {
            padding: 1.1rem 0.8rem;
        }
        .st-key-area-envio {
            padding: 0.8rem;
        }
        [data-testid="stChatMessageContent"] {
            max-width: calc(100% - 48px);
        }
    }
    @media (max-width: 420px) {
        .seguranca-atendimento > svg,
        .seguranca-atendimento span {
            display: none;
        }
        .icone-banco {
            height: 38px;
            width: 38px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "servico_atendimento" not in st.session_state:
    _iniciar_atendimento()

texto_usuario = st.session_state.pop("resposta_rapida_selecionada", None)
if texto_usuario and "mensagem_pendente" not in st.session_state:
    _agendar_mensagem(texto_usuario)
mensagem_pendente = st.session_state.get("mensagem_pendente")

with st.container(key="janela-atendimento"):
    st.markdown(CABECALHO_CHAT, unsafe_allow_html=True)

    historico_chat = st.container(
        height=420,
        border=False,
        key="historico-chat",
    )
    with historico_chat:
        for indice_mensagem, mensagem_atual in enumerate(
            st.session_state.mensagens
        ):
            _exibir_mensagem(mensagem_atual, indice_mensagem)
        if mensagem_pendente:
            _exibir_indicador_digitacao()
        _rolar_historico_se_necessario()

    with st.container(key="area-envio"):
        if st.session_state.atendimento_encerrado:
            st.info(
                "Este atendimento foi encerrado.",
                icon=":material/check_circle:",
            )
            if st.button(
                "Iniciar novo atendimento",
                icon=":material/refresh:",
                type="primary",
                use_container_width=True,
            ):
                _reiniciar_atendimento()
                st.rerun()
        else:
            texto_digitado = st.chat_input(
                "Digite sua mensagem",
                max_chars=1000,
                submit_mode="disable",
                disabled=bool(mensagem_pendente),
            )
            if texto_digitado and not mensagem_pendente:
                _agendar_mensagem(texto_digitado)
                st.rerun()

if mensagem_pendente:
    _processar_mensagem_pendente(mensagem_pendente)
    st.rerun()
