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
    <div class="icone-banco" aria-hidden="true">
        <svg viewBox="0 0 24 24">
            <path d="M12 3 2 8v2h20V8L12 3Zm-6 9v6H3v3h18v-3h-3v-6h-2v6h-3v-6h-2v6H8v-6H6Z"/>
        </svg>
    </div>
    <div>
        <h1>Banco Ágil</h1>
        <p><span class="status-online"></span>Atendimento online</p>
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
    const conteudo = historico.querySelector(
        ':scope > [data-testid="stVerticalBlock"]'
    );
    if (conteudo) {
        historico.observadorRolagem = new ResizeObserver(agendarRolagem);
        historico.observadorRolagem.observe(conteudo);
    }
    agendarRolagem();
}
</script>
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


def _iniciar_atendimento() -> None:
    st.session_state.servico_atendimento = ServicoAtendimento(root_agent)
    st.session_state.mensagens = []
    st.session_state.atendimento_encerrado = False

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
    }
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background:
            radial-gradient(circle at 15% 10%, #123c69 0, transparent 34%),
            radial-gradient(circle at 85% 90%, #0b4350 0, transparent 30%),
            #061a35;
    }
    [data-testid="stHeader"], #MainMenu, footer,
    [data-testid="stDecoration"] {
        display: none;
    }
    .stMainBlockContainer {
        align-items: center !important;
        box-sizing: border-box !important;
        display: flex !important;
        max-width: 940px;
        min-height: 100vh !important;
        padding: 1.5rem 1rem;
    }
    .stMainBlockContainer > [data-testid="stVerticalBlock"] {
        width: 100%;
    }
    .st-key-janela-atendimento {
        background: #f7f9fc;
        border: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 30px;
        box-shadow: 0 30px 80px rgba(0, 8, 24, 0.4);
        margin: 0 auto;
        max-width: 760px;
        overflow: hidden;
    }
    .cabecalho-banco {
        align-items: center;
        background: #ffffff;
        border-bottom: 1px solid #e3e8ef;
        display: flex;
        gap: 0.85rem;
        margin: 0;
        padding: 1.15rem 1.4rem;
    }
    .icone-banco {
        align-items: center;
        background: linear-gradient(145deg, #173f91, #08275e);
        border-radius: 50%;
        box-shadow: 0 8px 20px rgba(10, 44, 102, 0.2);
        display: flex;
        height: 46px;
        justify-content: center;
        width: 46px;
    }
    .icone-banco svg {
        fill: #ffffff;
        height: 23px;
        width: 23px;
    }
    .cabecalho-banco h1 {
        color: #102653;
        font-size: 1.08rem;
        font-weight: 750;
        letter-spacing: -0.01em;
        line-height: 1.2;
        margin: 0 0 0.3rem;
    }
    .cabecalho-banco p {
        align-items: center;
        color: #68758a;
        display: flex;
        font-size: 0.78rem;
        gap: 0.4rem;
        margin: 0;
    }
    .status-online {
        background: #23b77d;
        border-radius: 50%;
        box-shadow: 0 0 0 3px rgba(35, 183, 125, 0.13);
        height: 6px;
        width: 6px;
    }
    .st-key-historico-chat {
        background:
            linear-gradient(rgba(247, 249, 252, 0.88), rgba(247, 249, 252, 0.88)),
            radial-gradient(circle at 20% 20%, #dce8ff, transparent 35%),
            radial-gradient(circle at 80% 80%, #d7f2e7, transparent 32%);
        padding: 1.4rem 1.5rem;
        scrollbar-color: #b9c5d5 transparent;
        scrollbar-width: thin;
    }
    [data-testid="stChatMessage"] {
        align-items: flex-end !important;
        background: transparent;
        display: flex !important;
        gap: 12px !important;
        justify-content: flex-start !important;
        margin: 0;
        padding: 0;
        width: 100%;
    }
    [data-testid^="stChatMessageAvatar"] {
        align-items: center;
        background: #163d8d !important;
        border: 3px solid #ffffff;
        border-radius: 50% !important;
        box-shadow: 0 5px 14px rgba(20, 43, 95, 0.16);
        color: #ffffff !important;
        display: flex !important;
        flex: 0 0 40px !important;
        height: 40px !important;
        justify-content: center;
        width: 40px !important;
    }
    [data-testid^="stChatMessageAvatar"] * {
        color: #ffffff !important;
        fill: #ffffff !important;
    }
    [data-testid="stChatMessageContent"] {
        background: #e7eaff;
        border: 1px solid #d8ddfa;
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(25, 51, 105, 0.07);
        color: #102653;
        flex: 0 1 auto;
        font-size: 0.94rem;
        height: auto !important;
        line-height: 1.55;
        margin: 0 !important;
        max-width: min(75%, 560px);
        padding: 9px 14px;
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
        background: #08775d !important;
    }
    [class*="st-key-mensagem-cliente-"] [data-testid="stChatMessageContent"] {
        background: #d8f3e7;
        border-color: #c5e9da;
        border-radius: 12px;
        color: #123c35;
    }
    .st-key-area-envio {
        background: #ffffff;
        border-top: 1px solid #e3e8ef;
        padding: 1rem 1.3rem 1.2rem;
    }
    [class*="st-key-opcoes-mensagem-"] {
        margin-top: 0.7rem;
    }
    [class*="st-key-opcao-mensagem-"] button {
        background: #ffffff;
        border: 1px solid #5872bd;
        border-radius: 9px;
        color: #173f91;
        font-size: 0.8rem;
        font-weight: 650;
        min-height: 34px;
    }
    [class*="st-key-opcao-mensagem-"] button:hover {
        background: #edf3ff;
        border-color: #173f91;
        color: #102653;
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
        background: #f7f9fc;
        border: 1px solid #cbd4e0;
        border-radius: 18px;
        box-shadow: 0 6px 18px rgba(17, 42, 84, 0.07);
        min-height: 58px;
        width: 100%;
    }
    [data-testid="stChatInput"]:focus-within > div {
        border-color: #5872bd;
        box-shadow: 0 0 0 1px rgba(36, 70, 216, 0.12);
    }
    [data-testid="stChatInput"] textarea {
        align-self: center;
        color: #102653;
    }
    [data-testid="stChatInputSubmitButton"] {
        align-self: center;
        background: #173f91;
        border-radius: 13px;
        color: #ffffff;
        margin-right: 0.4rem;
    }
    [data-testid="stAlert"] {
        background: #edf3ff;
        border: 1px solid #d4e0f5;
        border-radius: 16px;
        color: #102653;
    }
    .stButton > button[kind="primary"] {
        background: #173f91;
        border-color: #173f91;
        border-radius: 15px;
        min-height: 46px;
    }
    @media (max-width: 640px) {
        .stMainBlockContainer {
            align-items: flex-start !important;
            padding: 0.7rem 0.55rem;
        }
        .st-key-janela-atendimento {
            border-radius: 22px;
        }
        .cabecalho-banco {
            padding: 1rem;
        }
        .st-key-historico-chat {
            padding: 1.1rem 0.85rem;
        }
        .st-key-area-envio {
            padding: 0.8rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "servico_atendimento" not in st.session_state:
    _iniciar_atendimento()

texto_usuario = st.session_state.pop("resposta_rapida_selecionada", None)
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
            )
            if texto_digitado:
                texto_usuario = texto_digitado

if texto_usuario:
    mensagem_usuario = {"autor": "cliente", "texto": texto_usuario}
    st.session_state.mensagens.append(mensagem_usuario)
    try:
        resultado = st.session_state.servico_atendimento.enviar_mensagem(
            texto_usuario
        )
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
    st.rerun()
