import asyncio
from dataclasses import dataclass
from uuid import uuid4

from google.adk.agents import BaseAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agentes.compartilhado.estado import OPCOES_RESPOSTA


@dataclass(frozen=True)
class ResultadoAtendimento:
    mensagens: tuple[str, ...]
    encerrado: bool
    opcoes_resposta: tuple[str, ...] = ()


class ServicoAtendimento:
    """Executa uma conversa com qualquer agente raiz do Banco Ágil."""

    def __init__(
        self,
        agente: BaseAgent,
        nome_aplicacao: str = "banco_agil",
    ) -> None:
        self._identificador_usuario = uuid4().hex
        self._identificador_sessao = uuid4().hex
        self._servico_sessao = InMemorySessionService()
        self._runner = Runner(
            app_name=nome_aplicacao,
            agent=agente,
            session_service=self._servico_sessao,
            auto_create_session=True,
        )

    def enviar_mensagem(self, texto: str) -> ResultadoAtendimento:
        texto = texto.strip()
        if not texto:
            raise ValueError("A mensagem não pode estar vazia.")

        mensagem = types.Content(
            role="user", parts=[types.Part.from_text(text=texto)]
        )
        respostas = []
        encerrado = False
        opcoes_resposta: tuple[str, ...] = ()

        for evento in self._runner.run(
            user_id=self._identificador_usuario,
            session_id=self._identificador_sessao,
            new_message=mensagem,
        ):
            if evento.actions and evento.actions.escalate:
                encerrado = True
            delta_estado = getattr(evento.actions, "state_delta", {})
            if OPCOES_RESPOSTA in delta_estado:
                opcoes = delta_estado[OPCOES_RESPOSTA]
                opcoes_resposta = (
                    tuple(str(opcao) for opcao in opcoes)
                    if isinstance(opcoes, (list, tuple))
                    else ()
                )

            if not evento.is_final_response() or not evento.content:
                continue

            resposta = "".join(
                parte.text or ""
                for parte in evento.content.parts or []
                if not parte.thought
            ).strip()
            if resposta:
                respostas.append(resposta)

        return ResultadoAtendimento(
            tuple(respostas), encerrado, opcoes_resposta
        )

    def fechar(self) -> None:
        asyncio.run(self._runner.close())
