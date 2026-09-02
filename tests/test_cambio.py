import asyncio
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from google.adk.models import LlmRequest, LlmResponse
from google.adk.models.base_llm import BaseLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agentes.cambio.agent import (
    agente_cambio,
    filtrar_resposta_mista,
    interceptar_fluxo_cambio,
)
from agentes.cambio.tools import cambio
from agentes.compartilhado.encerramento import solicitar_confirmacao_encerramento
from agentes.compartilhado.estado import OPCOES_RESPOSTA
from agentes.triagem.agent import agente_triagem


class RespostaHttpTeste:
    def __init__(self, dados):
        self.dados = dados

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        if isinstance(self.dados, bytes):
            return self.dados
        return json.dumps(self.dados).encode("utf-8")


class ContextoCambioTeste:
    def __init__(self, *, autenticado=True):
        self.state = {}
        if autenticado:
            self.state["cliente_autenticado"] = True
        self.user_content = None
        self.actions = SimpleNamespace(
            escalate=False,
            skip_summarization=False,
            transfer_to_agent=None,
        )


class ModeloCambioTeste(BaseLlm):
    async def generate_content_async(self, llm_request, stream=False):
        ferramentas = set(llm_request.tools_dict)
        if "processar_solicitacao_cambio" not in ferramentas:
            chamada = types.FunctionCall(
                name="transfer_to_agent", args={"agent_name": "cambio"}
            )
        else:
            chamada = types.FunctionCall(
                name="processar_solicitacao_cambio",
                args={
                    "status": "resolvido",
                    "ativo_base": "CNY",
                    "ativo_destino": "EUR",
                    "destino_explicito": True,
                    "evidencia_base": "moeda da china",
                    "evidencia_destino": "em euro",
                },
            )
        yield LlmResponse(
            content=types.Content(
                role="model", parts=[types.Part(function_call=chamada)]
            )
        )


class CambioTest(unittest.TestCase):
    def _requisicao(self, contexto, texto, *, retorno=None):
        contexto.user_content = types.Content(
            role="user", parts=[types.Part.from_text(text=texto)]
        )
        parte = (
            types.Part(
                function_response=types.FunctionResponse(
                    name="processar_solicitacao_cambio",
                    response=retorno,
                )
            )
            if retorno is not None
            else types.Part.from_text(text=texto)
        )
        return LlmRequest(contents=[types.Content(role="user", parts=[parte])])

    def _cotacao(self, base="CNY", destino="EUR"):
        return {
            f"{base}{destino}": {
                "code": base,
                "codein": destino,
                "name": f"{base}/{destino}",
                "bid": "0.1201",
                "ask": "0.1203",
                "timestamp": "1788221575",
            }
        }

    def test_consulta_envia_token_no_header_e_preserva_par_explicito(self):
        contexto = ContextoCambioTeste()
        with patch.dict(os.environ, {"AWESOMEAPI_TOKEN": "segredo"}), patch(
            "agentes.cambio.tools.cambio.urlopen",
            return_value=RespostaHttpTeste(self._cotacao()),
        ) as abrir:
            resultado = cambio.processar_solicitacao_cambio(
                status="resolvido",
                ativo_base="CNY",
                ativo_destino="EUR",
                destino_explicito=True,
                evidencia_base="moeda da china",
                evidencia_destino="em euro",
                tool_context=contexto,
            )

        requisicao = abrir.call_args.args[0]
        self.assertTrue(resultado["sucesso"])
        self.assertEqual(resultado["ativo_destino"], "EUR")
        self.assertEqual(requisicao.full_url, cambio.URL_COTACAO.format(par="CNY-EUR"))
        self.assertEqual(requisicao.get_header("X-api-key"), "segredo")
        self.assertNotIn("segredo", requisicao.full_url)

    def test_destino_ausente_aplica_brl_somente_quando_declarado(self):
        contexto = ContextoCambioTeste()
        resultado_api = {
            "sucesso": True,
            "ativo_base": "BTC",
            "ativo_destino": "BRL",
        }
        with patch.object(cambio, "_consultar_cotacao", return_value=resultado_api):
            resultado = cambio.processar_solicitacao_cambio(
                status="resolvido",
                ativo_base="BTC",
                ativo_destino=None,
                destino_explicito=False,
                evidencia_base="bitcoin",
                tool_context=contexto,
            )

        self.assertTrue(resultado["sucesso"])
        self.assertEqual(resultado["ativo_destino"], "BRL")

    def test_destino_explicito_exige_codigo_e_evidencia(self):
        contexto = ContextoCambioTeste()
        with patch.object(cambio, "_consultar_cotacao") as consultar:
            resultado = cambio.processar_solicitacao_cambio(
                status="resolvido",
                ativo_base="CNY",
                ativo_destino=None,
                destino_explicito=True,
                evidencia_base="moeda da china",
                evidencia_destino="em euro",
                tool_context=contexto,
            )

        self.assertEqual(resultado["erro"], "ativo_destino_explicito_invalido")
        consultar.assert_not_called()

    def test_quantidade_converte_compra_e_venda_com_cotacao_validada(self):
        contexto = ContextoCambioTeste()
        cotacao = {
            "sucesso": True,
            "ativo_base": "JPY",
            "ativo_destino": "BRL",
            "nome_par": "Iene Japonês/Real Brasileiro",
            "compra": "0.03226319",
            "venda": "0.03228197",
            "consultado_em": "02/09/2026 às 08:53 UTC",
            "fonte": "AwesomeAPI",
            "derivada_par_inverso": False,
            "derivada_par_cruzado": False,
        }
        with patch.object(cambio, "_consultar_cotacao", return_value=cotacao):
            resultado = cambio.processar_solicitacao_cambio(
                status="resolvido",
                ativo_base="JPY",
                ativo_destino="BRL",
                quantidade_base=10000,
                destino_explicito=True,
                evidencia_base="ienes",
                evidencia_destino="reais",
                evidencia_quantidade="10000",
                tool_context=contexto,
            )

        self.assertEqual(resultado["quantidade_base"], "10000")
        self.assertEqual(resultado["total_compra"], "322.6319")
        self.assertEqual(resultado["total_venda"], "322.8197")

    def test_quantidade_invalida_nao_consulta_api(self):
        contexto = ContextoCambioTeste()
        with patch.object(cambio, "_consultar_cotacao") as consultar:
            resultado = cambio.processar_solicitacao_cambio(
                status="resolvido",
                ativo_base="JPY",
                ativo_destino="BRL",
                quantidade_base=-100,
                destino_explicito=True,
                evidencia_base="ienes",
                evidencia_destino="reais",
                evidencia_quantidade="-100",
                tool_context=contexto,
            )

        self.assertEqual(resultado["erro"], "quantidade_invalida")
        consultar.assert_not_called()

    def test_ambiguidade_pergunta_sem_consultar_api(self):
        contexto = ContextoCambioTeste()
        with patch.object(cambio, "_consultar_cotacao") as consultar:
            resultado = cambio.processar_solicitacao_cambio(
                status="precisa_esclarecimento",
                ativo_base=None,
                ativo_destino=None,
                destino_explicito=False,
                pergunta_esclarecimento="De qual país você quer consultar o peso?",
                tool_context=contexto,
            )

        self.assertEqual(resultado["tipo"], "esclarecimento")
        self.assertIn("qual país", resultado["pergunta"])
        consultar.assert_not_called()

    def test_par_inverso_calcula_compra_e_venda_corretamente(self):
        indisponivel = {"sucesso": False, "erro": "par_nao_disponivel"}
        inversa = {
            "sucesso": True,
            "ativo_base": "USD",
            "ativo_destino": "BRL",
            "nome_par": "USD/BRL",
            "compra": "5",
            "venda": "5.2",
            "consultado_em": "01/09/2026 às 12:00 UTC",
            "fonte": "AwesomeAPI",
            "derivada_par_inverso": False,
        }
        with patch.object(cambio, "_buscar_par", side_effect=[indisponivel, inversa]):
            resultado = cambio._consultar_cotacao("BRL", "USD")

        self.assertTrue(resultado["derivada_par_inverso"])
        self.assertEqual(resultado["compra"], "0.1923076923076923")
        self.assertEqual(resultado["venda"], "0.2")

    def test_par_sem_cotacao_direta_e_calculado_pelas_referencias_brl(self):
        indisponivel = {"sucesso": False, "erro": "par_nao_disponivel"}
        cad_brl = {
            "sucesso": True,
            "ativo_base": "CAD",
            "ativo_destino": "BRL",
            "nome_par": "CAD/BRL",
            "compra": "3.75",
            "venda": "3.77",
            "consultado_em": "01/09/2026 às 12:00 UTC",
            "fonte": "AwesomeAPI",
            "derivada_par_inverso": False,
            "derivada_par_cruzado": False,
        }
        aud_brl = {
            "sucesso": True,
            "ativo_base": "AUD",
            "ativo_destino": "BRL",
            "nome_par": "AUD/BRL",
            "compra": "3.40",
            "venda": "3.42",
            "consultado_em": "01/09/2026 às 12:01 UTC",
            "fonte": "AwesomeAPI",
            "derivada_par_inverso": False,
            "derivada_par_cruzado": False,
        }
        with patch.object(
            cambio,
            "_buscar_par_ou_inverso",
            side_effect=[indisponivel, cad_brl, aud_brl],
        ):
            resultado = cambio._consultar_cotacao("CAD", "AUD")

        self.assertTrue(resultado["sucesso"])
        self.assertTrue(resultado["derivada_par_cruzado"])
        self.assertEqual(resultado["pares_referencia"], ["CAD/BRL", "AUD/BRL"])
        self.assertEqual(resultado["compra"], "1.096491228070175")
        self.assertEqual(resultado["venda"], "1.108823529411765")

    def test_falhas_da_api_sao_controladas(self):
        casos = (
            (404, "par_nao_disponivel"),
            (401, "token_invalido"),
            (429, "limite_api_excedido"),
        )
        for status, erro_esperado in casos:
            excecao = HTTPError("url", status, "erro", {}, None)
            with self.subTest(erro=erro_esperado), patch.dict(
                os.environ, {"AWESOMEAPI_TOKEN": "segredo"}
            ), patch("agentes.cambio.tools.cambio.urlopen", side_effect=excecao):
                resultado = cambio._buscar_par("USD", "BRL")
            excecao.close()
            self.assertEqual(resultado["erro"], erro_esperado)

        with patch.dict(os.environ, {"AWESOMEAPI_TOKEN": "segredo"}), patch(
            "agentes.cambio.tools.cambio.urlopen", side_effect=URLError("offline")
        ):
            resultado = cambio._buscar_par("USD", "BRL")
        self.assertEqual(resultado["erro"], "api_indisponivel")

    def test_token_ausente_nao_faz_requisicao(self):
        with patch.dict(os.environ, {}, clear=True), patch(
            "agentes.cambio.tools.cambio.urlopen"
        ) as abrir:
            resultado = cambio._buscar_par("USD", "BRL")
        self.assertEqual(resultado["erro"], "token_nao_configurado")
        abrir.assert_not_called()

    def test_resposta_malformada_e_rejeitada(self):
        with patch.dict(os.environ, {"AWESOMEAPI_TOKEN": "segredo"}), patch(
            "agentes.cambio.tools.cambio.urlopen",
            return_value=RespostaHttpTeste({"USDBRL": {"bid": "5"}}),
        ):
            resultado = cambio._buscar_par("USD", "BRL")
        self.assertEqual(resultado["erro"], "resposta_api_invalida")

    def test_callback_formata_somente_resultado_da_ferramenta(self):
        contexto = ContextoCambioTeste()
        resultado = {
            "sucesso": True,
            "ativo_base": "CNY",
            "ativo_destino": "EUR",
            "nome_par": "Yuan Chinês/Euro",
            "compra": "0.1201",
            "venda": "0.1203",
            "consultado_em": "01/09/2026 às 12:00 UTC",
            "fonte": "AwesomeAPI",
            "derivada_par_inverso": False,
        }
        contexto.state[cambio.RESULTADO_CAMBIO] = resultado

        resposta = interceptar_fluxo_cambio(
            contexto,
            self._requisicao(contexto, "mensagem", retorno=resultado),
        )

        texto = resposta.content.parts[0].text
        self.assertIn("para 1 CNY", texto)
        self.assertIn("EUR 0,1201", texto)
        self.assertIn("AwesomeAPI", texto)
        self.assertEqual(
            contexto.state[OPCOES_RESPOSTA],
            ["Nova cotação", "Encerrar atendimento"],
        )

    def test_callback_formata_conversao_da_quantidade(self):
        contexto = ContextoCambioTeste()
        resultado = {
            "sucesso": True,
            "ativo_base": "JPY",
            "ativo_destino": "BRL",
            "nome_par": "Iene Japonês/Real Brasileiro",
            "compra": "0.03226319",
            "venda": "0.03228197",
            "quantidade_base": "10000",
            "total_compra": "322.6319",
            "total_venda": "322.8197",
            "consultado_em": "02/09/2026 às 08:53 UTC",
            "fonte": "AwesomeAPI",
            "derivada_par_inverso": False,
            "derivada_par_cruzado": False,
        }
        contexto.state[cambio.RESULTADO_CAMBIO] = resultado

        resposta = interceptar_fluxo_cambio(
            contexto,
            self._requisicao(contexto, "mensagem", retorno=resultado),
        )

        texto = resposta.content.parts[0].text
        self.assertIn("Para 10.000 JPY", texto)
        self.assertIn("BRL 322,6319", texto)
        self.assertIn("BRL 322,8197", texto)

    def test_nova_cotacao_pede_novo_par_sem_chamar_modelo(self):
        contexto = ContextoCambioTeste()
        contexto.state[cambio.ETAPA_CAMBIO] = cambio.AGUARDANDO_PROXIMA_ACAO

        resposta = interceptar_fluxo_cambio(
            contexto, self._requisicao(contexto, "Nova cotação")
        )

        self.assertIn("Quais moedas", resposta.content.parts[0].text)
        self.assertIn("quantidade", resposta.content.parts[0].text)
        self.assertEqual(
            contexto.state[cambio.ETAPA_CAMBIO], cambio.AGUARDANDO_ESCLARECIMENTO
        )

    def test_botao_encerrar_solicita_confirmacao_localmente(self):
        contexto = ContextoCambioTeste()
        contexto.state[cambio.ETAPA_CAMBIO] = cambio.AGUARDANDO_PROXIMA_ACAO

        resposta = interceptar_fluxo_cambio(
            contexto, self._requisicao(contexto, "Encerrar atendimento")
        )

        self.assertIn("deseja encerrar", resposta.content.parts[0].text)
        self.assertEqual(
            contexto.state[OPCOES_RESPOSTA], ["Sim, encerrar", "Não, continuar"]
        )

    def test_callback_pede_esclarecimento_registrado_pelo_modelo(self):
        contexto = ContextoCambioTeste()
        resultado = {
            "sucesso": True,
            "tipo": "esclarecimento",
            "pergunta": "Você quis dizer o sol peruano ou a Solana?",
        }
        contexto.state[cambio.RESULTADO_CAMBIO] = resultado

        resposta = interceptar_fluxo_cambio(
            contexto,
            self._requisicao(contexto, "mensagem", retorno=resultado),
        )

        self.assertIn("sol peruano", resposta.content.parts[0].text)
        self.assertEqual(
            contexto.state[cambio.ETAPA_CAMBIO], cambio.AGUARDANDO_ESCLARECIMENTO
        )

    def test_cliente_nao_autenticado_retorna_triagem(self):
        contexto = ContextoCambioTeste(autenticado=False)
        resposta = interceptar_fluxo_cambio(
            contexto, self._requisicao(contexto, "cotação do dólar")
        )
        self.assertIn("autenticação", resposta.content.parts[0].text)
        self.assertEqual(contexto.actions.transfer_to_agent, "triagem")

    def test_encerramento_exige_confirmacao(self):
        contexto = ContextoCambioTeste()
        solicitar_confirmacao_encerramento(contexto)
        resposta = interceptar_fluxo_cambio(
            contexto, self._requisicao(contexto, "Sim, encerrar")
        )
        self.assertIn("Atendimento encerrado", resposta.content.parts[0].text)
        self.assertTrue(contexto.actions.escalate)

    def test_resposta_livre_sobre_recomendacao_e_preservada(self):
        contexto = ContextoCambioTeste()
        candidata = LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text=(
                            "Não posso recomendar a compra de bitcoin, pois isso "
                            "depende do seu perfil e dos riscos que aceita."
                        )
                    )
                ],
            )
        )

        resposta = filtrar_resposta_mista(contexto, candidata)

        self.assertIsNone(resposta)

    def test_texto_misturado_a_chamada_de_cotacao_e_removido(self):
        contexto = ContextoCambioTeste()
        chamada = types.Part(
            function_call=types.FunctionCall(
                name="processar_solicitacao_cambio",
                args={
                    "status": "resolvido",
                    "ativo_base": "BTC",
                    "destino_explicito": False,
                    "evidencia_base": "bitcoin",
                },
            )
        )
        resposta = filtrar_resposta_mista(
            contexto,
            LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part.from_text(text="Bitcoin está em alta."),
                        chamada,
                    ],
                )
            ),
        )
        self.assertEqual(resposta.content.parts, [chamada])

    def test_novo_turno_limpa_resultado_anterior_antes_do_modelo(self):
        contexto = ContextoCambioTeste()
        contexto.state[cambio.RESULTADO_CAMBIO] = {
            "sucesso": True,
            "ativo_base": "USD",
            "ativo_destino": "BRL",
        }
        contexto.state[cambio.ETAPA_CAMBIO] = cambio.AGUARDANDO_PROXIMA_ACAO

        resposta = interceptar_fluxo_cambio(
            contexto, self._requisicao(contexto, "e em euro?")
        )

        self.assertIsNone(resposta)
        self.assertIsNone(contexto.state[cambio.RESULTADO_CAMBIO])

    def test_agente_expoe_interpretacao_e_confirmacao(self):
        ferramentas = asyncio.run(agente_cambio.canonical_tools())
        self.assertEqual(
            {ferramenta.name for ferramenta in ferramentas},
            {"processar_solicitacao_cambio", "solicitar_confirmacao_encerramento"},
        )
        self.assertIn("moeda da China em euro", agente_cambio.instruction)
        self.assertIn("quantidade_base 10000", agente_cambio.instruction)

    def test_runner_transfere_e_preserva_par_interpretado_pelo_modelo(self):
        modelo = ModeloCambioTeste(model="modelo-cambio-teste")
        cambio_teste = agente_cambio.clone(update={"model": modelo})
        raiz = agente_triagem.clone(
            update={"model": modelo, "sub_agents": [cambio_teste]}
        )
        sessoes = InMemorySessionService()
        sessao = asyncio.run(
            sessoes.create_session(
                app_name="teste_cambio",
                user_id="cliente",
                state={"cliente_autenticado": True},
            )
        )
        runner = Runner(
            app_name="teste_cambio", agent=raiz, session_service=sessoes
        )
        cotacao = {
            "sucesso": True,
            "ativo_base": "CNY",
            "ativo_destino": "EUR",
            "nome_par": "Yuan Chinês/Euro",
            "compra": "0.1201",
            "venda": "0.1203",
            "consultado_em": "01/09/2026 às 12:00 UTC",
            "fonte": "AwesomeAPI",
            "derivada_par_inverso": False,
        }
        with patch.object(cambio, "_consultar_cotacao", return_value=cotacao):
            eventos = list(
                runner.run(
                    user_id=sessao.user_id,
                    session_id=sessao.id,
                    new_message=types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(
                                text="quero saber quanto tá a moeda da china em euro"
                            )
                        ],
                    ),
                )
            )
        asyncio.run(runner.close())

        textos = [
            "".join(parte.text or "" for parte in evento.content.parts or [])
            for evento in eventos
            if evento.content and evento.is_final_response()
        ]
        self.assertTrue(any("para 1 CNY" in texto for texto in textos), textos)
        self.assertTrue(any("EUR 0,1201" in texto for texto in textos), textos)
        self.assertTrue(any(evento.author == "cambio" for evento in eventos))


if __name__ == "__main__":
    unittest.main()
