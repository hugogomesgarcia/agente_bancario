import asyncio
import csv
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from google.adk.models import LlmRequest
from google.adk.models import LlmResponse
from google.adk.models.base_llm import BaseLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agentes.credito.agent import agente_credito
from agentes.credito.tools import credito
from agentes.compartilhado.estado import OPCOES_RESPOSTA
from agentes.compartilhado.encerramento import solicitar_confirmacao_encerramento
from agentes.entrevista_credito.agent import (
    AGUARDANDO_DEPENDENTES,
    AGUARDANDO_DESPESAS,
    AGUARDANDO_DIVIDAS,
    AGUARDANDO_EMPREGO,
    AGUARDANDO_RENDA,
    ETAPA_ENTREVISTA,
    TENTATIVAS_INTERPRETACAO_VALOR,
    agente_entrevista_credito,
    confirmar_valor_monetario_informado,
    interceptar_entrevista,
    registrar_resposta_entrevista,
    restringir_resposta_livre,
)
from agentes.entrevista_credito.tools import entrevista_credito
from agentes.triagem.agent import agente_triagem


class ModeloFluxoCompletoTeste(BaseLlm):
    async def generate_content_async(self, llm_request, stream=False):
        ferramentas = set(llm_request.tools_dict)
        if "solicitar_aumento_limite" in ferramentas:
            chamada = types.FunctionCall(
                name="solicitar_aumento_limite",
                args={"novo_limite_solicitado": "8000"},
            )
        elif "registrar_resposta_entrevista" not in ferramentas:
            chamada = types.FunctionCall(
                name="transfer_to_agent", args={"agent_name": "credito"}
            )
        else:
            raise AssertionError(f"Conjunto de ferramentas inesperado: {ferramentas}")
        yield LlmResponse(
            content=types.Content(
                role="model", parts=[types.Part(function_call=chamada)]
            )
        )


class ModeloEntrevistaDiretaTeste(BaseLlm):
    async def generate_content_async(self, llm_request, stream=False):
        ferramentas = set(llm_request.tools_dict)
        if "registrar_resposta_entrevista" in ferramentas:
            raise AssertionError(f"Conjunto de ferramentas inesperado: {ferramentas}")
        chamada = types.FunctionCall(
            name="transfer_to_agent",
            args={"agent_name": "entrevista_credito"},
        )
        yield LlmResponse(
            content=types.Content(
                role="model", parts=[types.Part(function_call=chamada)]
            )
        )


class ContextoEntrevistaTeste:
    def __init__(self, *, autenticado=True, invocation_id="entrevista-1"):
        self.state = {}
        if autenticado:
            self.state.update(
                {"cliente_autenticado": True, "cpf_cliente": "71048388050"}
            )
        self.invocation_id = invocation_id
        self.user_content = None
        self.actions = SimpleNamespace(
            escalate=False,
            skip_summarization=False,
            transfer_to_agent=None,
        )


class EntrevistaCreditoTest(unittest.TestCase):
    def setUp(self):
        self.diretorio = tempfile.TemporaryDirectory()
        self.clientes = Path(self.diretorio.name) / "clientes.csv"
        self.clientes.write_text(
            'CPF,"Data de Nascimento",Score,"Limite de Crédito"\n'
            '"710.483.880-50","29/07/1997",699,5000.00\n'
            '"529.982.247-25","15/03/1988",480,1500.00\n',
            encoding="utf-8",
        )
        self.patch_clientes = patch.object(
            entrevista_credito, "CLIENTES_CSV", self.clientes
        )
        self.patch_clientes.start()
        self.contexto = ContextoEntrevistaTeste()

    def tearDown(self):
        self.patch_clientes.stop()
        self.diretorio.cleanup()

    def _linhas(self):
        with self.clientes.open("r", encoding="utf-8", newline="") as arquivo:
            return list(csv.DictReader(arquivo))

    def _requisicao(self, texto):
        self.contexto.user_content = types.Content(
            role="user", parts=[types.Part.from_text(text=texto)]
        )
        return LlmRequest(
            contents=[
                types.Content(role="user", parts=[types.Part.from_text(text=texto)])
            ]
        )

    def _requisicao_retorno_ferramenta(self, nome, resultado):
        return LlmRequest(
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=nome,
                                response=resultado,
                            )
                        )
                    ],
                )
            ]
        )

    def _dados_validos(self):
        return {
            "renda_mensal": "600.00",
            "tipo_emprego": "formal",
            "despesas_fixas": "89.00",
            "numero_dependentes": 0,
            "tem_dividas": False,
        }

    def test_formula_aplica_pesos_arredondamento_e_limites(self):
        casos = (
            (("6000", "formal", "2999", 1, False), 540),
            (("8000", "formal", "2000", 0, False), 620),
            (("1", "desempregado", "59", 3, False), 131),
            (("0", "desempregado", "0", 3, True), 0),
            (("100000", "formal", "0", 0, False), 1000),
        )
        for argumentos, esperado in casos:
            with self.subTest(argumentos=argumentos):
                self.assertEqual(
                    entrevista_credito.calcular_score_credito(*argumentos),
                    esperado,
                )

    def test_formula_recusa_dados_invalidos(self):
        for argumentos in (
            ("-1", "formal", "0", 0, False),
            ("1", "inexistente", "0", 0, False),
            ("1", "formal", "-1", 0, False),
            ("1", "formal", "0", -1, False),
            ("1", "formal", "0", 1001, False),
            ("1", "formal", "0", 1.5, False),
            ("1", "formal", "0", True, False),
            ("1", "formal", "0", 0, "não"),
        ):
            with self.subTest(argumentos=argumentos):
                with self.assertRaises(ValueError):
                    entrevista_credito.calcular_score_credito(*argumentos)

    def test_conclusao_atualiza_somente_score_do_cliente_autenticado(self):
        self.contexto.state[entrevista_credito.DADOS_ENTREVISTA] = (
            self._dados_validos()
        )

        resultado = entrevista_credito.concluir_entrevista_credito(self.contexto)

        self.assertTrue(resultado["sucesso"])
        self.assertEqual(resultado["score_atualizado"], 700)
        clientes = self._linhas()
        self.assertEqual(clientes[0]["Score"], "700")
        self.assertEqual(clientes[0]["Limite de Crédito"], "5000.00")
        self.assertEqual(clientes[1]["Score"], "480")
        self.assertEqual(
            self.contexto.state[entrevista_credito.RETORNO_ENTREVISTA],
            {"score_atualizado": 700},
        )

    def test_conclusao_e_idempotente_na_mesma_invocacao(self):
        self.contexto.state[entrevista_credito.DADOS_ENTREVISTA] = (
            self._dados_validos()
        )
        with patch(
            "agentes.entrevista_credito.tools.entrevista_credito.substituir_csv",
            wraps=entrevista_credito.substituir_csv,
        ) as substituir:
            primeiro = entrevista_credito.concluir_entrevista_credito(
                self.contexto
            )
            segundo = entrevista_credito.concluir_entrevista_credito(
                self.contexto
            )

        self.assertEqual(segundo, primeiro)
        substituir.assert_called_once()

    def test_conclusao_recusa_cliente_nao_autenticado(self):
        contexto = ContextoEntrevistaTeste(autenticado=False)
        contexto.state[entrevista_credito.DADOS_ENTREVISTA] = self._dados_validos()

        resultado = entrevista_credito.concluir_entrevista_credito(contexto)

        self.assertEqual(resultado["erro"], "cliente_nao_autenticado")
        self.assertEqual(self._linhas()[0]["Score"], "699")

    def test_conclusao_falha_controlada_com_cpf_duplicado(self):
        with self.clientes.open("a", encoding="utf-8") as arquivo:
            arquivo.write('"71048388050","29/07/1997",500,1000.00\n')
        self.contexto.state[entrevista_credito.DADOS_ENTREVISTA] = (
            self._dados_validos()
        )

        resultado = entrevista_credito.concluir_entrevista_credito(self.contexto)

        self.assertEqual(resultado["erro"], "perfil_credito_indisponivel")

    def test_conclusao_nao_sobrepoe_transacao_de_credito_pendente(self):
        (self.clientes.parent / ".transacao_credito.json").write_text(
            "{}", encoding="utf-8"
        )
        self.contexto.state[entrevista_credito.DADOS_ENTREVISTA] = (
            self._dados_validos()
        )

        resultado = entrevista_credito.concluir_entrevista_credito(self.contexto)

        self.assertEqual(resultado["erro"], "base_clientes_indisponivel")
        self.assertEqual(self._linhas()[0]["Score"], "699")

    def test_fluxo_pergunta_um_campo_por_vez_e_nao_avanca_com_invalido(self):
        resposta = interceptar_entrevista(
            self.contexto, self._requisicao("iniciar")
        )
        self.assertIn("renda mensal", resposta.content.parts[0].text)
        self.assertEqual(
            self.contexto.state[ETAPA_ENTREVISTA], AGUARDANDO_RENDA
        )

        interceptar_entrevista(self.contexto, self._requisicao("valor inválido"))
        self.assertEqual(
            self.contexto.state[ETAPA_ENTREVISTA], AGUARDANDO_RENDA
        )
        interceptar_entrevista(self.contexto, self._requisicao("600"))
        self.assertEqual(
            self.contexto.state[ETAPA_ENTREVISTA], AGUARDANDO_EMPREGO
        )
        self.assertEqual(
            self.contexto.state[OPCOES_RESPOSTA],
            ["Formal", "Autônomo", "Desempregado"],
        )
        interceptar_entrevista(self.contexto, self._requisicao("formal"))
        self.assertEqual(
            self.contexto.state[ETAPA_ENTREVISTA], AGUARDANDO_DESPESAS
        )
        self.assertIsNone(self.contexto.state[OPCOES_RESPOSTA])
        interceptar_entrevista(self.contexto, self._requisicao("89"))
        self.assertEqual(
            self.contexto.state[ETAPA_ENTREVISTA], AGUARDANDO_DEPENDENTES
        )
        interceptar_entrevista(self.contexto, self._requisicao("0"))
        self.assertEqual(
            self.contexto.state[ETAPA_ENTREVISTA], AGUARDANDO_DIVIDAS
        )
        self.assertEqual(self.contexto.state[OPCOES_RESPOSTA], ["Sim", "Não"])
        resposta = interceptar_entrevista(
            self.contexto, self._requisicao("não")
        )

        self.assertIn("score foi atualizado para 700", resposta.content.parts[0].text)
        self.assertEqual(self.contexto.actions.transfer_to_agent, "credito")
        self.assertFalse(self.contexto.actions.escalate)
        self.assertIsNone(self.contexto.state[ETAPA_ENTREVISTA])
        self.assertIsNone(self.contexto.state[OPCOES_RESPOSTA])

    def test_valor_extraido_pelo_modelo_e_validado_antes_de_avancar(self):
        interceptar_entrevista(self.contexto, self._requisicao("iniciar"))
        resultado = confirmar_valor_monetario_informado("2 mil", self.contexto)
        requisicao = self._requisicao_retorno_ferramenta(
            "confirmar_valor_monetario_informado", resultado
        )

        resposta = interceptar_entrevista(self.contexto, requisicao)

        self.assertIn("tipo de emprego", resposta.content.parts[0].text)
        self.assertEqual(
            self.contexto.state[entrevista_credito.DADOS_ENTREVISTA]["renda_mensal"],
            "2000.00",
        )

    def test_falha_de_formato_orienta_modelo_a_tentar_valor_normalizado(self):
        interceptar_entrevista(self.contexto, self._requisicao("iniciar"))
        invalido = confirmar_valor_monetario_informado("doze k", self.contexto)
        requisicao_invalida = self._requisicao_retorno_ferramenta(
            "confirmar_valor_monetario_informado", invalido
        )

        self.assertIsNone(interceptar_entrevista(self.contexto, requisicao_invalida))
        self.assertEqual(self.contexto.state[TENTATIVAS_INTERPRETACAO_VALOR], 1)

        corrigido = confirmar_valor_monetario_informado("12000", self.contexto)
        requisicao_corrigida = self._requisicao_retorno_ferramenta(
            "confirmar_valor_monetario_informado", corrigido
        )

        resposta = interceptar_entrevista(self.contexto, requisicao_corrigida)
        self.assertIn("tipo de emprego", resposta.content.parts[0].text)
        self.assertEqual(self.contexto.state[TENTATIVAS_INTERPRETACAO_VALOR], 0)
        self.assertEqual(
            self.contexto.state[entrevista_credito.DADOS_ENTREVISTA]["renda_mensal"],
            "12000.00",
        )

    def test_dependentes_diretos_respeitam_intervalo_inclusivo(self):
        for texto, esperado in (("0", 0), ("1000", 1000), ("1001", None)):
            with self.subTest(texto=texto):
                self.contexto.state[ETAPA_ENTREVISTA] = AGUARDANDO_DEPENDENTES
                self.contexto.state[entrevista_credito.DADOS_ENTREVISTA] = {}

                resposta = interceptar_entrevista(
                    self.contexto, self._requisicao(texto)
                )

                if esperado is None:
                    self.assertIsNone(resposta)
                    self.assertEqual(
                        self.contexto.state[ETAPA_ENTREVISTA],
                        AGUARDANDO_DEPENDENTES,
                    )
                    self.assertEqual(
                        self.contexto.state[entrevista_credito.DADOS_ENTREVISTA],
                        {},
                    )
                else:
                    self.assertIn("dívidas ativas", resposta.content.parts[0].text)
                    self.assertEqual(
                        self.contexto.state[entrevista_credito.DADOS_ENTREVISTA][
                            "numero_dependentes"
                        ],
                        esperado,
                    )

    def test_resposta_monetaria_direta_limpa_tentativa_de_etapa_anterior(self):
        interceptar_entrevista(self.contexto, self._requisicao("iniciar"))
        invalido = confirmar_valor_monetario_informado("sem valor", self.contexto)
        self.assertIsNone(
            interceptar_entrevista(
                self.contexto,
                self._requisicao_retorno_ferramenta(
                    "confirmar_valor_monetario_informado", invalido
                ),
            )
        )
        self.assertEqual(self.contexto.state[TENTATIVAS_INTERPRETACAO_VALOR], 1)

        interceptar_entrevista(self.contexto, self._requisicao("600"))
        self.assertEqual(self.contexto.state[TENTATIVAS_INTERPRETACAO_VALOR], 0)
        interceptar_entrevista(self.contexto, self._requisicao("formal"))

        invalido = confirmar_valor_monetario_informado("sem valor", self.contexto)
        resposta = interceptar_entrevista(
            self.contexto,
            self._requisicao_retorno_ferramenta(
                "confirmar_valor_monetario_informado", invalido
            ),
        )

        self.assertIsNone(resposta)
        self.assertEqual(self.contexto.state[TENTATIVAS_INTERPRETACAO_VALOR], 1)
        self.assertEqual(self.contexto.state[ETAPA_ENTREVISTA], AGUARDANDO_DESPESAS)

    def test_encerramento_da_entrevista_exige_confirmacao_explicita(self):
        interceptar_entrevista(self.contexto, self._requisicao("iniciar"))
        solicitar_confirmacao_encerramento(self.contexto)

        resposta = interceptar_entrevista(
            self.contexto, self._requisicao("Sim, encerrar")
        )

        self.assertIn("Atendimento encerrado", resposta.content.parts[0].text)
        self.assertTrue(self.contexto.actions.escalate)
        self.assertEqual(self._linhas()[0]["Score"], "699")

    def test_agente_expoe_somente_ferramentas_deterministicas(self):
        ferramentas = asyncio.run(agente_entrevista_credito.canonical_tools())
        self.assertEqual(
            {ferramenta.name for ferramenta in ferramentas},
            {
                "confirmar_valor_monetario_informado",
                "registrar_resposta_entrevista",
                "solicitar_confirmacao_encerramento",
            },
        )

    def test_registro_interpretado_valida_e_avanca_apenas_a_etapa_atual(self):
        self.contexto.state[ETAPA_ENTREVISTA] = AGUARDANDO_EMPREGO
        self.contexto.state[entrevista_credito.DADOS_ENTREVISTA] = {}

        resultado = registrar_resposta_entrevista(
            self.contexto, tipo_emprego="autonomo"
        )

        self.assertTrue(resultado["sucesso"])
        self.assertEqual(resultado["campo_registrado"], "tipo_emprego")
        self.assertEqual(resultado["valor"], "autonomo")
        self.assertEqual(self.contexto.state[ETAPA_ENTREVISTA], AGUARDANDO_EMPREGO)

    def test_registro_recusa_campos_invalidos_produzidos_pelo_modelo(self):
        casos = (
            (AGUARDANDO_RENDA, {"renda_mensal": "inexistente"}, "valor_invalido"),
            (
                AGUARDANDO_EMPREGO,
                {"tipo_emprego": "aposentado"},
                "tipo_emprego_invalido",
            ),
            (
                AGUARDANDO_DESPESAS,
                {"despesas_fixas": "-1"},
                "valor_invalido",
            ),
            (
                AGUARDANDO_DEPENDENTES,
                {"numero_dependentes": -1},
                "numero_dependentes_invalido",
            ),
            (
                AGUARDANDO_DEPENDENTES,
                {"numero_dependentes": 1001},
                "numero_dependentes_invalido",
            ),
            (
                AGUARDANDO_DEPENDENTES,
                {"numero_dependentes": True},
                "numero_dependentes_invalido",
            ),
            (
                AGUARDANDO_DIVIDAS,
                {"tem_dividas": "não"},
                "dividas_invalido",
            ),
        )
        for etapa, argumentos, erro in casos:
            with self.subTest(etapa=etapa, argumentos=argumentos):
                self.contexto.state[ETAPA_ENTREVISTA] = etapa
                self.contexto.state[entrevista_credito.DADOS_ENTREVISTA] = {}

                resultado = registrar_resposta_entrevista(
                    self.contexto, **argumentos
                )

                self.assertEqual(resultado, {"sucesso": False, "erro": erro})
                self.assertEqual(self.contexto.state[ETAPA_ENTREVISTA], etapa)
                self.assertEqual(
                    self.contexto.state[entrevista_credito.DADOS_ENTREVISTA], {}
                )

    def test_callback_de_saida_preserva_chamadas_e_bloqueia_texto_livre(self):
        chamada = types.Part(
            function_call=types.FunctionCall(
                name="registrar_resposta_entrevista",
                args={"tipo_emprego": "formal"},
            )
        )
        somente_chamada = LlmResponse(
            content=types.Content(role="model", parts=[chamada])
        )
        resposta_mista = LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_text(text="Seu score será alto."),
                    chamada,
                ],
            )
        )
        somente_texto = LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text="Seu score será alto.")],
            )
        )

        self.assertIsNone(restringir_resposta_livre(self.contexto, somente_chamada))
        filtrada = restringir_resposta_livre(self.contexto, resposta_mista)
        bloqueada = restringir_resposta_livre(self.contexto, somente_texto)
        self.assertEqual(filtrada.content.parts, [chamada])
        self.assertNotIn("score será alto", bloqueada.content.parts[0].text)
        self.assertIn("dado solicitado", bloqueada.content.parts[0].text)

    def test_ultima_resposta_interpretada_mantem_etapa_ate_concluir(self):
        self.contexto.state[ETAPA_ENTREVISTA] = AGUARDANDO_DIVIDAS
        self.contexto.state[entrevista_credito.DADOS_ENTREVISTA] = {
            "renda_mensal": "8000.00",
            "tipo_emprego": "formal",
            "despesas_fixas": "2000.00",
            "numero_dependentes": 0,
        }

        resultado = registrar_resposta_entrevista(self.contexto, tem_dividas=False)

        self.assertTrue(resultado["sucesso"])
        self.assertEqual(self.contexto.state[ETAPA_ENTREVISTA], AGUARDANDO_DIVIDAS)

        resposta = interceptar_entrevista(
            self.contexto,
            LlmRequest(
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                function_response=types.FunctionResponse(
                                    name="registrar_resposta_entrevista",
                                    response=resultado,
                                )
                            )
                        ],
                    )
                ]
            ),
        )

        self.assertIn("score foi atualizado", resposta.content.parts[0].text)
        self.assertEqual(self.contexto.actions.transfer_to_agent, "credito")

    def test_runner_executa_rejeicao_entrevista_e_reanalise_aprovada(self):
        faixas = Path(self.diretorio.name) / "score_limite.csv"
        solicitacoes = Path(self.diretorio.name) / "solicitacoes.csv"
        faixas.write_text(
            "score_minimo,score_maximo,limite_maximo\n"
            "0,699,5000.00\n700,1000,10000.00\n",
            encoding="utf-8",
        )
        solicitacoes.write_text(
            ",".join(credito.COLUNAS_SOLICITACAO) + "\n",
            encoding="utf-8",
        )
        modelo = ModeloFluxoCompletoTeste(model="modelo-fluxo-completo")
        credito_teste = agente_credito.clone(update={"model": modelo})
        entrevista_teste = agente_entrevista_credito.clone(
            update={"model": modelo}
        )
        raiz = agente_triagem.clone(
            update={
                "model": modelo,
                "sub_agents": [credito_teste, entrevista_teste],
            }
        )
        servico_sessao = InMemorySessionService()
        sessao = asyncio.run(
            servico_sessao.create_session(
                app_name="teste_entrevista",
                user_id="cliente",
                state={
                    "cliente_autenticado": True,
                    "cpf_cliente": "71048388050",
                },
            )
        )
        runner = Runner(
            app_name="teste_entrevista",
            agent=raiz,
            session_service=servico_sessao,
        )

        def enviar(texto):
            return list(
                runner.run(
                    user_id=sessao.user_id,
                    session_id=sessao.id,
                    new_message=types.Content(
                        role="user", parts=[types.Part.from_text(text=texto)]
                    ),
                )
            )

        with patch.object(credito, "CLIENTES_CSV", self.clientes), patch.object(
            credito, "SCORE_LIMITE_CSV", faixas
        ), patch.object(
            credito, "SOLICITACOES_CSV", solicitacoes
        ):
            eventos_rejeicao = enviar("quero aumentar meu limite para 8000")
            eventos_inicio = enviar("sim")
            enviar("600")
            enviar("formal")
            enviar("89")
            enviar("0")
            eventos_finais = enviar("não")

        sessao_final = asyncio.run(
            servico_sessao.get_session(
                app_name="teste_entrevista",
                user_id=sessao.user_id,
                session_id=sessao.id,
            )
        )
        asyncio.run(runner.close())
        with solicitacoes.open("r", encoding="utf-8", newline="") as arquivo:
            registros = list(csv.DictReader(arquivo))

        self.assertTrue(
            any(
                "foi rejeitada" in "".join(
                    parte.text or "" for parte in evento.content.parts or []
                )
                for evento in eventos_rejeicao
                if evento.content
            ),
            eventos_rejeicao,
        )
        self.assertTrue(
            any(evento.author == "entrevista_credito" for evento in eventos_inicio)
        )
        textos_finais = [
            "".join(parte.text or "" for parte in evento.content.parts or [])
            for evento in eventos_finais
            if evento.content
        ]
        self.assertTrue(
            any("score foi atualizado para 700" in texto for texto in textos_finais),
            textos_finais,
        )
        self.assertTrue(
            any("solicitação foi aprovada" in texto for texto in textos_finais),
            textos_finais,
        )
        self.assertEqual(
            [registro["status_pedido"] for registro in registros],
            ["rejeitado", "aprovado"],
        )
        self.assertEqual(self._linhas()[0]["Score"], "700")
        self.assertEqual(self._linhas()[0]["Limite de Crédito"], "8000.00")
        self.assertTrue(sessao_final.state["cliente_autenticado"])
        self.assertIsNone(
            sessao_final.state.get(entrevista_credito.DADOS_ENTREVISTA)
        )
        self.assertIsNone(sessao_final.state.get(credito.PENDENCIA_REANALISE))

    def test_runner_executa_entrevista_direta_sem_criar_solicitacao(self):
        solicitacoes = Path(self.diretorio.name) / "solicitacoes_direta.csv"
        solicitacoes.write_text(
            ",".join(credito.COLUNAS_SOLICITACAO) + "\n",
            encoding="utf-8",
        )
        modelo = ModeloEntrevistaDiretaTeste(model="modelo-entrevista-direta")
        credito_teste = agente_credito.clone(update={"model": modelo})
        entrevista_teste = agente_entrevista_credito.clone(
            update={"model": modelo}
        )
        raiz = agente_triagem.clone(
            update={
                "model": modelo,
                "sub_agents": [credito_teste, entrevista_teste],
            }
        )
        servico_sessao = InMemorySessionService()
        sessao = asyncio.run(
            servico_sessao.create_session(
                app_name="teste_entrevista_direta",
                user_id="cliente",
                state={
                    "cliente_autenticado": True,
                    "cpf_cliente": "71048388050",
                },
            )
        )
        runner = Runner(
            app_name="teste_entrevista_direta",
            agent=raiz,
            session_service=servico_sessao,
        )

        def enviar(texto):
            return list(
                runner.run(
                    user_id=sessao.user_id,
                    session_id=sessao.id,
                    new_message=types.Content(
                        role="user", parts=[types.Part.from_text(text=texto)]
                    ),
                )
            )

        with patch.object(credito, "SOLICITACOES_CSV", solicitacoes):
            eventos_inicio = enviar("quero atualizar meu score")
            enviar("600")
            enviar("formal")
            enviar("89")
            enviar("0")
            eventos_finais = enviar("não")
        asyncio.run(runner.close())

        self.assertTrue(
            any(evento.author == "entrevista_credito" for evento in eventos_inicio)
        )
        textos_finais = [
            "".join(parte.text or "" for parte in evento.content.parts or [])
            for evento in eventos_finais
            if evento.content
        ]
        self.assertTrue(
            any("score foi atualizado para 700" in texto for texto in textos_finais),
            textos_finais,
        )
        self.assertTrue(
            any("consultar seu limite atual" in texto for texto in textos_finais),
            textos_finais,
        )
        with solicitacoes.open("r", encoding="utf-8", newline="") as arquivo:
            self.assertEqual(list(csv.DictReader(arquivo)), [])


if __name__ == "__main__":
    unittest.main()
