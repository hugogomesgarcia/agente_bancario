# Banco Ágil

## Visão Geral

Aplicação de atendimento bancário conversacional construída com Google ADK,
Gemini e Streamlit. O fluxo autentica o cliente, identifica sua necessidade e
transfere o atendimento entre agentes especializados sem expor a troca na
interface.

## Escopo

| Agente | Responsabilidade | Status |
| --- | --- | --- |
| Triagem | Autenticação por CPF e data de nascimento e roteamento | Implementado |
| Crédito | Consulta de limite e score e análise de aumento | Implementado |
| Entrevista de Crédito | Coleta financeira, recálculo de score e retorno ao Crédito | Implementado |
| Câmbio | Consulta de cotações | Não implementado |

Solicitações de câmbio são reconhecidas, mas não produzem cotações nem um
encaminhamento fictício.

## Funcionalidades

- Autenticação por CPF e data de nascimento com limite de três falhas.
- Confirmação explícita antes de encerrar uma conversa.
- Encerramento disponível durante autenticação, Crédito ou Entrevista.
- Transferências implícitas entre Triagem, Crédito e Entrevista de Crédito.
- Consulta do limite e do score do cliente autenticado.
- Solicitação de um novo limite total com validação monetária determinística.
- Aprovação ou rejeição conforme as faixas de `score_limite.csv`.
- Registro de cada análise em `solicitacoes_aumento_limite.csv`.
- Entrevista estruturada sobre renda, emprego, despesas, dependentes e dívidas.
- Persistência do novo score e reanálise automática de um pedido rejeitado.
- Respostas rápidas para escolhas fechadas na interface.
- Isolamento das conversas em sessões mantidas somente em memória.
- Fallbacks controlados para falhas de modelo e de armazenamento.

## Arquitetura

```text
agentes/
├── agent.py                    # inicialização e composição da árvore ADK
├── compartilhado/              # estado, encerramento, moeda e acesso a CSV
├── triagem/                    # autenticação, roteamento e guardrail de saída
├── credito/                    # fluxo e ferramentas de crédito
└── entrevista_credito/         # entrevista, cálculo e persistência do score
aplicacao/
└── servico_atendimento.py      # adaptação dos eventos do Runner
csv/
├── default/                    # dados fictícios usados como seed
└── local/                      # dados mutáveis gerados em execução
tests/                          # testes determinísticos e de integração ADK
interface.py                    # interface Streamlit
```

`agentes/agent.py` exporta `root_agent`, esperado pelo carregador do ADK. Na
árvore de execução, Triagem é a raiz e Crédito e Entrevista de Crédito são seus
subagentes.

`ServicoAtendimento` mantém a interface independente dos agentes concretos. Ele
envia mensagens ao `Runner`, reúne apenas respostas finais, remove partes de
raciocínio e reconhece o encerramento pelo sinal estruturado
`event.actions.escalate`.

Validações objetivas, cálculos e transições de estado são locais. O modelo é
usado somente para classificação e interpretação de linguagem livre. Respostas
livres da Triagem passam por uma revisão semântica separada antes de serem
exibidas.

## Dados

Na primeira importação, os arquivos ausentes de `csv/default/` são copiados para
`csv/local/`. Arquivos locais existentes não são sobrescritos, permitindo que o
estado sobreviva a reinicializações da aplicação.

As gravações de Crédito usam lock entre processos, arquivos temporários e um
journal de recuperação para reduzir o risco de inconsistência entre o cadastro
do cliente e o histórico de solicitações.

Os cadastros incluídos são fictícios. Um fluxo de demonstração pode usar:

- CPF: `710.483.880-50`
- Data de nascimento: `29/07/1997`
- Score inicial: `720`
- Limite inicial: `R$ 5.000,00`

Para esse cadastro, `R$ 8.000,00` demonstra uma aprovação e `R$ 12.000,00`
demonstra uma rejeição seguida da oferta de entrevista.

## Tutorial de Execução e Testes

### Configuração

O projeto foi validado com Python 3.14.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env-example .env
```

Configure a chave no arquivo `.env`:

```dotenv
GOOGLE_API_KEY=sua-chave
GOOGLE_MODEL=gemini-3.5-flash-lite
GOOGLE_GUARD_MODEL=gemini-3.5-flash-lite
```

`GOOGLE_GUARD_MODEL` é opcional. Quando ausente, o guardrail usa o mesmo modelo
definido em `GOOGLE_MODEL`.

### Execução

Interface web:

```bash
.venv/bin/streamlit run interface.py
```

Runner interativo do ADK:

```bash
.venv/bin/adk run agentes
```

### Testes

```bash
.venv/bin/python -m unittest discover -s tests -v
```

A suíte usa CSVs temporários para operações de escrita e modelos controlados
para os testes de integração. Ela não requer chamadas reais ao Gemini.

## Limitações

- O agente de Câmbio e a consulta a uma API externa de cotações ainda não foram
  implementados.
- O armazenamento em CSV é adequado ao escopo demonstrativo, mas não substitui
  um banco transacional em produção.
- As sessões da interface ficam somente em memória e são descartadas ao
  reiniciar a aplicação.
- O limite de tentativas de autenticação vale para a conversa em memória; um
  novo atendimento inicia uma nova sessão.
