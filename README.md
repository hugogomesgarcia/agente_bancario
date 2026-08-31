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
estado sobreviva a reinicializações da aplicação. A única exceção é a migração
de políticas de limite anteriormente distribuídas pelo projeto: elas são
atualizadas somente quando o conteúdo local ainda corresponde exatamente a uma
versão conhecida, preservando tabelas customizadas.

As gravações de Crédito usam lock entre processos, arquivos temporários e um
journal de recuperação para reduzir o risco de inconsistência entre o cadastro
do cliente e o histórico de solicitações.

Os cadastros incluídos são fictícios. Um fluxo de demonstração pode usar:

- CPF: `710.483.880-50`
- Data de nascimento: `29/07/1997`
- Score inicial: `720`
- Limite inicial: `R$ 5.000,00`

Para esse cadastro, cujo score inicial é `720`, novos limites acima do atual e
de até `R$ 16.000,00` são aprovados. `R$ 17.000,00` demonstra uma rejeição
seguida da oferta de entrevista.

A política demonstrativa usa faixas menores a partir do limite de `R$ 10 mil`
para evitar saltos grandes entre perfis próximos e mantém exposição máxima de
`R$ 20 mil`:

| Score | Limite máximo |
| --- | ---: |
| 0–299 | R$ 1.000 |
| 300–399 | R$ 2.500 |
| 400–499 | R$ 5.000 |
| 500–549 | R$ 7.500 |
| 550–574 | R$ 10.000 |
| 575–599 | R$ 11.000 |
| 600–624 | R$ 12.000 |
| 625–649 | R$ 13.000 |
| 650–674 | R$ 14.000 |
| 675–699 | R$ 15.000 |
| 700–724 | R$ 16.000 |
| 725–749 | R$ 17.000 |
| 750–774 | R$ 18.000 |
| 775–799 | R$ 19.000 |
| 800–1000 | R$ 20.000 |

Com renda de `R$ 12.000`, emprego formal, despesas de `R$ 2.000`, nenhum
dependente e nenhuma dívida, a fórmula especificada produz score `680`. Nessa
faixa, um novo limite total de `R$ 13.000` é aprovado e o teto é `R$ 15.000`.

## Desafios Enfrentados

A fórmula fornecida combina a razão entre renda e despesas com emprego,
dependentes e dívidas, mas não preserva a renda absoluta. Por isso, duas pessoas
com rendas diferentes e a mesma proporção de despesas podem obter scores
semelhantes. Sem alterar a fórmula, a política foi calibrada com perfis
representativos, faixas menores e um teto conservador; os testes registram tanto
o cenário esperado quanto os principais fatores de redução.

A evolução da política também precisava alcançar instalações existentes sem
apagar ajustes deliberados. A inicialização migra apenas as versões anteriores
reconhecidas por conteúdo exato e deixa qualquer outra tabela local intacta.

## Escolhas Técnicas

A fórmula de score permanece isolada e inalterada. A conversão entre score e
limite fica no CSV configurável, enquanto aprovação, persistência e mensagens
são determinísticas. Em uma rejeição, a resposta informa o score utilizado e o
teto da faixa, tornando a decisão auditável sem delegar valores financeiros ao
modelo.

No chat, a rolagem automática acompanha o término do layout das novas mensagens
e respostas rápidas. Se o cliente rolar o histórico para cima, a posição de
leitura é preservada até que ele retorne ao final da conversa.

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
- Como a fórmula considera a razão entre renda e despesas, as faixas não
  garantem proporcionalidade perfeita entre renda absoluta e limite disponível.
