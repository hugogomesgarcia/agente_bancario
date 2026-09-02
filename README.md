# Banco Ágil

Atendimento bancário conversacional construído com Google ADK, Gemini e
Streamlit. A aplicação autentica o cliente e conduz, em uma única conversa,
fluxos de Crédito, Entrevista de Crédito e Câmbio.

## Executar no Windows

### 1. Instale o Python

Baixe o instalador em [Python para Windows](https://www.python.org/downloads/windows/).
O projeto aceita Python 3.10 ou superior e foi validado com Python 3.14.

Na primeira tela do instalador, marque **Add python.exe to PATH** antes de
selecionar **Install Now**. Ao final da instalação, feche e abra novamente o
terminal.

### 2. Abra um terminal na pasta do projeto

Você pode usar PowerShell ou Prompt de Comando (CMD). No Explorador de Arquivos,
abra a pasta do projeto, clique na barra de endereço, digite `powershell` ou
`cmd` e pressione Enter.

Se preferir clonar o repositório, instale o
[Git para Windows](https://git-scm.com/download/win) e execute:

```powershell
git clone https://github.com/hugogomesgarcia/agente_bancario.git
cd agente_bancario
```

Confirme que o Python está disponível:

```powershell
py --version
```

Se `py` não for reconhecido, tente `python --version`. Se nenhum dos dois
funcionar, reinstale o Python com a opção de adicionar ao `PATH` marcada.

### 3. Crie o ambiente virtual e instale as dependências

Os comandos abaixo chamam os executáveis da `.venv` diretamente. Não é
necessário ativar o ambiente virtual, o que também evita problemas com a
política de execução do PowerShell.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Configure as chaves

No PowerShell:

```powershell
Copy-Item .env-example .env
notepad .env
```

No CMD:

```bat
copy .env-example .env
notepad .env
```

Preencha o arquivo aberto pelo Bloco de Notas:

```dotenv
GOOGLE_API_KEY=sua-chave-gemini
GOOGLE_MODEL=gemini-3.5-flash-lite
GOOGLE_GUARD_MODEL=gemini-3.5-flash-lite
AWESOMEAPI_TOKEN=seu-token-awesomeapi
```

A chave do Gemini pode ser criada no
[Google AI Studio](https://aistudio.google.com/app/apikey). O token da API de
cotações é descrito na
[documentação da AwesomeAPI](https://docs.awesomeapi.com.br/api-de-moedas).
`GOOGLE_GUARD_MODEL` é opcional; quando ausente, o revisor usa o valor de
`GOOGLE_MODEL`.

### 5. Inicie a interface

```powershell
.\.venv\Scripts\streamlit.exe run interface.py
```

O navegador deve abrir automaticamente. Se isso não acontecer, acesse
[http://localhost:8501](http://localhost:8501). Para parar a aplicação, volte
ao terminal e pressione `Ctrl+C`.

### 6. Execute os testes

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Os testes automatizados usam diretórios temporários e não consomem chamadas
reais do Gemini ou da AwesomeAPI.

### 7. Reinicie os dados de demonstração

Pare a aplicação antes do reset. No PowerShell:

```powershell
Remove-Item -Recurse -Force csv\local
```

No CMD:

```bat
rmdir /s /q csv\local
```

Na próxima inicialização, a aplicação recriará `csv/local/` a partir dos
templates de `csv/default/`.

## Executar no Linux

Em Debian ou Ubuntu, instale o Python e as ferramentas do ambiente virtual:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git
```

Na raiz do projeto:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env-example .env
```

Edite `.env` com as mesmas variáveis mostradas na seção de Windows e inicie:

```bash
.venv/bin/streamlit run interface.py
```

Testes:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Reset dos dados de demonstração, com a aplicação parada:

```bash
rm -rf csv/local
```

## Execução Alternativa pelo ADK

Para conversar com a árvore de agentes diretamente no terminal:

Windows:

```powershell
.\.venv\Scripts\adk.exe run agentes
```

Linux:

```bash
.venv/bin/adk run agentes
```

## Visão Geral

O Banco Ágil simula o atendimento de um banco digital fictício. A Triagem é a
porta de entrada, autentica o cliente por CPF e data de nascimento e transfere a
conversa para o especialista adequado sem expor a troca na interface.

| Agente | Responsabilidade |
| --- | --- |
| Triagem | Autenticação, identificação do assunto, roteamento e encerramento |
| Crédito | Consulta de score e limite e análise de aumento de limite |
| Entrevista de Crédito | Coleta financeira, recálculo do score e retorno ao Crédito |
| Câmbio | Interpretação de pares, cotação e conversão de moedas e criptomoedas |

## Funcionalidades Implementadas

- Autenticação por CPF e data de nascimento, com validação de CPF e limite de
  três falhas por conversa.
- Transferências implícitas entre a Triagem e os três especialistas.
- Confirmação explícita antes de encerrar o atendimento.
- Consulta determinística do score e do limite do cliente autenticado.
- Solicitação de um novo limite total com normalização de valores monetários.
- Aprovação ou rejeição baseada nas faixas configuráveis de
  `score_limite.csv`.
- Registro do histórico em `solicitacoes_aumento_limite.csv`.
- Entrevista sobre renda, emprego, despesas, dependentes e dívidas.
- Atualização do score e uma única reanálise automática do pedido rejeitado.
- Cotação em tempo real pela AwesomeAPI, incluindo compra, venda e horário da
  fonte.
- Conversão de quantidades, pares inversos e taxas cruzadas por BRL.
- Perguntas de esclarecimento antes da API quando o par é incompleto ou
  ambíguo.
- Respostas rápidas para escolhas fechadas e indicador visual de digitação.
- Sessões isoladas por conversa na interface Streamlit.
- Tratamento controlado de falhas de modelo, API e armazenamento.

## Arquitetura

```text
interface.py
└── aplicacao/servico_atendimento.py
    └── Google ADK Runner
        └── agentes/agent.py: root_agent (Triagem)
            ├── agentes/credito/
            ├── agentes/entrevista_credito/
            └── agentes/cambio/

agentes/compartilhado/       estado, encerramento, valores, locks e CSV
csv/default/                 templates fictícios versionados
csv/local/                   dados mutáveis de execução, ignorados pelo Git
tests/                       testes unitários e integrações controladas
```

`agentes/agent.py` carrega o `.env`, prepara os dados locais e exporta
`root_agent`, conforme esperado pelo carregador do Google ADK. A ordem de
inicialização é intencional porque os módulos dos agentes fixam configurações e
caminhos durante a importação.

`ServicoAtendimento` isola a interface dos agentes concretos. Ele mantém uma
sessão ADK em memória por conversa, envia mensagens ao `Runner`, reúne apenas as
respostas finais, omite partes de raciocínio e reconhece o encerramento pelo
sinal estruturado `event.actions.escalate`.

### Fluxo de autenticação e triagem

1. A interface envia uma mensagem interna para obter a saudação inicial.
2. A Triagem coleta e valida o CPF localmente.
3. A data de nascimento é comparada com `clientes.csv`.
4. Após a autenticação, a intenção do cliente é encaminhada ao especialista.
5. A interface continua exibindo todos os agentes como Banco Ágil.

### Fluxo de crédito e entrevista

1. Crédito consulta o perfil usando o CPF autenticado no estado da sessão.
2. O novo limite total é comparado ao teto da faixa de score.
3. Uma aprovação atualiza o cliente e registra a decisão no histórico.
4. Uma rejeição preserva o pedido e oferece a entrevista financeira.
5. A entrevista atualiza somente o score e retorna ao Crédito.
6. O mesmo valor rejeitado é reanalisado exatamente uma vez.

### Fluxo de câmbio

1. O Gemini interpreta a mensagem inteira em uma estrutura com ativo de
   origem, destino, quantidade e evidências textuais.
2. Solicitações ambíguas geram uma pergunta sem chamada à API.
3. O código valida os códigos e consulta a AwesomeAPI com o token em
   `x-api-key`.
4. Compra, venda, par e timestamp são validados antes da resposta.
5. Pares inversos e taxas cruzadas por BRL são calculados localmente com
   `Decimal`.

## Dados CSV

### `csv/default/`

Esse diretório é a fonte versionada dos dados fictícios. Os arquivos funcionam
como templates para uma instalação limpa:

| Arquivo | Conteúdo |
| --- | --- |
| `clientes.csv` | CPF, data de nascimento, score e limite inicial |
| `score_limite.csv` | Faixas de score e limite máximo permitido |
| `solicitacoes_aumento_limite.csv` | Cabeçalho do histórico de solicitações |

A aplicação não edita os arquivos de `csv/default/`. Assim, testes manuais e
demonstrações podem alterar dados sem modificar o repositório ou os templates
originais.

### `csv/local/`

Na primeira importação de `agentes/agent.py`, cada template ausente é copiado
para `csv/local/`. A aplicação lê e grava somente essas cópias locais. Arquivos
locais existentes não são sobrescritos, portanto alterações de score, limite e
histórico sobrevivem a reinicializações.

`csv/local/` é ignorado pelo Git. Para reiniciar um roteiro de demonstração ou
teste manual, pare a aplicação e exclua o diretório conforme os comandos do
início deste README. A próxima execução o recriará a partir de `csv/default/`.

A suíte automatizada não depende desse reset: seus testes de escrita substituem
os caminhos por CSVs temporários e não alteram os dados locais do usuário.

As gravações são protegidas por um lock exclusivo multiplataforma fornecido por
`portalocker`, com timeout controlado. Arquivos temporários e substituição
atômica evitam conteúdo parcial. Aprovações que alteram `clientes.csv` e o
histórico usam também um journal para recuperação se a segunda gravação falhar.

### Cadastro de demonstração

Um fluxo completo pode ser testado com dados fictícios já incluídos:

| Campo | Valor |
| --- | --- |
| CPF | `710.483.880-50` |
| Data de nascimento | `29/07/1997` |
| Score inicial | `720` |
| Limite inicial | `R$ 5.000,00` |

Com esse perfil, um novo limite total de até `R$ 16.000,00` é permitido pela
política inicial. Um pedido de `R$ 17.000,00` demonstra rejeição, oferta de
entrevista e reanálise.

## Desafios Enfrentados

### Regras de crédito parcialmente especificadas

A especificação exige uma comparação com `score_limite.csv`, mas não fornece o
arquivo, suas colunas, faixas ou tetos. Também não define se o valor representa
um incremento ou o novo limite total, nem se uma aprovação deve alterar o
cadastro imediatamente. A implementação adotou uma política demonstrativa e
configurável: o pedido representa o novo limite total, é aprovado até o teto da
faixa e atualiza o cadastro quando aprovado.

A fórmula de score também não define como converter resultados fracionários
para o inteiro armazenado em `clientes.csv`. O cálculo usa `Decimal`, arredonda
metade para cima somente ao final e limita o resultado ao intervalo de 0 a 1000.

### Estado composto no Google ADK

Mutações internas em um dicionário aninhado não geravam deltas persistidos pelo
estado do ADK. A entrevista passou a copiar e reatribuir o rascunho completo a
cada resposta. Um teste com `Runner` cobre os cinco turnos, a persistência, a
transferência de volta e a reanálise.

### Consistência entre arquivos CSV

Uma aprovação precisa atualizar o cadastro e o histórico, mas CSV não oferece
uma transação conjunta. O projeto combina lock entre processos, temporários,
substituição atômica e journal durável. Se a segunda publicação falhar, o estado
anterior pode ser restaurado no mesmo atendimento ou na próxima inicialização.

### Ambiguidade em pedidos de câmbio

Termos como “peso” podem representar várias moedas, e frases como “moeda da
China em euro” possuem origem e destino expressos de maneiras diferentes.
Tabelas locais de palavras parciais poderiam reconhecer apenas uma parte da
mensagem. Por isso, uma única interpretação semântica considera a solicitação
completa e fornece evidências; o código pergunta antes de consultar quando a
interpretação não é segura.

## Escolhas Técnicas

### IA para linguagem, determinismo para operações

O Gemini é usado onde há ambiguidade linguística: classificação de intenção,
interpretação de respostas livres e identificação semântica de moedas. CPF,
datas, cálculos financeiros, decisões de crédito, transições de estado,
persistência e formatação de valores permanecem em código local. Essa divisão
torna os fluxos críticos reproduzíveis e testáveis sem chamadas reais ao modelo.

### Respostas baseadas em evidências

Crédito e Câmbio não permitem que o modelo invente valores ou conclusões. Textos
com score, limite, aprovação, rejeição, compra, venda ou conversão são montados
localmente a partir dos CSVs ou de uma resposta validada da AwesomeAPI. Texto
livre misturado a uma chamada de ferramenta é descartado.

### Guardrail independente na Triagem

Respostas livres da Triagem passam por um segundo Gemini, que compara o texto
proposto com as ferramentas disponíveis e as evidências reais do turno. Uma
reprovação permite uma única reescrita. Falha do revisor, JSON inválido ou uma
segunda reprovação produzem uma resposta local segura, sem exibir conteúdo não
revisado.

### Portabilidade do armazenamento local

`portalocker` abstrai o lock de arquivo entre Linux, macOS e Windows. O projeto
mantém um lock por diretório para coordenar threads e processos que cooperam com
o mesmo conjunto de CSVs, sem depender diretamente de `fcntl` ou de extensões
específicas do Windows.

### Interface desacoplada dos especialistas

A interface injeta somente `root_agent` em `ServicoAtendimento`. Respostas
rápidas são publicadas pelos agentes como metadados genéricos e retornam pelo
mesmo caminho de uma mensagem digitada. Os handoffs e os nomes internos dos
especialistas não vazam para o cliente.
