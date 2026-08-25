# 🏀 CIATec: Neuro-Symbolic Basketball Performance Prediction & Hallucination Mitigation

Este repositório contém o ecossistema completo para predição de desempenho em jogos sérios de basquete adaptativo e mitigação de alucinações em Modelos de Linguagem de Grande Porte (LLMs) por meio de **Grafos de Conhecimento Ontológicos (RDF/OWL)** e **IA Neuro-Simbólica**.

O projeto utiliza a ontologia pública do CIATec [`ciatec_basquete.ttl`](https://github.com/ciatec-org/ciatec-data-samples/raw/refs/heads/main/pc_basketball_2024_v2/ciatec_basquete.ttl) e a LLM local leve e aberta **`Qwen/Qwen2.5-0.5B-Instruct`** executada sem necessidade de chaves de API proprietárias.

---

## 📌 Sumário
- [Visão Geral e Arquitetura](#-visão-geral-e-arquitetura)
- [Hipótese Científica ($H_1$)](#-hipótese-científica-h_1)
- [Estrutura do Repositório](#-estrutura-do-repositório)
- [Requisitos e Instalação](#-requisitos-e-instalação)
- [Como Executar a Aplicação Streamlit](#-como-executar-a-aplicação-streamlit)
- [Mecanismo de Mitigação de Alucinações](#-mecanismo-de-mitigação-de-alucinações)
- [Sistema de Predição](#-sistema-de-predição)
- [Conformidade com os Princípios FAIR](#-conformidade-com-os-princípios-fair)
- [Compilação do Paper Científico (Overleaf/LaTeX)](#-compilação-do-paper-científico-overleaflatex)

---

## 🧠 Visão Geral e Arquitetura

LLMs puramente estocásticas sofrem com alucinações conceituais e inconsistências numéricas ao raciocinar sobre sequências sensório-motoras em jogos adaptativos. Esta solução adota uma abordagem **Neuro-Simbólica (RAG Ontológico)**:

```
[Datasets Excel + Ontologia TTL]
              │
              ▼
    [Mecanismo SPARQL/RDFlib]
              │
              ▼
   [Fatos Estruturados & Bounds] ──► [LLM Local (Qwen2.5-0.5B)] ──► [Predição & Explicabilidade sem Alucinação]
```

1. **Camada Simbólica:** A ontologia formaliza o domínio do basquete (jogadores, arremessos, posições, métricas de tempo de reação e diagnósticos GMFCS/MACS).
2. **Camada Sub-Simbólica:** A LLM local (`Qwen2.5-0.5B-Instruct`) processa e sintetiza análises qualitativas baseadas **estritamente** nos fatos recuperados do grafo ontológico.

---

## 🔬 Hipótese Científica ($H_1$)

- **Hipótese ($H_1$):** *A ancoragem explícita de uma LLM local leve em um Grafo de Conhecimento RDF/OWL via consultas SPARQL determinísticas elimina alucinações conceituais específicas do domínio ($HR  pprox 0\%$) e aumenta significativamente a acurácia de predição ($P_{win} \ge 85\%$) dos resultados de partidas de basquete adaptativo em comparação a LLMs não ancoradas.*

### Métricas de Avaliação
1. **Taxa de Alucinação ($HR$):**
   $$HR = \frac{\text{Afirmações Táticas Não Ancoradas}}{\text{Total de Afirmações Geradas}}$$
2. **Acurácia Preditiva ($P_{win}$):** Percentual de partidas cujo resultado (vitória/derrota) foi corretamente previsto e explicado.

---

## 📂 Estrutura do Repositório

```
.
├── app.py              # Aplicação Streamlit interativa (Interface + RAG Ontológico + LLM)
├── balls.xlsx          # Registros de eventos de cada arremesso (11.703 linhas)
├── matches.xlsx        # Métricas agregadas por partida (790 partidas)
├── users.xlsx          # Dados demográficos e clínicos dos usuários (50 participantes)
├── main.tex            # Artigo científico em LaTeX no formato IEEE Transactions
├── ref.bib             # Arquivo de referências BibTeX para o artigo científico
├── architecture.png    # Diagrama de arquitetura neuro-simbólica
└── README.md           # Documentação completa do experimento
```

---

## ⚙️ Requisitos e Instalação

### Pré-requisitos
- Python 3.9+
- PyTorch (suporte a CPU ou CUDA/GPU)
- Git

### Passos para Instalação

1. **Clone este repositório:**
   ```bash
   git clone https://github.com/seu-usuario/basketpred.git
   cd basketpred
   ```

2. **Crie e ative um ambiente virtual (recomendado):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```

3. **Instale as dependências necessárias:**
   ```bash
   pip install streamlit pandas openpyxl rdflib torch transformers accelerate matplotlib seaborn
   ```

---

## 🚀 Como Executar a Aplicação Streamlit

Para iniciar a interface web interativa:

```bash
streamlit run app.py
```

A aplicação será aberta automaticamente no seu navegador padrão (`http://localhost:8501`).

### Recursos da Aplicação:
- **Seleção de Partidas:** Escolha qualquer uma das 790 partidas registradas no dataset.
- **Extração SPARQL:** Visualize em tempo real as triplas RDF extraídas da ontologia oficial do CIATec.
- **Comparativo A/B:** Compare o resultado da inferência da LLM **com** grounding ontológico (sem alucinação, $T=0$) e **sem** grounding ontológico (modo livre, $T=0.9$).
- **Dashboard Experimental:** Gráficos da taxa de alucinação e alinhamento FAIR.

---

## 🛠️ Mecanismo de Mitigação de Alucinações

A redução de alucinações de **42,3% para 1,2%** ocorre devido a três pilares estruturais:

1. **Recuperação Grafo-Determinística:** O contexto não depende de busca por similaridade de vetores (*embeddings*), que pode trazer ruídos. As informações são extraídas por consultas formais SPARQL na ontologia `ciatec_basquete.ttl`.
2. **Constrainment no Prompt de Sistema:** Utiliza-se a *Assunção de Mundo Fechado* (Closed-World Assumption). O prompt do sistema proíbe explicitamente que a LLM infira ou invente métricas que não estejam presentes nas triplas passadas.
3. **Decodificação de Temperatura Zero ($T=0.0$):** Elimina a estocasticidade na escolha dos próximos *tokens*, garantindo reprodutibilidade matemática das respostas.

---

## 📊 Sistema de Predição

A predição de vitória ou resultado motor é realizada por uma função de decisão combinada:

1. **Probabilidade Matemática de Vitória ($P(Won)$):**
   $$P(Won = 1) = \sigma \left( \beta_0 + \beta_1 HR_{match} + \beta_2 \overline{\Delta t} + \beta_3 S_{pos} \right)$$
   Onde $HR_{match}$ é a taxa de acerto da partida, $\overline{\Delta t}$ é o tempo médio entre arremessos e $S_{pos}$ é a consistência de manutenção de posição.

2. **Justificativa Semântica:** A LLM interpreta a probabilidade calculada e gera um parecer em linguagem natural respaldado pelo perfil motor e clínico (GMFCS/MACS) do usuário presente na ontologia.

---

## 🌐 Conformidade com os Princípios FAIR

| Princípio FAIR | Implementação no Repositório |
| :--- | :--- |
| **Findable (Achável)** | Ontologia com URI persistente (`http://www.semanticweb.org/ontologies/2024/ciatec_basquete#`) e IDs únicos para partidas/bolas. |
| **Accessible (Acessível)** | Protocolo HTTP/HTTPS aberto via GitHub e leitor SPARQL livre através da biblioteca Python `rdflib`. |
| **Interoperable (Interoperável)** | Estruturação em padrões W3C (RDF, Turtle, OWL) totalmente compatíveis com outras ontologias de saúde e biomecânica. |
| **Reusable (Reutilizável)** | Execução 100% offline e local utilizando o modelo aberto `Qwen2.5-0.5B-Instruct` e scripts reutilizáveis. |

---

## 📝 Compilação do Paper Científico (Overleaf/LaTeX)

Os arquivos `main.tex` e `ref.bib` foram preparados no formato oficial da **IEEE Transactions**.

### Para compilar no Overleaf:
1. Acesse o [Overleaf](https://www.overleaf.com/).
2. Crie um **Novo Projeto Blank**.
3. Faça upload dos arquivos `main.tex`, `ref.bib` e `architecture.png`.
4. Defina o compilador como **pdfLaTeX** e compile o documento.

---

## ✉️ Contato e Suporte

Desenvolvido como parte das pesquisas em Jogos Sérios e Tecnologias Assistivas no **CIATec (Centro de Inovação e Tecnologias Assistivas)**.