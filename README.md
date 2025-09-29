# 🎯 Roteirizador de Entrevistas – Match de Vagas

## 📌 Sobre o Projeto

Este projeto foi desenvolvido como parte do Datathon POSTECH – Fase 5, com o desafio de aplicar Inteligência Artificial em recrutamento e seleção.
A solução é um aplicativo em Streamlit que ajuda recrutadores a encontrar o candidato ideal para cada vaga, considerando:
- Similaridade entre descrição de vagas e perfil técnico dos candidatos
- Senioridade
- Nível de idiomas (inglês e espanhol)
- Localização
- Tipo de contrato

Além de sugerir os melhores matches, o sistema armazena candidatos sem vaga compatível para futuras oportunidades.

---

## ⚡ Principais Funcionalidades

- Upload da base de vagas no formato vagas.json
- Formulário simples para registrar candidatos
- Cálculo de match com múltiplos critérios (texto, senioridade, idiomas, localização, contrato)
- Ajuste de pesos para calibrar a relevância de cada fator
- Visualização dos resultados com scores e detalhamento por vaga
- Salvamento automático de candidatos sem match atual

---

## 🛠️ Tecnologias Utilizadas

- Python 3.10+
- Streamlit
- scikit-learn
- pandas
- numpy
- unidecode
- matplotlib

---

## 📂 Estrutura dos Dados

A aplicação utiliza apenas um arquivo de entrada:
- vagas.json → contém as informações das vagas abertas, incluindo:
  - Título da vaga
  - Área de atuação
  - Principais atividades
  - Competências técnicas e comportamentais
  - Localização (cidade, estado, país)
  - Senioridade exigida
  - Requisitos de inglês e espanhol
  - Tipo de contratação

Esse arquivo deve estar na raiz do repositório e ser carregado pela aplicação no Streamlit.

---

## ▶️ Como Executar Localmente

1. Clone o repositório:
```bash
git clone https://github.com/seuusuario/seurepositorio.git
cd seurepositorio
```

2. Crie e ative um ambiente virtual (opcional, mas recomendado):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Rode a aplicação:
```bash
streamlit run streamlit_rotas_match_app.py
```

5. Acesse no navegador:
```bash
http://localhost:8501
```

---

## 🚀 Deploy

A aplicação pode ser facilmente publicada no Streamlit Cloud.
Certifique-se de que:
- O requirements.txt está na raiz do repositório
- O arquivo de entrada se chama exatamente vagas.json

---

## 📹 Pitch do Projeto

O vídeo de apresentação do projeto explica o problema, a solução e os benefícios. (link do vídeo aqui quando disponível)

---
