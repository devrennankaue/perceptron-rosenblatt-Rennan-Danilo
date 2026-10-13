# Projeto Prático 2.1 — Perceptron de Rosenblatt

Implementação do Perceptron de Rosenblatt para a disciplina de Redes Neurais Artificiais.

A dupla possui identificador **0**, calculado por `(6 + 6) % 4`; assim, a Parte II usa o arquivo `data020262.txt` e a Parte III usa a semente `0`.

## Execução

Na pasta deste repositório, instale as dependências e escolha a parte desejada:

```bash
python3 -m pip install -r requirements.txt
python3 perceptron.py --parte 1
python3 perceptron.py --parte 2
python3 perceptron.py --parte 3
```

Também há um notebook, `PP2_1_Perceptron_Rosenblatt.ipynb`, que apresenta o código, resultados e gráficos.

## Organização

- `perceptron.py`: implementação do algoritmo e das três partes do projeto;
- `PP2_1_Perceptron_Rosenblatt.ipynb`: relatório executável;
- `requirements.txt`: dependências do projeto.

Os arquivos de dados fornecidos ficam em `dados/`.
