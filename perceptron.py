"""Implementação do Perceptron de Rosenblatt para o Projeto Prático 2.1."""

from pathlib import Path
import argparse
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


DADOS = Path(__file__).resolve().parent / "dados"
CORES = np.array(["tab:red", "tab:blue"])
IDENTIFICADOR_EQUIPE = 0


def carregar_dados(nome_arquivo: str) -> Tuple[np.ndarray, np.ndarray]:
    """Lê um arquivo NumPy fornecido e devolve X=(x1,x2) e y=(0,1)."""
    dados = np.load(DADOS / nome_arquivo).reshape(-1, 3)
    return dados[:, :2].astype(float), dados[:, 2].astype(int)


def normalizar(X: np.ndarray, escala: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Divide cada atributo pelo seu maior valor absoluto, levando-o a [-1, 1]."""
    if escala is None:
        escala = np.abs(X).max(axis=0)
    return X / escala, escala


def pesos_escala_original(pesos: np.ndarray, escala: np.ndarray) -> np.ndarray:
    """Converte pesos treinados em X/escala para o espaço original dos atributos."""
    return np.concatenate(([pesos[0]], pesos[1:] / escala))


def degrau(ativacao: np.ndarray) -> np.ndarray:
    """Função degrau com limiar theta = 0."""
    return (ativacao >= 0).astype(int)


def prever(X: np.ndarray, pesos: np.ndarray) -> np.ndarray:
    X_bias = np.column_stack((np.ones(X.shape[0]), X))
    return degrau(X_bias @ pesos)


def treinar_perceptron(
    X: np.ndarray,
    y: np.ndarray,
    taxa_aprendizado: float = 0.1,
    intervalo_pesos: Tuple[float, float] = (-0.5, 0.5),
    max_epocas: Optional[int] = None,
    embaralhar: bool = False,
    semente: int | np.random.Generator | None = None,
    parar_na_convergencia: bool = True,
) -> Dict:
    """Treina pelo algoritmo online de Rosenblatt.

    Cada erro realiza o ajuste w <- w + eta * (yd - y) * x. Quando
    ``max_epocas`` é None, o treinamento só termina ao não haver erros.
    """
    gerador = np.random.default_rng(semente)
    X_bias = np.column_stack((np.ones(X.shape[0]), X))
    pesos = gerador.uniform(*intervalo_pesos, size=X_bias.shape[1])
    pesos_iniciais = pesos.copy()
    ajustes = 0
    erros_por_epoca = []
    epoca = 0

    while max_epocas is None or epoca < max_epocas:
        indices = gerador.permutation(X.shape[0]) if embaralhar else np.arange(X.shape[0])
        erros = 0
        for indice in indices:
            saida = int(X_bias[indice] @ pesos >= 0)
            erro = y[indice] - saida
            if erro:
                pesos += taxa_aprendizado * erro * X_bias[indice]
                ajustes += 1
                erros += 1
        erros_por_epoca.append(erros)
        epoca += 1
        if erros == 0 and parar_na_convergencia:
            break

    return {
        "pesos": pesos,
        "pesos_iniciais": pesos_iniciais,
        "ajustes": ajustes,
        "epocas": epoca,
        "erros_por_epoca": erros_por_epoca,
    }


def plotar_reta(X: np.ndarray, y: np.ndarray, pesos: np.ndarray, ax, titulo: str) -> None:
    """Desenha os exemplos binários e a fronteira de decisão aprendida."""
    ax.scatter(X[:, 0], X[:, 1], c=CORES[y], alpha=0.7, edgecolors="none")
    x_linha = np.array([X[:, 0].min(), X[:, 0].max()])
    if np.isclose(pesos[2], 0):
        ax.axvline(-pesos[0] / pesos[1], color="black", label="Reta de separação")
    else:
        y_linha = -(pesos[0] + pesos[1] * x_linha) / pesos[2]
        ax.plot(x_linha, y_linha, color="black", linewidth=2, label="Reta de separação")
    ax.scatter([], [], color="tab:red", label="Classe 0")
    ax.scatter([], [], color="tab:blue", label="Classe 1")
    ax.set(title=titulo, xlabel="$x_1$", ylabel="$x_2$")
    ax.legend()
    ax.grid(alpha=0.25)


def executar_parte_1() -> Dict:
    """Resolve dataAll20262.txt até a convergência."""
    X, y = carregar_dados("dataAll20262.txt")
    resultado = treinar_perceptron(X, y, taxa_aprendizado=0.1)
    print("Pesos iniciais:", resultado["pesos_iniciais"])
    print("Pesos finais:", resultado["pesos"])
    print("Ajustes:", resultado["ajustes"])
    print("Épocas:", resultado["epocas"])
    print("Acurácia:", f"{accuracy_score(y, prever(X, resultado['pesos'])):.2%}")
    fig, ax = plt.subplots(figsize=(9, 6))
    plotar_reta(X, y, resultado["pesos"], ax, "Parte I: solução do Perceptron")
    plt.show()
    return resultado


def executar_parte_2(identificador: int = IDENTIFICADOR_EQUIPE) -> List[Dict]:
    """Executa os seis cenários exigidos, dez vezes cada, até convergirem."""
    X, y = carregar_dados(f"data{identificador}20262.txt")
    X, _ = normalizar(X)
    configuracoes = [
        (taxa, intervalo)
        for taxa in (0.4, 0.1, 0.01)
        for intervalo in ((-100, 100), (-0.5, 0.5))
    ]
    resumo = []
    for numero, (taxa, intervalo) in enumerate(configuracoes):
        resultados = [
            treinar_perceptron(X, y, taxa, intervalo, semente=1000 * numero + repeticao)
            for repeticao in range(10)
        ]
        ajustes = np.array([resultado["ajustes"] for resultado in resultados])
        epocas = np.array([resultado["epocas"] for resultado in resultados])
        resumo.append({
            "taxa": taxa,
            "intervalo": intervalo,
            "media_ajustes": ajustes.mean(),
            "desvio_ajustes": ajustes.std(ddof=1),
            "min_epocas": epocas.min(),
            "max_epocas": epocas.max(),
            "ajustes": ajustes,
        })
    print("\nParte II — resumo das 10 repetições por configuração")
    print("η       intervalo          ajustes (média ± desvio)   min épocas   max épocas")
    for linha in resumo:
        print(
            f"{linha['taxa']:<7} {str(linha['intervalo']):<18} "
            f"{linha['media_ajustes']:8.2f} ± {linha['desvio_ajustes']:<8.2f} "
            f"{linha['min_epocas']:^12} {linha['max_epocas']:^10}"
        )
    fig, ax = plt.subplots(figsize=(12, 6))
    rotulos = [f"η={linha['taxa']}\nU{linha['intervalo']}" for linha in resumo]
    ax.boxplot([linha["ajustes"] for linha in resumo], tick_labels=rotulos, showmeans=True)
    ax.set(
        title="Parte II: distribuição da quantidade de ajustes",
        xlabel="Configuração",
        ylabel="Quantidade de ajustes",
    )
    ax.grid(axis="y", alpha=0.25)
    plt.show()
    return resumo


def executar_parte_3(identificador: int = IDENTIFICADOR_EQUIPE) -> Dict:
    """Executa a validação holdout 70/30 e treina por exatamente 100 épocas."""
    X, y = carregar_dados("dataHoldout20262.txt")
    gerador = np.random.default_rng(identificador)
    indices = gerador.permutation(X.shape[0])
    corte = int(0.7 * X.shape[0])
    treino, teste = indices[:corte], indices[corte:]
    X_treino, y_treino = X[treino], y[treino]
    X_teste, y_teste = X[teste], y[teste]
    X_treino_norm, escala = normalizar(X_treino)
    X_teste_norm, _ = normalizar(X_teste, escala)
    resultado = treinar_perceptron(
        X_treino_norm, y_treino, 0.1, (-0.5, 0.5), max_epocas=100,
        embaralhar=True, semente=gerador, parar_na_convergencia=False,
    )
    y_previsto = prever(X_teste_norm, resultado["pesos"])
    pesos_originais = pesos_escala_original(resultado["pesos"], escala)
    metricas = {
        "matriz_confusao": confusion_matrix(y_teste, y_previsto, labels=[0, 1]),
        "acuracia": accuracy_score(y_teste, y_previsto),
        "precisao": precision_score(y_teste, y_previsto, zero_division=0),
        "revocacao": recall_score(y_teste, y_previsto, zero_division=0),
        "fscore": f1_score(y_teste, y_previsto, zero_division=0),
    }
    print("Matriz de confusão (linhas: real; colunas: previsto):")
    print(metricas["matriz_confusao"])
    print(f"Acurácia:  {metricas['acuracia']:.2%}")
    print(f"Precisão:  {metricas['precisao']:.2%}")
    print(f"Revocação: {metricas['revocacao']:.2%}")
    print(f"F1-Score:  {metricas['fscore']:.2%}")
    fig, eixos = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
    plotar_reta(X_treino, y_treino, pesos_originais, eixos[0], "Parte III: treinamento")
    plotar_reta(X_teste, y_teste, pesos_originais, eixos[1], "Parte III: teste")
    plt.tight_layout()
    plt.show()
    return {"resultado": resultado, "metricas": metricas, "dados": (X_treino, y_treino, X_teste, y_teste)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Projeto Prático 2.1 — Perceptron de Rosenblatt")
    parser.add_argument("--parte", choices=("1", "2", "3", "todas"), default="1")
    argumentos = parser.parse_args()
    if argumentos.parte in ("1", "todas"):
        executar_parte_1()
    if argumentos.parte in ("2", "todas"):
        executar_parte_2()
    if argumentos.parte in ("3", "todas"):
        executar_parte_3()
